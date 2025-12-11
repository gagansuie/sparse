#!/usr/bin/env python3
"""
Fast quantization tuning using vectorized PyTorch operations.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=20, max_length=256):
    """Compute perplexity (fast version with fewer samples)."""
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts[:max_samples]:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def quantize_tensor_fast(weight, group_size=16, iterations=3, percentile=0.0):
    """
    Fast vectorized quantization.
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad to multiple of group_size
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    # Reshape to [num_groups, group_size]
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
    # Get min/max per group (with optional percentile clipping)
    if percentile > 0:
        g_min = torch.quantile(groups, percentile, dim=1)
        g_max = torch.quantile(groups, 1.0 - percentile, dim=1)
    else:
        g_min = groups.min(dim=1).values
        g_max = groups.max(dim=1).values
    
    # Iterative refinement (vectorized)
    for _ in range(iterations):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        
        # Quantize
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        
        # Adjust range
        g_min = g_min + err.min(dim=1).values * 0.5
        g_max = g_max + err.max(dim=1).values * 0.5
    
    # Final quantization
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    dequantized = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Unpad and reshape
    dequantized = dequantized.flatten()[:n]
    return dequantized.view(original_shape).to(weight.dtype)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Fast Quantization Tuning: {args.model}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:50]
    
    # Save original weights
    print("Saving original weights...")
    original_weights = {}
    for i, layer in enumerate(model.model.layers):
        original_weights[i] = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, name):
                proj = getattr(layer.mlp, name)
                original_weights[i][name] = proj.weight.data.clone()
    
    # Baseline PPL
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Test configurations (reduced set for speed)
    configs = [
        # (name, group_size, iterations, percentile)
        ("g8_i3", 8, 3, 0.0),
        ("g8_i5", 8, 5, 0.0),
        ("g8_i10", 8, 10, 0.0),
        ("g16_i3", 16, 3, 0.0),  # Current int4_opt_v1
        ("g16_i5", 16, 5, 0.0),
        ("g16_i10", 16, 10, 0.0),
        ("g8_i5_p1", 8, 5, 0.01),
        ("g4_i5", 4, 5, 0.0),
    ]
    
    results = []
    
    print("\n" + "=" * 70)
    print("Testing Configurations")
    print("=" * 70)
    
    for name, g, iters, pct in configs:
        print(f"\n  {name} (g={g}, iter={iters}, pct={pct})...", end=" ", flush=True)
        
        # Restore original weights
        for i, layer in enumerate(model.model.layers):
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if proj_name in original_weights[i]:
                    proj = getattr(layer.mlp, proj_name)
                    proj.weight.data = original_weights[i][proj_name].clone()
        
        # Quantize
        for layer in model.model.layers:
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    proj.weight.data = quantize_tensor_fast(proj.weight.data, g, iters, pct)
        
        # Measure PPL
        ppl = compute_ppl(model, tokenizer, texts)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"PPL: {ppl:.4f} (Δ {delta:+.2f}%)")
        
        # Compression ratio
        if g == 4:
            compression = 3.2
        elif g == 8:
            compression = 4.0
        elif g == 16:
            compression = 5.33
        elif g == 32:
            compression = 6.4
        else:
            compression = 5.33
        
        results.append((name, g, iters, pct, ppl, delta, compression))
        gc.collect()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Config':<15} {'G':>3} {'Iter':>4} {'PPL':>10} {'Δ':>8} {'Comp':>6}")
    print("-" * 60)
    
    results.sort(key=lambda x: x[5])
    
    for name, g, iters, pct, ppl, delta, comp in results:
        status = "🎯" if delta == results[0][5] else ""
        print(f"{name:<15} {g:>3} {iters:>4} {ppl:>10.4f} {delta:>+7.2f}% {comp:>5.2f}x {status}")
    
    best = results[0]
    print(f"\n✅ Best: {best[0]} with {best[5]:+.2f}% PPL delta at {best[6]:.2f}x compression")


if __name__ == "__main__":
    main()
