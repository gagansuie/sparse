#!/usr/bin/env python3
"""
Test sparse + quantized compression for higher compression ratios.
Target: 8x-10x compression with <1% PPL delta.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=20, max_length=256):
    """Compute perplexity."""
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


def sparse_quantize(weight, group_size=8, iterations=5, sparsity=0.0):
    """
    Sparse + quantized compression.
    
    Args:
        weight: Input tensor
        group_size: Quantization group size
        iterations: Scale refinement iterations
        sparsity: Fraction of weights to prune (0.0 = none, 0.5 = 50%)
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Apply sparsity - prune smallest magnitude weights
    if sparsity > 0:
        threshold = torch.quantile(weight_flat.abs(), sparsity)
        mask = weight_flat.abs() >= threshold
        weight_flat = weight_flat * mask.float()
    
    # Pad to multiple of group_size
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    # Reshape to [num_groups, group_size]
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
    # Get min/max per group
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    
    # Iterative refinement
    for _ in range(iterations):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        
        g_min = g_min + err.min(dim=1).values * 0.5
        g_max = g_max + err.max(dim=1).values * 0.5
    
    # Final quantization
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    dequantized = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    dequantized = dequantized.flatten()[:n]
    return dequantized.view(original_shape).to(weight.dtype)


def test_config(model, original_weights, tokenizer, texts, name, group_size, iterations, sparsity, baseline_ppl):
    """Test a configuration."""
    print(f"  {name}...", end=" ", flush=True)
    
    # Restore and apply quantization
    for i, layer in enumerate(model.model.layers):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if proj_name in original_weights[i]:
                proj = getattr(layer.mlp, proj_name)
                orig = original_weights[i][proj_name].clone()
                proj.weight.data = sparse_quantize(orig, group_size, iterations, sparsity)
    
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    # Calculate compression
    # Base: 4 bits/weight + overhead for g=8
    # Sparsity reduces effective bits
    base_bits = 4 + (32 / group_size)  # 4 bits + scale overhead
    effective_bits = base_bits * (1 - sparsity * 0.5)  # Sparse weights still need index
    compression = 32 / effective_bits
    
    status = "✅" if delta < 1.0 else "⚠️" if delta < 2.0 else "❌"
    print(f"PPL: {ppl:.4f} (Δ {delta:+.2f}%) ~{compression:.1f}x {status}")
    
    return ppl, delta, compression


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Sparse + Quantized Compression Test: {args.model}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    # Baseline
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Test configurations
    configs = [
        # (name, group_size, iterations, sparsity)
        ("g8_i5_s0", 8, 5, 0.0),      # Current best (4x)
        ("g8_i5_s10", 8, 5, 0.10),    # 10% sparsity
        ("g8_i5_s20", 8, 5, 0.20),    # 20% sparsity  
        ("g8_i5_s30", 8, 5, 0.30),    # 30% sparsity
        ("g8_i5_s40", 8, 5, 0.40),    # 40% sparsity
        ("g8_i5_s50", 8, 5, 0.50),    # 50% sparsity (~8x target)
        ("g16_i5_s30", 16, 5, 0.30),  # Larger groups + sparsity
        ("g16_i5_s50", 16, 5, 0.50),  # Higher compression target
    ]
    
    print("\n" + "=" * 70)
    print("Testing Configurations (Target: <1% PPL delta)")
    print("=" * 70)
    
    results = []
    for name, g, iters, sparsity in configs:
        ppl, delta, comp = test_config(
            model, original_weights, tokenizer, texts,
            name, g, iters, sparsity, baseline_ppl
        )
        results.append((name, g, sparsity, ppl, delta, comp))
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by compression)")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Config':<15} {'Sparse':>6} {'PPL':>10} {'Δ':>8} {'Comp':>6} {'Status'}")
    print("-" * 60)
    
    results.sort(key=lambda x: -x[5])  # Sort by compression desc
    
    for name, g, sparsity, ppl, delta, comp in results:
        status = "✅ OK" if delta < 1.0 else "⚠️ ~OK" if delta < 2.0 else "❌ BAD"
        print(f"{name:<15} {sparsity*100:>5.0f}% {ppl:>10.4f} {delta:>+7.2f}% {comp:>5.1f}x {status}")
    
    # Find best config meeting <1% target
    valid = [r for r in results if r[4] < 1.0]
    if valid:
        best = max(valid, key=lambda x: x[5])
        print(f"\n✅ Best config meeting <1% target: {best[0]}")
        print(f"   Compression: {best[5]:.1f}x, PPL Delta: {best[4]:+.2f}%")
    else:
        print("\n⚠️ No config achieved <1% PPL delta")


if __name__ == "__main__":
    main()
