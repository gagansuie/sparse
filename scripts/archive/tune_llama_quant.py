#!/usr/bin/env python3
"""
Tune quantization parameters for Llama-architecture models.
Tests different group sizes, iterations, and strategies.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=30, max_length=512):
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


def quantize_tensor_python(weight, group_size=16, iterations=3, percentile=0.0):
    """
    Quantize a tensor with configurable parameters.
    
    Args:
        weight: Input tensor
        group_size: Number of weights per group
        iterations: Number of scale refinement iterations
        percentile: Percentile clipping (0 = none, 0.01 = 1% outliers)
    """
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    quantized = torch.zeros_like(weight_flat, dtype=torch.uint8)
    scales = torch.zeros(num_groups)
    offsets = torch.zeros(num_groups)
    
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, n)
        group = weight_flat[start:end]
        
        # Apply percentile clipping
        if percentile > 0:
            low = torch.quantile(group, percentile)
            high = torch.quantile(group, 1.0 - percentile)
        else:
            low = group.min()
            high = group.max()
        
        g_min = low.item()
        g_max = high.item()
        
        # Iterative refinement
        for _ in range(iterations):
            scale = (g_max - g_min) / 15.0 if abs(g_max - g_min) > 1e-8 else 1.0
            inv_scale = 1.0 / scale if abs(scale) > 1e-8 else 1.0
            
            # Compute quantization error
            q = ((group - g_min) * inv_scale).round().clamp(0, 15)
            deq = q * scale + g_min
            err = group - deq
            
            # Adjust range based on error
            g_min += err.min().item() * 0.5
            g_max += err.max().item() * 0.5
        
        scale = (g_max - g_min) / 15.0 if abs(g_max - g_min) > 1e-8 else 1.0
        inv_scale = 1.0 / scale if abs(scale) > 1e-8 else 1.0
        
        scales[g] = scale
        offsets[g] = g_min
        
        # Quantize
        q = ((group - g_min) * inv_scale).round().clamp(0, 15).to(torch.uint8)
        quantized[start:end] = q
    
    # Dequantize
    dequantized = torch.zeros_like(weight_flat)
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, n)
        dequantized[start:end] = quantized[start:end].float() * scales[g] + offsets[g]
    
    return dequantized.view_as(weight).to(weight.dtype)


def test_config(model, tokenizer, texts, config_name, group_size, iterations, percentile, baseline_ppl):
    """Test a specific quantization configuration."""
    print(f"\n  Testing: {config_name} (g={group_size}, iter={iterations}, pct={percentile})")
    
    # Quantize all MLP weights
    for layer in model.model.layers:
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, name):
                proj = getattr(layer.mlp, name)
                original = proj.weight.data.clone()
                quantized = quantize_tensor_python(original, group_size, iterations, percentile)
                proj.weight.data = quantized
    
    # Measure PPL
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"    PPL: {ppl:.4f} (Δ {delta:+.2f}%)")
    
    # Restore original weights
    gc.collect()
    
    return ppl, delta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Tuning Quantization for: {args.model}")
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
    texts = [x["text"] for x in ds if x["text"].strip()][:100]
    
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
    
    # Test configurations
    configs = [
        # (name, group_size, iterations, percentile)
        ("g8_iter3", 8, 3, 0.0),
        ("g8_iter5", 8, 5, 0.0),
        ("g8_iter10", 8, 10, 0.0),
        ("g16_iter3", 16, 3, 0.0),
        ("g16_iter5", 16, 5, 0.0),
        ("g16_iter10", 16, 10, 0.0),
        ("g8_iter5_p1", 8, 5, 0.01),    # 1% outlier clipping
        ("g8_iter5_p05", 8, 5, 0.005),  # 0.5% outlier clipping
        ("g4_iter5", 4, 5, 0.0),         # Very small groups
        ("g32_iter5", 32, 5, 0.0),       # Larger groups
    ]
    
    results = []
    
    print("\n" + "=" * 70)
    print("Testing Configurations")
    print("=" * 70)
    
    for name, g, iters, pct in configs:
        # Restore original weights
        for i, layer in enumerate(model.model.layers):
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if proj_name in original_weights[i]:
                    proj = getattr(layer.mlp, proj_name)
                    proj.weight.data = original_weights[i][proj_name].clone()
        
        ppl, delta = test_config(model, tokenizer, texts, name, g, iters, pct, baseline_ppl)
        
        # Calculate compression
        if g == 4:
            compression = 3.2  # 4/2 bytes data + 4/4 bytes scales
        elif g == 8:
            compression = 4.0
        elif g == 16:
            compression = 5.33
        elif g == 32:
            compression = 6.4
        else:
            compression = 5.33
        
        results.append((name, g, iters, pct, ppl, delta, compression))
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Config':<20} {'G':>3} {'Iter':>4} {'Pct':>5} {'PPL':>10} {'Δ':>8} {'Comp':>6}")
    print("-" * 70)
    
    # Sort by PPL delta
    results.sort(key=lambda x: x[5])
    
    for name, g, iters, pct, ppl, delta, comp in results:
        status = "🎯 BEST" if delta == results[0][5] else ""
        print(f"{name:<20} {g:>3} {iters:>4} {pct:>5.3f} {ppl:>10.4f} {delta:>+7.2f}% {comp:>5.2f}x {status}")
    
    print("-" * 70)
    
    best = results[0]
    print(f"\n✅ Best config: {best[0]} (g={best[1]}, iter={best[2]}, pct={best[3]})")
    print(f"   PPL Delta: {best[5]:+.2f}%")
    print(f"   Compression: {best[6]:.2f}x")


if __name__ == "__main__":
    main()
