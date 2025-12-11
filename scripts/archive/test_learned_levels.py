#!/usr/bin/env python3
"""
Creative Approach #2: Learned Optimal Quantization Levels

Instead of uniform INT4 (0-15), learn optimal quantization points
from the actual weight distribution using Lloyd's algorithm (k-means on 1D).

This is similar to NF4 (normalized float 4-bit) but:
1. Per-tensor levels instead of global
2. Adaptive to actual distribution, not assuming Gaussian
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=20, max_length=256):
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


def lloyd_1d(data, num_levels, iterations=20):
    """
    Lloyd's algorithm for optimal 1D quantization.
    Finds num_levels centroids that minimize MSE.
    """
    data = data.flatten().float()
    
    # Initialize with quantiles
    quantiles = torch.linspace(0, 1, num_levels + 2)[1:-1]
    levels = torch.quantile(data, quantiles)
    
    for _ in range(iterations):
        # Assign each point to nearest level
        dists = (data.unsqueeze(1) - levels.unsqueeze(0)).abs()
        assignments = dists.argmin(dim=1)
        
        # Update levels to mean of assigned points
        new_levels = torch.zeros_like(levels)
        for i in range(num_levels):
            mask = assignments == i
            if mask.sum() > 0:
                new_levels[i] = data[mask].mean()
            else:
                new_levels[i] = levels[i]
        
        # Check convergence
        if (new_levels - levels).abs().max() < 1e-6:
            break
        levels = new_levels
    
    return levels.sort().values


def learned_quantize(weight, num_levels=16, group_size=16, lloyd_iters=10):
    """
    Quantize using learned optimal levels per group.
    
    Storage per group:
    - num_levels centroids × 2 bytes (FP16) = 32 bytes for 16 levels
    - N/group_size indices × log2(num_levels) bits
    
    For num_levels=16, 4 bits per index (same as INT4)
    But levels are optimal, not uniform!
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
    dequantized = torch.zeros_like(groups)
    
    for g in range(num_groups):
        group = groups[g]
        
        # Learn optimal levels for this group
        levels = lloyd_1d(group, num_levels, lloyd_iters)
        
        # Quantize to nearest level
        dists = (group.unsqueeze(1) - levels.unsqueeze(0)).abs()
        indices = dists.argmin(dim=1)
        
        # Dequantize
        dequantized[g] = levels[indices]
    
    result = dequantized.flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


def learned_quantize_global(weight, num_levels=16, group_size=16, lloyd_iters=20):
    """
    Learn levels globally (per-tensor), then apply per-group scaling.
    
    More efficient storage: only one codebook per tensor.
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Learn global levels from a sample of the weights
    sample_size = min(100000, n)
    sample_indices = torch.randperm(n)[:sample_size]
    sample = weight_flat[sample_indices]
    
    # Normalize to [-1, 1] range
    sample_std = sample.std()
    sample_norm = sample / (sample_std + 1e-8)
    
    # Learn levels on normalized data
    global_levels = lloyd_1d(sample_norm, num_levels, lloyd_iters)
    
    # Pad
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
    dequantized = torch.zeros_like(groups)
    
    for g in range(num_groups):
        group = groups[g]
        
        # Per-group scale
        g_std = group.std()
        g_norm = group / (g_std + 1e-8)
        
        # Quantize using global levels
        dists = (g_norm.unsqueeze(1) - global_levels.unsqueeze(0)).abs()
        indices = dists.argmin(dim=1)
        
        # Dequantize with per-group scale
        dequantized[g] = global_levels[indices] * g_std
    
    result = dequantized.flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


def iterative_int4(weight, group_size=8, iterations=5):
    """Our current best: iterative scale refinement (baseline)."""
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    
    for _ in range(iterations):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(dim=1).values * 0.5
        g_max = g_max + err.max(dim=1).values * 0.5
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    dequantized = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    result = dequantized.flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


def test_approach(model, original_weights, tokenizer, texts, name, quant_fn, baseline_ppl):
    """Test a quantization approach."""
    print(f"  {name}...", end=" ", flush=True)
    
    for i, layer in enumerate(model.model.layers):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if proj_name in original_weights[i]:
                proj = getattr(layer.mlp, proj_name)
                orig = original_weights[i][proj_name].clone()
                proj.weight.data = quant_fn(orig)
    
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    status = "✅" if delta < 1.0 else "⚠️" if delta < 2.0 else "❌"
    print(f"PPL: {ppl:.4f} (Δ {delta:+.2f}%) {status}")
    
    return ppl, delta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Learned Quantization Levels Test: {args.model}")
    print("=" * 70)
    
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
    
    print("Saving original weights...")
    original_weights = {}
    for i, layer in enumerate(model.model.layers):
        original_weights[i] = {}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, name):
                proj = getattr(layer.mlp, name)
                original_weights[i][name] = proj.weight.data.clone()
    
    def restore():
        for i, layer in enumerate(model.model.layers):
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if proj_name in original_weights[i]:
                    proj = getattr(layer.mlp, proj_name)
                    proj.weight.data = original_weights[i][proj_name].clone()
    
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    print("\n" + "=" * 70)
    print("Testing Learned Quantization Approaches")
    print("=" * 70)
    
    results = []
    
    # Baseline
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "INT4 g8 i5 (baseline)",
        lambda w: iterative_int4(w, 8, 5),
        baseline_ppl
    )
    results.append(("INT4 baseline", ppl, delta))
    gc.collect()
    
    # Learned levels (per-group)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "Learned 16-levels g16",
        lambda w: learned_quantize(w, 16, 16, 10),
        baseline_ppl
    )
    results.append(("Learned g16", ppl, delta))
    gc.collect()
    
    # Learned levels (global + scale)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "Global learned + scale",
        lambda w: learned_quantize_global(w, 16, 16, 20),
        baseline_ppl
    )
    results.append(("Global learned", ppl, delta))
    gc.collect()
    
    # 8 levels (INT3 equivalent but optimal)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "Learned 8-levels g16 (INT3)",
        lambda w: learned_quantize(w, 8, 16, 10),
        baseline_ppl
    )
    results.append(("Learned 8-lvl", ppl, delta))
    gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Approach':<25} {'PPL':>10} {'Δ':>8} {'Status'}")
    print("-" * 50)
    
    for name, ppl, delta in results:
        status = "✅ OK" if delta < 1.0 else "⚠️ ~OK" if delta < 2.0 else "❌ BAD"
        print(f"{name:<25} {ppl:>10.4f} {delta:>+7.2f}% {status}")


if __name__ == "__main__":
    main()
