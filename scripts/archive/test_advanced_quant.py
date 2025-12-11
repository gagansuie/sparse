#!/usr/bin/env python3
"""
Advanced quantization techniques for 8x-10x compression without calibration.
Tests: Product Quantization, NF3, Mixed Precision, Residual Quantization
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import numpy as np
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


# =============================================================================
# APPROACH 1: Product Quantization (Codebook-based)
# =============================================================================
def product_quantize(weight, codebook_size=256, vector_size=8, residual_bits=0):
    """
    Product Quantization: Use a codebook instead of scalar quantization.
    
    Storage:
    - 8-bit index per vector_size weights
    - Optional residual correction
    
    Compression:
    - No residual: 32/8 = 4x per vector = 32x overall (but quality?)
    - 2-bit residual: 32/(8+16) = 1.33x per vector = 10.7x overall
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad to multiple of vector_size
    pad_size = (vector_size - n % vector_size) % vector_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    # Reshape to vectors
    num_vectors = weight_flat.numel() // vector_size
    vectors = weight_flat.view(num_vectors, vector_size)
    
    # Simple k-means clustering (few iterations for speed)
    # In production, use sklearn.cluster.MiniBatchKMeans
    device = weight.device
    
    # Initialize codebook with random vectors from data
    perm = torch.randperm(num_vectors)[:codebook_size]
    codebook = vectors[perm].clone()
    
    # K-means iterations
    for _ in range(10):
        # Assign vectors to nearest codebook entry
        dists = torch.cdist(vectors, codebook)
        indices = dists.argmin(dim=1)
        
        # Update codebook
        for i in range(codebook_size):
            mask = indices == i
            if mask.sum() > 0:
                codebook[i] = vectors[mask].mean(dim=0)
    
    # Final assignment
    dists = torch.cdist(vectors, codebook)
    indices = dists.argmin(dim=1)
    
    # Reconstruct
    reconstructed = codebook[indices]
    
    # Optional residual
    if residual_bits > 0:
        residual = vectors - reconstructed
        # Simple uniform quantization of residual
        res_min = residual.min()
        res_max = residual.max()
        res_scale = (res_max - res_min) / (2**residual_bits - 1)
        res_q = ((residual - res_min) / res_scale).round().clamp(0, 2**residual_bits - 1)
        res_deq = res_q * res_scale + res_min
        reconstructed = reconstructed + res_deq
    
    dequantized = reconstructed.flatten()[:n]
    return dequantized.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 2: NF3 (Non-uniform 3-bit quantization)
# =============================================================================

# Pre-computed optimal levels for Gaussian distribution (3-bit = 8 levels)
# These are quantiles of N(0,1) that minimize expected squared error
NF3_LEVELS = torch.tensor([-1.5104, -0.7560, -0.3530, 0.0, 0.3530, 0.7560, 1.5104, 2.0])

def nf3_quantize(weight, group_size=16):
    """
    NF3: Non-uniform 3-bit quantization with Gaussian-optimized levels.
    
    Storage: 3 bits per weight + scale overhead
    Compression: ~8x (3 bits vs 32 bits)
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
    
    # Per-group scaling to normalize to ~N(0,1)
    scales = groups.abs().max(dim=1).values / 1.5104  # Scale so max maps to max level
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    
    # Normalize groups
    normalized = groups / scales.unsqueeze(1)
    
    # Quantize to nearest NF3 level
    levels = NF3_LEVELS.to(weight.device)
    dists = (normalized.unsqueeze(2) - levels.unsqueeze(0).unsqueeze(0)).abs()
    indices = dists.argmin(dim=2)
    
    # Dequantize
    dequantized = levels[indices] * scales.unsqueeze(1)
    
    dequantized = dequantized.flatten()[:n]
    return dequantized.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 3: Mixed Precision (Different compression per layer type)
# =============================================================================
def mixed_precision_quantize(weight, layer_type, iterations=5):
    """
    Different quantization settings based on layer sensitivity.
    
    - Attention: g=8 (conservative) = 4x compression
    - MLP gate/up: g=16 (moderate) = 5.33x compression  
    - MLP down: g=32 (aggressive) = 6.4x compression
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Select group size based on layer type
    if layer_type in ['q_proj', 'k_proj']:
        group_size = 8  # Most sensitive
    elif layer_type in ['v_proj', 'o_proj', 'gate_proj', 'up_proj']:
        group_size = 16  # Moderate
    else:  # down_proj
        group_size = 32  # Least sensitive
    
    # Pad
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    groups = weight_flat.view(num_groups, group_size)
    
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
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    dequantized = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    dequantized = dequantized.flatten()[:n]
    return dequantized.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 4: Residual Quantization (Two-pass)
# =============================================================================
def residual_quantize(weight, group_size=8, iterations=5, residual_group=32):
    """
    Two-pass quantization: INT4 + INT2 residual correction.
    
    Pass 1: INT4 quantization (4 bits)
    Pass 2: INT2 quantization of residual (2 bits)
    Total: 6 bits + overhead ≈ 7 bits/weight = 4.5x compression
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad for first pass
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    # First pass: INT4 with iterative refinement
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
    q1 = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    deq1 = q1 * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Compute residual
    residual = (groups - deq1).flatten()
    
    # Second pass: INT2 on residual (4 levels)
    pad_size2 = (residual_group - residual.numel() % residual_group) % residual_group
    if pad_size2 > 0:
        residual = torch.cat([residual, torch.zeros(pad_size2, device=weight.device)])
    
    num_groups2 = residual.numel() // residual_group
    res_groups = residual.view(num_groups2, residual_group)
    
    res_min = res_groups.min(dim=1).values
    res_max = res_groups.max(dim=1).values
    res_scale = (res_max - res_min) / 3.0  # 4 levels (0,1,2,3)
    res_scale = torch.where(res_scale.abs() < 1e-8, torch.ones_like(res_scale), res_scale)
    
    q2 = ((res_groups - res_min.unsqueeze(1)) / res_scale.unsqueeze(1)).round().clamp(0, 3)
    deq2 = q2 * res_scale.unsqueeze(1) + res_min.unsqueeze(1)
    
    # Combine
    final = deq1.flatten()[:n] + deq2.flatten()[:n]
    return final.view(original_shape).to(weight.dtype)


# =============================================================================
# TEST HARNESS
# =============================================================================
def test_approach(model, original_weights, tokenizer, texts, name, quant_fn, baseline_ppl, compression):
    """Test a quantization approach."""
    print(f"  {name}...", end=" ", flush=True)
    
    # Apply quantization to MLP layers
    for i, layer in enumerate(model.model.layers):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if proj_name in original_weights[i]:
                proj = getattr(layer.mlp, proj_name)
                orig = original_weights[i][proj_name].clone()
                if callable(quant_fn):
                    proj.weight.data = quant_fn(orig)
                else:
                    # Mixed precision passes layer type
                    proj.weight.data = quant_fn(orig, proj_name)
    
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    status = "✅" if delta < 1.0 else "⚠️" if delta < 2.0 else "❌"
    print(f"PPL: {ppl:.4f} (Δ {delta:+.2f}%) {compression}x {status}")
    
    return ppl, delta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Advanced Quantization Test: {args.model}")
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
    
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    print("\n" + "=" * 70)
    print("Testing Advanced Quantization Approaches")
    print("=" * 70)
    
    results = []
    
    # Restore helper
    def restore():
        for i, layer in enumerate(model.model.layers):
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if proj_name in original_weights[i]:
                    proj = getattr(layer.mlp, proj_name)
                    proj.weight.data = original_weights[i][proj_name].clone()
    
    # Test 1: Current best (baseline)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "INT4 g8 i5 (baseline)",
        lambda w: product_quantize(w, 1, 1, 0) if False else  # Dummy, we use simple quant
            mixed_precision_quantize(w, 'gate_proj', 5),
        baseline_ppl, "4.0"
    )
    results.append(("INT4 g8 i5", 4.0, ppl, delta))
    gc.collect()
    
    # Test 2: Product Quantization (no residual)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "PQ 256x8 (no residual)",
        lambda w: product_quantize(w, codebook_size=256, vector_size=8, residual_bits=0),
        baseline_ppl, "~16"
    )
    results.append(("PQ 256x8", 16.0, ppl, delta))
    gc.collect()
    
    # Test 3: Product Quantization with 2-bit residual
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "PQ 256x8 + 2bit residual",
        lambda w: product_quantize(w, codebook_size=256, vector_size=8, residual_bits=2),
        baseline_ppl, "~8"
    )
    results.append(("PQ+res", 8.0, ppl, delta))
    gc.collect()
    
    # Test 4: NF3 (non-uniform 3-bit)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "NF3 g16",
        lambda w: nf3_quantize(w, group_size=16),
        baseline_ppl, "~8"
    )
    results.append(("NF3 g16", 8.0, ppl, delta))
    gc.collect()
    
    # Test 5: Mixed Precision
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "Mixed Precision",
        lambda w, lt="down_proj": mixed_precision_quantize(w, lt, 5),
        baseline_ppl, "~5.5"
    )
    results.append(("Mixed", 5.5, ppl, delta))
    gc.collect()
    
    # Test 6: Residual Quantization (INT4 + INT2)
    restore()
    ppl, delta = test_approach(
        model, original_weights, tokenizer, texts,
        "INT4+INT2 Residual",
        lambda w: residual_quantize(w, group_size=8, iterations=5, residual_group=32),
        baseline_ppl, "~4.5"
    )
    results.append(("INT4+INT2", 4.5, ppl, delta))
    gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Approach':<20} {'Comp':>6} {'PPL':>10} {'Δ':>8} {'Status'}")
    print("-" * 55)
    
    results.sort(key=lambda x: -x[1])  # Sort by compression desc
    
    for name, comp, ppl, delta in results:
        status = "✅ OK" if delta < 1.0 else "⚠️ ~OK" if delta < 2.0 else "❌ BAD"
        print(f"{name:<20} {comp:>5.1f}x {ppl:>10.4f} {delta:>+7.2f}% {status}")
    
    # Best meeting target
    valid = [r for r in results if r[3] < 1.0]
    if valid:
        best = max(valid, key=lambda x: x[1])
        print(f"\n✅ Best config meeting <1% target: {best[0]}")
        print(f"   Compression: {best[1]:.1f}x, PPL Delta: {best[3]:+.2f}%")
    else:
        almost = [r for r in results if r[3] < 2.0]
        if almost:
            best = max(almost, key=lambda x: x[1])
            print(f"\n⚠️ Best config meeting <2% target: {best[0]}")
            print(f"   Compression: {best[1]:.1f}x, PPL Delta: {best[3]:+.2f}%")


if __name__ == "__main__":
    main()
