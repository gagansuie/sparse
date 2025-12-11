#!/usr/bin/env python3
"""
Residual Quantization: Push compression higher with two-pass approach.
INT4 + INT2 = 6 bits effective ≈ 5.3x compression
INT4 + INT1 = 5 bits effective ≈ 6.4x compression
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


def residual_quantize_optimized(weight, 
                                 group_size=8, 
                                 iterations=5,
                                 residual_bits=2,
                                 residual_group=16):
    """
    Optimized two-pass residual quantization.
    
    Pass 1: INT4 with iterative refinement (best quality)
    Pass 2: INTx residual correction
    
    Compression calculation:
    - INT4: 4 bits + (32/group_size) overhead bits
    - Residual: residual_bits + overhead
    
    g=8, res_bits=2, res_g=16:
      Main: 4 + 4 = 8 bits per weight (4x)
      Residual: 2 + 2 = 4 bits per weight (8x)
      Combined: effectively 4 + 0.5 = 4.5 bits ≈ 7x? No...
      
    Actually:
      Total bits = main_bits + residual_bits with shared overhead
      = 4 (packed) + 4/8 (scale) + 4/8 (offset) + 2 (res packed) + 4/16 (res scale)
      = 4 + 0.5 + 0.5 + 2 + 0.25 = 7.25 bits → 4.4x compression
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad for first pass
    pad_size = (group_size - n % group_size) % group_size
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
    
    # ===== PASS 1: INT4 with iterative refinement =====
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
    
    # ===== PASS 2: Residual quantization =====
    residual = (groups - deq1).flatten()
    
    # Pad for residual groups
    pad_size2 = (residual_group - residual.numel() % residual_group) % residual_group
    if pad_size2 > 0:
        residual = torch.cat([residual, torch.zeros(pad_size2, device=weight.device)])
    
    num_groups2 = residual.numel() // residual_group
    res_groups = residual.view(num_groups2, residual_group)
    
    res_min = res_groups.min(dim=1).values
    res_max = res_groups.max(dim=1).values
    
    levels = 2**residual_bits - 1
    res_scale = (res_max - res_min) / levels
    res_scale = torch.where(res_scale.abs() < 1e-8, torch.ones_like(res_scale), res_scale)
    
    q2 = ((res_groups - res_min.unsqueeze(1)) / res_scale.unsqueeze(1)).round().clamp(0, levels)
    deq2 = q2 * res_scale.unsqueeze(1) + res_min.unsqueeze(1)
    
    # ===== Combine =====
    final = deq1.flatten()[:n] + deq2.flatten()[:n]
    return final.view(original_shape).to(weight.dtype)


def calculate_compression(group_size, residual_bits, residual_group):
    """Calculate actual compression ratio."""
    # Main quantization: 4 bits packed + FP16 scale + FP16 offset per group
    main_bits_per_weight = 4 + (16 / group_size) + (16 / group_size)  # 4 + 2 + 2 = 8 for g=8
    
    # Residual: residual_bits packed + FP16 scale per residual_group
    res_bits_per_weight = residual_bits + (16 / residual_group)
    
    # Total
    total_bits = main_bits_per_weight + res_bits_per_weight
    compression = 32 / total_bits
    
    return compression, total_bits


def test_config(model, original_weights, tokenizer, texts, 
                name, group_size, iterations, res_bits, res_group, baseline_ppl):
    """Test a residual quantization configuration."""
    comp, bits = calculate_compression(group_size, res_bits, res_group)
    print(f"  {name} ({bits:.1f} bits)...", end=" ", flush=True)
    
    for i, layer in enumerate(model.model.layers):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            if proj_name in original_weights[i]:
                proj = getattr(layer.mlp, proj_name)
                orig = original_weights[i][proj_name].clone()
                proj.weight.data = residual_quantize_optimized(
                    orig, group_size, iterations, res_bits, res_group
                )
    
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    status = "✅" if delta < 1.0 else "⚠️" if delta < 2.0 else "❌"
    print(f"PPL: {ppl:.4f} (Δ {delta:+.2f}%) {comp:.2f}x {status}")
    
    return ppl, delta, comp, bits


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Residual Quantization Test: {args.model}")
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
    print("Testing Residual Quantization Configurations")
    print("=" * 70)
    
    configs = [
        # (name, group_size, iterations, residual_bits, residual_group)
        ("g8_i5 (no res)", 8, 5, 0, 16),         # Baseline: ~4x
        ("g8_i5_r2_rg16", 8, 5, 2, 16),          # INT4 + INT2
        ("g8_i5_r2_rg32", 8, 5, 2, 32),          # INT4 + INT2 (larger res groups)
        ("g8_i5_r1_rg16", 8, 5, 1, 16),          # INT4 + INT1 (more aggressive)
        ("g8_i5_r1_rg32", 8, 5, 1, 32),          # INT4 + INT1 (larger res groups)
        ("g16_i5_r2_rg16", 16, 5, 2, 16),        # Larger main groups
        ("g16_i5_r2_rg32", 16, 5, 2, 32),        # Larger main + res groups
        ("g16_i5_r1_rg32", 16, 5, 1, 32),        # Most aggressive
    ]
    
    results = []
    
    for name, g, i, rb, rg in configs:
        restore()
        if rb == 0:
            # No residual - just regular quantization
            ppl, delta, comp, bits = test_config(
                model, original_weights, tokenizer, texts,
                name, g, i, 0, 16, baseline_ppl
            )
        else:
            ppl, delta, comp, bits = test_config(
                model, original_weights, tokenizer, texts,
                name, g, i, rb, rg, baseline_ppl
            )
        results.append((name, comp, bits, ppl, delta))
        gc.collect()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (sorted by compression)")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Config':<20} {'Bits':>5} {'Comp':>6} {'PPL':>10} {'Δ':>8} {'Status'}")
    print("-" * 65)
    
    results.sort(key=lambda x: -x[1])  # Sort by compression desc
    
    for name, comp, bits, ppl, delta in results:
        status = "✅ OK" if delta < 1.0 else "⚠️ ~OK" if delta < 2.0 else "❌ BAD"
        print(f"{name:<20} {bits:>5.1f} {comp:>5.2f}x {ppl:>10.4f} {delta:>+7.2f}% {status}")
    
    # Find best
    valid = [r for r in results if r[4] < 1.0]
    if valid:
        best = max(valid, key=lambda x: x[1])
        print(f"\n✅ Best config meeting <1% target: {best[0]}")
        print(f"   Compression: {best[1]:.2f}x ({best[2]:.1f} bits), PPL Delta: {best[4]:+.2f}%")
    
    almost = [r for r in results if 1.0 <= r[4] < 2.0]
    if almost:
        best_almost = max(almost, key=lambda x: x[1])
        print(f"\n⚠️ Best config at <2% target: {best_almost[0]}")
        print(f"   Compression: {best_almost[1]:.2f}x ({best_almost[2]:.1f} bits), PPL Delta: {best_almost[4]:+.2f}%")


if __name__ == "__main__":
    main()
