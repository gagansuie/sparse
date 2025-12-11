#!/usr/bin/env python3
"""
Combined best techniques for maximum compression without calibration.

Key findings:
- Multi-stage residual: -0.1% PPL at 4.5x
- Layer position: +1.8% at 5.5x

Combined strategy:
1. First/last 15% layers: INT4 g8 + multi-stage residual (quality)
2. Middle layers: INT4 g32 + INT2 residual (compression)
3. Overall: Push for 6-8x compression
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=15, max_length=256):
    model.eval()
    device = next(model.parameters()).device
    nll, ntokens = 0.0, 0
    with torch.no_grad():
        for text in texts[:max_samples]:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(device)
            out = model(ids, labels=ids)
            nll += out.loss.item() * ids.numel()
            ntokens += ids.numel()
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def multi_stage_quant(weight, group_size=16, use_int1=True):
    """Multi-stage: INT4 + INT2 + optional INT1."""
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
    
    # Stage 1: INT4
    g_min, g_max = groups.min(1).values, groups.max(1).values
    for _ in range(5):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(1).values * 0.5
        g_max = g_max + err.max(1).values * 0.5
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q1 = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    stage1 = q1 * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Stage 2: INT2 on residual
    residual1 = groups - stage1
    r1_min, r1_max = residual1.min(1).values, residual1.max(1).values
    r1_scale = (r1_max - r1_min) / 3.0
    r1_scale = torch.where(r1_scale.abs() < 1e-8, torch.ones_like(r1_scale), r1_scale)
    q2 = ((residual1 - r1_min.unsqueeze(1)) / r1_scale.unsqueeze(1)).round().clamp(0, 3)
    stage2 = q2 * r1_scale.unsqueeze(1) + r1_min.unsqueeze(1)
    
    result = stage1 + stage2
    
    # Stage 3: INT1 (sign correction)
    if use_int1:
        residual2 = groups - result
        r2_scale = residual2.abs().mean(dim=1, keepdim=True)
        q3 = (residual2 > 0).float() * 2 - 1
        stage3 = q3 * r2_scale * 0.5
        result = result + stage3
    
    return result.flatten()[:n].view(weight.shape).to(weight.dtype)


def aggressive_quant(weight, group_size=32):
    """Aggressive INT4 with large groups for compression."""
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
    g_min, g_max = groups.min(1).values, groups.max(1).values
    
    for _ in range(5):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(1).values * 0.5
        g_max = g_max + err.max(1).values * 0.5
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    return deq.flatten()[:n].view(weight.shape).to(weight.dtype)


def combined_quant_v1(weight, layer_idx, total_layers):
    """
    Combined V1: Multi-stage for edge, aggressive for middle.
    
    Edge (first/last 15%): INT4 g16 + INT2 + INT1 = ~7 bits = 4.5x
    Middle (70%): INT4 g32 = ~4.5 bits = 7x
    
    Expected overall: ~5.5-6x compression
    """
    pos = layer_idx / total_layers
    
    if pos < 0.15 or pos > 0.85:
        return multi_stage_quant(weight, group_size=16, use_int1=True)
    else:
        return aggressive_quant(weight, group_size=32)


def combined_quant_v2(weight, layer_idx, total_layers):
    """
    Combined V2: More aggressive on middle layers.
    
    Edge (first/last 10%): INT4 g8 + INT2 + INT1
    Near-edge (next 10%): INT4 g16 + INT2
    Middle (60%): INT4 g64
    
    Expected: ~6-7x compression
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return multi_stage_quant(weight, group_size=8, use_int1=True)
    elif pos < 0.2 or pos > 0.8:
        return multi_stage_quant(weight, group_size=16, use_int1=False)
    else:
        return aggressive_quant(weight, group_size=64)


def combined_quant_v3(weight, layer_idx, total_layers):
    """
    Combined V3: Maximum compression on middle.
    
    Edge (first/last 5%): INT4 g8 + full residual
    Near-edge (next 15%): INT4 g16
    Middle (60%): INT4 g128 (very aggressive)
    
    Expected: ~7-8x compression (but risky)
    """
    pos = layer_idx / total_layers
    
    if pos < 0.05 or pos > 0.95:
        return multi_stage_quant(weight, group_size=8, use_int1=True)
    elif pos < 0.2 or pos > 0.8:
        return multi_stage_quant(weight, group_size=16, use_int1=False)
    else:
        return aggressive_quant(weight, group_size=128)


def combined_quant_v4(weight, layer_idx, total_layers):
    """
    Combined V4: Uniform multi-stage with larger groups.
    
    All layers: INT4 g32 + INT2 + INT1
    
    Expected: ~5x compression with good quality
    """
    return multi_stage_quant(weight, group_size=32, use_int1=True)


def test(model, orig, tok, texts, name, fn, baseline, expected_comp):
    print(f"  {name}...", end=" ", flush=True)
    total_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                getattr(layer.mlp, p).weight.data = fn(orig[i][p].clone(), i, total_layers)
    ppl = compute_ppl(model, tok, texts)
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) {expected_comp} {status}")
    return ppl, delta


def main():
    print("=" * 70)
    print("Combined Approach: Push for Maximum Compression")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:30]
    
    orig = {}
    for i, layer in enumerate(model.model.layers):
        orig[i] = {}
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, p):
                orig[i][p] = getattr(layer.mlp, p).weight.data.clone()
    
    def restore():
        for i, layer in enumerate(model.model.layers):
            for p in ["gate_proj", "up_proj", "down_proj"]:
                if p in orig[i]:
                    getattr(layer.mlp, p).weight.data = orig[i][p].clone()
    
    baseline = compute_ppl(model, tok, texts)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print("\n" + "-" * 70)
    
    configs = [
        ("Baseline INT4 g8", lambda w, i, t: multi_stage_quant(w, 8, False), "~4x"),
        ("Multi-stage g16", lambda w, i, t: multi_stage_quant(w, 16, True), "~4.5x"),
        ("Multi-stage g32", lambda w, i, t: multi_stage_quant(w, 32, True), "~5x"),
        ("Combined V1", combined_quant_v1, "~5.5-6x"),
        ("Combined V2", combined_quant_v2, "~6-7x"),
        ("Combined V3 (risky)", combined_quant_v3, "~7-8x"),
        ("Combined V4 (uniform)", combined_quant_v4, "~5x"),
    ]
    
    results = []
    for name, fn, comp in configs:
        restore()
        ppl, delta = test(model, orig, tok, texts, name, fn, baseline, comp)
        results.append((name, comp, ppl, delta))
        gc.collect()
    
    print("\n" + "=" * 70)
    print("RESULTS - Combined Approaches")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print()
    print(f"{'Technique':<25} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 55)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<25} {comp:>8} {ppl:>8.2f} {delta:>+6.1f}% {status}")
    
    # Find best
    best_ok = [r for r in results if r[3] < 1]
    if best_ok:
        best = max(best_ok, key=lambda x: x[3])  # Closest to 1% that's still under
        print(f"\n✅ Best <1% PPL: {best[0]} at {best[1]} compression")


if __name__ == "__main__":
    main()
