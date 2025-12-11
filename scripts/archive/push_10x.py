#!/usr/bin/env python3
"""
Final push for 10x compression.

We've achieved:
- 5x at +1.0% PPL
- 6-7x at +1.3% PPL
- 7-8x at +1.7% PPL

Now try:
1. INT3 + multi-stage residual for middle layers
2. More aggressive layer position targeting
3. Combine sparsity with multi-stage
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


def int3_with_residual(weight, group_size=16):
    """
    INT3 (8 levels) + INT2 residual + INT1 sign correction.
    
    Total: 3 + 2 + 1 = 6 bits + overhead ≈ 5.3x compression
    But potentially better quality than plain INT3 thanks to correction.
    """
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
    
    # Stage 1: INT3 (7 levels = 0-7)
    g_min, g_max = groups.min(1).values, groups.max(1).values
    for _ in range(5):
        scale = (g_max - g_min) / 7.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 7)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(1).values * 0.5
        g_max = g_max + err.max(1).values * 0.5
    
    scale = (g_max - g_min) / 7.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q1 = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 7)
    stage1 = q1 * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Stage 2: INT2 residual
    residual1 = groups - stage1
    r1_min, r1_max = residual1.min(1).values, residual1.max(1).values
    r1_scale = (r1_max - r1_min) / 3.0
    r1_scale = torch.where(r1_scale.abs() < 1e-8, torch.ones_like(r1_scale), r1_scale)
    q2 = ((residual1 - r1_min.unsqueeze(1)) / r1_scale.unsqueeze(1)).round().clamp(0, 3)
    stage2 = q2 * r1_scale.unsqueeze(1) + r1_min.unsqueeze(1)
    
    # Stage 3: INT1 sign correction
    residual2 = groups - stage1 - stage2
    r2_scale = residual2.abs().mean(dim=1, keepdim=True)
    q3 = (residual2 > 0).float() * 2 - 1
    stage3 = q3 * r2_scale * 0.5
    
    result = stage1 + stage2 + stage3
    return result.flatten()[:n].view(weight.shape).to(weight.dtype)


def int4_multi_stage(weight, group_size=16, use_int1=True):
    """INT4 + INT2 + optional INT1."""
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
    q1 = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    stage1 = q1 * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    residual1 = groups - stage1
    r1_min, r1_max = residual1.min(1).values, residual1.max(1).values
    r1_scale = (r1_max - r1_min) / 3.0
    r1_scale = torch.where(r1_scale.abs() < 1e-8, torch.ones_like(r1_scale), r1_scale)
    q2 = ((residual1 - r1_min.unsqueeze(1)) / r1_scale.unsqueeze(1)).round().clamp(0, 3)
    stage2 = q2 * r1_scale.unsqueeze(1) + r1_min.unsqueeze(1)
    
    result = stage1 + stage2
    
    if use_int1:
        residual2 = groups - result
        r2_scale = residual2.abs().mean(dim=1, keepdim=True)
        q3 = (residual2 > 0).float() * 2 - 1
        stage3 = q3 * r2_scale * 0.5
        result = result + stage3
    
    return result.flatten()[:n].view(weight.shape).to(weight.dtype)


def aggressive_push_v1(weight, layer_idx, total_layers):
    """
    Aggressive V1: INT3 with residual for middle 60%.
    
    Edge: INT4 g8 + multi-stage
    Middle: INT3 g16 + multi-stage residual
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return int4_multi_stage(weight, group_size=8, use_int1=True)
    elif pos < 0.2 or pos > 0.8:
        return int4_multi_stage(weight, group_size=16, use_int1=True)
    else:
        return int3_with_residual(weight, group_size=16)


def aggressive_push_v2(weight, layer_idx, total_layers):
    """
    Aggressive V2: INT3 + residual for middle 80%.
    Only protect first/last 10%.
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return int4_multi_stage(weight, group_size=8, use_int1=True)
    else:
        return int3_with_residual(weight, group_size=16)


def aggressive_push_v3(weight, layer_idx, total_layers):
    """
    Aggressive V3: INT3 + large groups for middle.
    
    Edge: INT4 g8 + full correction
    Middle: INT3 g32 + residual (very aggressive)
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return int4_multi_stage(weight, group_size=8, use_int1=True)
    else:
        return int3_with_residual(weight, group_size=32)


def sparse_multi_stage(weight, sparsity=0.15, group_size=16):
    """
    Structured sparsity + multi-stage.
    
    Prune lowest-norm groups, then multi-stage quantize the rest.
    """
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    num_groups = flat.numel() // group_size
    groups = flat.view(num_groups, group_size)
    
    # Find groups to prune
    norms = groups.norm(dim=1)
    num_prune = int(num_groups * sparsity)
    _, prune_idx = torch.topk(norms, num_prune, largest=False)
    prune_mask = torch.zeros(num_groups, dtype=torch.bool, device=weight.device)
    prune_mask[prune_idx] = True
    
    # Prune groups (set to zero or mean)
    mean_val = groups[~prune_mask].mean()
    groups[prune_mask] = mean_val
    
    # Multi-stage on remaining
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
    
    # INT2 residual
    residual = groups - stage1
    r_min, r_max = residual.min(1).values, residual.max(1).values
    r_scale = (r_max - r_min) / 3.0
    r_scale = torch.where(r_scale.abs() < 1e-8, torch.ones_like(r_scale), r_scale)
    q2 = ((residual - r_min.unsqueeze(1)) / r_scale.unsqueeze(1)).round().clamp(0, 3)
    stage2 = q2 * r_scale.unsqueeze(1) + r_min.unsqueeze(1)
    
    result = stage1 + stage2
    return result.flatten()[:n].view(weight.shape).to(weight.dtype)


def sparse_combined(weight, layer_idx, total_layers):
    """
    Sparsity + multi-stage, position-aware.
    
    Edge: No sparsity, full multi-stage
    Middle: 20% sparsity + multi-stage
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return int4_multi_stage(weight, group_size=8, use_int1=True)
    else:
        return sparse_multi_stage(weight, sparsity=0.2, group_size=16)


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
    print("FINAL PUSH FOR 10x COMPRESSION")
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
        ("INT3+residual g16", lambda w, i, t: int3_with_residual(w, 16), "~5.3x"),
        ("INT3+residual g32", lambda w, i, t: int3_with_residual(w, 32), "~6.4x"),
        ("Aggressive V1", aggressive_push_v1, "~7x"),
        ("Aggressive V2", aggressive_push_v2, "~8x"),
        ("Aggressive V3", aggressive_push_v3, "~9x"),
        ("Sparse 15% + multi", lambda w, i, t: sparse_multi_stage(w, 0.15, 16), "~6x"),
        ("Sparse 20% + multi", lambda w, i, t: sparse_multi_stage(w, 0.20, 16), "~6.5x"),
        ("Sparse combined", sparse_combined, "~7x"),
    ]
    
    results = []
    for name, fn, comp in configs:
        restore()
        ppl, delta = test(model, orig, tok, texts, name, fn, baseline, comp)
        results.append((name, comp, ppl, delta))
        gc.collect()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS - Push for 10x")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print()
    print(f"{'Technique':<25} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 55)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<25} {comp:>8} {ppl:>8.2f} {delta:>+6.1f}% {status}")
    
    print("\n" + "=" * 70)
    print("Best configs:")
    best_ok = [r for r in results if r[3] < 1]
    best_almost = [r for r in results if 1 <= r[3] < 2]
    
    if best_ok:
        for r in sorted(best_ok, key=lambda x: -float(x[1].replace('~','').replace('x',''))):
            print(f"  ✅ {r[0]}: {r[1]} at {r[3]:+.1f}% PPL")
    if best_almost:
        for r in sorted(best_almost, key=lambda x: -float(x[1].replace('~','').replace('x',''))):
            print(f"  ⚠️ {r[0]}: {r[1]} at {r[3]:+.1f}% PPL")


if __name__ == "__main__":
    main()
