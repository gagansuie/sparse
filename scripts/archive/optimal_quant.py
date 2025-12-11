#!/usr/bin/env python3
"""
Optimal Quantization Experiments (No Calibration)

Goal: Achieve <1% PPL delta with high compression using only weight statistics.

Approaches:
1. Percentile clipping - Use 0.1/99.9 percentiles instead of min/max
2. MSE-optimal scale - Grid search for scale minimizing reconstruction error
3. Symmetric quantization - Might work better for some layers
4. Iterative refinement - Refine scale based on MSE feedback
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def quantize_minmax(weight, group_size=16, bits=4):
    """Standard min-max quantization."""
    max_val = 2 ** bits - 1
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    # Pad
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    scale = (g_max - g_min).clamp(min=1e-8) / max_val
    
    q = ((groups - g_min) / scale).round().clamp(0, max_val)
    deq = q * scale + g_min
    
    return deq.flatten()[:n].view_as(weight)


def quantize_percentile(weight, group_size=16, bits=4, pct_lo=0.1, pct_hi=99.9):
    """Percentile-based clipping to handle outliers."""
    max_val = 2 ** bits - 1
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    # Use percentiles instead of min/max
    g_min = torch.quantile(groups.float(), pct_lo / 100, dim=1, keepdim=True)
    g_max = torch.quantile(groups.float(), pct_hi / 100, dim=1, keepdim=True)
    
    scale = (g_max - g_min).clamp(min=1e-8) / max_val
    
    # Clip then quantize
    clipped = groups.clamp(g_min, g_max)
    q = ((clipped - g_min) / scale).round().clamp(0, max_val)
    deq = q * scale + g_min
    
    return deq.flatten()[:n].view_as(weight)


def quantize_mse_optimal(weight, group_size=16, bits=4, n_search=20):
    """Grid search for MSE-optimal scale."""
    max_val = 2 ** bits - 1
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    g_min_base = groups.min(dim=1, keepdim=True).values
    g_max_base = groups.max(dim=1, keepdim=True).values
    g_range = g_max_base - g_min_base
    
    best_deq = None
    best_mse = float('inf')
    
    # Search over different scale factors
    for alpha in torch.linspace(0.9, 1.1, n_search):
        for beta in torch.linspace(-0.05, 0.05, 5):
            g_min = g_min_base - beta * g_range
            g_max = g_max_base + beta * g_range
            scale = ((g_max - g_min) * alpha).clamp(min=1e-8) / max_val
            
            q = ((groups - g_min) / scale).round().clamp(0, max_val)
            deq = q * scale + g_min
            
            mse = ((groups - deq) ** 2).mean().item()
            if mse < best_mse:
                best_mse = mse
                best_deq = deq.clone()
    
    return best_deq.flatten()[:n].view_as(weight)


def quantize_symmetric(weight, group_size=16, bits=4):
    """Symmetric quantization around zero."""
    max_val = 2 ** (bits - 1) - 1  # -7 to 7 for 4-bit
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    # Symmetric: scale based on max absolute value
    g_abs_max = groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = g_abs_max / max_val
    
    q = (groups / scale).round().clamp(-max_val, max_val)
    deq = q * scale
    
    return deq.flatten()[:n].view_as(weight)


def quantize_iterative(weight, group_size=16, bits=4, iterations=3):
    """Iterative refinement of quantization parameters."""
    max_val = 2 ** bits - 1
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    # Start with min/max
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    
    for _ in range(iterations):
        scale = (g_max - g_min).clamp(min=1e-8) / max_val
        q = ((groups - g_min) / scale).round().clamp(0, max_val)
        deq = q * scale + g_min
        
        # Compute error and adjust
        error = groups - deq
        
        # Adjust min/max based on error distribution
        error_lo = error.min(dim=1, keepdim=True).values
        error_hi = error.max(dim=1, keepdim=True).values
        
        # Shrink range slightly based on error
        g_min = g_min + error_lo * 0.5
        g_max = g_max + error_hi * 0.5
    
    scale = (g_max - g_min).clamp(min=1e-8) / max_val
    q = ((groups - g_min) / scale).round().clamp(0, max_val)
    deq = q * scale + g_min
    
    return deq.flatten()[:n].view_as(weight)


def quantize_hybrid(weight, group_size=16, bits=4):
    """Hybrid: Use percentile + MSE refinement."""
    max_val = 2 ** bits - 1
    w_flat = weight.flatten()
    n = w_flat.numel()
    num_groups = (n + group_size - 1) // group_size
    
    padded = F.pad(w_flat, (0, num_groups * group_size - n))
    groups = padded.view(num_groups, group_size)
    
    # Start with percentile clipping
    g_min = torch.quantile(groups.float(), 0.001, dim=1, keepdim=True)
    g_max = torch.quantile(groups.float(), 0.999, dim=1, keepdim=True)
    
    best_deq = None
    best_mse = float('inf')
    
    # Small grid search around percentile values
    for scale_mult in [0.95, 0.98, 1.0, 1.02, 1.05]:
        g_range = g_max - g_min
        adj_min = g_min - (scale_mult - 1) * g_range * 0.5
        adj_max = g_max + (scale_mult - 1) * g_range * 0.5
        
        scale = (adj_max - adj_min).clamp(min=1e-8) / max_val
        clipped = groups.clamp(adj_min, adj_max)
        q = ((clipped - adj_min) / scale).round().clamp(0, max_val)
        deq_clipped = q * scale + adj_min
        
        # For values outside clip range, use original
        deq = torch.where(
            (groups >= adj_min) & (groups <= adj_max),
            deq_clipped,
            groups  # Keep outliers as-is (will be clipped in actual impl)
        )
        
        mse = ((groups - deq_clipped) ** 2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_deq = deq_clipped.clone()
    
    return best_deq.flatten()[:n].view_as(weight)


def test_method(model, tokenizer, texts, baseline_ppl, method_fn, name, group_size=16):
    """Test a quantization method."""
    from transformers.pytorch_utils import Conv1D
    
    print(f"\nTesting: {name} (g={group_size})")
    
    # Quantize MLP layers
    total_mse = 0
    total_weights = 0
    
    for block in model.transformer.h:
        for layer_name in ["c_fc", "c_proj"]:
            layer = getattr(block.mlp, layer_name)
            
            if isinstance(layer, Conv1D):
                weight = layer.weight.data.T.clone()
                weight_q = method_fn(weight, group_size=group_size)
                layer.weight.data = weight_q.T
            else:
                weight = layer.weight.data.clone()
                weight_q = method_fn(weight, group_size=group_size)
                layer.weight.data = weight_q
            
            mse = ((weight - weight_q) ** 2).mean().item()
            total_mse += mse * weight.numel()
            total_weights += weight.numel()
    
    avg_mse = total_mse / total_weights
    
    # Evaluate
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    print(f"  Avg MSE: {avg_mse:.6f}")
    print(f"  PPL: {ppl:.4f} (Δ {delta:+.2f}%)")
    
    return {
        'name': name,
        'group_size': group_size,
        'mse': avg_mse,
        'ppl': ppl,
        'ppl_delta': delta,
    }


def main():
    print("=" * 70)
    print("Optimal Quantization (No Calibration)")
    print("Goal: <1% PPL delta with 5x+ compression")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:80]
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Test methods
    methods = [
        (quantize_minmax, "Min-Max (baseline)"),
        (quantize_percentile, "Percentile (0.1-99.9)"),
        (quantize_mse_optimal, "MSE-Optimal Search"),
        (quantize_symmetric, "Symmetric"),
        (quantize_iterative, "Iterative Refine"),
        (quantize_hybrid, "Hybrid (pct+MSE)"),
    ]
    
    results = []
    
    for group_size in [16, 12, 10, 8]:
        print(f"\n{'='*60}")
        print(f"GROUP SIZE = {group_size}")
        print('='*60)
        
        for method_fn, name in methods:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            result = test_method(model, tokenizer, texts, baseline_ppl, method_fn, name, group_size)
            results.append(result)
            del model
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: <1% PPL delta")
    print()
    print(f"{'Method':<25} {'g':<5} {'MSE':<12} {'PPL Δ':<10} {'Status'}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['ppl_delta']):
        status = "🎯 <1%!" if r['ppl_delta'] < 1.0 else "✓ Good" if r['ppl_delta'] < 1.5 else "~ OK" if r['ppl_delta'] < 2.0 else ""
        print(f"{r['name']:<25} {r['group_size']:<5} {r['mse']:.6f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 70)
    
    # Best result
    best = min(results, key=lambda r: r['ppl_delta'])
    print(f"\nBest: {best['name']} (g={best['group_size']}): {best['ppl_delta']:+.2f}% PPL delta")
    print("=" * 80)


if __name__ == "__main__":
    main()
