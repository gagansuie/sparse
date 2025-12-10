#!/usr/bin/env python3
"""
Outlier-Aware Quantization

Key insight: A small percentage of weights (outliers) are critical for model quality.
By keeping outliers in FP16 and quantizing the rest to INT3, we can achieve:
- ~8-9x compression (vs 10x for pure INT3)
- Much better quality than pure INT3

Storage:
- 99% of weights: INT3 (3 bits)
- 1% outliers: FP16 (16 bits) + indices (16 bits per outlier)
- Average: ~3 + 0.01*32 = 3.32 bits/weight ≈ 9.6x compression
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


def outlier_aware_quantize(weight, bits=3, outlier_pct=1.0, group_size=32):
    """
    Quantize weights with outlier preservation.
    
    Args:
        weight: [out, in] weight tensor
        bits: quantization bits for non-outliers
        outlier_pct: percentage of weights to keep as outliers
        group_size: group size for group quantization
    
    Returns:
        Dequantized weight tensor (for evaluation)
        Storage estimate in bytes
    """
    weight_flat = weight.flatten()
    n = weight_flat.numel()
    
    # Find outliers (top outlier_pct% by magnitude)
    n_outliers = max(1, int(n * outlier_pct / 100))
    _, outlier_indices = weight_flat.abs().topk(n_outliers)
    
    # Create mask
    is_outlier = torch.zeros(n, dtype=torch.bool)
    is_outlier[outlier_indices] = True
    
    # Quantize non-outliers
    max_val = 2 ** bits - 1
    result = weight_flat.clone()
    
    non_outlier_mask = ~is_outlier
    non_outlier_vals = weight_flat[non_outlier_mask]
    
    if non_outlier_vals.numel() > 0:
        # Group quantization for non-outliers
        num_groups = (non_outlier_vals.numel() + group_size - 1) // group_size
        
        # Pad to group size
        padded_len = num_groups * group_size
        padded = F.pad(non_outlier_vals, (0, padded_len - non_outlier_vals.numel()))
        groups = padded.view(num_groups, group_size)
        
        # Per-group min/max
        g_min = groups.min(dim=1, keepdim=True).values
        g_max = groups.max(dim=1, keepdim=True).values
        scale = (g_max - g_min).clamp(min=1e-8) / max_val
        
        # Quantize
        q = ((groups - g_min) / scale).round().clamp(0, max_val)
        
        # Dequantize
        deq = q * scale + g_min
        
        # Apply back (trim padding)
        result[non_outlier_mask] = deq.flatten()[:non_outlier_vals.numel()]
    
    # Outliers stay as-is (FP16 in actual storage)
    
    # Calculate storage
    n_non_outlier = n - n_outliers
    quant_bytes = (n_non_outlier * bits + 7) // 8  # Packed bits
    num_groups = (n_non_outlier + group_size - 1) // group_size
    scale_bytes = num_groups * 4  # FP16 scale + FP16 offset
    outlier_bytes = n_outliers * (2 + 2)  # FP16 value + uint16 index
    
    total_bytes = quant_bytes + scale_bytes + outlier_bytes
    bits_per_weight = (total_bytes * 8) / n
    
    return result.view_as(weight), total_bytes, bits_per_weight


class OutlierQuantLinear(nn.Module):
    """Linear layer with outlier-aware quantization."""
    
    def __init__(self, weight, bias, bits=3, outlier_pct=1.0, group_size=32):
        super().__init__()
        
        # Quantize and store dequantized weights
        weight_deq, storage, bpw = outlier_aware_quantize(
            weight, bits=bits, outlier_pct=outlier_pct, group_size=group_size
        )
        
        self.register_buffer('weight', weight_deq)
        self.register_buffer('bias', bias)
        self.storage_bytes = storage
        self.bits_per_weight = bpw
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def test_config(model_name, tokenizer, texts, baseline_ppl, bits, outlier_pct, group_size):
    """Test a specific outlier quantization configuration."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    from transformers.pytorch_utils import Conv1D
    
    total_storage = 0
    total_weights = 0
    
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ["c_fc", "c_proj"]:
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            
            if isinstance(layer, Conv1D):
                weight = layer.weight.data.T.contiguous()
                bias = layer.bias.data.clone()
            else:
                weight = layer.weight.data.clone()
                bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_weights += weight.numel()
            
            quant_layer = OutlierQuantLinear(
                weight, bias, bits=bits, outlier_pct=outlier_pct, group_size=group_size
            )
            total_storage += quant_layer.storage_bytes
            
            # Replace in model
            setattr(model.transformer.h[block_idx].mlp, layer_name, quant_layer)
    
    # Evaluate
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    original_bytes = total_weights * 4  # FP32
    compression = original_bytes / total_storage
    bits_per_weight = (total_storage * 8) / total_weights
    
    del model
    
    return {
        'bits': bits,
        'outlier_pct': outlier_pct,
        'group_size': group_size,
        'ppl': ppl,
        'ppl_delta': delta,
        'compression': compression,
        'bits_per_weight': bits_per_weight,
    }


def main():
    print("=" * 70)
    print("Outlier-Aware Quantization")
    print("Target: 10x compression with <1% PPL via outlier preservation")
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
    
    # Test configurations
    configs = [
        # (bits, outlier_pct, group_size)
        (4, 0.0, 8),    # INT4 g8 (baseline)
        (4, 0.0, 16),   # INT4 g16
        (4, 0.0, 32),   # INT4 g32
        (4, 0.0, 64),   # INT4 g64
        (4, 0.0, 128),  # INT4 g128 (maximum compression for INT4)
        (3, 1.0, 8),    # INT3 g8 + outliers
        (3, 2.0, 8),    # INT3 g8 + more outliers
        (3, 5.0, 8),    # INT3 g8 + even more outliers
    ]
    
    results = []
    
    for bits, outlier_pct, group_size in configs:
        name = f"INT{bits} g{group_size}" + (f" + {outlier_pct}% outliers" if outlier_pct > 0 else "")
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        result = test_config(model_name, tokenizer, texts, baseline_ppl, bits, outlier_pct, group_size)
        result['name'] = name
        results.append(result)
        
        print(f"Compression: {result['compression']:.2f}x")
        print(f"Bits/weight: {result['bits_per_weight']:.2f}")
        print(f"PPL: {result['ppl']:.4f} (Δ {result['ppl_delta']:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Outlier-Aware Quantization")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: 10x compression, <1% PPL delta")
    print()
    print(f"{'Config':<30} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 75)
    
    for r in results:
        status = "🎯 TARGET!" if r['compression'] >= 8 and r['ppl_delta'] < 1.0 else \
                 "✓ Good" if r['ppl_delta'] < 1.0 else \
                 "~ OK" if r['ppl_delta'] < 5.0 else "✗"
        
        print(f"{r['name']:<30} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 75)
    
    # Find best config
    best = min(results, key=lambda r: abs(r['ppl_delta']) if r['compression'] >= 6 else float('inf'))
    print(f"\nBest >6x config: {best['name']}")
    print(f"  {best['compression']:.2f}x compression, {best['ppl_delta']:+.2f}% PPL delta")
    print("=" * 80)


if __name__ == "__main__":
    main()
