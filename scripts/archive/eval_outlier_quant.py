#!/usr/bin/env python3
"""
Outlier-Aware Quantization for 10x Compression

Key insight: A small fraction of weights (outliers) have disproportionate 
impact on model quality. By keeping these in higher precision, we can 
aggressively quantize the rest.

Strategy:
1. Identify top 1% of weights by magnitude (outliers)
2. Keep outliers in FP16
3. Quantize remaining 99% to INT4 with large group size (g=128)

Storage math for 1% outliers:
- 1% weights in FP16: 0.01 × 2 bytes = 0.02 bytes/weight
- 99% weights in INT4 g=128: 0.99 × 0.53 bytes = 0.52 bytes/weight
- Total: ~0.54 bytes/weight

From FP32 (4 bytes): 4 / 0.54 = 7.4x
From FP16 (2 bytes): 2 / 0.54 = 3.7x

Still not 10x. Let's try INT2 for non-outliers with outlier protection.

Strategy 2: 5% outliers in FP16, 95% in INT2
- 5% weights in FP16: 0.05 × 2 bytes = 0.1 bytes/weight  
- 95% weights in INT2 g=64: 0.95 × 0.31 bytes = 0.30 bytes/weight
- Total: ~0.40 bytes/weight

From FP32: 4 / 0.40 = 10x ✓
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TENPAK_ROOT = Path(__file__).parent.parent


def get_wikitext2_subset(tokenizer, num_tokens=20000):
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", max_length=num_tokens, truncation=True)
    print(f"Using {encodings.input_ids.size(1)} tokens")
    return encodings


def evaluate_ppl(model, encodings, max_steps=30):
    print(f"Evaluating PPL ({max_steps} steps)...")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    stride = 256
    max_length = 512
    
    for step, begin_loc in enumerate(range(0, seq_len - 1, stride)):
        if step >= max_steps:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.cpu())
        
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break
    
    return torch.exp(torch.stack(nlls).mean()).item()


def outlier_quant_int4(weight, outlier_fraction=0.01, group_size=128):
    """
    Outlier-aware INT4 quantization.
    
    1. Find outliers (top outlier_fraction by magnitude)
    2. Store outliers separately in FP16
    3. Quantize non-outliers to INT4
    """
    shape = weight.shape
    weight_flat = weight.flatten().float()
    n = len(weight_flat)
    
    # Find outlier threshold
    abs_weights = weight_flat.abs()
    num_outliers = max(1, int(n * outlier_fraction))
    threshold = torch.kthvalue(abs_weights, n - num_outliers + 1).values
    
    # Create outlier mask
    outlier_mask = abs_weights >= threshold
    num_actual_outliers = outlier_mask.sum().item()
    
    # Extract outliers
    outlier_indices = outlier_mask.nonzero().squeeze(-1).to(torch.int32)
    outlier_values = weight_flat[outlier_mask].half()
    
    # Zero out outliers for quantization
    weight_no_outliers = weight_flat.clone()
    weight_no_outliers[outlier_mask] = 0
    
    # Pad for group quantization
    pad_len = (group_size - n % group_size) % group_size
    if pad_len > 0:
        weight_no_outliers = torch.cat([weight_no_outliers, torch.zeros(pad_len)])
    
    num_groups = len(weight_no_outliers) // group_size
    weight_groups = weight_no_outliers.view(num_groups, group_size)
    
    # Quantize to INT4
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return {
        'packed': packed,
        'scales': scales.half(),
        'offsets': offsets.half(),
        'outlier_indices': outlier_indices,
        'outlier_values': outlier_values,
        'shape': shape,
        'pad_len': pad_len,
        'group_size': group_size,
        'num_outliers': num_actual_outliers,
    }


def dequantize_outlier_int4(data):
    """Dequantize outlier-aware INT4 weights."""
    packed = data['packed']
    scales = data['scales']
    offsets = data['offsets']
    shape = data['shape']
    pad_len = data['pad_len']
    group_size = data['group_size']
    
    # Unpack INT4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    
    # Dequantize
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    # Flatten and remove padding
    weight_flat = weight_deq.flatten()
    n = shape[0] * shape[1]
    weight_flat = weight_flat[:n]
    
    # Restore outliers
    outlier_indices = data['outlier_indices']
    outlier_values = data['outlier_values']
    weight_flat[outlier_indices.long()] = outlier_values.float()
    
    return weight_flat.view(shape)


def compute_outlier_bytes(data):
    """Compute storage for outlier-aware quantization."""
    # Packed INT4
    bytes_packed = data['packed'].numel() * 1  # uint8
    # Scales and offsets
    bytes_scales = data['scales'].numel() * 2  # fp16
    bytes_offsets = data['offsets'].numel() * 2  # fp16
    # Outliers: indices (int32) + values (fp16)
    bytes_outlier_idx = data['outlier_indices'].numel() * 4  # int32
    bytes_outlier_val = data['outlier_values'].numel() * 2  # fp16
    
    return bytes_packed + bytes_scales + bytes_offsets + bytes_outlier_idx + bytes_outlier_val


class OutlierQuantLinear(nn.Module):
    """Linear layer with outlier-aware quantization."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data = None
        self.original_bytes = 0
        self.register_buffer('bias_data', None)
    
    @classmethod
    def from_linear(cls, linear, outlier_fraction=0.01, group_size=128):
        layer = cls(linear.in_features, linear.out_features)
        layer.original_bytes = linear.weight.numel() * linear.weight.element_size()
        layer.data = outlier_quant_int4(linear.weight.data, outlier_fraction, group_size)
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        
        return layer
    
    def quantized_bytes(self):
        return compute_outlier_bytes(self.data)
    
    def forward(self, x):
        weight = dequantize_outlier_int4(self.data).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model_outlier(model, outlier_fraction=0.01, group_size=128):
    """Apply outlier-aware quantization."""
    print(f"Applying outlier-aware quantization (outliers={outlier_fraction*100:.1f}%, g={group_size})...")
    
    original_bytes = 0
    quantized_bytes = 0
    total_outliers = 0
    total_weights = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                original_bytes += original.weight.numel() * original.weight.element_size()
                total_weights += original.weight.numel()
                
                quantized = OutlierQuantLinear.from_linear(original, outlier_fraction, group_size)
                setattr(mlp, proj_name, quantized)
                quantized_bytes += quantized.quantized_bytes()
                total_outliers += quantized.data['num_outliers']
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    compression = original_bytes / quantized_bytes
    print(f"\nStats:")
    print(f"  Outliers: {total_outliers:,} / {total_weights:,} ({100*total_outliers/total_weights:.2f}%)")
    print(f"  Compression: {original_bytes/1e9:.3f} GB → {quantized_bytes/1e9:.3f} GB = {compression:.2f}x")
    
    return model, original_bytes, quantized_bytes


def run_experiment(model_name, encodings, outlier_fraction, group_size, baseline_ppl):
    """Run outlier quantization experiment."""
    print(f"\n{'='*70}")
    print(f"Outliers: {outlier_fraction*100:.1f}%, Group Size: {group_size}")
    print('='*70)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    model, orig_bytes, quant_bytes = quantize_model_outlier(model, outlier_fraction, group_size)
    
    quant_ppl = evaluate_ppl(model, encodings)
    ppl_delta = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    
    fp16_compression = orig_bytes / quant_bytes
    fp32_compression = (orig_bytes * 2) / quant_bytes
    
    print(f"\nResults:")
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")
    print(f"  Quantized PPL:    {quant_ppl:.2f}")
    print(f"  PPL Delta:        {ppl_delta:+.2f}%")
    print(f"  Compression (FP16): {fp16_compression:.2f}x")
    print(f"  Compression (FP32): {fp32_compression:.2f}x")
    
    del model
    import gc
    gc.collect()
    
    return {
        'outlier_fraction': outlier_fraction,
        'group_size': group_size,
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quant_ppl,
        'ppl_delta': ppl_delta,
        'compression_fp16': fp16_compression,
        'compression_fp32': fp32_compression,
    }


def main():
    print("="*70)
    print("OUTLIER-AWARE QUANTIZATION EXPERIMENTS")
    print("="*70)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = get_wikitext2_subset(tokenizer, num_tokens=20000)
    
    # Get baseline
    print("\nComputing baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    baseline_ppl = evaluate_ppl(model, encodings)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    del model
    import gc
    gc.collect()
    
    # Experiments: (outlier_fraction, group_size)
    experiments = [
        # Conservative: more outliers, smaller groups
        (0.01, 8),    # 1% outliers, g=8
        (0.02, 8),    # 2% outliers, g=8
        (0.05, 8),    # 5% outliers, g=8
        
        # Aggressive: fewer outliers, larger groups (higher compression)
        (0.01, 128),  # 1% outliers, g=128
        (0.02, 128),  # 2% outliers, g=128
        (0.05, 128),  # 5% outliers, g=128
        
        # Very aggressive
        (0.10, 128),  # 10% outliers, g=128
        (0.10, 256),  # 10% outliers, g=256
    ]
    
    results = []
    for outlier_frac, group_size in experiments:
        try:
            result = run_experiment(model_name, encodings, outlier_frac, group_size, baseline_ppl)
            results.append(result)
            
            # Early exit if we find a good solution
            if result['compression_fp32'] >= 10 and result['ppl_delta'] < 1.0:
                print("\n*** FOUND TARGET: 10x compression with <1% PPL! ***")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Outliers':<10} {'Group':<6} {'PPL Δ':>8} {'FP16→':>8} {'FP32→':>8} {'Target':>12}")
    print("-"*62)
    
    for r in results:
        target = "✓ 10x+<1%" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ 10x" if r['compression_fp32'] >= 10 else \
                 "◐ <1%" if r['ppl_delta'] < 1.0 else \
                 "✗"
        print(f"{r['outlier_fraction']*100:>6.1f}%   {r['group_size']:<6} {r['ppl_delta']:>+7.2f}% {r['compression_fp16']:>7.2f}x {r['compression_fp32']:>7.2f}x {target:>12}")
    
    # Save
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "outlier_quant_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/outlier_quant_experiments.json")


if __name__ == "__main__":
    main()
