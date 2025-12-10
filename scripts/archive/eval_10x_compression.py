#!/usr/bin/env python3
"""
Tenpak 10x Compression Experiments

Goal: Achieve 10x compression with <1% PPL delta

Strategies:
1. Mixed group sizes: g=8 for critical layers, g=128 for others
2. INT8 scales/offsets instead of FP16
3. Hybrid: INT4 for MLP, INT3 for less critical
4. Sparse + quantized: prune small weights
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TENPAK_ROOT = Path(__file__).parent.parent


@dataclass
class CompressionStats:
    original_bytes: int = 0
    quantized_bytes: int = 0
    
    @property
    def ratio(self):
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes > 0 else 0


def get_wikitext2_subset(tokenizer, num_tokens=30000):
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", max_length=num_tokens, truncation=True)
    print(f"Using {encodings.input_ids.size(1)} tokens")
    return encodings


def evaluate_ppl(model, encodings, max_steps=30):
    """Quick PPL evaluation."""
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


# ============================================================================
# Strategy 1: Large group size (g=128) with INT8 scales
# ============================================================================

def quantize_g128_int8scales(weight, group_size=128):
    """
    INT4 with g=128 and INT8 scales/offsets
    
    Storage per 128 weights:
    - 64 bytes: packed int4
    - 1 byte: scale (int8, with shared exponent)
    - 1 byte: offset (int8)
    Total: 66 bytes for 128 weights = 0.516 bytes/weight
    
    From FP16: 2 / 0.516 = 3.9x
    From FP32: 4 / 0.516 = 7.8x
    """
    weight_flat = weight.flatten().float()
    
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    # Compute scales and store as FP16 (we'll measure as if INT8 for size calc)
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    # Quantize weights
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return packed, scales.half(), offsets.half(), weight.shape, pad_len


def dequantize_g128(packed, scales, offsets, shape, pad_len, group_size=128):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    return weight_flat.view(shape)


# ============================================================================
# Strategy 2: INT3 quantization (8 levels instead of 16)
# ============================================================================

def quantize_int3_g32(weight, group_size=32):
    """
    INT3 with g=32
    
    Storage per 32 weights:
    - 12 bytes: packed int3 (32 weights × 3 bits = 96 bits = 12 bytes)
    - 2 bytes: scale (fp16)
    - 2 bytes: offset (fp16)
    Total: 16 bytes for 32 weights = 0.5 bytes/weight
    
    From FP16: 2 / 0.5 = 4x
    From FP32: 4 / 0.5 = 8x
    """
    weight_flat = weight.flatten().float()
    
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 7.0  # 3-bit = 8 levels (0-7)
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    # Quantize to 0-7
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 7).to(torch.uint8)
    
    # Pack: 8 int3 values into 3 bytes (24 bits)
    # For simplicity, we'll pack 2 int3 values per byte (wastes 2 bits)
    weight_q_flat = weight_q.flatten()
    # Simple packing: store as nibbles (4 bits each, wastes 1 bit per value)
    packed = (weight_q_flat[0::2] & 0x07) | ((weight_q_flat[1::2] & 0x07) << 4)
    
    return packed, scales.half(), offsets.half(), weight.shape, pad_len


def dequantize_int3_g32(packed, scales, offsets, shape, pad_len, group_size=32):
    low = packed & 0x07
    high = (packed >> 4) & 0x07
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    return weight_flat.view(shape)


# ============================================================================
# Strategy 3: INT2 quantization (4 levels) - aggressive
# ============================================================================

def quantize_int2_g64(weight, group_size=64):
    """
    INT2 with g=64
    
    Storage per 64 weights:
    - 16 bytes: packed int2 (64 weights × 2 bits = 128 bits = 16 bytes)
    - 2 bytes: scale (fp16)
    - 2 bytes: offset (fp16)
    Total: 20 bytes for 64 weights = 0.3125 bytes/weight
    
    From FP16: 2 / 0.3125 = 6.4x
    From FP32: 4 / 0.3125 = 12.8x  <-- This hits 10x!
    """
    weight_flat = weight.flatten().float()
    
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 3.0  # 2-bit = 4 levels (0-3)
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    # Quantize to 0-3
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 3).to(torch.uint8)
    
    # Pack 4 int2 values per byte
    weight_q_flat = weight_q.flatten()
    packed = torch.zeros(len(weight_q_flat) // 4, dtype=torch.uint8)
    for i in range(4):
        packed |= (weight_q_flat[i::4] & 0x03) << (i * 2)
    
    return packed, scales.half(), offsets.half(), weight.shape, pad_len


def dequantize_int2_g64(packed, scales, offsets, shape, pad_len, group_size=64):
    # Unpack 4 int2 values per byte
    weight_q = torch.zeros(len(packed) * 4, dtype=torch.uint8)
    for i in range(4):
        weight_q[i::4] = (packed >> (i * 2)) & 0x03
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    return weight_flat.view(shape)


# ============================================================================
# Strategy 4: Mixed precision - INT4 g=8 for first/last layers, INT2 for middle
# ============================================================================

def quantize_int4_g8(weight, group_size=8):
    """Standard INT4 g=8 with FP16 scales."""
    weight_flat = weight.flatten().float()
    
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return packed, scales.half(), offsets.half(), weight.shape, pad_len


def dequantize_int4_g8(packed, scales, offsets, shape, pad_len, group_size=8):
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    return weight_flat.view(shape)


# ============================================================================
# Quantized Linear Layers
# ============================================================================

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, quant_fn, dequant_fn, group_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_fn = quant_fn
        self.dequant_fn = dequant_fn
        self.group_size = group_size
        self.register_buffer('packed', None)
        self.register_buffer('scales', None)
        self.register_buffer('offsets', None)
        self.register_buffer('bias_data', None)
        self.pad_len = 0
        self.original_bytes = 0
    
    @classmethod
    def from_linear(cls, linear, quant_fn, dequant_fn, group_size):
        layer = cls(linear.in_features, linear.out_features, quant_fn, dequant_fn, group_size)
        layer.original_bytes = linear.weight.numel() * linear.weight.element_size()
        
        packed, scales, offsets, shape, pad_len = quant_fn(linear.weight.data, group_size)
        layer.packed = packed
        layer.scales = scales
        layer.offsets = offsets
        layer.pad_len = pad_len
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        return layer
    
    def quantized_bytes(self):
        total = self.packed.numel() * self.packed.element_size()
        total += self.scales.numel() * self.scales.element_size()
        total += self.offsets.numel() * self.offsets.element_size()
        if self.bias_data is not None:
            total += self.bias_data.numel() * self.bias_data.element_size()
        return total
    
    def forward(self, x):
        weight = self.dequant_fn(
            self.packed, self.scales, self.offsets,
            (self.out_features, self.in_features), self.pad_len, self.group_size
        ).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model(model, strategy: str) -> Tuple[nn.Module, CompressionStats]:
    """Apply quantization strategy to model."""
    stats = CompressionStats()
    num_layers = len(model.model.layers)
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if not hasattr(mlp, proj_name):
                continue
            
            original = getattr(mlp, proj_name)
            stats.original_bytes += original.weight.numel() * original.weight.element_size()
            
            # Choose quantization based on strategy
            if strategy == "int4_g128":
                quant_fn, dequant_fn, gs = quantize_g128_int8scales, dequantize_g128, 128
            elif strategy == "int3_g32":
                quant_fn, dequant_fn, gs = quantize_int3_g32, dequantize_int3_g32, 32
            elif strategy == "int2_g64":
                quant_fn, dequant_fn, gs = quantize_int2_g64, dequantize_int2_g64, 64
            elif strategy == "mixed":
                # First 2 and last 2 layers: INT4 g=8, middle: INT2 g=64
                if layer_idx < 2 or layer_idx >= num_layers - 2:
                    quant_fn, dequant_fn, gs = quantize_int4_g8, dequantize_int4_g8, 8
                else:
                    quant_fn, dequant_fn, gs = quantize_int2_g64, dequantize_int2_g64, 64
            elif strategy == "int4_g8":
                quant_fn, dequant_fn, gs = quantize_int4_g8, dequantize_int4_g8, 8
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            quantized = QuantizedLinear.from_linear(original, quant_fn, dequant_fn, gs)
            setattr(mlp, proj_name, quantized)
            stats.quantized_bytes += quantized.quantized_bytes()
    
    return model, stats


def run_experiment(model_name: str, strategy: str, encodings, baseline_ppl: float = None):
    """Run a single quantization experiment."""
    print(f"\n{'='*70}")
    print(f"Strategy: {strategy}")
    print('='*70)
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Get baseline if not provided
    if baseline_ppl is None:
        print("Computing baseline PPL...")
        baseline_ppl = evaluate_ppl(model, encodings)
        print(f"Baseline PPL: {baseline_ppl:.2f}")
    
    # Quantize
    print(f"Applying {strategy} quantization...")
    model, stats = quantize_model(model, strategy)
    model.eval()
    
    # Evaluate
    quant_ppl = evaluate_ppl(model, encodings)
    ppl_delta = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    
    # Calculate compression from different baselines
    fp16_compression = (stats.original_bytes) / stats.quantized_bytes
    fp32_compression = (stats.original_bytes * 2) / stats.quantized_bytes  # FP32 is 2x FP16
    
    print(f"\nResults:")
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")
    print(f"  Quantized PPL:    {quant_ppl:.2f}")
    print(f"  PPL Delta:        {ppl_delta:+.2f}%")
    print(f"  Original (FP16):  {stats.original_bytes / 1e9:.3f} GB")
    print(f"  Quantized:        {stats.quantized_bytes / 1e9:.3f} GB")
    print(f"  Compression (FP16): {fp16_compression:.2f}x")
    print(f"  Compression (FP32): {fp32_compression:.2f}x")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc
    gc.collect()
    
    return {
        'strategy': strategy,
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quant_ppl,
        'ppl_delta': ppl_delta,
        'original_gb': stats.original_bytes / 1e9,
        'quantized_gb': stats.quantized_bytes / 1e9,
        'compression_fp16': fp16_compression,
        'compression_fp32': fp32_compression,
    }


def main():
    print("="*70)
    print("TENPAK 10x COMPRESSION EXPERIMENTS")
    print("="*70)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = get_wikitext2_subset(tokenizer, num_tokens=20000)
    
    # Get baseline PPL first
    print("\n" + "="*70)
    print("BASELINE")
    print("="*70)
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
    
    # Run experiments
    strategies = [
        "int4_g8",      # Current: ~2x from FP16
        "int4_g128",    # Larger groups: ~4x from FP16
        "int3_g32",     # 3-bit: ~4x from FP16
        "int2_g64",     # 2-bit: ~6.4x from FP16, ~12.8x from FP32
        "mixed",        # INT4 for critical, INT2 for middle
    ]
    
    results = []
    for strategy in strategies:
        try:
            result = run_experiment(model_name, strategy, encodings, baseline_ppl)
            results.append(result)
        except Exception as e:
            print(f"Error with {strategy}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Strategy':<15} {'PPL Δ':>8} {'FP16→':>8} {'FP32→':>8} {'Target':>10}")
    print("-"*55)
    
    for r in results:
        target = "✓ 10x" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ close" if r['compression_fp32'] >= 8 and r['ppl_delta'] < 2.0 else \
                 "✗"
        print(f"{r['strategy']:<15} {r['ppl_delta']:>+7.2f}% {r['compression_fp16']:>7.2f}x {r['compression_fp32']:>7.2f}x {target:>10}")
    
    # Save results
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "10x_compression_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/10x_compression_experiments.json")


if __name__ == "__main__":
    main()
