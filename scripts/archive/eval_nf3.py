#!/usr/bin/env python3
"""
NF3 (Non-uniform Float 3-bit) Quantization

The key insight: uniform INT3 fails because it assumes uniform weight distribution.
But neural network weights follow approximately Gaussian distribution.

NF3 uses quantization levels optimized for Gaussian distribution, placing more
levels where weights are dense (near 0) and fewer in the tails.

This is similar to bitsandbytes' NF4, but with 3 bits (8 levels instead of 16).

Target: 3 bits/weight = 10.67x compression vs FP32
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


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


# =============================================================================
# OPTIMAL QUANTIZATION LEVELS
# =============================================================================

def compute_nf_levels(n_bits, distribution='gaussian'):
    """
    Compute optimal quantization levels for a given distribution.
    
    For Gaussian, we use quantiles that minimize expected squared error.
    """
    n_levels = 2 ** n_bits
    
    if distribution == 'gaussian':
        # Use quantiles of standard normal distribution
        # This places levels where probability density is highest
        quantiles = np.linspace(0, 1, n_levels + 1)[1:-1]  # Interior points
        levels = stats.norm.ppf(quantiles)
        
        # Add endpoints (approximate -inf and +inf with reasonable values)
        levels = np.concatenate([[-3.0], levels, [3.0]])
        
        # Compute midpoints as reconstruction values
        midpoints = (levels[:-1] + levels[1:]) / 2
        midpoints[0] = levels[1] - 0.5  # Adjust first
        midpoints[-1] = levels[-2] + 0.5  # Adjust last
        
        return midpoints
    
    elif distribution == 'uniform':
        # Standard uniform quantization
        return np.linspace(-1, 1, n_levels)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# Pre-computed NF3 levels (8 levels optimized for Gaussian)
# These minimize expected squared error for N(0,1) distribution
NF3_LEVELS = np.array([
    -1.5104,  # bin 0: weights << -1
    -0.7560,  # bin 1: weights ~ -0.75
    -0.3829,  # bin 2: weights ~ -0.38  
    -0.1068,  # bin 3: weights ~ -0.1 (near zero, high density)
     0.1068,  # bin 4: weights ~ +0.1 (near zero, high density)
     0.3829,  # bin 5: weights ~ +0.38
     0.7560,  # bin 6: weights ~ +0.75
     1.5104,  # bin 7: weights >> +1
], dtype=np.float32)

# NF4 levels for comparison (16 levels)
NF4_LEVELS = np.array([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
], dtype=np.float32)


def quantize_nf(weight: torch.Tensor, levels: np.ndarray, group_size: int = 8):
    """
    Quantize weights using non-uniform levels with per-group scaling.
    """
    weight = weight.float()
    original_shape = weight.shape
    
    # Flatten and pad
    flat = weight.flatten()
    numel = flat.numel()
    padded_len = ((numel + group_size - 1) // group_size) * group_size
    if padded_len > numel:
        flat = F.pad(flat, (0, padded_len - numel))
    
    # Reshape to groups
    groups = flat.view(-1, group_size)
    num_groups = groups.shape[0]
    
    # Per-group absmax scaling (like bitsandbytes)
    absmax = groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    normalized = groups / absmax
    
    # Quantize: find nearest level for each weight
    levels_t = torch.from_numpy(levels).float()
    
    # Compute distances to all levels and find minimum
    dists = (normalized.unsqueeze(-1) - levels_t).abs()
    indices = dists.argmin(dim=-1)
    
    # Pack indices (for 3-bit, we can pack into bytes differently)
    indices_flat = indices.flatten().to(torch.uint8)
    
    return {
        'indices': indices_flat.numpy(),
        'scales': absmax.flatten().half().numpy(),
        'levels': levels,
        'shape': original_shape,
        'numel': numel,
        'group_size': group_size,
        'n_bits': int(np.log2(len(levels))),
    }


def dequantize_nf(data: dict) -> torch.Tensor:
    """Dequantize NF-quantized weights."""
    indices = torch.from_numpy(data['indices']).long()
    scales = torch.from_numpy(data['scales'].astype(np.float32))
    levels = torch.from_numpy(data['levels'])
    shape = data['shape']
    numel = data['numel']
    group_size = data['group_size']
    
    # Lookup levels
    values = levels[indices]
    
    # Reshape to groups and scale
    values = values.view(-1, group_size)
    values = values * scales.unsqueeze(1)
    
    # Flatten and trim
    flat = values.flatten()[:numel]
    
    return flat.view(shape)


def nf_storage_bytes(data: dict) -> int:
    """Calculate storage for NF-quantized data."""
    n_bits = data['n_bits']
    numel = data['numel']
    group_size = data['group_size']
    num_groups = (numel + group_size - 1) // group_size
    
    # Indices: n_bits per weight (packed)
    indices_bytes = (numel * n_bits + 7) // 8
    
    # Scales: FP16 per group
    scales_bytes = num_groups * 2
    
    # Levels: small constant (8 or 16 floats)
    levels_bytes = len(data['levels']) * 4
    
    return indices_bytes + scales_bytes + levels_bytes


class NFLinear(nn.Module):
    """Linear layer with NF quantization."""
    
    def __init__(self, data, bias=None):
        super().__init__()
        self.data = data
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = dequantize_nf(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


def test_nf_config(model_name, tokenizer, texts, baseline_ppl, 
                   name, levels, group_size):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"  Levels: {len(levels)}, Group size: {group_size}")
    print('='*60)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    total_orig = 0
    total_quant = 0
    total_weights = 0
    
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ['c_fc', 'c_proj']:
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_weights += weight.numel()
            total_orig += weight.numel() * 4  # FP32
            
            data = quantize_nf(weight, levels, group_size)
            total_quant += nf_storage_bytes(data)
            
            setattr(model.transformer.h[block_idx].mlp, layer_name,
                   NFLinear(data, bias))
    
    compress_fp32 = total_orig / total_quant
    bits_per_weight = (total_quant * 8) / total_weights
    
    print(f"Compression: {compress_fp32:.2f}x vs FP32, {bits_per_weight:.2f} bits/weight")
    print("Computing PPL...", end=" ", flush=True)
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
    
    del model
    
    return {
        'name': name,
        'compression_fp32': compress_fp32,
        'bits_per_weight': bits_per_weight,
        'ppl': ppl,
        'ppl_delta': delta,
    }


def main():
    print("=" * 70)
    print("NF3/NF4 - Non-uniform Quantization for 10x Compression")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:80]
    
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Compute optimal levels
    nf3_optimal = compute_nf_levels(3, 'gaussian')
    nf4_optimal = compute_nf_levels(4, 'gaussian')
    uniform_3bit = compute_nf_levels(3, 'uniform')
    
    print(f"\nNF3 levels: {NF3_LEVELS}")
    print(f"NF3 computed: {nf3_optimal}")
    print(f"Uniform 3-bit: {uniform_3bit}")
    
    # Test configurations
    configs = [
        # NF3 variants (target: 10x compression)
        ("NF3 g=8 (optimal)", NF3_LEVELS, 8),
        ("NF3 g=16", NF3_LEVELS, 16),
        ("NF3 g=32", NF3_LEVELS, 32),
        
        # Uniform 3-bit for comparison
        ("Uniform INT3 g=8", uniform_3bit, 8),
        
        # NF4 for quality comparison
        ("NF4 g=8 (optimal)", NF4_LEVELS, 8),
    ]
    
    results = []
    for name, levels, group_size in configs:
        result = test_nf_config(model_name, tokenizer, texts, baseline_ppl,
                                name, levels, group_size)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Non-uniform Quantization")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: 10x compression, <1% PPL delta")
    print()
    print(f"{'Config':<25} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "🎯 10x!" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "✓ GREAT" if r['ppl_delta'] < 1.0 else \
                 "~ OK" if r['ppl_delta'] < 5.0 else "✗"
        
        print(f"{r['name']:<25} {r['compression_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 70)
    print("Comparison: INT4 g8_fp16 = 4.00x, +0.59% PPL")
    print("=" * 80)


if __name__ == "__main__":
    main()
