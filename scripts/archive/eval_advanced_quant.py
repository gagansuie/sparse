#!/usr/bin/env python3
"""
ADVANCED QUANTIZATION TECHNIQUES

Inspired by llama.cpp Q4_K_M and other methods:

1. K-QUANT STYLE: Higher precision for critical layers
   - First/last layers: INT8
   - Attention layers: INT6  
   - MLP middle layers: INT4

2. SUPER-BLOCKS: Nested quantization
   - Quantize scales to INT8 instead of FP16
   - Reduces scale overhead by 2x

3. IMPORTANCE-WEIGHTED: Use weight magnitude
   - Keep top 1% weights in higher precision
   - Quantize rest more aggressively

4. OPTIMAL BLOCK SIZES: Match llama.cpp
   - Block size 32 (like Q4_0)
   - With FP16 scale per block

Target: Beat AWQ's 4x compression with <2% PPL delta
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity."""
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
# TECHNIQUE 1: SUPER-BLOCKS (Quantized Scales)
# =============================================================================

def int4_superblock_quantize(weight: torch.Tensor, block_size: int = 32):
    """
    INT4 with super-blocks: scales are quantized to INT8.
    
    Structure (like llama.cpp Q4_K):
    - Super-block of 256 weights
    - Contains 8 sub-blocks of 32 weights each
    - One FP16 super-scale for the whole super-block
    - 8 INT8 sub-scales (relative to super-scale)
    - Packed INT4 weights
    
    Storage per 256 weights:
    - 128 bytes (packed int4)
    - 2 bytes (FP16 super-scale)
    - 8 bytes (INT8 sub-scales)
    - 2 bytes (FP16 min for asymmetric)
    = 140 bytes for 256 weights = 4.375 bits/weight
    
    vs FP16: 512 bytes → 3.66x compression
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    super_block_size = 256
    sub_block_size = block_size
    num_sub_blocks = super_block_size // sub_block_size
    
    # Pad to multiple of super_block_size
    padded = ((numel + super_block_size - 1) // super_block_size) * super_block_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    # Reshape to super-blocks
    super_blocks = flat.view(-1, super_block_size)
    num_super_blocks = super_blocks.shape[0]
    
    # For each super-block, compute global min/max
    sb_min = super_blocks.min(dim=1).values
    sb_max = super_blocks.max(dim=1).values
    sb_range = sb_max - sb_min
    sb_range = torch.clamp(sb_range, min=1e-8)
    
    # Normalize super-blocks to 0-1 range
    normalized = (super_blocks - sb_min.unsqueeze(1)) / sb_range.unsqueeze(1)
    
    # Reshape to sub-blocks within each super-block
    sub_blocks = normalized.view(num_super_blocks, num_sub_blocks, sub_block_size)
    
    # For each sub-block, compute local min/max (within 0-1 range)
    sub_min = sub_blocks.min(dim=2).values  # (num_super, num_sub)
    sub_max = sub_blocks.max(dim=2).values
    sub_range = sub_max - sub_min
    sub_range = torch.clamp(sub_range, min=1e-8)
    
    # Quantize sub-block scales to INT8 (0-255)
    sub_scales_q = (sub_range * 255).round().clamp(0, 255).to(torch.uint8)
    sub_offsets_q = (sub_min * 255).round().clamp(0, 255).to(torch.uint8)
    
    # Dequantize for actual quantization
    sub_range_dq = sub_scales_q.float() / 255
    sub_min_dq = sub_offsets_q.float() / 255
    
    # Quantize weights within sub-blocks
    sub_normalized = (sub_blocks - sub_min_dq.unsqueeze(2)) / sub_range_dq.unsqueeze(2).clamp(min=1e-8)
    q = (sub_normalized * 15).round().clamp(0, 15).to(torch.uint8)
    
    # Pack INT4
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'super_scales': sb_range.half(),  # FP16
        'super_offsets': sb_min.half(),   # FP16
        'sub_scales': sub_scales_q,       # INT8
        'sub_offsets': sub_offsets_q,     # INT8
        'shape': weight.shape,
        'block_size': block_size,
        'super_block_size': super_block_size,
        'numel': numel,
    }


def int4_superblock_dequantize(data: dict) -> torch.Tensor:
    """Dequantize super-block INT4."""
    packed = data['packed']
    super_scales = data['super_scales'].float()
    super_offsets = data['super_offsets'].float()
    sub_scales = data['sub_scales'].float() / 255
    sub_offsets = data['sub_offsets'].float() / 255
    shape = data['shape']
    block_size = data['block_size']
    super_block_size = data['super_block_size']
    numel = data['numel']
    
    num_sub_blocks = super_block_size // block_size
    
    # Unpack INT4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Reshape to sub-blocks
    num_super_blocks = super_scales.numel()
    sub_blocks = q.view(num_super_blocks, num_sub_blocks, block_size)
    
    # Dequantize within sub-blocks (to 0-1 normalized range)
    sub_range = sub_scales.view(num_super_blocks, num_sub_blocks, 1)
    sub_min = sub_offsets.view(num_super_blocks, num_sub_blocks, 1)
    normalized = (sub_blocks / 15) * sub_range + sub_min
    
    # Dequantize super-blocks (to original range)
    super_range = super_scales.view(num_super_blocks, 1, 1)
    super_min = super_offsets.view(num_super_blocks, 1, 1)
    weight = normalized * super_range + super_min
    
    weight = weight.flatten()[:numel]
    return weight.view(shape)


def superblock_storage_bytes(data: dict) -> int:
    """Calculate storage for super-block quantization."""
    return (data['packed'].numel() +
            data['super_scales'].numel() * 2 +
            data['super_offsets'].numel() * 2 +
            data['sub_scales'].numel() +
            data['sub_offsets'].numel())


# =============================================================================
# TECHNIQUE 2: IMPORTANCE-WEIGHTED QUANTIZATION
# =============================================================================

def int4_importance_quantize(weight: torch.Tensor, group_size: int = 32, keep_top_pct: float = 0.01):
    """
    Keep top weights in FP16, quantize rest to INT4.
    
    The idea: Large magnitude weights are most important.
    Keep top 1% in FP16, quantize rest.
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    # Find top weights by magnitude
    num_keep = max(1, int(numel * keep_top_pct))
    magnitudes = flat.abs()
    _, top_indices = magnitudes.topk(num_keep)
    
    # Create mask for top weights
    mask = torch.zeros(numel, dtype=torch.bool)
    mask[top_indices] = True
    
    # Store top weights in FP16
    top_weights = flat[mask].half()
    top_indices_stored = top_indices.to(torch.int32)
    
    # Zero out top weights for quantization
    flat_for_quant = flat.clone()
    flat_for_quant[mask] = 0
    
    # Quantize remaining weights
    padded = ((numel + group_size - 1) // group_size) * group_size
    if padded > numel:
        flat_for_quant = F.pad(flat_for_quant, (0, padded - numel))
    
    groups = flat_for_quant.view(-1, group_size)
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    scale = (g_max - g_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)
    offset = g_min
    
    q = ((groups - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'scales': scale.half(),
        'offsets': offset.half(),
        'top_weights': top_weights,
        'top_indices': top_indices_stored,
        'shape': weight.shape,
        'group_size': group_size,
        'numel': numel,
    }


def int4_importance_dequantize(data: dict) -> torch.Tensor:
    """Dequantize importance-weighted INT4."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    top_weights = data['top_weights'].float()
    top_indices = data['top_indices']
    shape = data['shape']
    group_size = data['group_size']
    numel = data['numel']
    
    # Unpack INT4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Dequantize
    groups = q.view(-1, group_size)
    weight = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    # Restore top weights
    weight[top_indices] = top_weights
    
    return weight.view(shape)


def importance_storage_bytes(data: dict) -> int:
    """Calculate storage for importance-weighted quantization."""
    return (data['packed'].numel() +
            data['scales'].numel() * 2 +
            data['offsets'].numel() * 2 +
            data['top_weights'].numel() * 2 +
            data['top_indices'].numel() * 4)


# =============================================================================
# TECHNIQUE 3: LLAMA.CPP Q4_0 STYLE (Simple, block=32, FP16 scale only)
# =============================================================================

def q4_0_quantize(weight: torch.Tensor):
    """
    llama.cpp Q4_0 style: symmetric quantization, block=32, FP16 scale only.
    
    No offset (symmetric around 0), just scale.
    Simpler but may lose accuracy for asymmetric distributions.
    
    Storage: 32 weights → 16 bytes (packed) + 2 bytes (scale) = 18 bytes
    = 4.5 bits/weight → 3.56x compression vs FP16
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    block_size = 32
    
    # Pad
    padded = ((numel + block_size - 1) // block_size) * block_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, block_size)
    
    # Symmetric: scale based on max absolute value
    max_abs = groups.abs().max(dim=1).values
    scale = max_abs / 7.0  # Range: -7 to +7 (shifted to 0-15)
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize to -7..+7, then shift to 0..15
    q = (groups / scale.unsqueeze(1)).round().clamp(-7, 7)
    q = (q + 8).to(torch.uint8)  # Shift to 0-15
    
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'scales': scale.half(),
        'shape': weight.shape,
        'numel': numel,
    }


def q4_0_dequantize(data: dict) -> torch.Tensor:
    """Dequantize Q4_0."""
    packed = data['packed']
    scales = data['scales'].float()
    shape = data['shape']
    numel = data['numel']
    block_size = 32
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Shift back and dequantize
    q = q - 8  # Back to -7..+7
    groups = q.view(-1, block_size)
    weight = groups * scales.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


def q4_0_storage_bytes(data: dict) -> int:
    return data['packed'].numel() + data['scales'].numel() * 2


# =============================================================================
# TECHNIQUE 4: Q4_1 STYLE (Asymmetric, block=32)
# =============================================================================

def q4_1_quantize(weight: torch.Tensor):
    """
    llama.cpp Q4_1 style: asymmetric, block=32, FP16 scale + FP16 min.
    
    Storage: 32 weights → 16 bytes + 2 + 2 = 20 bytes
    = 5 bits/weight → 3.2x compression vs FP16
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    block_size = 32
    
    padded = ((numel + block_size - 1) // block_size) * block_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, block_size)
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    scale = (g_max - g_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)
    
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'scales': scale.half(),
        'mins': g_min.half(),
        'shape': weight.shape,
        'numel': numel,
    }


def q4_1_dequantize(data: dict) -> torch.Tensor:
    packed = data['packed']
    scales = data['scales'].float()
    mins = data['mins'].float()
    shape = data['shape']
    numel = data['numel']
    block_size = 32
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    groups = q.view(-1, block_size)
    weight = groups * scales.unsqueeze(1) + mins.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


def q4_1_storage_bytes(data: dict) -> int:
    return data['packed'].numel() + data['scales'].numel() * 2 + data['mins'].numel() * 2


# =============================================================================
# QUANTIZED LINEAR
# =============================================================================

class QuantLinear(nn.Module):
    def __init__(self, data, method, bias=None):
        super().__init__()
        self.data = data
        self.method = method
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            if self.method == 'superblock':
                self._w = int4_superblock_dequantize(self.data)
            elif self.method == 'importance':
                self._w = int4_importance_dequantize(self.data)
            elif self.method == 'q4_0':
                self._w = q4_0_dequantize(self.data)
            elif self.method == 'q4_1':
                self._w = q4_1_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


# =============================================================================
# QUANTIZE MODEL
# =============================================================================

def quantize_gpt2(model, method, **kwargs):
    """Quantize GPT-2 MLP layers."""
    total_orig = 0
    total_quant = 0
    
    for block in model.transformer.h:
        for name in ['c_fc', 'c_proj']:
            layer = getattr(block.mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig += weight.numel() * 2  # FP16 baseline
            
            if method == 'superblock':
                data = int4_superblock_quantize(weight, **kwargs)
                total_quant += superblock_storage_bytes(data)
            elif method == 'importance':
                data = int4_importance_quantize(weight, **kwargs)
                total_quant += importance_storage_bytes(data)
            elif method == 'q4_0':
                data = q4_0_quantize(weight)
                total_quant += q4_0_storage_bytes(data)
            elif method == 'q4_1':
                data = q4_1_quantize(weight)
                total_quant += q4_1_storage_bytes(data)
            
            setattr(block.mlp, name, QuantLinear(data, method, bias))
    
    return total_orig, total_quant


def main():
    print("=" * 70)
    print("ADVANCED QUANTIZATION TECHNIQUES")
    print("Inspired by llama.cpp Q4_K_M")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:128]
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Configurations
    configs = [
        # llama.cpp style
        ("Q4_0 (symmetric)", 'q4_0', {}),
        ("Q4_1 (asymmetric)", 'q4_1', {}),
        
        # Super-blocks (quantized scales)
        ("SuperBlock-32", 'superblock', {'block_size': 32}),
        ("SuperBlock-64", 'superblock', {'block_size': 64}),
        
        # Importance-weighted
        ("Importance-1%", 'importance', {'group_size': 32, 'keep_top_pct': 0.01}),
        ("Importance-2%", 'importance', {'group_size': 32, 'keep_top_pct': 0.02}),
        ("Importance-5%", 'importance', {'group_size': 32, 'keep_top_pct': 0.05}),
    ]
    
    results = []
    
    for name, method, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        orig_bytes, quant_bytes = quantize_gpt2(model, method, **kwargs)
        
        compression = orig_bytes / quant_bytes
        bits_per_weight = (quant_bytes * 8) / (orig_bytes / 2)
        
        print(f"Compression: {orig_bytes/1e6:.2f} MB → {quant_bytes/1e6:.2f} MB = {compression:.2f}x")
        print(f"Bits/weight: {bits_per_weight:.2f}")
        
        print("Computing PPL...", end=" ", flush=True)
        ppl = compute_perplexity(model, tokenizer, texts)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
        
        results.append({
            'name': name,
            'compression': compression,
            'bits_per_weight': bits_per_weight,
            'ppl': ppl,
            'ppl_delta': delta,
        })
        
        del model
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Method':<20} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status':<10} {'vs AWQ'}")
    print("-" * 75)
    
    for r in results:
        if r['ppl_delta'] < 1.0:
            status = "✓ GREAT"
        elif r['ppl_delta'] < 2.0:
            status = "~ GOOD"
        elif r['ppl_delta'] < 5.0:
            status = "~ OK"
        else:
            status = "✗ FAIL"
        
        awq = "🎯 BEATS!" if r['compression'] >= 4.0 and r['ppl_delta'] < 2.0 else ""
        print(f"{r['name']:<20} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status:<10} {awq}")
    
    print("-" * 75)
    print("AWQ reference: 4x compression, <1% PPL delta")
    print("=" * 70)
    
    # Best results
    print("\n📊 BEST RESULTS:")
    
    good_results = [r for r in results if r['ppl_delta'] < 5.0]
    if good_results:
        best = max(good_results, key=lambda x: x['compression'])
        print(f"  Best compression (<5% PPL): {best['name']}")
        print(f"    Compression: {best['compression']:.2f}x")
        print(f"    PPL Delta: {best['ppl_delta']:+.2f}%")
        print(f"    Bits/weight: {best['bits_per_weight']:.2f}")


if __name__ == "__main__":
    main()
