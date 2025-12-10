#!/usr/bin/env python3
"""
BEST COMBINATION: Trying to maximize compression while keeping PPL reasonable.

Combining:
1. Super-blocks with quantized scales (best compression)
2. Larger block sizes for less overhead
3. Mixed precision: INT4 for MLP, INT8 for attention (if we add it)

Also trying:
- NF4 (normalized float 4-bit) - used by bitsandbytes
- Stochastic rounding
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
# NF4 (Normalized Float 4-bit) - Used by QLoRA/bitsandbytes
# =============================================================================

# NF4 quantiles for normal distribution (from bitsandbytes)
NF4_QUANTILES = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230850219727, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
])

def nf4_quantize(weight: torch.Tensor, block_size: int = 64):
    """
    NF4 quantization - maps to quantiles of normal distribution.
    Better for normally distributed weights.
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    # Pad
    padded = ((numel + block_size - 1) // block_size) * block_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, block_size)
    
    # Compute absmax per group (symmetric)
    absmax = groups.abs().max(dim=1).values
    absmax = torch.clamp(absmax, min=1e-8)
    
    # Normalize to [-1, 1]
    normalized = groups / absmax.unsqueeze(1)
    
    # Find nearest NF4 quantile
    # Expand for broadcasting: (num_groups, block_size, 1) vs (16,)
    nf4 = NF4_QUANTILES.to(normalized.device)
    distances = (normalized.unsqueeze(-1) - nf4).abs()
    q = distances.argmin(dim=-1).to(torch.uint8)
    
    # Pack
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'absmax': absmax.half(),
        'shape': weight.shape,
        'block_size': block_size,
        'numel': numel,
    }


def nf4_dequantize(data: dict) -> torch.Tensor:
    packed = data['packed']
    absmax = data['absmax'].float()
    shape = data['shape']
    block_size = data['block_size']
    numel = data['numel']
    
    # Unpack
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten()
    
    # Map back to NF4 values
    nf4 = NF4_QUANTILES.to(q.device)
    normalized = nf4[q.long()]
    
    # Denormalize
    groups = normalized.view(-1, block_size)
    weight = groups * absmax.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


def nf4_storage_bytes(data: dict) -> int:
    return data['packed'].numel() + data['absmax'].numel() * 2


# =============================================================================
# OPTIMAL INT4: Best settings from previous experiments
# =============================================================================

def optimal_int4_quantize(weight: torch.Tensor, block_size: int = 64):
    """
    Optimal INT4: asymmetric, block=64, FP16 scale+offset.
    
    Block 64 gives good balance of compression vs quality.
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
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
        'block_size': block_size,
        'numel': numel,
    }


def optimal_int4_dequantize(data: dict) -> torch.Tensor:
    packed = data['packed']
    scales = data['scales'].float()
    mins = data['mins'].float()
    shape = data['shape']
    block_size = data['block_size']
    numel = data['numel']
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    groups = q.view(-1, block_size)
    weight = groups * scales.unsqueeze(1) + mins.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


def optimal_int4_storage_bytes(data: dict) -> int:
    return data['packed'].numel() + data['scales'].numel() * 2 + data['mins'].numel() * 2


# =============================================================================
# SUPER-BLOCK V2: Optimized version
# =============================================================================

def superblock_v2_quantize(weight: torch.Tensor, super_size: int = 256, sub_size: int = 32):
    """
    Improved super-block with better scale quantization.
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    # Pad to super_size
    padded = ((numel + super_size - 1) // super_size) * super_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    super_blocks = flat.view(-1, super_size)
    num_super = super_blocks.shape[0]
    num_sub = super_size // sub_size
    
    # Global min/max per super-block
    sb_min = super_blocks.min(dim=1).values
    sb_max = super_blocks.max(dim=1).values
    sb_range = torch.clamp(sb_max - sb_min, min=1e-8)
    
    # Normalize to 0-1
    normalized = (super_blocks - sb_min.unsqueeze(1)) / sb_range.unsqueeze(1)
    
    # Sub-blocks
    sub_blocks = normalized.view(num_super, num_sub, sub_size)
    sub_min = sub_blocks.min(dim=2).values
    sub_max = sub_blocks.max(dim=2).values
    sub_range = torch.clamp(sub_max - sub_min, min=1e-8)
    
    # Quantize sub-scales to 6-bit (0-63) - fits 2 per byte with 4 bits spare
    sub_scales_q = (sub_range * 63).round().clamp(0, 63).to(torch.uint8)
    sub_mins_q = (sub_min * 63).round().clamp(0, 63).to(torch.uint8)
    
    # Dequantize for weight quantization
    sub_range_dq = sub_scales_q.float() / 63
    sub_min_dq = sub_mins_q.float() / 63
    
    # Quantize weights
    sub_normalized = (sub_blocks - sub_min_dq.unsqueeze(2)) / sub_range_dq.unsqueeze(2).clamp(min=1e-8)
    q = (sub_normalized * 15).round().clamp(0, 15).to(torch.uint8)
    
    # Pack INT4
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    # Pack sub-scales: 2 x 6-bit per byte (with 4 bits wasted, or use 8-bit)
    # For simplicity, use 8-bit
    
    return {
        'packed': packed,
        'super_mins': sb_min.half(),
        'super_ranges': sb_range.half(),
        'sub_scales': sub_scales_q,
        'sub_mins': sub_mins_q,
        'shape': weight.shape,
        'super_size': super_size,
        'sub_size': sub_size,
        'numel': numel,
    }


def superblock_v2_dequantize(data: dict) -> torch.Tensor:
    packed = data['packed']
    super_mins = data['super_mins'].float()
    super_ranges = data['super_ranges'].float()
    sub_scales = data['sub_scales'].float() / 63
    sub_mins = data['sub_mins'].float() / 63
    shape = data['shape']
    super_size = data['super_size']
    sub_size = data['sub_size']
    numel = data['numel']
    
    num_super = super_mins.numel()
    num_sub = super_size // sub_size
    
    # Unpack
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Reshape
    sub_blocks = q.view(num_super, num_sub, sub_size)
    
    # Dequantize within sub-blocks
    sub_range = sub_scales.view(num_super, num_sub, 1)
    sub_min = sub_mins.view(num_super, num_sub, 1)
    normalized = (sub_blocks / 15) * sub_range + sub_min
    
    # Dequantize super-blocks
    super_range = super_ranges.view(num_super, 1, 1)
    super_min = super_mins.view(num_super, 1, 1)
    weight = normalized * super_range + super_min
    
    weight = weight.flatten()[:numel]
    return weight.view(shape)


def superblock_v2_storage_bytes(data: dict) -> int:
    return (data['packed'].numel() +
            data['super_mins'].numel() * 2 +
            data['super_ranges'].numel() * 2 +
            data['sub_scales'].numel() +
            data['sub_mins'].numel())


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
            if self.method == 'nf4':
                self._w = nf4_dequantize(self.data)
            elif self.method == 'optimal_int4':
                self._w = optimal_int4_dequantize(self.data)
            elif self.method == 'superblock_v2':
                self._w = superblock_v2_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


def quantize_gpt2(model, method, **kwargs):
    total_orig = 0
    total_quant = 0
    
    for block in model.transformer.h:
        for name in ['c_fc', 'c_proj']:
            layer = getattr(block.mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig += weight.numel() * 2
            
            if method == 'nf4':
                data = nf4_quantize(weight, **kwargs)
                total_quant += nf4_storage_bytes(data)
            elif method == 'optimal_int4':
                data = optimal_int4_quantize(weight, **kwargs)
                total_quant += optimal_int4_storage_bytes(data)
            elif method == 'superblock_v2':
                data = superblock_v2_quantize(weight, **kwargs)
                total_quant += superblock_v2_storage_bytes(data)
            
            setattr(block.mlp, name, QuantLinear(data, method, bias))
    
    return total_orig, total_quant


def main():
    print("=" * 70)
    print("BEST COMBINATION QUANTIZATION")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:128]
    
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    configs = [
        # NF4 variants
        ("NF4-32", 'nf4', {'block_size': 32}),
        ("NF4-64", 'nf4', {'block_size': 64}),
        ("NF4-128", 'nf4', {'block_size': 128}),
        
        # Optimal INT4
        ("INT4-32", 'optimal_int4', {'block_size': 32}),
        ("INT4-64", 'optimal_int4', {'block_size': 64}),
        ("INT4-128", 'optimal_int4', {'block_size': 128}),
        
        # Super-block v2
        ("SuperV2-256/32", 'superblock_v2', {'super_size': 256, 'sub_size': 32}),
        ("SuperV2-256/64", 'superblock_v2', {'super_size': 256, 'sub_size': 64}),
        ("SuperV2-512/64", 'superblock_v2', {'super_size': 512, 'sub_size': 64}),
    ]
    
    results = []
    
    for name, method, kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
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
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'name': name,
                'compression': 0,
                'bits_per_weight': 0,
                'ppl': float('inf'),
                'ppl_delta': float('inf'),
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Method':<18} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<12} {'Status'}")
    print("-" * 65)
    
    for r in results:
        if r['ppl_delta'] == float('inf'):
            status = "ERROR"
        elif r['ppl_delta'] < 1.0:
            status = "✓ GREAT"
        elif r['ppl_delta'] < 2.0:
            status = "~ GOOD"
        elif r['ppl_delta'] < 5.0:
            status = "~ OK"
        else:
            status = "✗ FAIL"
        
        print(f"{r['name']:<18} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%       {status}")
    
    print("-" * 65)
    print("AWQ: 4x compression, <1% PPL | llama.cpp Q4_K_M: 4x, <1%")
    print("=" * 70)
    
    # Best
    valid = [r for r in results if r['ppl_delta'] < 100]
    if valid:
        best_compress = max(valid, key=lambda x: x['compression'])
        best_quality = min(valid, key=lambda x: x['ppl_delta'])
        
        print(f"\n📊 BEST RESULTS:")
        print(f"  Best compression: {best_compress['name']} ({best_compress['compression']:.2f}x, {best_compress['ppl_delta']:+.2f}%)")
        print(f"  Best quality: {best_quality['name']} ({best_quality['compression']:.2f}x, {best_quality['ppl_delta']:+.2f}%)")


if __name__ == "__main__":
    main()
