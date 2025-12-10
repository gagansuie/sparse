#!/usr/bin/env python3
"""
INT3 and Mixed Precision Quantization

Optimizations:
1. INT3 quantization (3-bit, 8 levels) - more aggressive compression
2. Mixed precision: INT4 for critical layers (first/last), INT3 for middle

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
# INT3 QUANTIZATION (3-bit, 8 levels: 0-7)
# =============================================================================

def int3_quantize(weight: torch.Tensor, group_size: int = 16):
    """
    Quantize to INT3 (3-bit, 8 levels).
    
    Packing: 8 int3 values → 3 bytes (24 bits)
    - More complex packing but better compression
    
    For simplicity, we'll pack 2 int3 values per byte (wastes 2 bits)
    or pack 8 values into 3 bytes for optimal storage.
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    # Pad to multiple of group_size
    padded = ((numel + group_size - 1) // group_size) * group_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, group_size)
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    
    scale = (g_max - g_min) / 7.0  # 3-bit range: 0-7
    scale = torch.clamp(scale, min=1e-8)
    offset = g_min
    
    q = ((groups - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 7).to(torch.uint8)
    q_flat = q.flatten()
    
    # Pack 8 int3 values into 3 bytes (optimal)
    # Pad to multiple of 8
    pack_padded = ((q_flat.numel() + 7) // 8) * 8
    if pack_padded > q_flat.numel():
        q_flat = F.pad(q_flat, (0, pack_padded - q_flat.numel()))
    
    q_flat = q_flat.view(-1, 8)
    
    # Pack: 8 x 3-bit = 24 bits = 3 bytes
    # byte0 = v0[2:0] | v1[2:0] << 3 | v2[1:0] << 6
    # byte1 = v2[2] | v3[2:0] << 1 | v4[2:0] << 4 | v5[0] << 7
    # byte2 = v5[2:1] | v6[2:0] << 2 | v7[2:0] << 5
    
    v = q_flat
    byte0 = (v[:, 0] | (v[:, 1] << 3) | ((v[:, 2] & 0x3) << 6)).to(torch.uint8)
    byte1 = ((v[:, 2] >> 2) | (v[:, 3] << 1) | (v[:, 4] << 4) | ((v[:, 5] & 0x1) << 7)).to(torch.uint8)
    byte2 = ((v[:, 5] >> 1) | (v[:, 6] << 2) | (v[:, 7] << 5)).to(torch.uint8)
    
    packed = torch.stack([byte0, byte1, byte2], dim=1).flatten()
    
    return {
        'packed': packed,
        'scales': scale.half(),
        'offsets': offset.half(),
        'shape': weight.shape,
        'group_size': group_size,
        'numel': numel,
        'bits': 3,
    }


def int3_dequantize(data: dict) -> torch.Tensor:
    """Dequantize INT3."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    shape = data['shape']
    group_size = data['group_size']
    numel = data['numel']
    
    # Unpack 3 bytes → 8 int3 values
    packed = packed.view(-1, 3)
    byte0, byte1, byte2 = packed[:, 0], packed[:, 1], packed[:, 2]
    
    v0 = byte0 & 0x7
    v1 = (byte0 >> 3) & 0x7
    v2 = ((byte0 >> 6) | ((byte1 & 0x1) << 2)) & 0x7
    v3 = (byte1 >> 1) & 0x7
    v4 = (byte1 >> 4) & 0x7
    v5 = ((byte1 >> 7) | ((byte2 & 0x3) << 1)) & 0x7
    v6 = (byte2 >> 2) & 0x7
    v7 = (byte2 >> 5) & 0x7
    
    q = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1).flatten().float()
    
    # Reshape to groups and dequantize
    groups = q.view(-1, group_size)
    weight = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


# =============================================================================
# INT4 QUANTIZATION (for comparison and mixed precision)
# =============================================================================

def int4_quantize(weight: torch.Tensor, group_size: int = 16):
    """Quantize to INT4."""
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    padded = ((numel + group_size - 1) // group_size) * group_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, group_size)
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
        'shape': weight.shape,
        'group_size': group_size,
        'numel': numel,
        'bits': 4,
    }


def int4_dequantize(data: dict) -> torch.Tensor:
    """Dequantize INT4."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    shape = data['shape']
    group_size = data['group_size']
    numel = data['numel']
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    groups = q.view(-1, group_size)
    weight = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


# =============================================================================
# INT2 QUANTIZATION (2-bit, 4 levels: 0-3) - very aggressive
# =============================================================================

def int2_quantize(weight: torch.Tensor, group_size: int = 16):
    """Quantize to INT2 (2-bit, 4 levels)."""
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    padded = ((numel + group_size - 1) // group_size) * group_size
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    groups = flat.view(-1, group_size)
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    
    scale = (g_max - g_min) / 3.0  # 2-bit range: 0-3
    scale = torch.clamp(scale, min=1e-8)
    offset = g_min
    
    q = ((groups - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 3).to(torch.uint8)
    q_flat = q.flatten()
    
    # Pack 4 int2 values per byte
    padded_len = ((q_flat.numel() + 3) // 4) * 4
    if padded_len > q_flat.numel():
        q_flat = F.pad(q_flat, (0, padded_len - q_flat.numel()))
    
    q_flat = q_flat.view(-1, 4)
    packed = (q_flat[:, 0] | (q_flat[:, 1] << 2) | (q_flat[:, 2] << 4) | (q_flat[:, 3] << 6)).to(torch.uint8)
    
    return {
        'packed': packed,
        'scales': scale.half(),
        'offsets': offset.half(),
        'shape': weight.shape,
        'group_size': group_size,
        'numel': numel,
        'bits': 2,
    }


def int2_dequantize(data: dict) -> torch.Tensor:
    """Dequantize INT2."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    shape = data['shape']
    group_size = data['group_size']
    numel = data['numel']
    
    v0 = packed & 0x3
    v1 = (packed >> 2) & 0x3
    v2 = (packed >> 4) & 0x3
    v3 = (packed >> 6) & 0x3
    q = torch.stack([v0, v1, v2, v3], dim=1).flatten().float()
    
    groups = q.view(-1, group_size)
    weight = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    weight = weight.flatten()[:numel]
    
    return weight.view(shape)


# =============================================================================
# STORAGE CALCULATION
# =============================================================================

def storage_bytes(data: dict) -> int:
    """Calculate storage bytes."""
    return (data['packed'].numel() + 
            data['scales'].numel() * 2 +  # FP16
            data['offsets'].numel() * 2)  # FP16


# =============================================================================
# QUANTIZED LINEAR LAYER
# =============================================================================

class QuantLinear(nn.Module):
    def __init__(self, data, bias=None):
        super().__init__()
        self.data = data
        self.bits = data['bits']
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            if self.bits == 4:
                self._w = int4_dequantize(self.data)
            elif self.bits == 3:
                self._w = int3_dequantize(self.data)
            elif self.bits == 2:
                self._w = int2_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


# =============================================================================
# QUANTIZE MODEL
# =============================================================================

def quantize_gpt2(model, config):
    """
    Quantize GPT-2 with given config.
    
    config can be:
    - {'bits': 4, 'group_size': 16}  # uniform
    - {'mixed': True, 'critical_bits': 4, 'other_bits': 3, 'group_size': 16}  # mixed
    """
    total_orig = 0
    total_quant = 0
    num_layers = len(model.transformer.h)
    
    for layer_idx, block in enumerate(model.transformer.h):
        # Determine bits for this layer
        if config.get('mixed', False):
            # First 2 and last 2 layers are critical
            if layer_idx < 2 or layer_idx >= num_layers - 2:
                bits = config['critical_bits']
            else:
                bits = config['other_bits']
        else:
            bits = config['bits']
        
        group_size = config['group_size']
        
        for name in ['c_fc', 'c_proj']:
            layer = getattr(block.mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig += weight.numel() * 2  # FP16 baseline
            
            if bits == 4:
                data = int4_quantize(weight, group_size)
            elif bits == 3:
                data = int3_quantize(weight, group_size)
            elif bits == 2:
                data = int2_quantize(weight, group_size)
            
            total_quant += storage_bytes(data)
            setattr(block.mlp, name, QuantLinear(data, bias))
    
    return total_orig, total_quant


def main():
    print("=" * 70)
    print("INT3 AND MIXED PRECISION QUANTIZATION")
    print("Target: Beat AWQ's 4x compression")
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
    
    # Configurations to test
    configs = [
        # Reference: INT4
        ("int4_g16", {'bits': 4, 'group_size': 16}),
        ("int4_g32", {'bits': 4, 'group_size': 32}),
        
        # INT3 (more aggressive)
        ("int3_g16", {'bits': 3, 'group_size': 16}),
        ("int3_g32", {'bits': 3, 'group_size': 32}),
        ("int3_g64", {'bits': 3, 'group_size': 64}),
        
        # INT2 (very aggressive)
        ("int2_g16", {'bits': 2, 'group_size': 16}),
        ("int2_g32", {'bits': 2, 'group_size': 32}),
        
        # Mixed precision: INT4 for critical, INT3 for others
        ("mixed_4_3_g16", {'mixed': True, 'critical_bits': 4, 'other_bits': 3, 'group_size': 16}),
        ("mixed_4_3_g32", {'mixed': True, 'critical_bits': 4, 'other_bits': 3, 'group_size': 32}),
        
        # Mixed: INT4 for critical, INT2 for others
        ("mixed_4_2_g32", {'mixed': True, 'critical_bits': 4, 'other_bits': 2, 'group_size': 32}),
    ]
    
    results = []
    
    for name, config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        orig_bytes, quant_bytes = quantize_gpt2(model, config)
        
        compression = orig_bytes / quant_bytes
        total_weights = orig_bytes / 2
        bits_per_weight = (quant_bytes * 8) / total_weights
        
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
    print(f"{'Config':<18} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status':<8} {'vs AWQ'}")
    print("-" * 70)
    
    for r in results:
        if r['ppl_delta'] < 1.0:
            status = "✓ GREAT"
        elif r['ppl_delta'] < 2.0:
            status = "~ GOOD"
        elif r['ppl_delta'] < 5.0:
            status = "~ OK"
        else:
            status = "✗ FAIL"
        
        awq = "🎯 BEATS!" if r['compression'] > 4.0 and r['ppl_delta'] < 2.0 else ""
        print(f"{r['name']:<18} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status:<8} {awq}")
    
    print("-" * 70)
    print("AWQ reference: 4x compression, <1% PPL delta")
    print("=" * 70)
    
    # Find best results
    print("\n📊 ANALYSIS:")
    
    # Best <1% PPL
    best_quality = [r for r in results if r['ppl_delta'] < 1.0]
    if best_quality:
        best = max(best_quality, key=lambda x: x['compression'])
        print(f"  Best <1% PPL: {best['name']} ({best['compression']:.2f}x, {best['ppl_delta']:+.2f}%)")
    
    # Best <2% PPL
    best_ok = [r for r in results if r['ppl_delta'] < 2.0]
    if best_ok:
        best = max(best_ok, key=lambda x: x['compression'])
        print(f"  Best <2% PPL: {best['name']} ({best['compression']:.2f}x, {best['ppl_delta']:+.2f}%)")
    
    # Best that beats AWQ (>4x, <2% PPL)
    beats_awq = [r for r in results if r['compression'] > 4.0 and r['ppl_delta'] < 2.0]
    if beats_awq:
        best = min(beats_awq, key=lambda x: x['ppl_delta'])
        print(f"\n  🎯 BEATS AWQ: {best['name']} ({best['compression']:.2f}x, {best['ppl_delta']:+.2f}%)")
    else:
        # Closest to beating AWQ
        close = [r for r in results if r['ppl_delta'] < 5.0]
        if close:
            best = max(close, key=lambda x: x['compression'])
            print(f"\n  Closest to AWQ: {best['name']} ({best['compression']:.2f}x, {best['ppl_delta']:+.2f}%)")
            print(f"  Need {4.0 - best['compression']:.2f}x more compression OR better PPL")


if __name__ == "__main__":
    main()
