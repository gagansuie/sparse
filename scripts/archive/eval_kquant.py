#!/usr/bin/env python3
"""
K-QUANT IMPLEMENTATION (llama.cpp style)

Key techniques from llama.cpp Q4_K_M:

1. SUPER-BLOCKS: 256 weights = 8 blocks of 32
   - One FP16 super-scale and super-min for the whole super-block
   - 8 x 6-bit sub-scales and sub-mins (quantized!)
   - Packed INT4 weights

2. MIXED PRECISION BY LAYER TYPE:
   - Q4_K_M uses Q6_K for attention.wv and feed_forward.w2 (half of them)
   - Q4_K for everything else
   - Output layer always Q6

3. STORAGE CALCULATION for Q4_K:
   - 256 weights → 128 bytes (packed int4)
   - 2 bytes (FP16 super-scale) + 2 bytes (FP16 super-min)
   - 8 bytes (8 x 6-bit scales, packed) + 8 bytes (8 x 6-bit mins)
   - Total: 128 + 4 + 16 = 148 bytes for 256 weights
   - = 4.625 bits/weight → 3.46x compression vs FP16

Let's implement this EXACTLY like llama.cpp!
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
# Q4_K: 4-bit with 6-bit quantized scales (llama.cpp style)
# =============================================================================

def q4_k_quantize(weight: torch.Tensor):
    """
    Q4_K quantization (llama.cpp style):
    - Super-block: 256 weights = 8 blocks of 32
    - Scales and mins quantized to 6-bit
    - 4.5 bits per weight
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    SUPER_BLOCK_SIZE = 256
    BLOCK_SIZE = 32
    NUM_BLOCKS = SUPER_BLOCK_SIZE // BLOCK_SIZE  # 8
    
    # Pad to super-block size
    padded = ((numel + SUPER_BLOCK_SIZE - 1) // SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    # Reshape to super-blocks
    super_blocks = flat.view(-1, SUPER_BLOCK_SIZE)
    num_super = super_blocks.shape[0]
    
    # Compute super-block min and max (for the whole 256 weights)
    sb_min = super_blocks.min(dim=1).values
    sb_max = super_blocks.max(dim=1).values
    sb_range = sb_max - sb_min
    sb_range = torch.clamp(sb_range, min=1e-8)
    
    # Normalize super-blocks to 0-1
    normalized = (super_blocks - sb_min.unsqueeze(1)) / sb_range.unsqueeze(1)
    
    # Reshape to blocks within super-blocks
    blocks = normalized.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    # Compute per-block min and max (within 0-1 range)
    block_min = blocks.min(dim=2).values  # (num_super, 8)
    block_max = blocks.max(dim=2).values
    block_range = block_max - block_min
    block_range = torch.clamp(block_range, min=1e-8)
    
    # Quantize block scales and mins to 6-bit (0-63)
    # This is the key innovation of k-quants!
    block_scales_q = (block_range * 63).round().clamp(0, 63).to(torch.uint8)
    block_mins_q = (block_min * 63).round().clamp(0, 63).to(torch.uint8)
    
    # Dequantize for actual weight quantization
    block_range_dq = block_scales_q.float() / 63
    block_min_dq = block_mins_q.float() / 63
    
    # Quantize weights within blocks to 4-bit
    block_normalized = (blocks - block_min_dq.unsqueeze(2)) / block_range_dq.unsqueeze(2).clamp(min=1e-8)
    q = (block_normalized * 15).round().clamp(0, 15).to(torch.uint8)
    
    # Pack INT4 (2 per byte)
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    # Pack 6-bit values: 4 x 6-bit = 24 bits = 3 bytes
    # For simplicity, store as uint8 (wastes 2 bits per value)
    # In production, would pack more efficiently
    
    return {
        'packed': packed,
        'd': sb_range.half(),      # FP16 super-scale
        'dmin': sb_min.half(),     # FP16 super-min  
        'scales': block_scales_q,  # 6-bit (stored as uint8)
        'mins': block_mins_q,      # 6-bit (stored as uint8)
        'shape': weight.shape,
        'numel': numel,
    }


def q4_k_dequantize(data: dict) -> torch.Tensor:
    """Dequantize Q4_K."""
    packed = data['packed']
    d = data['d'].float()
    dmin = data['dmin'].float()
    scales = data['scales'].float() / 63
    mins = data['mins'].float() / 63
    shape = data['shape']
    numel = data['numel']
    
    SUPER_BLOCK_SIZE = 256
    BLOCK_SIZE = 32
    NUM_BLOCKS = 8
    
    num_super = d.numel()
    
    # Unpack INT4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Reshape to blocks
    blocks = q.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    # Dequantize within blocks (to 0-1 normalized)
    block_range = scales.view(num_super, NUM_BLOCKS, 1)
    block_min = mins.view(num_super, NUM_BLOCKS, 1)
    normalized = (blocks / 15) * block_range + block_min
    
    # Dequantize super-blocks (to original range)
    super_range = d.view(num_super, 1, 1)
    super_min = dmin.view(num_super, 1, 1)
    weight = normalized * super_range + super_min
    
    weight = weight.flatten()[:numel]
    return weight.view(shape)


def q4_k_storage_bytes(data: dict) -> int:
    """
    Storage for Q4_K:
    - packed: numel/2 bytes
    - d, dmin: 2 + 2 = 4 bytes per super-block
    - scales, mins: 8 + 8 = 16 bytes per super-block (could be 12 with proper packing)
    """
    num_super = data['d'].numel()
    return (data['packed'].numel() +  # INT4 packed
            num_super * 4 +            # FP16 d and dmin
            num_super * 16)            # 8 scales + 8 mins (uint8 each)


# =============================================================================
# Q6_K: 6-bit for critical layers
# =============================================================================

def q6_k_quantize(weight: torch.Tensor):
    """
    Q6_K quantization for critical layers.
    - Super-block: 256 weights = 16 blocks of 16
    - 8-bit scales
    - 6.5625 bits per weight
    """
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    SUPER_BLOCK_SIZE = 256
    BLOCK_SIZE = 16
    NUM_BLOCKS = SUPER_BLOCK_SIZE // BLOCK_SIZE  # 16
    
    padded = ((numel + SUPER_BLOCK_SIZE - 1) // SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    super_blocks = flat.view(-1, SUPER_BLOCK_SIZE)
    num_super = super_blocks.shape[0]
    
    # Super-block scale (symmetric)
    sb_absmax = super_blocks.abs().max(dim=1).values
    sb_absmax = torch.clamp(sb_absmax, min=1e-8)
    
    # Normalize to -1..1
    normalized = super_blocks / sb_absmax.unsqueeze(1)
    
    # Reshape to blocks
    blocks = normalized.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    # Per-block scales (8-bit)
    block_absmax = blocks.abs().max(dim=2).values
    block_scales_q = (block_absmax * 127).round().clamp(0, 127).to(torch.uint8)
    
    # Dequantize for quantization
    block_absmax_dq = block_scales_q.float() / 127
    
    # Quantize to 6-bit symmetric (-32..31 → 0..63)
    block_normalized = blocks / block_absmax_dq.unsqueeze(2).clamp(min=1e-8)
    q = (block_normalized * 31).round().clamp(-32, 31)
    q = (q + 32).to(torch.uint8)  # Shift to 0..63
    
    # Pack 6-bit: 4 values → 3 bytes
    q_flat = q.flatten()
    # Pad to multiple of 4
    pad_len = (4 - q_flat.numel() % 4) % 4
    if pad_len > 0:
        q_flat = F.pad(q_flat, (0, pad_len))
    
    q_flat = q_flat.view(-1, 4)
    # Pack: v0[5:0] | v1[5:0] | v2[5:0] | v3[5:0] → 3 bytes
    byte0 = (q_flat[:, 0] | ((q_flat[:, 1] & 0x3) << 6)).to(torch.uint8)
    byte1 = ((q_flat[:, 1] >> 2) | ((q_flat[:, 2] & 0xF) << 4)).to(torch.uint8)
    byte2 = ((q_flat[:, 2] >> 4) | (q_flat[:, 3] << 2)).to(torch.uint8)
    packed = torch.stack([byte0, byte1, byte2], dim=1).flatten()
    
    return {
        'packed': packed,
        'd': sb_absmax.half(),
        'scales': block_scales_q,
        'shape': weight.shape,
        'numel': numel,
        'num_super': num_super,
    }


def q6_k_dequantize(data: dict) -> torch.Tensor:
    """Dequantize Q6_K."""
    packed = data['packed']
    d = data['d'].float()
    scales = data['scales'].float() / 127
    shape = data['shape']
    numel = data['numel']
    num_super = data['num_super']
    
    BLOCK_SIZE = 16
    NUM_BLOCKS = 16
    
    # Unpack 6-bit
    packed = packed.view(-1, 3)
    byte0, byte1, byte2 = packed[:, 0], packed[:, 1], packed[:, 2]
    
    v0 = byte0 & 0x3F
    v1 = ((byte0 >> 6) | ((byte1 & 0xF) << 2)) & 0x3F
    v2 = ((byte1 >> 4) | ((byte2 & 0x3) << 4)) & 0x3F
    v3 = (byte2 >> 2) & 0x3F
    
    q = torch.stack([v0, v1, v2, v3], dim=1).flatten().float()
    q = q[:num_super * 256]  # Trim padding
    
    # Shift back to -32..31
    q = q - 32
    
    # Reshape
    blocks = q.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    # Dequantize
    block_scale = scales.view(num_super, NUM_BLOCKS, 1)
    normalized = (blocks / 31) * block_scale
    
    super_scale = d.view(num_super, 1, 1)
    weight = normalized * super_scale
    
    weight = weight.flatten()[:numel]
    return weight.view(shape)


def q6_k_storage_bytes(data: dict) -> int:
    num_super = data['num_super']
    # 256 weights → 192 bytes (packed 6-bit) + 2 (d) + 16 (scales)
    return data['packed'].numel() + num_super * 2 + data['scales'].numel()


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
            if self.method == 'q4_k':
                self._w = q4_k_dequantize(self.data)
            elif self.method == 'q6_k':
                self._w = q6_k_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


# =============================================================================
# QUANTIZE MODEL WITH MIXED PRECISION (Q4_K_M style)
# =============================================================================

def quantize_gpt2_kquant(model, mode='q4_k_s'):
    """
    Quantize GPT-2 with k-quant style.
    
    Modes:
    - q4_k_s: All Q4_K (simple)
    - q4_k_m: Q6_K for c_proj (output projection), Q4_K for c_fc
    - q6_k: All Q6_K (highest quality)
    """
    total_orig = 0
    total_quant = 0
    
    num_layers = len(model.transformer.h)
    
    for layer_idx, block in enumerate(model.transformer.h):
        for name in ['c_fc', 'c_proj']:
            layer = getattr(block.mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig += weight.numel() * 2  # FP16 baseline
            
            # Determine quantization method
            if mode == 'q6_k':
                method = 'q6_k'
            elif mode == 'q4_k_m':
                # Q4_K_M: Use Q6_K for c_proj in half the layers
                if name == 'c_proj' and layer_idx % 2 == 0:
                    method = 'q6_k'
                else:
                    method = 'q4_k'
            else:  # q4_k_s
                method = 'q4_k'
            
            if method == 'q4_k':
                data = q4_k_quantize(weight)
                total_quant += q4_k_storage_bytes(data)
            else:
                data = q6_k_quantize(weight)
                total_quant += q6_k_storage_bytes(data)
            
            setattr(block.mlp, name, QuantLinear(data, method, bias))
    
    return total_orig, total_quant


def main():
    print("=" * 70)
    print("K-QUANT IMPLEMENTATION (llama.cpp style)")
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
        ("Q4_K_S (all Q4_K)", 'q4_k_s'),
        ("Q4_K_M (mixed Q4/Q6)", 'q4_k_m'),
        ("Q6_K (all Q6_K)", 'q6_k'),
    ]
    
    results = []
    
    for name, mode in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        orig_bytes, quant_bytes = quantize_gpt2_kquant(model, mode)
        
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
    print("SUMMARY - K-QUANT RESULTS")
    print("=" * 70)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Method':<25} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 65)
    
    for r in results:
        if r['ppl_delta'] < 1.0:
            status = "✓ GREAT"
        elif r['ppl_delta'] < 2.0:
            status = "~ GOOD"
        elif r['ppl_delta'] < 5.0:
            status = "~ OK"
        else:
            status = "✗ FAIL"
        
        print(f"{r['name']:<25} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 65)
    print("llama.cpp Q4_K_M reference: ~4x compression, <1% PPL delta")
    print("=" * 70)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("The gap between our implementation and llama.cpp could be due to:")
    print("1. llama.cpp quantizes ALL layers (attention + MLP + embeddings)")
    print("2. They use importance matrix (imatrix) for better scale selection")
    print("3. Their 4x compression is vs FP32, not FP16")
    print("4. Different PPL evaluation methodology")


if __name__ == "__main__":
    main()
