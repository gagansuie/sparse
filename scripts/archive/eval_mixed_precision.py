#!/usr/bin/env python3
"""
MIXED PRECISION QUANTIZATION

Strategy (like llama.cpp Q4_K_M):
- Embeddings: Q4_K (less sensitive)
- MLP layers: Q4_K  
- Attention: Q8 or FP16 (most sensitive)
- Output: Q6_K

This should give us the best of both worlds:
- High compression on MLP (bulk of parameters)
- High quality on attention (critical for PPL)
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
# QUANTIZATION METHODS
# =============================================================================

def q4_k_quantize(weight: torch.Tensor):
    """Q4_K: 4-bit with 6-bit quantized scales."""
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    SUPER_BLOCK_SIZE = 256
    BLOCK_SIZE = 32
    NUM_BLOCKS = 8
    
    padded = ((numel + SUPER_BLOCK_SIZE - 1) // SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    super_blocks = flat.view(-1, SUPER_BLOCK_SIZE)
    num_super = super_blocks.shape[0]
    
    sb_min = super_blocks.min(dim=1).values
    sb_max = super_blocks.max(dim=1).values
    sb_range = torch.clamp(sb_max - sb_min, min=1e-8)
    
    normalized = (super_blocks - sb_min.unsqueeze(1)) / sb_range.unsqueeze(1)
    blocks = normalized.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    block_min = blocks.min(dim=2).values
    block_max = blocks.max(dim=2).values
    block_range = torch.clamp(block_max - block_min, min=1e-8)
    
    block_scales_q = (block_range * 63).round().clamp(0, 63).to(torch.uint8)
    block_mins_q = (block_min * 63).round().clamp(0, 63).to(torch.uint8)
    
    block_range_dq = block_scales_q.float() / 63
    block_min_dq = block_mins_q.float() / 63
    
    block_normalized = (blocks - block_min_dq.unsqueeze(2)) / block_range_dq.unsqueeze(2).clamp(min=1e-8)
    q = (block_normalized * 15).round().clamp(0, 15).to(torch.uint8)
    
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'd': sb_range.half(),
        'dmin': sb_min.half(),
        'scales': block_scales_q,
        'mins': block_mins_q,
        'shape': weight.shape,
        'numel': numel,
        'method': 'q4_k',
    }


def q8_quantize(weight: torch.Tensor):
    """Q8: 8-bit symmetric quantization (high quality)."""
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    BLOCK_SIZE = 32
    
    padded = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    blocks = flat.view(-1, BLOCK_SIZE)
    
    # Symmetric: use absmax
    absmax = blocks.abs().max(dim=1).values
    absmax = torch.clamp(absmax, min=1e-8)
    
    # Quantize to -127..127
    q = (blocks / absmax.unsqueeze(1) * 127).round().clamp(-127, 127).to(torch.int8)
    
    return {
        'quants': q.flatten(),
        'scales': absmax.half(),
        'shape': weight.shape,
        'numel': numel,
        'method': 'q8',
    }


def dequantize(data: dict) -> torch.Tensor:
    """Dequantize based on method."""
    method = data['method']
    
    if method == 'q4_k':
        packed = data['packed']
        d = data['d'].float()
        dmin = data['dmin'].float()
        scales = data['scales'].float() / 63
        mins = data['mins'].float() / 63
        shape = data['shape']
        numel = data['numel']
        
        num_super = d.numel()
        
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q = torch.stack([low, high], dim=1).flatten().float()
        
        blocks = q.view(num_super, 8, 32)
        
        block_range = scales.view(num_super, 8, 1)
        block_min = mins.view(num_super, 8, 1)
        normalized = (blocks / 15) * block_range + block_min
        
        super_range = d.view(num_super, 1, 1)
        super_min = dmin.view(num_super, 1, 1)
        weight = normalized * super_range + super_min
        
        weight = weight.flatten()[:numel]
        return weight.view(shape)
    
    elif method == 'q8':
        quants = data['quants'].float()
        scales = data['scales'].float()
        shape = data['shape']
        numel = data['numel']
        
        blocks = quants.view(-1, 32)
        weight = blocks * scales.unsqueeze(1) / 127
        weight = weight.flatten()[:numel]
        return weight.view(shape)


def storage_bytes(data: dict) -> int:
    method = data['method']
    
    if method == 'q4_k':
        num_super = data['d'].numel()
        return data['packed'].numel() + num_super * 4 + num_super * 16
    elif method == 'q8':
        num_blocks = data['scales'].numel()
        return data['quants'].numel() + num_blocks * 2


# =============================================================================
# QUANTIZED LAYERS
# =============================================================================

class QuantLinear(nn.Module):
    def __init__(self, data, bias=None):
        super().__init__()
        self.data = data
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


class QuantEmbedding(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = dequantize(self.data)
        return F.embedding(x, self._w.to(torch.float32))


# =============================================================================
# MIXED PRECISION QUANTIZATION
# =============================================================================

def quantize_mixed(model, attn_method='q8'):
    """
    Mixed precision quantization:
    - Embeddings: Q4_K
    - MLP: Q4_K
    - Attention: q8 or fp16 (configurable)
    """
    total_orig_fp16 = 0
    total_orig_fp32 = 0
    total_quant = 0
    
    # Embeddings: Q4_K
    print("  Quantizing embeddings (Q4_K)...")
    for name in ['wte', 'wpe']:
        emb = getattr(model.transformer, name)
        weight = emb.weight.data
        total_orig_fp16 += weight.numel() * 2
        total_orig_fp32 += weight.numel() * 4
        
        data = q4_k_quantize(weight)
        total_quant += storage_bytes(data)
        
        setattr(model.transformer, name, QuantEmbedding(data))
    
    # Transformer blocks
    print("  Quantizing transformer blocks...")
    for layer_idx, block in enumerate(model.transformer.h):
        # Attention: Q8 or keep FP16
        attn = block.attn
        for name in ['c_attn', 'c_proj']:
            layer = getattr(attn, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig_fp16 += weight.numel() * 2
            total_orig_fp32 += weight.numel() * 4
            
            if attn_method == 'q8':
                data = q8_quantize(weight)
                total_quant += storage_bytes(data)
                setattr(attn, name, QuantLinear(data, bias))
            elif attn_method == 'q4_k':
                data = q4_k_quantize(weight)
                total_quant += storage_bytes(data)
                setattr(attn, name, QuantLinear(data, bias))
            else:  # fp16 - don't quantize
                total_quant += weight.numel() * 2
        
        # MLP: Q4_K
        mlp = block.mlp
        for name in ['c_fc', 'c_proj']:
            layer = getattr(mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig_fp16 += weight.numel() * 2
            total_orig_fp32 += weight.numel() * 4
            
            data = q4_k_quantize(weight)
            total_quant += storage_bytes(data)
            
            setattr(mlp, name, QuantLinear(data, bias))
    
    return total_orig_fp16, total_orig_fp32, total_quant


def main():
    print("=" * 70)
    print("MIXED PRECISION QUANTIZATION")
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
        ("All Q4_K", 'q4_k'),
        ("Attn Q8, MLP Q4_K", 'q8'),
        ("Attn FP16, MLP Q4_K", 'fp16'),
    ]
    
    results = []
    
    for name, attn_method in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        orig_fp16, orig_fp32, quant_bytes = quantize_mixed(model, attn_method)
        
        compress_fp16 = orig_fp16 / quant_bytes
        compress_fp32 = orig_fp32 / quant_bytes
        bits_per_weight = (quant_bytes * 8) / (orig_fp16 / 2)
        
        print(f"\nCompression vs FP16: {compress_fp16:.2f}x")
        print(f"Compression vs FP32: {compress_fp32:.2f}x")
        print(f"Bits/weight: {bits_per_weight:.2f}")
        
        print("Computing PPL...", end=" ", flush=True)
        ppl = compute_perplexity(model, tokenizer, texts)
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
        
        results.append({
            'name': name,
            'compress_fp16': compress_fp16,
            'compress_fp32': compress_fp32,
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
    print(f"{'Method':<25} {'vs FP16':<10} {'vs FP32':<10} {'Bits/W':<8} {'PPL Δ':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<25} {r['compress_fp16']:.2f}x      {r['compress_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%")
    
    print("-" * 70)
    print("llama.cpp Q4_K_M: ~4x vs FP32, <1% PPL delta")
    print("=" * 70)
    
    # Best result
    best = min(results, key=lambda x: abs(x['ppl_delta']))
    print(f"\n🎯 Best quality: {best['name']}")
    print(f"   Compression: {best['compress_fp32']:.2f}x vs FP32")
    print(f"   PPL Delta: {best['ppl_delta']:+.2f}%")


if __name__ == "__main__":
    main()
