#!/usr/bin/env python3
"""
Optimized int4 quantization - v2
Using same PPL methodology as original eval_g8.py for fair comparison.

Key optimizations:
1. FP16 scales/offsets (not FP32)
2. Test multiple group sizes
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
    """Same PPL computation as original eval_g8.py."""
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def int4_quantize(weight: torch.Tensor, group_size: int = 8, use_fp16: bool = True):
    """Quantize weight to int4."""
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
    
    scale = (g_max - g_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)
    offset = g_min
    
    q = ((groups - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    dtype = torch.float16 if use_fp16 else torch.float32
    
    return {
        'packed': packed,
        'scales': scale.to(dtype),
        'offsets': offset.to(dtype),
        'shape': weight.shape,
        'group_size': group_size,
        'numel': numel,
    }


def int4_dequantize(data: dict) -> torch.Tensor:
    """Dequantize int4 back to float."""
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


def storage_bytes(data: dict) -> int:
    """Calculate storage bytes."""
    return (data['packed'].numel() + 
            data['scales'].numel() * data['scales'].element_size() +
            data['offsets'].numel() * data['offsets'].element_size())


class QuantLinear(nn.Module):
    def __init__(self, data, bias=None):
        super().__init__()
        self.data = data
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = int4_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


def quantize_gpt2(model, group_size=8, use_fp16=True):
    """Quantize GPT-2 MLP layers."""
    total_orig = 0
    total_quant = 0
    
    for block in model.transformer.h:
        for name in ['c_fc', 'c_proj']:
            layer = getattr(block.mlp, name)
            weight = layer.weight.data.t().contiguous()  # Conv1D transpose
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig += weight.numel() * 2  # FP16 baseline
            
            data = int4_quantize(weight, group_size, use_fp16)
            total_quant += storage_bytes(data)
            
            setattr(block.mlp, name, QuantLinear(data, bias))
    
    return total_orig, total_quant


def main():
    print("=" * 60)
    print("INT4 QUANTIZATION - ACCURATE BENCHMARK")
    print("=" * 60)
    
    model_name = "gpt2"
    device = "cpu"
    
    # Load data (same as original eval_g8.py)
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:128]  # Same as original
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Test configurations
    configs = [
        ("g8_fp32", 8, False),
        ("g8_fp16", 8, True),
        ("g16_fp16", 16, True),
        ("g32_fp16", 32, True),
        ("g64_fp16", 64, True),
    ]
    
    results = []
    
    for name, group_size, use_fp16 in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        orig_bytes, quant_bytes = quantize_gpt2(model, group_size, use_fp16)
        
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
            'group_size': group_size,
            'use_fp16': use_fp16,
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
    print(f"{'Config':<15} {'Group':<6} {'FP16':<5} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r['ppl_delta'] < 1.0 else "~" if r['ppl_delta'] < 2.0 else "✗"
        fp16 = "Yes" if r['use_fp16'] else "No"
        print(f"{r['name']:<15} {r['group_size']:<6} {fp16:<5} {r['compression']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 70)
    print("Target: AWQ = 4x compression, <1% PPL delta")
    print("=" * 70)
    
    # Best result
    best = min(results, key=lambda x: abs(x['ppl_delta']))
    print(f"\nBest quality: {best['name']} with {best['ppl_delta']:+.2f}% PPL delta, {best['compression']:.2f}x compression")
    
    best_compress = max([r for r in results if r['ppl_delta'] < 2.0], key=lambda x: x['compression'], default=None)
    if best_compress:
        print(f"Best compression (<2% PPL): {best_compress['name']} with {best_compress['compression']:.2f}x, {best_compress['ppl_delta']:+.2f}% PPL")


if __name__ == "__main__":
    main()
