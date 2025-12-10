#!/usr/bin/env python3
"""
FAST PRODUCT QUANTIZATION - Using sklearn and sampling for speed.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Use sklearn for fast k-means
try:
    from sklearn.cluster import MiniBatchKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using slow k-means")


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


def fast_kmeans(data, k, max_samples=100000):
    """Fast k-means using sklearn MiniBatchKMeans with sampling."""
    n = data.shape[0]
    
    # Sample if too many vectors
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        sample = data[indices]
    else:
        sample = data
    
    if HAS_SKLEARN:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init=3, 
                                  max_iter=100, random_state=42)
        kmeans.fit(sample)
        codebook = kmeans.cluster_centers_
        # Assign all points to nearest centroid
        assignments = kmeans.predict(data)
    else:
        # Fallback: simple k-means with fewer iterations
        codebook = data[np.random.choice(n, k, replace=False)]
        for _ in range(20):
            dists = np.sum((data[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)
            for i in range(k):
                mask = assignments == i
                if mask.sum() > 0:
                    codebook[i] = data[mask].mean(axis=0)
    
    return codebook.astype(np.float32), assignments


def product_quantize(weight: torch.Tensor, codebook_size=256, vector_size=8):
    """Fast Product Quantization."""
    weight = weight.float()
    original_shape = weight.shape
    
    flat = weight.flatten()
    numel = flat.numel()
    padded_len = ((numel + vector_size - 1) // vector_size) * vector_size
    if padded_len > numel:
        flat = F.pad(flat, (0, padded_len - numel))
    
    vectors = flat.view(-1, vector_size).numpy()
    
    print(f"    Clustering {vectors.shape[0]} vectors...", end=" ", flush=True)
    codebook, indices = fast_kmeans(vectors, codebook_size)
    print("done")
    
    return {
        'indices': indices.astype(np.uint8) if codebook_size <= 256 else indices.astype(np.uint16),
        'codebook': codebook.astype(np.float16),
        'shape': original_shape,
        'numel': numel,
        'vector_size': vector_size,
    }


def product_dequantize(data: dict) -> torch.Tensor:
    indices = data['indices']
    codebook = data['codebook'].astype(np.float32)
    shape = data['shape']
    numel = data['numel']
    
    vectors = codebook[indices]
    flat = vectors.flatten()[:numel]
    
    return torch.from_numpy(flat).view(shape)


def pq_storage_bytes(data: dict) -> int:
    indices_bytes = data['indices'].nbytes
    codebook_bytes = data['codebook'].nbytes
    return indices_bytes + codebook_bytes


def pq_with_residual(weight: torch.Tensor, codebook_size=256, vector_size=8, residual_bits=2):
    """PQ with residual quantization."""
    weight = weight.float()
    original_shape = weight.shape
    
    pq_data = product_quantize(weight, codebook_size, vector_size)
    pq_recon = product_dequantize(pq_data)
    
    residual = weight - pq_recon
    flat_residual = residual.flatten()
    numel = flat_residual.numel()
    
    # Per-group residual quantization (groups of 8 for efficiency)
    GROUP_SIZE = 8
    num_groups = (numel + GROUP_SIZE - 1) // GROUP_SIZE
    
    padded = F.pad(flat_residual, (0, num_groups * GROUP_SIZE - numel))
    groups = padded.view(num_groups, GROUP_SIZE)
    
    # Per-group scale for residual
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    g_range = torch.clamp(g_max - g_min, min=1e-8)
    
    num_levels = (1 << residual_bits) - 1
    
    # Quantize residual per group
    normalized = (groups - g_min.unsqueeze(1)) / g_range.unsqueeze(1)
    r_quant = (normalized * num_levels).round().clamp(0, num_levels).to(torch.uint8)
    
    return {
        'pq': pq_data,
        'residual_quant': r_quant.numpy(),
        'residual_scales': g_range.half().numpy(),
        'residual_mins': g_min.half().numpy(),
        'residual_bits': residual_bits,
        'shape': original_shape,
        'numel': numel,
    }


def pq_residual_dequantize(data: dict) -> torch.Tensor:
    pq_recon = product_dequantize(data['pq'])
    
    r_quant = torch.from_numpy(data['residual_quant']).float()
    scales = torch.from_numpy(data['residual_scales'].astype(np.float32))
    mins = torch.from_numpy(data['residual_mins'].astype(np.float32))
    
    num_levels = (1 << data['residual_bits']) - 1
    
    residual = (r_quant / num_levels) * scales.unsqueeze(1) + mins.unsqueeze(1)
    residual = residual.flatten()[:data['numel']].view(data['shape'])
    
    return pq_recon + residual


def pq_residual_storage_bytes(data: dict) -> int:
    pq_bytes = pq_storage_bytes(data['pq'])
    
    # Residual: packed bits
    residual_bits = data['residual_bits']
    numel = data['numel']
    residual_bytes = (numel * residual_bits + 7) // 8
    
    # Scales and mins (FP16, per group of 8)
    num_groups = (numel + 7) // 8
    scales_bytes = num_groups * 4  # 2 bytes scale + 2 bytes min
    
    return pq_bytes + residual_bytes + scales_bytes


class PQLinear(nn.Module):
    def __init__(self, data, bias=None, use_residual=True):
        super().__init__()
        self.data = data
        self.use_residual = use_residual
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            if self.use_residual and 'residual_quant' in self.data:
                self._w = pq_residual_dequantize(self.data)
            else:
                self._w = product_dequantize(self.data['pq'] if 'pq' in self.data else self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


def test_config(model_name, tokenizer, texts, baseline_ppl, name, cb_size, vec_size, res_bits=None):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
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
            total_orig += weight.numel() * 4
            
            print(f"  Layer {block_idx}.{layer_name}:", end=" ")
            
            if res_bits:
                data = pq_with_residual(weight, cb_size, vec_size, res_bits)
                total_quant += pq_residual_storage_bytes(data)
            else:
                pq_data = product_quantize(weight, cb_size, vec_size)
                data = {'pq': pq_data, 'numel': pq_data['numel'], 'shape': pq_data['shape']}
                total_quant += pq_storage_bytes(pq_data)
            
            setattr(model.transformer.h[block_idx].mlp, layer_name, 
                   PQLinear(data, bias, use_residual=bool(res_bits)))
    
    compress_fp32 = total_orig / total_quant
    bits_per_weight = (total_quant * 8) / total_weights
    
    print(f"\nCompression: {compress_fp32:.2f}x vs FP32, {bits_per_weight:.2f} bits/weight")
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
    print("FAST PRODUCT QUANTIZATION - Path to 10x Compression")
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
    
    # Key configurations to test
    configs = [
        # Pure PQ - maximum compression
        ("PQ-256 v=8 (pure)", 256, 8, None),
        
        # PQ with residual - balance quality and compression  
        ("PQ-256 v=8 + 2-bit res", 256, 8, 2),
        ("PQ-256 v=8 + 3-bit res", 256, 8, 3),
        
        # Larger codebook
        ("PQ-1024 v=8 + 2-bit res", 1024, 8, 2),
        
        # Larger vectors = more compression
        ("PQ-256 v=16 + 2-bit res", 256, 16, 2),
    ]
    
    results = []
    for name, cb, vec, res in configs:
        result = test_config(model_name, tokenizer, texts, baseline_ppl, name, cb, vec, res)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: 10x compression, <1% PPL delta")
    print()
    print(f"{'Config':<30} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'10x Goal'}")
    print("-" * 75)
    
    for r in results:
        goal = "🎯 YES!" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
               "Close" if r['compression_fp32'] >= 8 and r['ppl_delta'] < 2.0 else ""
        print(f"{r['name']:<30} {r['compression_fp32']:.1f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {goal}")
    
    print("-" * 75)
    
    # Analysis
    best = max(results, key=lambda x: x['compression_fp32'] if x['ppl_delta'] < 5 else 0)
    print(f"\n📊 Best result: {best['name']}")
    print(f"   {best['compression_fp32']:.1f}x compression, {best['ppl_delta']:+.2f}% PPL")
    
    if best['compression_fp32'] >= 10 and best['ppl_delta'] < 1.0:
        print("\n🎯 10x TARGET ACHIEVED!")
    else:
        print(f"\n📍 Path to 10x:")
        print(f"   Current: {best['compression_fp32']:.1f}x")
        print(f"   Need: Combine PQ with mixed precision + attention optimization")


if __name__ == "__main__":
    main()
