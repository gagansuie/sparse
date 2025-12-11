#!/usr/bin/env python3
"""
PRODUCT QUANTIZATION (PQ) - Path to 10x Compression

Product Quantization treats groups of weights as vectors and uses
a learned codebook to represent them. This can achieve extreme compression:

- 8-bit index per 8 weights = 1 bit/weight baseline
- Plus optional residual quantization for quality
- Target: 3-4 bits/weight = 8-10x compression vs FP32

This script tests PQ on GPT-2 to validate the approach.
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


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on text samples."""
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
# PRODUCT QUANTIZATION
# =============================================================================

def kmeans_plusplus_init(data, k, random_state=42):
    """K-means++ initialization for better codebook init."""
    np.random.seed(random_state)
    n = data.shape[0]
    
    # First centroid: random
    centroids = [data[np.random.randint(n)]]
    
    for _ in range(1, k):
        # Compute distances to nearest centroid
        dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
        # Sample proportional to distance squared
        probs = dists / dists.sum()
        idx = np.random.choice(n, p=probs)
        centroids.append(data[idx])
    
    return np.array(centroids)


def kmeans(data, k, max_iters=50, tol=1e-4):
    """Simple k-means clustering."""
    # Initialize with k-means++
    centroids = kmeans_plusplus_init(data, k)
    
    for iteration in range(max_iters):
        # Assign points to nearest centroid
        dists = np.sum((data[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Check convergence
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        
        if shift < tol:
            break
    
    return centroids, assignments


def product_quantize(weight: torch.Tensor, codebook_size=256, vector_size=8):
    """
    Quantize weight tensor using Product Quantization.
    
    Args:
        weight: Weight tensor to quantize
        codebook_size: Number of codebook entries (256 = 8-bit indices)
        vector_size: Number of weights per vector
    
    Returns:
        dict with indices, codebook, shape, etc.
    """
    weight = weight.float()
    original_shape = weight.shape
    
    # Flatten and pad to multiple of vector_size
    flat = weight.flatten()
    numel = flat.numel()
    padded_len = ((numel + vector_size - 1) // vector_size) * vector_size
    if padded_len > numel:
        flat = F.pad(flat, (0, padded_len - numel))
    
    # Reshape to vectors
    vectors = flat.view(-1, vector_size).numpy()
    num_vectors = vectors.shape[0]
    
    # K-means clustering to create codebook
    codebook, indices = kmeans(vectors, codebook_size)
    
    return {
        'indices': indices.astype(np.uint8),
        'codebook': codebook.astype(np.float16),
        'shape': original_shape,
        'numel': numel,
        'vector_size': vector_size,
    }


def product_dequantize(data: dict) -> torch.Tensor:
    """Reconstruct weight from PQ data."""
    indices = data['indices']
    codebook = data['codebook'].astype(np.float32)
    shape = data['shape']
    numel = data['numel']
    
    # Lookup codebook entries
    vectors = codebook[indices]
    
    # Flatten and trim to original size
    flat = vectors.flatten()[:numel]
    
    return torch.from_numpy(flat).view(shape)


def pq_storage_bytes(data: dict) -> int:
    """Calculate storage for PQ data."""
    indices_bytes = data['indices'].nbytes
    codebook_bytes = data['codebook'].nbytes
    return indices_bytes + codebook_bytes


# =============================================================================
# PQ WITH RESIDUAL QUANTIZATION
# =============================================================================

def pq_with_residual(weight: torch.Tensor, codebook_size=256, vector_size=8, residual_bits=2):
    """
    PQ with residual quantization for better quality.
    
    After PQ, quantize the residual (error) to recover more precision.
    """
    weight = weight.float()
    original_shape = weight.shape
    
    # First pass: PQ
    pq_data = product_quantize(weight, codebook_size, vector_size)
    pq_recon = product_dequantize(pq_data)
    
    # Compute residual
    residual = weight - pq_recon
    
    # Quantize residual to low precision
    flat_residual = residual.flatten()
    numel = flat_residual.numel()
    
    # Per-tensor asymmetric quantization of residual
    r_min = flat_residual.min().item()
    r_max = flat_residual.max().item()
    r_range = r_max - r_min if abs(r_max - r_min) > 1e-8 else 1.0
    
    num_levels = (1 << residual_bits) - 1  # e.g., 2 bits = 3 levels
    r_scale = r_range / num_levels
    
    # Quantize residual
    r_quant = ((flat_residual - r_min) / r_scale).round().clamp(0, num_levels)
    
    # Pack residual bits (simplified: store as uint8 for now)
    # In production, would pack 2-bit values into bytes
    r_packed = r_quant.to(torch.uint8)
    
    return {
        'pq': pq_data,
        'residual_packed': r_packed.numpy(),
        'residual_scale': r_scale,
        'residual_min': r_min,
        'residual_bits': residual_bits,
        'shape': original_shape,
        'numel': numel,
    }


def pq_residual_dequantize(data: dict) -> torch.Tensor:
    """Reconstruct from PQ + residual."""
    # Reconstruct PQ
    pq_recon = product_dequantize(data['pq'])
    
    # Reconstruct residual
    r_packed = torch.from_numpy(data['residual_packed']).float()
    residual = r_packed * data['residual_scale'] + data['residual_min']
    residual = residual.view(data['shape'])
    
    return pq_recon + residual


def pq_residual_storage_bytes(data: dict) -> int:
    """Calculate storage for PQ + residual."""
    pq_bytes = pq_storage_bytes(data['pq'])
    
    # Residual: ideally packed, but using uint8 for simplicity
    # True storage would be: numel * residual_bits / 8
    residual_bits = data['residual_bits']
    numel = data['numel']
    true_residual_bytes = (numel * residual_bits + 7) // 8
    
    # Add scale and min (8 bytes)
    return pq_bytes + true_residual_bytes + 8


# =============================================================================
# QUANTIZED LINEAR LAYER
# =============================================================================

class PQLinear(nn.Module):
    """Linear layer with Product Quantization."""
    
    def __init__(self, data, bias=None, use_residual=True):
        super().__init__()
        self.data = data
        self.use_residual = use_residual
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            if self.use_residual and 'pq' in self.data:
                self._w = pq_residual_dequantize(self.data)
            else:
                self._w = product_dequantize(self.data)
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def test_pq_config(model_name, tokenizer, texts, baseline_ppl, config_name, 
                   codebook_size, vector_size, residual_bits=None):
    """Test a PQ configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"  Codebook size: {codebook_size}, Vector size: {vector_size}")
    if residual_bits:
        print(f"  Residual bits: {residual_bits}")
    print('='*60)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    total_orig_bytes = 0
    total_quant_bytes = 0
    total_weights = 0
    
    # Quantize MLP layers
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ['c_fc', 'c_proj']:
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            weight = layer.weight.data.t().contiguous()  # GPT-2 Conv1D transpose
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_weights += weight.numel()
            total_orig_bytes += weight.numel() * 4  # FP32
            
            if residual_bits:
                data = pq_with_residual(weight, codebook_size, vector_size, residual_bits)
                total_quant_bytes += pq_residual_storage_bytes(data)
                new_layer = PQLinear(data, bias, use_residual=True)
            else:
                data = product_quantize(weight, codebook_size, vector_size)
                total_quant_bytes += pq_storage_bytes(data)
                new_layer = PQLinear({'pq': data, 'numel': data['numel'], 'shape': data['shape']}, 
                                     bias, use_residual=False)
            
            setattr(model.transformer.h[block_idx].mlp, layer_name, new_layer)
    
    # Calculate metrics
    compression_fp32 = total_orig_bytes / total_quant_bytes
    compression_fp16 = (total_orig_bytes / 2) / total_quant_bytes
    bits_per_weight = (total_quant_bytes * 8) / total_weights
    
    print(f"\nCompression: {total_orig_bytes/1e6:.2f} MB → {total_quant_bytes/1e6:.2f} MB")
    print(f"  vs FP32: {compression_fp32:.2f}x")
    print(f"  vs FP16: {compression_fp16:.2f}x")
    print(f"  Bits/weight: {bits_per_weight:.2f}")
    
    # Evaluate PPL
    print("Computing PPL...", end=" ", flush=True)
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
    
    del model
    
    return {
        'name': config_name,
        'compression_fp32': compression_fp32,
        'compression_fp16': compression_fp16,
        'bits_per_weight': bits_per_weight,
        'ppl': ppl,
        'ppl_delta': delta,
    }


def main():
    print("=" * 70)
    print("PRODUCT QUANTIZATION - Path to 10x Compression")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:100]  # Reduced for speed
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Test configurations
    configs = [
        # Pure PQ (no residual)
        ("PQ-256 (pure)", 256, 8, None),
        ("PQ-512 (pure)", 512, 8, None),
        ("PQ-1024 (pure)", 1024, 8, None),
        
        # PQ with residual
        ("PQ-256 + 1-bit residual", 256, 8, 1),
        ("PQ-256 + 2-bit residual", 256, 8, 2),
        ("PQ-256 + 3-bit residual", 256, 8, 3),
        
        # Different vector sizes
        ("PQ-256 v=4 + 2-bit", 256, 4, 2),
        ("PQ-256 v=16 + 2-bit", 256, 16, 2),
    ]
    
    results = []
    for name, cb_size, vec_size, res_bits in configs:
        result = test_pq_config(model_name, tokenizer, texts, baseline_ppl,
                                name, cb_size, vec_size, res_bits)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Product Quantization Results")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: 10x compression with <1% PPL delta")
    print()
    print(f"{'Configuration':<30} {'vs FP32':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
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
        
        target_met = "🎯" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else ""
        
        print(f"{r['name']:<30} {r['compression_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status} {target_met}")
    
    print("-" * 75)
    print("Target: 10.00x compression, <1% PPL delta")
    print("=" * 80)
    
    # Best result toward 10x goal
    viable = [r for r in results if r['ppl_delta'] < 5.0]
    if viable:
        best = max(viable, key=lambda x: x['compression_fp32'])
        print(f"\n📊 Best viable result: {best['name']}")
        print(f"   Compression: {best['compression_fp32']:.2f}x vs FP32")
        print(f"   PPL Delta: {best['ppl_delta']:+.2f}%")
        
        if best['compression_fp32'] >= 10 and best['ppl_delta'] < 1.0:
            print("\n🎯 TARGET ACHIEVED!")
        else:
            gap = 10.0 / best['compression_fp32']
            print(f"\n   Gap to 10x: Need {gap:.2f}x more compression")
            print("   Next steps: Try larger vector sizes, optimize codebook, combine with mixed precision")


if __name__ == "__main__":
    main()
