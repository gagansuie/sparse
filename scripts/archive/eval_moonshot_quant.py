#!/usr/bin/env python3
"""
MOONSHOT COMPRESSION EXPERIMENTS

Novel approaches inspired by other sciences:

1. FRACTAL COMPRESSION (Mathematics)
   - Weight matrices often have self-similar structure at different scales
   - Store "seed" patterns + affine transformations
   - Like JPEG but for weight matrices

2. LEARNED CODEBOOK (Biology/Genetics)  
   - DNA: 3 nucleotides → 1 amino acid (64 codons → 20 amino acids)
   - Learn a codebook of common weight "patterns"
   - Store indices into codebook instead of weights
   - Product Quantization on steroids

3. MANIFOLD PROJECTION (Neuroscience)
   - Neural activity lies on low-dimensional manifolds
   - Project weights onto learned basis vectors
   - Store only coefficients in the low-dim space

4. SYMMETRY EXPLOITATION (Crystallography)
   - Crystals: unit cell → infinite lattice via symmetry
   - Find approximate symmetries in weight matrices
   - Store only the "asymmetric unit" + symmetry operations

5. RESIDUAL CODING (Signal Processing)
   - Like how MP3 works: perceptual coding
   - Quantize aggressively, but keep a sparse residual for "important" weights
   - The residual is highly compressible

6. HASH-BASED WEIGHT SHARING (Computer Science)
   - HashedNets: use hash function to share weights
   - Extreme compression with learned hash collisions
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TENPAK_ROOT = Path(__file__).parent.parent


def get_wikitext2_subset(tokenizer, num_tokens=20000):
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", max_length=num_tokens, truncation=True)
    print(f"Using {encodings.input_ids.size(1)} tokens")
    return encodings


def evaluate_ppl(model, encodings, max_steps=30):
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
# METHOD 1: LEARNED CODEBOOK (Product Quantization++)
# ============================================================================

def learn_codebook(weights_list: List[torch.Tensor], num_codes=256, subvec_size=8):
    """
    Learn a codebook from weight matrices using k-means.
    
    Memory-efficient version: sample subvectors instead of using all.
    """
    print(f"Learning codebook (codes={num_codes}, subvec={subvec_size})...")
    
    # Collect SAMPLED weight subvectors (memory efficient)
    max_samples = 500000  # Limit samples for memory
    all_subvecs = []
    total_subvecs = 0
    
    for w in weights_list:
        w_flat = w.flatten().float()
        pad_len = (subvec_size - len(w_flat) % subvec_size) % subvec_size
        if pad_len > 0:
            w_flat = torch.cat([w_flat, torch.zeros(pad_len)])
        subvecs = w_flat.view(-1, subvec_size)
        total_subvecs += len(subvecs)
        
        # Sample from this weight matrix
        sample_size = min(len(subvecs), max_samples // len(weights_list))
        indices = torch.randperm(len(subvecs))[:sample_size]
        all_subvecs.append(subvecs[indices])
    
    all_subvecs = torch.cat(all_subvecs, dim=0)
    print(f"  Sampled {len(all_subvecs):,} / {total_subvecs:,} subvectors")
    
    # K-means clustering (mini-batch for memory efficiency)
    indices = torch.randperm(len(all_subvecs))[:num_codes]
    codebook = all_subvecs[indices].clone()
    
    batch_size = 50000
    for iteration in range(20):
        new_codebook = torch.zeros_like(codebook)
        counts = torch.zeros(num_codes)
        
        # Process in batches
        for start in range(0, len(all_subvecs), batch_size):
            end = min(start + batch_size, len(all_subvecs))
            batch = all_subvecs[start:end]
            
            # Assign to nearest centroid
            dists = torch.cdist(batch, codebook)
            assignments = dists.argmin(dim=1)
            
            # Update centroids
            for i, a in enumerate(assignments):
                new_codebook[a] += batch[i]
                counts[a] += 1
        
        counts = counts.clamp(min=1)
        new_codebook = new_codebook / counts.unsqueeze(1)
        
        diff = (new_codebook - codebook).abs().mean()
        codebook = new_codebook
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: diff={diff:.6f}")
        
        if diff < 1e-6:
            break
    
    return codebook


def codebook_quantize(weight, codebook, subvec_size=8):
    """Quantize weight using learned codebook (memory efficient)."""
    shape = weight.shape
    w_flat = weight.flatten().float()
    
    pad_len = (subvec_size - len(w_flat) % subvec_size) % subvec_size
    if pad_len > 0:
        w_flat = torch.cat([w_flat, torch.zeros(pad_len)])
    
    subvecs = w_flat.view(-1, subvec_size)
    
    # Find nearest codebook entry in batches (memory efficient)
    batch_size = 50000
    all_indices = []
    for start in range(0, len(subvecs), batch_size):
        end = min(start + batch_size, len(subvecs))
        batch = subvecs[start:end]
        dists = torch.cdist(batch, codebook)
        indices = dists.argmin(dim=1)
        all_indices.append(indices)
    
    indices = torch.cat(all_indices)
    
    # Use uint16 if more than 256 codes
    if len(codebook) <= 256:
        indices = indices.to(torch.uint8)
    else:
        indices = indices.to(torch.int16)
    
    return {
        'indices': indices,
        'shape': shape,
        'pad_len': pad_len,
    }


def codebook_dequantize(data, codebook, subvec_size=8):
    """Dequantize using codebook."""
    indices = data['indices'].long()
    shape = data['shape']
    pad_len = data['pad_len']
    
    # Look up codebook entries
    subvecs = codebook[indices]
    w_flat = subvecs.flatten()
    
    # Remove padding
    n = shape[0] * shape[1]
    w_flat = w_flat[:n]
    
    return w_flat.view(shape)


# ============================================================================
# METHOD 2: MANIFOLD PROJECTION (Low-rank + Sparse Residual)
# ============================================================================

def manifold_compress(weight, rank_ratio=0.1, residual_sparsity=0.01):
    """
    Project weight onto low-dimensional manifold + sparse residual.
    
    Inspired by neuroscience: neural activity lies on low-dim manifolds.
    
    W ≈ U @ V + sparse_residual
    
    Where:
    - U @ V is low-rank approximation (the "manifold")
    - sparse_residual captures critical outliers
    """
    m, n = weight.shape
    rank = max(1, int(min(m, n) * rank_ratio))
    
    # SVD for low-rank approximation
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    
    # Keep top-k singular values
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vh_k = Vh[:rank, :]
    
    # Low-rank reconstruction
    low_rank = U_k @ torch.diag(S_k) @ Vh_k
    
    # Residual
    residual = weight.float() - low_rank
    
    # Keep only top residual_sparsity fraction of residual
    residual_flat = residual.flatten()
    num_keep = max(1, int(len(residual_flat) * residual_sparsity))
    
    # Find top-k by magnitude
    _, top_indices = residual_flat.abs().topk(num_keep)
    top_values = residual_flat[top_indices]
    
    return {
        'U': U_k.half(),
        'S': S_k.half(),
        'Vh': Vh_k.half(),
        'residual_indices': top_indices.to(torch.int32),
        'residual_values': top_values.half(),
        'shape': weight.shape,
    }


def manifold_decompress(data):
    """Decompress manifold-projected weight."""
    U = data['U'].float()
    S = data['S'].float()
    Vh = data['Vh'].float()
    shape = data['shape']
    
    # Reconstruct low-rank
    weight = U @ torch.diag(S) @ Vh
    
    # Add sparse residual
    residual_indices = data['residual_indices'].long()
    residual_values = data['residual_values'].float()
    
    weight_flat = weight.flatten()
    weight_flat[residual_indices] += residual_values
    
    return weight_flat.view(shape)


def manifold_bytes(data):
    """Compute storage for manifold compression."""
    bytes_U = data['U'].numel() * 2
    bytes_S = data['S'].numel() * 2
    bytes_Vh = data['Vh'].numel() * 2
    bytes_res_idx = data['residual_indices'].numel() * 4
    bytes_res_val = data['residual_values'].numel() * 2
    return bytes_U + bytes_S + bytes_Vh + bytes_res_idx + bytes_res_val


# ============================================================================
# METHOD 3: HASH-BASED WEIGHT SHARING
# ============================================================================

def hash_compress(weight, num_buckets=65536):
    """
    Hash-based weight sharing (HashedNets-inspired).
    
    Instead of storing each weight, use a hash function to map
    weight positions to a small set of shared values.
    
    Extreme compression: all weights share from a small pool.
    """
    shape = weight.shape
    m, n = shape
    
    # Create hash table of shared values
    # Initialize with weight statistics
    w_flat = weight.flatten().float()
    
    # Cluster weights into buckets
    min_val, max_val = w_flat.min(), w_flat.max()
    
    # Create bucket centers (uniform quantization of value range)
    bucket_centers = torch.linspace(min_val, max_val, num_buckets)
    
    # For each position, find which bucket it hashes to
    # Use a deterministic hash based on position
    hash_table = torch.zeros(num_buckets)
    hash_counts = torch.zeros(num_buckets)
    
    for i in range(len(w_flat)):
        # Simple hash: position mod num_buckets
        # In practice, use a better hash function
        bucket = i % num_buckets
        hash_table[bucket] += w_flat[i]
        hash_counts[bucket] += 1
    
    # Average values in each bucket
    hash_counts = hash_counts.clamp(min=1)
    hash_table = hash_table / hash_counts
    
    return {
        'hash_table': hash_table.half(),
        'shape': shape,
        'num_buckets': num_buckets,
    }


def hash_decompress(data):
    """Decompress hash-based weights."""
    hash_table = data['hash_table'].float()
    shape = data['shape']
    num_buckets = data['num_buckets']
    
    m, n = shape
    weight = torch.zeros(m * n)
    
    for i in range(len(weight)):
        bucket = i % num_buckets
        weight[i] = hash_table[bucket]
    
    return weight.view(shape)


# ============================================================================
# METHOD 4: FRACTAL COMPRESSION (Iterated Function Systems)
# ============================================================================

def fractal_compress(weight, block_size=8, num_transforms=256):
    """
    Fractal compression inspired by IFS (Iterated Function Systems).
    
    Key insight: Parts of the weight matrix may be approximate
    affine transformations of other parts.
    
    Store: domain blocks + affine transforms
    Decode: Apply transforms iteratively
    """
    shape = weight.shape
    m, n = shape
    w = weight.float()
    
    # Divide into range blocks (what we want to encode)
    range_block_size = block_size
    # Domain blocks are larger (we'll downsample)
    domain_block_size = block_size * 2
    
    # Collect domain blocks (source patterns)
    domain_blocks = []
    for i in range(0, m - domain_block_size + 1, block_size):
        for j in range(0, n - domain_block_size + 1, block_size):
            block = w[i:i+domain_block_size, j:j+domain_block_size]
            # Downsample to range block size
            block_down = F.avg_pool2d(block.unsqueeze(0).unsqueeze(0), 2).squeeze()
            domain_blocks.append(block_down.flatten())
    
    if len(domain_blocks) == 0:
        # Fallback for small matrices
        return {
            'fallback': weight.half(),
            'shape': shape,
            'is_fallback': True,
        }
    
    domain_blocks = torch.stack(domain_blocks)
    
    # For each range block, find best matching domain block + transform
    transforms = []
    for i in range(0, m - range_block_size + 1, range_block_size):
        for j in range(0, n - range_block_size + 1, range_block_size):
            range_block = w[i:i+range_block_size, j:j+range_block_size].flatten()
            
            # Find best domain block (minimize MSE after affine transform)
            best_idx = 0
            best_scale = 1.0
            best_offset = 0.0
            best_error = float('inf')
            
            for idx, domain in enumerate(domain_blocks[:num_transforms]):
                # Solve for optimal scale and offset: range ≈ scale * domain + offset
                # Using least squares
                d_mean = domain.mean()
                r_mean = range_block.mean()
                
                d_centered = domain - d_mean
                r_centered = range_block - r_mean
                
                denom = (d_centered ** 2).sum()
                if denom > 1e-8:
                    scale = (d_centered * r_centered).sum() / denom
                else:
                    scale = 0.0
                
                scale = torch.clamp(torch.tensor(scale), -2.0, 2.0).item()
                offset = r_mean - scale * d_mean
                
                # Compute error
                reconstructed = scale * domain + offset
                error = ((range_block - reconstructed) ** 2).mean()
                
                if error < best_error:
                    best_error = error
                    best_idx = idx
                    best_scale = scale
                    best_offset = offset
            
            transforms.append({
                'domain_idx': best_idx,
                'scale': best_scale,
                'offset': best_offset,
            })
    
    # Pack transforms
    domain_indices = torch.tensor([t['domain_idx'] for t in transforms], dtype=torch.int16)
    scales = torch.tensor([t['scale'] for t in transforms], dtype=torch.float16)
    offsets = torch.tensor([t['offset'] for t in transforms], dtype=torch.float16)
    
    return {
        'domain_blocks': domain_blocks[:num_transforms].half(),
        'domain_indices': domain_indices,
        'scales': scales,
        'offsets': offsets,
        'shape': shape,
        'block_size': block_size,
        'is_fallback': False,
    }


def fractal_decompress(data):
    """Decompress fractal-encoded weights."""
    if data.get('is_fallback', False):
        return data['fallback'].float()
    
    shape = data['shape']
    m, n = shape
    block_size = data['block_size']
    
    domain_blocks = data['domain_blocks'].float()
    domain_indices = data['domain_indices']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    
    # Reconstruct
    weight = torch.zeros(m, n)
    
    idx = 0
    for i in range(0, m - block_size + 1, block_size):
        for j in range(0, n - block_size + 1, block_size):
            if idx >= len(domain_indices):
                break
            
            domain = domain_blocks[domain_indices[idx]]
            scale = scales[idx]
            offset = offsets[idx]
            
            reconstructed = scale * domain + offset
            weight[i:i+block_size, j:j+block_size] = reconstructed.view(block_size, block_size)
            idx += 1
    
    return weight


# ============================================================================
# METHOD 5: RESIDUAL PERCEPTUAL CODING (like MP3 for weights)
# ============================================================================

def perceptual_compress(weight, base_bits=2, residual_fraction=0.02):
    """
    Perceptual coding inspired by MP3/JPEG.
    
    Key insight: Not all quantization errors are equally "perceptible"
    to the model. Errors on high-magnitude weights matter more.
    
    Strategy:
    1. Very aggressive base quantization (2-bit)
    2. Keep sparse residual for high-impact weights
    """
    shape = weight.shape
    w = weight.float()
    w_flat = w.flatten()
    
    # Compute "perceptual importance" (magnitude-based)
    importance = w_flat.abs()
    
    # Base quantization (very aggressive: 2-bit = 4 levels)
    num_levels = 2 ** base_bits
    min_val, max_val = w_flat.min(), w_flat.max()
    
    scale = (max_val - min_val) / (num_levels - 1)
    if scale < 1e-8:
        scale = 1.0
    
    w_quant = ((w_flat - min_val) / scale).round().clamp(0, num_levels - 1)
    w_dequant = w_quant * scale + min_val
    
    # Compute residual
    residual = w_flat - w_dequant
    
    # Keep residual for most important weights
    num_residual = max(1, int(len(w_flat) * residual_fraction))
    
    # Importance-weighted residual selection
    weighted_residual = residual.abs() * importance
    _, top_indices = weighted_residual.topk(num_residual)
    
    # Pack base quantization
    if base_bits == 2:
        # Pack 4 values per byte
        packed_len = (len(w_quant) + 3) // 4
        packed = torch.zeros(packed_len, dtype=torch.uint8)
        for i in range(len(w_quant)):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= (int(w_quant[i]) & 0x3) << bit_offset
    else:
        packed = w_quant.to(torch.uint8)
    
    return {
        'packed': packed,
        'scale': torch.tensor(scale).half(),
        'min_val': torch.tensor(min_val).half(),
        'residual_indices': top_indices.to(torch.int32),
        'residual_values': residual[top_indices].half(),
        'shape': shape,
        'base_bits': base_bits,
        'num_weights': len(w_flat),
    }


def perceptual_decompress(data):
    """Decompress perceptual-coded weights."""
    packed = data['packed']
    scale = data['scale'].float()
    min_val = data['min_val'].float()
    shape = data['shape']
    base_bits = data['base_bits']
    num_weights = data['num_weights']
    
    # Unpack base quantization
    if base_bits == 2:
        w_quant = torch.zeros(num_weights)
        for i in range(num_weights):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            w_quant[i] = (packed[byte_idx] >> bit_offset) & 0x3
    else:
        w_quant = packed.float()
    
    # Dequantize
    w_flat = w_quant * scale + min_val
    
    # Add residual
    residual_indices = data['residual_indices'].long()
    residual_values = data['residual_values'].float()
    w_flat[residual_indices] += residual_values
    
    return w_flat.view(shape)


def perceptual_bytes(data):
    """Compute storage for perceptual coding."""
    bytes_packed = data['packed'].numel()
    bytes_scale = 2
    bytes_min = 2
    bytes_res_idx = data['residual_indices'].numel() * 4
    bytes_res_val = data['residual_values'].numel() * 2
    return bytes_packed + bytes_scale + bytes_min + bytes_res_idx + bytes_res_val


# ============================================================================
# UNIFIED COMPRESSION LAYER
# ============================================================================

class MoonshotLinear(nn.Module):
    """Linear layer with moonshot compression."""
    
    def __init__(self, in_features, out_features, method='codebook'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.data = None
        self.codebook = None
        self.original_bytes = 0
        self.register_buffer('bias_data', None)
    
    def quantized_bytes(self):
        if self.method == 'codebook':
            return self.data['indices'].numel() + self.codebook.numel() * 2
        elif self.method == 'manifold':
            return manifold_bytes(self.data)
        elif self.method == 'perceptual':
            return perceptual_bytes(self.data)
        elif self.method == 'fractal':
            if self.data.get('is_fallback', False):
                return self.data['fallback'].numel() * 2
            return (self.data['domain_blocks'].numel() * 2 + 
                    self.data['domain_indices'].numel() * 2 +
                    self.data['scales'].numel() * 2 +
                    self.data['offsets'].numel() * 2)
        return 0
    
    def forward(self, x):
        if self.method == 'codebook':
            weight = codebook_dequantize(self.data, self.codebook)
        elif self.method == 'manifold':
            weight = manifold_decompress(self.data)
        elif self.method == 'perceptual':
            weight = perceptual_decompress(self.data)
        elif self.method == 'fractal':
            weight = fractal_decompress(self.data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return F.linear(x, weight.to(x.dtype).to(x.device), self.bias_data)


def quantize_model_moonshot(model, method, **kwargs):
    """Apply moonshot compression to model."""
    print(f"Applying {method} compression...")
    
    original_bytes = 0
    quantized_bytes = 0
    
    # Collect all weights first (for codebook learning)
    if method == 'codebook':
        all_weights = []
        for layer in model.model.layers:
            mlp = layer.mlp
            for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(mlp, proj_name):
                    all_weights.append(getattr(mlp, proj_name).weight.data)
        
        codebook = learn_codebook(all_weights, **kwargs)
    else:
        codebook = None
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if not hasattr(mlp, proj_name):
                continue
            
            original = getattr(mlp, proj_name)
            original_bytes += original.weight.numel() * original.weight.element_size()
            
            new_layer = MoonshotLinear(original.in_features, original.out_features, method)
            new_layer.original_bytes = original.weight.numel() * original.weight.element_size()
            
            if method == 'codebook':
                new_layer.data = codebook_quantize(original.weight.data, codebook, kwargs.get('subvec_size', 8))
                new_layer.codebook = codebook
            elif method == 'manifold':
                new_layer.data = manifold_compress(original.weight.data, 
                                                   kwargs.get('rank_ratio', 0.1),
                                                   kwargs.get('residual_sparsity', 0.01))
            elif method == 'perceptual':
                new_layer.data = perceptual_compress(original.weight.data,
                                                     kwargs.get('base_bits', 2),
                                                     kwargs.get('residual_fraction', 0.02))
            elif method == 'fractal':
                new_layer.data = fractal_compress(original.weight.data,
                                                  kwargs.get('block_size', 8),
                                                  kwargs.get('num_transforms', 256))
            
            if original.bias is not None:
                new_layer.bias_data = original.bias.data.clone()
            
            setattr(mlp, proj_name, new_layer)
            quantized_bytes += new_layer.quantized_bytes()
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    compression = original_bytes / quantized_bytes if quantized_bytes > 0 else 0
    print(f"\nCompression: {original_bytes/1e9:.3f} GB → {quantized_bytes/1e9:.3f} GB = {compression:.2f}x")
    
    return model, original_bytes, quantized_bytes


def run_experiment(model_name, encodings, method, params, baseline_ppl):
    """Run moonshot experiment."""
    print(f"\n{'='*70}")
    print(f"Method: {method.upper()}")
    print(f"Params: {params}")
    print('='*70)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    model, orig_bytes, quant_bytes = quantize_model_moonshot(model, method, **params)
    
    quant_ppl = evaluate_ppl(model, encodings)
    ppl_delta = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    
    fp16_compression = orig_bytes / quant_bytes if quant_bytes > 0 else 0
    fp32_compression = (orig_bytes * 2) / quant_bytes if quant_bytes > 0 else 0
    
    print(f"\nResults:")
    print(f"  Baseline PPL:     {baseline_ppl:.2f}")
    print(f"  Quantized PPL:    {quant_ppl:.2f}")
    print(f"  PPL Delta:        {ppl_delta:+.2f}%")
    print(f"  Compression (FP16): {fp16_compression:.2f}x")
    print(f"  Compression (FP32): {fp32_compression:.2f}x")
    
    del model
    import gc
    gc.collect()
    
    return {
        'method': method,
        'params': str(params),
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quant_ppl,
        'ppl_delta': ppl_delta,
        'compression_fp16': fp16_compression,
        'compression_fp32': fp32_compression,
    }


def main():
    print("="*70)
    print("MOONSHOT COMPRESSION EXPERIMENTS")
    print("="*70)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encodings = get_wikitext2_subset(tokenizer, num_tokens=20000)
    
    # Get baseline
    print("\nComputing baseline...")
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
    
    # Experiments
    experiments = [
        # Codebook (DNA-inspired)
        ('codebook', {'num_codes': 256, 'subvec_size': 8}),
        ('codebook', {'num_codes': 512, 'subvec_size': 4}),
        ('codebook', {'num_codes': 1024, 'subvec_size': 8}),
        
        # Manifold projection (Neuroscience-inspired)
        ('manifold', {'rank_ratio': 0.05, 'residual_sparsity': 0.01}),
        ('manifold', {'rank_ratio': 0.10, 'residual_sparsity': 0.02}),
        ('manifold', {'rank_ratio': 0.15, 'residual_sparsity': 0.01}),
        
        # Perceptual coding (MP3-inspired)
        ('perceptual', {'base_bits': 2, 'residual_fraction': 0.02}),
        ('perceptual', {'base_bits': 2, 'residual_fraction': 0.05}),
        ('perceptual', {'base_bits': 3, 'residual_fraction': 0.02}),
    ]
    
    results = []
    for method, params in experiments:
        try:
            result = run_experiment(model_name, encodings, method, params, baseline_ppl)
            results.append(result)
            
            if result['compression_fp32'] >= 10 and result['ppl_delta'] < 1.0:
                print("\n*** MOONSHOT SUCCESS: 10x compression with <1% PPL! ***")
        except Exception as e:
            print(f"Error with {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("MOONSHOT RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Method':<12} {'Params':<35} {'PPL Δ':>8} {'FP32→':>8} {'Target':>12}")
    print("-"*80)
    
    for r in results:
        target = "✓ 10x+<1%" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ 10x" if r['compression_fp32'] >= 10 else \
                 "◐ <1%" if r['ppl_delta'] < 1.0 else \
                 "✗"
        params_short = r['params'][:33] + '..' if len(r['params']) > 35 else r['params']
        print(f"{r['method']:<12} {params_short:<35} {r['ppl_delta']:>+7.2f}% {r['compression_fp32']:>7.2f}x {target:>12}")
    
    # Save
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "moonshot_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/moonshot_experiments.json")


if __name__ == "__main__":
    main()
