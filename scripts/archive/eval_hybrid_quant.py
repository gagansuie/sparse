#!/usr/bin/env python3
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
"""
HYBRID COMPRESSION: Manifold + Codebook

Novel approach combining two compression techniques:

1. MANIFOLD PROJECTION (Low-rank decomposition)
   - W ≈ U @ S @ V^T  (SVD)
   - Reduces dimensionality dramatically
   
2. CODEBOOK QUANTIZATION on the factors
   - Instead of storing U, S, V in FP16, quantize them with learned codebook
   - Much higher compression on already-compressed data

The key insight: After SVD, the factors U and V have much more structure
than the original weight matrix. Codebook quantization works better on
structured data.

Also includes:
- K-means++ initialization (better than random)
- Residual correction for critical weights
- Adaptive rank selection based on singular value decay

Target: 10x+ compression with <1% PPL
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


def evaluate_ppl(model, encodings, max_steps=30, early_stop_ppl=None):
    """Evaluate PPL with optional early stopping if PPL exceeds threshold."""
    print(f"Evaluating PPL ({max_steps} steps)...", end="", flush=True)
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
        
        # Early stopping: check PPL every 5 steps
        if early_stop_ppl and step > 0 and step % 5 == 0:
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            if current_ppl > early_stop_ppl:
                print(f" EARLY STOP at step {step} (PPL={current_ppl:.1f} > {early_stop_ppl:.1f})")
                return current_ppl
        
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f" done (PPL={ppl:.2f})")
    return ppl


# ============================================================================
# K-MEANS++ INITIALIZATION
# ============================================================================

def kmeans_plusplus_init(data: torch.Tensor, num_codes: int) -> torch.Tensor:
    """
    K-means++ initialization for better codebook starting points.
    
    Much better than random initialization - proven to give O(log k) 
    competitive ratio vs optimal clustering.
    """
    n = len(data)
    
    # First center: random
    centers = [data[torch.randint(n, (1,)).item()].clone()]
    
    for _ in range(num_codes - 1):
        # Compute distances to nearest center
        centers_tensor = torch.stack(centers)
        dists = torch.cdist(data, centers_tensor)
        min_dists = dists.min(dim=1).values
        
        # Sample proportional to distance squared
        probs = min_dists ** 2
        probs = probs / probs.sum()
        
        # Sample next center
        idx = torch.multinomial(probs, 1).item()
        centers.append(data[idx].clone())
    
    return torch.stack(centers)


def learn_codebook_plusplus(vectors: torch.Tensor, num_codes: int, max_iters: int = 15) -> torch.Tensor:
    """
    Learn codebook with k-means++ initialization.
    """
    print(f"  Learning codebook (k-means++, codes={num_codes})...")
    
    # K-means++ initialization
    codebook = kmeans_plusplus_init(vectors, num_codes)
    
    batch_size = 50000
    for iteration in range(max_iters):
        new_codebook = torch.zeros_like(codebook)
        counts = torch.zeros(num_codes)
        
        # Process in batches
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            batch = vectors[start:end]
            
            dists = torch.cdist(batch, codebook)
            assignments = dists.argmin(dim=1)
            
            for i, a in enumerate(assignments):
                new_codebook[a] += batch[i]
                counts[a] += 1
        
        counts = counts.clamp(min=1)
        new_codebook = new_codebook / counts.unsqueeze(1)
        
        diff = (new_codebook - codebook).abs().mean()
        codebook = new_codebook
        
        if iteration % 10 == 0:
            print(f"    Iter {iteration}: diff={diff:.6f}")
        
        if diff < 1e-6:
            break
    
    return codebook


# ============================================================================
# HYBRID COMPRESSION: MANIFOLD + CODEBOOK
# ============================================================================

def adaptive_rank(S: torch.Tensor, energy_threshold: float = 0.99) -> int:
    """
    Adaptively select rank based on singular value energy.
    
    Keep enough singular values to capture `energy_threshold` of total energy.
    """
    total_energy = (S ** 2).sum()
    cumulative_energy = (S ** 2).cumsum(dim=0)
    
    # Find rank where we capture threshold of energy
    rank = (cumulative_energy < energy_threshold * total_energy).sum().item() + 1
    return max(1, min(rank, len(S)))


def hybrid_compress(weight: torch.Tensor, 
                    rank_ratio: float = 0.15,
                    num_codes: int = 256,
                    subvec_size: int = 4,
                    residual_fraction: float = 0.005) -> Dict:
    """
    Hybrid compression: Manifold projection + Codebook quantization.
    
    Steps:
    1. SVD decomposition: W ≈ U @ diag(S) @ V^T
    2. Quantize U and V using learned codebooks
    3. Keep sparse residual for critical weights
    
    This is novel because:
    - SVD factors have more structure than raw weights
    - Codebook works better on structured data
    - Residual catches what both methods miss
    """
    shape = weight.shape
    m, n = shape
    w = weight.float()
    
    # Step 1: SVD decomposition
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    
    # Adaptive or fixed rank
    if rank_ratio < 1.0:
        rank = max(1, int(min(m, n) * rank_ratio))
    else:
        rank = adaptive_rank(S, energy_threshold=0.99)
    
    U_k = U[:, :rank]      # m x rank
    S_k = S[:rank]         # rank
    Vh_k = Vh[:rank, :]    # rank x n
    
    # Step 2: Quantize U and V with codebooks
    # Flatten into subvectors
    def quantize_factor(factor, name):
        flat = factor.flatten()
        pad_len = (subvec_size - len(flat) % subvec_size) % subvec_size
        if pad_len > 0:
            flat = torch.cat([flat, torch.zeros(pad_len)])
        
        subvecs = flat.view(-1, subvec_size)
        
        # Sample for codebook learning (memory efficient)
        sample_size = min(len(subvecs), 100000)
        sample_idx = torch.randperm(len(subvecs))[:sample_size]
        sample = subvecs[sample_idx]
        
        # Learn codebook with k-means++
        codebook = learn_codebook_plusplus(sample, num_codes)
        
        # Quantize all subvectors
        batch_size = 50000
        all_indices = []
        for start in range(0, len(subvecs), batch_size):
            end = min(start + batch_size, len(subvecs))
            batch = subvecs[start:end]
            dists = torch.cdist(batch, codebook)
            indices = dists.argmin(dim=1)
            all_indices.append(indices)
        
        indices = torch.cat(all_indices)
        
        if num_codes <= 256:
            indices = indices.to(torch.uint8)
        else:
            indices = indices.to(torch.int16)
        
        return {
            'indices': indices,
            'codebook': codebook.half(),
            'shape': factor.shape,
            'pad_len': pad_len,
        }
    
    U_quant = quantize_factor(U_k, "U")
    Vh_quant = quantize_factor(Vh_k, "Vh")
    
    # Step 3: Compute residual
    # Reconstruct from quantized factors
    def dequantize_factor(data, subvec_size):
        indices = data['indices'].long()
        codebook = data['codebook'].float()
        shape = data['shape']
        pad_len = data['pad_len']
        
        subvecs = codebook[indices]
        flat = subvecs.flatten()
        n = shape[0] * shape[1]
        flat = flat[:n]
        return flat.view(shape)
    
    U_deq = dequantize_factor(U_quant, subvec_size)
    Vh_deq = dequantize_factor(Vh_quant, subvec_size)
    
    # Low-rank reconstruction
    low_rank = U_deq @ torch.diag(S_k) @ Vh_deq
    
    # Residual
    residual = w - low_rank
    residual_flat = residual.flatten()
    
    # Keep top residual_fraction by magnitude
    num_residual = max(1, int(len(residual_flat) * residual_fraction))
    _, top_indices = residual_flat.abs().topk(num_residual)
    top_values = residual_flat[top_indices]
    
    return {
        'U': U_quant,
        'S': S_k.half(),
        'Vh': Vh_quant,
        'residual_indices': top_indices.to(torch.int32),
        'residual_values': top_values.half(),
        'shape': shape,
        'rank': rank,
        'subvec_size': subvec_size,
    }


def hybrid_decompress(data: Dict) -> torch.Tensor:
    """Decompress hybrid-encoded weight."""
    subvec_size = data['subvec_size']
    shape = data['shape']
    
    def dequantize_factor(factor_data):
        indices = factor_data['indices'].long()
        codebook = factor_data['codebook'].float()
        fshape = factor_data['shape']
        
        subvecs = codebook[indices]
        flat = subvecs.flatten()
        n = fshape[0] * fshape[1]
        flat = flat[:n]
        return flat.view(fshape)
    
    U = dequantize_factor(data['U'])
    S = data['S'].float()
    Vh = dequantize_factor(data['Vh'])
    
    # Reconstruct
    weight = U @ torch.diag(S) @ Vh
    
    # Add residual
    residual_indices = data['residual_indices'].long()
    residual_values = data['residual_values'].float()
    
    weight_flat = weight.flatten()
    weight_flat[residual_indices] += residual_values
    
    return weight_flat.view(shape)


def hybrid_bytes(data: Dict) -> int:
    """Compute storage for hybrid compression."""
    # U factor
    bytes_U = (data['U']['indices'].numel() * data['U']['indices'].element_size() +
               data['U']['codebook'].numel() * 2)
    # Vh factor
    bytes_Vh = (data['Vh']['indices'].numel() * data['Vh']['indices'].element_size() +
                data['Vh']['codebook'].numel() * 2)
    # S (singular values)
    bytes_S = data['S'].numel() * 2
    # Residual
    bytes_res = (data['residual_indices'].numel() * 4 +
                 data['residual_values'].numel() * 2)
    
    return bytes_U + bytes_Vh + bytes_S + bytes_res


# ============================================================================
# ALSO TEST: Pure INT4 with optimal group size (baseline comparison)
# ============================================================================

def int4_g8_compress(weight: torch.Tensor, group_size: int = 8) -> Dict:
    """Tenpak's proven int4_g8_v1 codec for comparison."""
    shape = weight.shape
    w = weight.float().flatten()
    
    pad_len = (group_size - len(w) % group_size) % group_size
    if pad_len > 0:
        w = torch.cat([w, torch.zeros(pad_len)])
    
    num_groups = len(w) // group_size
    groups = w.view(num_groups, group_size)
    
    min_vals = groups.min(dim=1).values
    max_vals = groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    w_q = ((groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    w_q_flat = w_q.flatten()
    packed = (w_q_flat[0::2] & 0x0F) | ((w_q_flat[1::2] & 0x0F) << 4)
    
    return {
        'packed': packed,
        'scales': scales.half(),
        'offsets': offsets.half(),
        'shape': shape,
        'pad_len': pad_len,
        'group_size': group_size,
    }


def int4_g8_decompress(data: Dict) -> torch.Tensor:
    """Decompress int4_g8."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    shape = data['shape']
    pad_len = data['pad_len']
    group_size = data['group_size']
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    w_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    w_q[0::2] = low
    w_q[1::2] = high
    
    num_groups = len(w_q) // group_size
    groups = w_q.view(num_groups, group_size).float()
    
    w = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    w = w.flatten()
    
    n = shape[0] * shape[1]
    w = w[:n]
    
    return w.view(shape)


def int4_g8_bytes(data: Dict) -> int:
    return (data['packed'].numel() + 
            data['scales'].numel() * 2 + 
            data['offsets'].numel() * 2)


# ============================================================================
# COMPRESSION LAYER
# ============================================================================

class HybridLinear(nn.Module):
    """Linear layer with hybrid compression."""
    
    def __init__(self, in_features, out_features, method='hybrid', needs_transpose=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method
        self.needs_transpose = needs_transpose  # For GPT-2 Conv1D compatibility
        self.data = None
        self.original_bytes = 0
        self.register_buffer('bias_data', None)
    
    def quantized_bytes(self):
        if self.method == 'hybrid':
            return hybrid_bytes(self.data)
        elif self.method == 'int4_g8':
            return int4_g8_bytes(self.data)
        return 0
    
    def forward(self, x):
        if self.method == 'hybrid':
            weight = hybrid_decompress(self.data)
        elif self.method == 'int4_g8':
            weight = int4_g8_decompress(self.data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # GPT-2 Conv1D stores weights transposed compared to nn.Linear
        if self.needs_transpose:
            weight = weight.t()
        
        return F.linear(x, weight.to(x.dtype).to(x.device), self.bias_data)


def quantize_model(model, method, **kwargs):
    """Apply compression to model. Supports GPT-2 and LLaMA-style architectures."""
    print(f"Applying {method} compression...")
    
    original_bytes = 0
    quantized_bytes = 0
    
    # Detect architecture
    is_gpt2 = False
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style: model.transformer.h[i].mlp.c_fc, c_proj
        layers = model.transformer.h
        mlp_attr = 'mlp'
        proj_names = ['c_fc', 'c_proj']  # GPT-2 MLP projections
        is_gpt2 = True
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA style: model.model.layers[i].mlp.gate_proj, up_proj, down_proj
        layers = model.model.layers
        mlp_attr = 'mlp'
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")
    
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, mlp_attr)
        for proj_name in proj_names:
            if not hasattr(mlp, proj_name):
                continue
            
            original = getattr(mlp, proj_name)
            original_bytes += original.weight.numel() * original.weight.element_size()
            
            # Handle both nn.Linear and Conv1D (GPT-2)
            # GPT-2 Conv1D: weight is (in_features, out_features) - need to transpose for F.linear
            # nn.Linear: weight is (out_features, in_features) - already correct
            if hasattr(original, 'in_features'):
                in_features, out_features = original.in_features, original.out_features
                weight_for_compress = original.weight.data  # Already (out, in)
            else:
                # Conv1D: weight is (in_features, out_features), nf is out_features
                in_features = original.weight.shape[0]
                out_features = original.nf
                # Transpose to (out, in) for compression, will transpose back in forward
                weight_for_compress = original.weight.data.t().contiguous()
            
            # We already transposed GPT-2 weights during compression, so no need to transpose in forward
            new_layer = HybridLinear(in_features, out_features, method, needs_transpose=False)
            new_layer.original_bytes = original.weight.numel() * original.weight.element_size()
            
            if method == 'hybrid':
                new_layer.data = hybrid_compress(
                    weight_for_compress,
                    rank_ratio=kwargs.get('rank_ratio', 0.15),
                    num_codes=kwargs.get('num_codes', 256),
                    subvec_size=kwargs.get('subvec_size', 4),
                    residual_fraction=kwargs.get('residual_fraction', 0.005),
                )
            elif method == 'int4_g8':
                new_layer.data = int4_g8_compress(
                    weight_for_compress,
                    group_size=kwargs.get('group_size', 8),
                )
            
            if original.bias is not None:
                new_layer.bias_data = original.bias.data.clone()
            
            setattr(mlp, proj_name, new_layer)
            quantized_bytes += new_layer.quantized_bytes()
        
        if layer_idx % 3 == 0:
            print(f"  Layer {layer_idx}/{len(layers)}")
    
    compression = original_bytes / quantized_bytes if quantized_bytes > 0 else 0
    print(f"\nCompression: {original_bytes/1e9:.3f} GB → {quantized_bytes/1e9:.3f} GB = {compression:.2f}x")
    
    return model, original_bytes, quantized_bytes


def run_experiment(model_name, encodings, method, params, baseline_ppl, early_stop_ppl=None):
    """Run experiment with optional early stopping."""
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
    
    model, orig_bytes, quant_bytes = quantize_model(model, method, **params)
    
    quant_ppl = evaluate_ppl(model, encodings, early_stop_ppl=early_stop_ppl)
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
    print("HYBRID COMPRESSION: MANIFOLD + CODEBOOK")
    print("="*70)
    
    # Use GPT-2 for faster iteration (124M vs 1.1B)
    model_name = "gpt2"
    
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
    
    # Early stop threshold: 2x baseline PPL (way beyond 1% target)
    early_stop_ppl = baseline_ppl * 2.0
    print(f"Early stop threshold: {early_stop_ppl:.2f} (2x baseline)")
    
    del model
    import gc
    gc.collect()
    
    # Experiments - ordered from conservative to aggressive
    # Start with high quality, progressively increase compression
    experiments = [
        # Baseline: Tenpak int4_g8_v1 (known good: 14.4x, +0.43%)
        ('int4_g8', {'group_size': 8}),
        
        # Hybrid: Very conservative (establish if method works at all)
        ('hybrid', {'rank_ratio': 0.50, 'num_codes': 1024, 'subvec_size': 4, 'residual_fraction': 0.05}),
        ('hybrid', {'rank_ratio': 0.40, 'num_codes': 512, 'subvec_size': 4, 'residual_fraction': 0.02}),
        ('hybrid', {'rank_ratio': 0.30, 'num_codes': 512, 'subvec_size': 4, 'residual_fraction': 0.01}),
        
        # Hybrid: Moderate (if conservative works, try these)
        ('hybrid', {'rank_ratio': 0.25, 'num_codes': 512, 'subvec_size': 4, 'residual_fraction': 0.01}),
        ('hybrid', {'rank_ratio': 0.20, 'num_codes': 512, 'subvec_size': 4, 'residual_fraction': 0.01}),
    ]
    
    results = []
    for method, params in experiments:
        try:
            result = run_experiment(model_name, encodings, method, params, baseline_ppl, early_stop_ppl)
            results.append(result)
            
            if result['compression_fp32'] >= 10 and result['ppl_delta'] < 1.0:
                print("\n" + "!"*70)
                print("!!! MOONSHOT SUCCESS: 10x+ compression with <1% PPL !!!")
                print("!"*70)
        except Exception as e:
            print(f"Error with {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("HYBRID COMPRESSION RESULTS")
    print("="*70)
    print(f"\n{'Method':<10} {'Params':<50} {'PPL Δ':>8} {'FP32→':>8} {'Target':>10}")
    print("-"*90)
    
    for r in results:
        target = "✓ 10x+<1%" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ 10x" if r['compression_fp32'] >= 10 else \
                 "◐ <1%" if r['ppl_delta'] < 1.0 else \
                 "✗"
        params_short = r['params'][:48] + '..' if len(r['params']) > 50 else r['params']
        print(f"{r['method']:<10} {params_short:<50} {r['ppl_delta']:>+7.2f}% {r['compression_fp32']:>7.2f}x {target:>10}")
    
    # Save
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "hybrid_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/hybrid_experiments.json")


if __name__ == "__main__":
    main()
