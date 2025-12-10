#!/usr/bin/env python3
"""
Sparse + Quantized Compression

Strategy: Prune small-magnitude weights + INT4 quantize the rest

Math for 10x compression:
- Prune 60% of weights (stored as sparse indices)
- Remaining 40% quantized to INT4
- Sparse storage: indices (16-bit) + values (4-bit)

Example for 1M weights:
- Keep 400k weights
- Indices: 400k × 2 bytes = 800KB
- Values (INT4): 400k × 0.5 bytes = 200KB  
- Scales (g=8): 50k × 2 bytes = 100KB
- Total: 1.1MB vs 4MB (FP32) = 3.6x

Hmm, that's not 10x. Let's try a different approach:
- Use block sparsity (prune entire groups)
- Or use magnitude-based pruning with higher sparsity
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
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
# Strategy: Block-wise Pruning + INT4 Quantization
# ============================================================================

def sparse_quant_block(weight, block_size=8, sparsity=0.5, group_size=8):
    """
    Block-sparse quantization:
    1. Divide weights into blocks of size block_size
    2. Compute L2 norm of each block
    3. Prune blocks with lowest norms (set to zero)
    4. Quantize remaining blocks to INT4
    
    Storage:
    - Bitmap: 1 bit per block (which blocks are non-zero)
    - Non-zero blocks: INT4 packed + scales/offsets
    
    For sparsity=0.5, block_size=8:
    - 50% of blocks pruned
    - Remaining 50%: 4 bytes (packed INT4) + 2 bytes (scale) + 2 bytes (offset) = 8 bytes per 8 weights
    - Bitmap: 1 bit per 8 weights = 0.125 bytes per 8 weights
    - Average: 0.5 * (8 + 0.125) + 0.5 * 0.125 = 4.125 bytes per 8 weights = 0.516 bytes/weight
    
    From FP32 (4 bytes): 4 / 0.516 = 7.75x
    
    Need higher sparsity for 10x...
    """
    out_features, in_features = weight.shape
    weight_flat = weight.flatten().float()
    
    # Pad to multiple of block_size
    pad_len = (block_size - len(weight_flat) % block_size) % block_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_blocks = len(weight_flat) // block_size
    weight_blocks = weight_flat.view(num_blocks, block_size)
    
    # Compute block importance (L2 norm)
    block_norms = weight_blocks.norm(dim=1)
    
    # Find threshold for sparsity
    num_prune = int(num_blocks * sparsity)
    if num_prune > 0:
        threshold = torch.kthvalue(block_norms, num_prune).values
        mask = block_norms > threshold  # True = keep, False = prune
    else:
        mask = torch.ones(num_blocks, dtype=torch.bool)
    
    # Get non-zero blocks
    nonzero_blocks = weight_blocks[mask]
    num_nonzero = nonzero_blocks.shape[0]
    
    # Quantize non-zero blocks to INT4
    if num_nonzero > 0:
        min_vals = nonzero_blocks.min(dim=1).values
        max_vals = nonzero_blocks.max(dim=1).values
        
        scales = (max_vals - min_vals) / 15.0
        scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
        offsets = min_vals
        
        weight_q = ((nonzero_blocks - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
        weight_q_flat = weight_q.flatten()
        packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    else:
        packed = torch.tensor([], dtype=torch.uint8)
        scales = torch.tensor([], dtype=torch.float16)
        offsets = torch.tensor([], dtype=torch.float16)
    
    return {
        'packed': packed,
        'scales': scales.half(),
        'offsets': offsets.half(),
        'mask': mask,  # Boolean mask for which blocks are non-zero
        'shape': weight.shape,
        'pad_len': pad_len,
        'block_size': block_size,
        'num_blocks': num_blocks,
        'num_nonzero': num_nonzero,
    }


def dequantize_sparse_block(data):
    """Dequantize sparse block-quantized weights."""
    packed = data['packed']
    scales = data['scales']
    offsets = data['offsets']
    mask = data['mask']
    shape = data['shape']
    pad_len = data['pad_len']
    block_size = data['block_size']
    num_blocks = data['num_blocks']
    
    # Unpack INT4
    if len(packed) > 0:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
        weight_q[0::2] = low
        weight_q[1::2] = high
        
        num_nonzero = data['num_nonzero']
        weight_groups = weight_q.view(num_nonzero, block_size).float()
        
        # Dequantize
        weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    # Reconstruct full weight tensor
    weight_full = torch.zeros(num_blocks, block_size)
    if len(packed) > 0:
        weight_full[mask] = weight_deq
    
    # Flatten and remove padding
    weight_flat = weight_full.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    
    return weight_flat.view(shape)


def compute_sparse_bytes(data):
    """Compute storage size for sparse quantized weights."""
    # Packed INT4 values
    bytes_packed = data['packed'].numel() * data['packed'].element_size()
    # Scales and offsets (FP16)
    bytes_scales = data['scales'].numel() * data['scales'].element_size()
    bytes_offsets = data['offsets'].numel() * data['offsets'].element_size()
    # Mask (1 bit per block, stored as uint8)
    bytes_mask = (data['num_blocks'] + 7) // 8
    
    return bytes_packed + bytes_scales + bytes_offsets + bytes_mask


class SparseQuantLinear(nn.Module):
    """Linear layer with sparse block quantization."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data = None
        self.original_bytes = 0
        self.register_buffer('bias_data', None)
    
    @classmethod
    def from_linear(cls, linear, sparsity=0.5, block_size=8):
        layer = cls(linear.in_features, linear.out_features)
        layer.original_bytes = linear.weight.numel() * linear.weight.element_size()
        layer.data = sparse_quant_block(linear.weight.data, block_size=block_size, sparsity=sparsity)
        
        # Register buffers for the quantized data
        layer.register_buffer('packed', layer.data['packed'])
        layer.register_buffer('scales', layer.data['scales'])
        layer.register_buffer('offsets', layer.data['offsets'])
        layer.register_buffer('mask', layer.data['mask'])
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        
        return layer
    
    def quantized_bytes(self):
        return compute_sparse_bytes(self.data)
    
    def forward(self, x):
        weight = dequantize_sparse_block(self.data).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model_sparse(model, sparsity=0.5, block_size=8):
    """Apply sparse quantization to MLP layers."""
    print(f"Applying sparse quantization (sparsity={sparsity}, block_size={block_size})...")
    
    original_bytes = 0
    quantized_bytes = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                original_bytes += original.weight.numel() * original.weight.element_size()
                
                quantized = SparseQuantLinear.from_linear(original, sparsity=sparsity, block_size=block_size)
                setattr(mlp, proj_name, quantized)
                quantized_bytes += quantized.quantized_bytes()
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    compression = original_bytes / quantized_bytes
    print(f"\nCompression: {original_bytes/1e9:.3f} GB → {quantized_bytes/1e9:.3f} GB = {compression:.2f}x")
    
    return model, original_bytes, quantized_bytes


def run_experiment(model_name, encodings, sparsity, block_size, baseline_ppl):
    """Run a single sparse quantization experiment."""
    print(f"\n{'='*70}")
    print(f"Sparsity: {sparsity*100:.0f}%, Block Size: {block_size}")
    print('='*70)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    model, orig_bytes, quant_bytes = quantize_model_sparse(model, sparsity=sparsity, block_size=block_size)
    
    quant_ppl = evaluate_ppl(model, encodings)
    ppl_delta = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    
    fp16_compression = orig_bytes / quant_bytes
    fp32_compression = (orig_bytes * 2) / quant_bytes
    
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
        'sparsity': sparsity,
        'block_size': block_size,
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quant_ppl,
        'ppl_delta': ppl_delta,
        'compression_fp16': fp16_compression,
        'compression_fp32': fp32_compression,
    }


def main():
    print("="*70)
    print("SPARSE + QUANTIZED COMPRESSION EXPERIMENTS")
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
    
    # Test different sparsity levels
    experiments = [
        # (sparsity, block_size)
        (0.3, 8),   # 30% pruned
        (0.4, 8),   # 40% pruned
        (0.5, 8),   # 50% pruned
        (0.6, 8),   # 60% pruned
        (0.7, 8),   # 70% pruned - target for 10x
        (0.5, 16),  # 50% pruned, larger blocks
        (0.6, 16),  # 60% pruned, larger blocks
    ]
    
    results = []
    for sparsity, block_size in experiments:
        try:
            result = run_experiment(model_name, encodings, sparsity, block_size, baseline_ppl)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Sparsity':<10} {'Block':<6} {'PPL Δ':>8} {'FP16→':>8} {'FP32→':>8} {'Target':>10}")
    print("-"*60)
    
    for r in results:
        target = "✓ 10x+<1%" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ 10x" if r['compression_fp32'] >= 10 else \
                 "◐ <1%" if r['ppl_delta'] < 1.0 else \
                 "✗"
        print(f"{r['sparsity']*100:>6.0f}%   {r['block_size']:<6} {r['ppl_delta']:>+7.2f}% {r['compression_fp16']:>7.2f}x {r['compression_fp32']:>7.2f}x {target:>10}")
    
    # Save results
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "sparse_quant_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/sparse_quant_experiments.json")


if __name__ == "__main__":
    main()
