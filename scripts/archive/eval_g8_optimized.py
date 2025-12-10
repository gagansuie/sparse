#!/usr/bin/env python3
"""
OPTIMIZED int4 quantization with:
1. FP16 scales/offsets (not FP32) - halves overhead
2. Mixed group sizes - g=8 for critical, g=32 for others

Target: Beat AWQ's 4x compression with <1% PPL delta
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time


def evaluate_ppl(model, encodings, max_steps=30):
    """Evaluate perplexity."""
    print(f"  Evaluating PPL ({max_steps} steps)...", end="", flush=True)
    model.eval()
    seq_len = encodings.input_ids.size(1)
    nlls = []
    stride = 256
    max_length = 512
    
    for step, begin_loc in enumerate(range(0, seq_len - 1, stride)):
        if step >= max_steps:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - (0 if step == 0 else begin_loc)
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.cpu())
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f" {ppl:.2f}", flush=True)
    return ppl


# =============================================================================
# OPTIMIZED INT4 QUANTIZATION
# =============================================================================

def int4_quantize_optimized(weight: torch.Tensor, group_size: int = 8, use_fp16: bool = True):
    """
    Quantize weight to int4 with optimized storage.
    
    Args:
        weight: (out_features, in_features) tensor
        group_size: Number of weights per group (8, 16, 32, etc.)
        use_fp16: Use FP16 for scales/offsets (vs FP32)
    
    Returns:
        dict with packed data, scales, offsets
    """
    weight = weight.float()
    out_features, in_features = weight.shape
    
    # Flatten and pad to multiple of group_size
    flat = weight.flatten()
    numel = flat.numel()
    padded_numel = ((numel + group_size - 1) // group_size) * group_size
    if padded_numel > numel:
        flat = F.pad(flat, (0, padded_numel - numel))
    
    # Reshape to groups
    groups = flat.view(-1, group_size)
    num_groups = groups.shape[0]
    
    # Compute per-group min/max
    g_min = groups.min(dim=1).values
    g_max = groups.max(dim=1).values
    
    # Compute scale and offset (asymmetric quantization)
    scale = (g_max - g_min) / 15.0  # 4-bit range: 0-15
    scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
    offset = g_min
    
    # Quantize
    q = ((groups - offset.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    
    # Pack two int4 values per byte
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    # Convert scales/offsets to FP16 if requested
    dtype = torch.float16 if use_fp16 else torch.float32
    
    return {
        'packed': packed,
        'scales': scale.to(dtype),
        'offsets': offset.to(dtype),
        'shape': weight.shape,
        'group_size': group_size,
        'original_numel': numel,
    }


def int4_dequantize_optimized(data: dict) -> torch.Tensor:
    """Dequantize int4 back to float."""
    packed = data['packed']
    scales = data['scales'].float()
    offsets = data['offsets'].float()
    shape = data['shape']
    group_size = data['group_size']
    original_numel = data['original_numel']
    
    # Unpack
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    # Reshape to groups
    groups = q.view(-1, group_size)
    
    # Dequantize
    weight = groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    
    # Flatten and trim padding
    weight = weight.flatten()[:original_numel]
    
    return weight.view(shape)


def compute_storage_bytes(data: dict) -> int:
    """Compute actual storage bytes."""
    packed_bytes = data['packed'].numel()
    scale_bytes = data['scales'].numel() * data['scales'].element_size()
    offset_bytes = data['offsets'].numel() * data['offsets'].element_size()
    return packed_bytes + scale_bytes + offset_bytes


# =============================================================================
# QUANTIZED LINEAR LAYER
# =============================================================================

class QuantizedLinear(nn.Module):
    """Linear layer with int4 quantization."""
    
    def __init__(self, in_features, out_features, data, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data = data
        self.register_buffer('bias', bias)
        self._weight_cache = None
    
    def forward(self, x):
        if self._weight_cache is None:
            self._weight_cache = int4_dequantize_optimized(self.data)
        weight = self._weight_cache.to(x.dtype).to(x.device)
        return F.linear(x, weight, self.bias)


# =============================================================================
# QUANTIZE MODEL
# =============================================================================

def quantize_model_optimized(model, group_size=8, use_fp16=True, mixed_groups=False):
    """
    Quantize model MLP layers with optimized int4.
    
    Args:
        model: HuggingFace model
        group_size: Default group size
        use_fp16: Use FP16 for scales/offsets
        mixed_groups: Use g=8 for first/last layers, g=32 for middle
    """
    print(f"Quantizing with group_size={group_size}, fp16_scales={use_fp16}, mixed={mixed_groups}")
    
    total_original = 0
    total_quantized = 0
    
    # Detect architecture
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        proj_names = ['c_fc', 'c_proj']
        is_gpt2 = True
    else:
        layers = model.model.layers
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
        is_gpt2 = False
    
    num_layers = len(layers)
    
    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        
        # Mixed group size strategy
        if mixed_groups:
            # First 2 and last 2 layers: g=8 (most critical)
            # Middle layers: g=32 (less critical, more compression)
            if layer_idx < 2 or layer_idx >= num_layers - 2:
                g = 8
            else:
                g = 32
        else:
            g = group_size
        
        for proj_name in proj_names:
            if not hasattr(mlp, proj_name):
                continue
            
            original = getattr(mlp, proj_name)
            
            # Get weight (handle GPT-2 Conv1D)
            if is_gpt2:
                weight = original.weight.data.t().contiguous()
                bias = original.bias.data.clone() if original.bias is not None else None
                in_f, out_f = original.weight.shape[0], original.nf
            else:
                weight = original.weight.data
                bias = original.bias.data.clone() if original.bias is not None else None
                in_f, out_f = original.in_features, original.out_features
            
            # Original size (FP16)
            original_bytes = weight.numel() * 2
            total_original += original_bytes
            
            # Quantize
            data = int4_quantize_optimized(weight, group_size=g, use_fp16=use_fp16)
            quantized_bytes = compute_storage_bytes(data)
            total_quantized += quantized_bytes
            
            # Replace layer
            new_layer = QuantizedLinear(in_f, out_f, data, bias)
            setattr(mlp, proj_name, new_layer)
        
        if layer_idx % 4 == 0:
            print(f"  Layer {layer_idx}/{num_layers} (g={g})")
    
    compression = total_original / total_quantized
    print(f"\nTotal: {total_original/1e6:.2f} MB → {total_quantized/1e6:.2f} MB = {compression:.2f}x")
    
    return model, total_original, total_quantized


# =============================================================================
# MAIN
# =============================================================================

def run_experiment(name, model_name, encodings, baseline_ppl, **kwargs):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print('='*60)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    
    model, orig_bytes, quant_bytes = quantize_model_optimized(model, **kwargs)
    
    ppl = evaluate_ppl(model, encodings)
    ppl_delta = (ppl - baseline_ppl) / baseline_ppl * 100
    compression = orig_bytes / quant_bytes
    
    # Calculate bits per weight
    total_weights = orig_bytes / 2  # FP16 = 2 bytes per weight
    bits_per_weight = (quant_bytes * 8) / total_weights
    
    print(f"\nResults:")
    print(f"  PPL: {ppl:.2f} (Δ {ppl_delta:+.2f}%)")
    print(f"  Compression: {compression:.2f}x vs FP16")
    print(f"  Bits/weight: {bits_per_weight:.2f}")
    
    del model
    
    return {
        'name': name,
        'ppl': ppl,
        'ppl_delta': ppl_delta,
        'compression': compression,
        'bits_per_weight': bits_per_weight,
    }


def main():
    print("=" * 60)
    print("OPTIMIZED INT4 QUANTIZATION")
    print("Target: Beat AWQ's 4x compression with <1% PPL")
    print("=" * 60)
    
    model_name = "gpt2"
    
    # Load data
    print(f"\nLoading {model_name} and WikiText-2...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=20000)
    
    # Baseline
    print("\n--- BASELINE ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    baseline_ppl = evaluate_ppl(model, encodings)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    del model
    
    # Experiments
    experiments = [
        # Original (FP32 scales, g=8) - for comparison
        ("g8_fp32 (original)", {'group_size': 8, 'use_fp16': False, 'mixed_groups': False}),
        
        # Optimization 1: FP16 scales/offsets
        ("g8_fp16", {'group_size': 8, 'use_fp16': True, 'mixed_groups': False}),
        
        # Optimization 2: Larger group size
        ("g16_fp16", {'group_size': 16, 'use_fp16': True, 'mixed_groups': False}),
        ("g32_fp16", {'group_size': 32, 'use_fp16': True, 'mixed_groups': False}),
        
        # Optimization 3: Mixed group sizes
        ("mixed_g8_g32_fp16", {'group_size': 8, 'use_fp16': True, 'mixed_groups': True}),
    ]
    
    results = []
    for name, kwargs in experiments:
        result = run_experiment(name, model_name, encodings, baseline_ppl, **kwargs)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'PPL Δ':<10} {'Compress':<10} {'Bits/W':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓ GOOD" if r['ppl_delta'] < 1.0 else "~ OK" if r['ppl_delta'] < 2.0 else "✗ FAIL"
        awq_beat = "🎯 >4x!" if r['compression'] > 4.0 else ""
        print(f"{r['name']:<25} {r['ppl_delta']:+.2f}%     {r['compression']:.2f}x      {r['bits_per_weight']:.2f}       {status} {awq_beat}")
    
    print("-" * 70)
    print("AWQ reference: ~4x compression, <1% PPL delta")
    print("=" * 70)
    
    # Find best result that beats AWQ
    best = None
    for r in results:
        if r['ppl_delta'] < 1.0 and r['compression'] > 4.0:
            if best is None or r['compression'] > best['compression']:
                best = r
    
    if best:
        print(f"\n🎉 SUCCESS: {best['name']} beats AWQ!")
        print(f"   Compression: {best['compression']:.2f}x (vs AWQ's 4x)")
        print(f"   PPL Delta: {best['ppl_delta']:+.2f}% (vs AWQ's <1%)")
    else:
        # Find best <1% PPL
        best_quality = max([r for r in results if r['ppl_delta'] < 1.0], 
                          key=lambda x: x['compression'], default=None)
        if best_quality:
            print(f"\n⚠️  Best <1% PPL: {best_quality['name']}")
            print(f"   Compression: {best_quality['compression']:.2f}x")
            print(f"   Need {4.0/best_quality['compression']:.1f}x more compression to beat AWQ")


if __name__ == "__main__":
    main()
