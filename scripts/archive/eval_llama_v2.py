#!/usr/bin/env python3
"""
Tenpak Evaluation v2 - Accurate compression measurement + FP16 scales/offsets

Fixes:
1. Measures weight-only compression (not full model)
2. Uses FP16 for scales/offsets (halves overhead)
3. Reports both MLP-only and full-model metrics
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TENPAK_ROOT = Path(__file__).parent.parent


def get_wikitext2_subset(tokenizer, num_tokens=30000):
    """Load a subset of WikiText-2 for quick testing."""
    print("Loading WikiText-2 dataset (subset)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", max_length=num_tokens, truncation=True)
    print(f"Using {encodings.input_ids.size(1)} tokens")
    return encodings


def evaluate_perplexity_quick(model, encodings, device, max_length=512, stride=256, max_steps=50):
    """Quick perplexity evaluation."""
    print(f"Evaluating perplexity ({max_steps} steps)...")
    
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    step = 0
    
    for begin_loc in range(0, seq_len - 1, stride):
        if step >= max_steps:
            break
            
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood.cpu())
        prev_end_loc = end_loc
        step += 1
        
        if step % 10 == 0:
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  Step {step}/{max_steps}, PPL: {current_ppl:.2f}")
        
        if end_loc >= seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def quantize_linear_g8_fp16(weight, group_size=8):
    """
    Quantize weights using int4_g8_v1 with FP16 scales/offsets.
    
    Storage per group of 8 weights:
    - 4 bytes: packed int4 (8 weights × 0.5 bytes)
    - 2 bytes: scale (fp16)
    - 2 bytes: offset (fp16)
    Total: 8 bytes for 8 weights = 1 byte/weight
    
    vs FP16: 2 bytes/weight → 2x compression per weight
    vs FP32: 4 bytes/weight → 4x compression per weight
    """
    weight_flat = weight.flatten().float()
    
    # Pad to multiple of group_size
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    # Compute per-group min/max
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    # Compute scales and offsets (store as FP16)
    scales = ((max_vals - min_vals) / 15.0).half()
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals.half()
    
    # Quantize to 0-15
    scales_f32 = scales.float()
    offsets_f32 = offsets.float()
    weight_q = ((weight_groups - offsets_f32.unsqueeze(1)) / scales_f32.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    
    # Pack two int4 values per byte
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return packed, scales, offsets, weight.shape, pad_len


def dequantize_linear_g8_fp16(packed, scales, offsets, shape, pad_len, group_size=8):
    """Dequantize int4_g8_v1 weights with FP16 scales/offsets."""
    # Unpack
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    # Reshape to groups
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    
    # Dequantize (convert scales/offsets to float for computation)
    scales_f32 = scales.float()
    offsets_f32 = offsets.float()
    weight_deq = weight_groups * scales_f32.unsqueeze(1) + offsets_f32.unsqueeze(1)
    
    # Flatten and remove padding
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    
    return weight_flat.view(shape)


class QuantizedLinearV2(nn.Module):
    """Linear layer with int4_g8 quantized weights and FP16 scales/offsets."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('packed', None)
        self.register_buffer('scales', None)  # FP16
        self.register_buffer('offsets', None)  # FP16
        self.register_buffer('bias_data', None)
        self.pad_len = 0
        self.original_weight_bytes = 0  # Track original size
    
    @classmethod
    def from_linear(cls, linear):
        layer = cls(linear.in_features, linear.out_features)
        
        # Track original weight size
        layer.original_weight_bytes = linear.weight.numel() * linear.weight.element_size()
        
        # Quantize
        packed, scales, offsets, shape, pad_len = quantize_linear_g8_fp16(linear.weight.data)
        layer.packed = packed
        layer.scales = scales  # Already FP16
        layer.offsets = offsets  # Already FP16
        layer.pad_len = pad_len
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        
        return layer
    
    def quantized_bytes(self):
        """Return actual storage size in bytes."""
        total = self.packed.numel() * self.packed.element_size()  # uint8
        total += self.scales.numel() * self.scales.element_size()  # fp16
        total += self.offsets.numel() * self.offsets.element_size()  # fp16
        if self.bias_data is not None:
            total += self.bias_data.numel() * self.bias_data.element_size()
        return total
    
    def forward(self, x):
        weight = dequantize_linear_g8_fp16(
            self.packed, self.scales, self.offsets,
            (self.out_features, self.in_features), self.pad_len
        ).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model_mlp_v2(model):
    """Quantize MLP layers and track compression stats."""
    print("Quantizing MLP layers with int4_g8 (FP16 scales/offsets)...")
    
    stats = {
        'original_mlp_bytes': 0,
        'quantized_mlp_bytes': 0,
        'mlp_params': 0,
        'total_params': sum(p.numel() for p in model.parameters()),
    }
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                
                # Track original size
                orig_bytes = original.weight.numel() * original.weight.element_size()
                stats['original_mlp_bytes'] += orig_bytes
                stats['mlp_params'] += original.weight.numel()
                
                # Quantize
                quantized = QuantizedLinearV2.from_linear(original)
                setattr(mlp, proj_name, quantized)
                
                # Track quantized size
                stats['quantized_mlp_bytes'] += quantized.quantized_bytes()
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    stats['mlp_compression'] = stats['original_mlp_bytes'] / stats['quantized_mlp_bytes']
    
    print(f"\nMLP Quantization Stats:")
    print(f"  Original MLP:  {stats['original_mlp_bytes'] / 1e9:.3f} GB")
    print(f"  Quantized MLP: {stats['quantized_mlp_bytes'] / 1e9:.3f} GB")
    print(f"  MLP Compression: {stats['mlp_compression']:.2f}x")
    print(f"  MLP params: {stats['mlp_params']:,} / {stats['total_params']:,} ({100*stats['mlp_params']/stats['total_params']:.1f}%)")
    
    return model, stats


def measure_model_bytes(model):
    """Measure total model size."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def main():
    print("="*70)
    print("TENPAK EVALUATION v2 - Accurate Compression Measurement")
    print("="*70)
    
    device = "cpu"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading dataset...")
    encodings = get_wikitext2_subset(tokenizer, num_tokens=30000)
    
    # === BASELINE ===
    print(f"\n{'='*70}")
    print("BASELINE (FP16)")
    print("="*70)
    
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    baseline_bytes = measure_model_bytes(model)
    print(f"Full model size: {baseline_bytes / 1e9:.2f} GB")
    
    # Measure MLP-only baseline
    mlp_baseline_bytes = 0
    for layer in model.model.layers:
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.mlp, proj_name):
                w = getattr(layer.mlp, proj_name).weight
                mlp_baseline_bytes += w.numel() * w.element_size()
    print(f"MLP weights only: {mlp_baseline_bytes / 1e9:.2f} GB")
    
    start = time.time()
    baseline_ppl = evaluate_perplexity_quick(model, encodings, device, max_steps=50)
    baseline_time = time.time() - start
    print(f"\nBaseline PPL: {baseline_ppl:.2f} ({baseline_time:.1f}s)")
    
    # === QUANTIZED ===
    print(f"\n{'='*70}")
    print("TENPAK int4_g8_v1 (FP16 scales/offsets)")
    print("="*70)
    
    model, quant_stats = quantize_model_mlp_v2(model)
    model.eval()
    
    quantized_bytes = measure_model_bytes(model)
    print(f"\nFull model size: {quantized_bytes / 1e9:.2f} GB")
    
    start = time.time()
    quantized_ppl = evaluate_perplexity_quick(model, encodings, device, max_steps=50)
    quantized_time = time.time() - start
    print(f"\nQuantized PPL: {quantized_ppl:.2f} ({quantized_time:.1f}s)")
    
    # === RESULTS ===
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("="*70)
    
    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    full_compression = baseline_bytes / quantized_bytes
    mlp_compression = quant_stats['mlp_compression']
    
    # Calculate theoretical FP32 → int4_g8 compression
    # FP32: 4 bytes/weight
    # int4_g8 with FP16 scales: 1 byte/weight (4 packed + 2 scale + 2 offset per 8 weights)
    theoretical_compression = 4.0 / 1.0  # 4x from FP32
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                         QUALITY                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Baseline PPL:      {baseline_ppl:>8.2f}                                      │
│  Quantized PPL:     {quantized_ppl:>8.2f}                                      │
│  PPL Delta:         {ppl_delta:>+7.2f}%                                       │
├─────────────────────────────────────────────────────────────────────┤
│                       COMPRESSION                                    │
├─────────────────────────────────────────────────────────────────────┤
│  MLP Weights (FP16 baseline):                                        │
│    Original:        {quant_stats['original_mlp_bytes']/1e9:>8.3f} GB                                   │
│    Quantized:       {quant_stats['quantized_mlp_bytes']/1e9:>8.3f} GB                                   │
│    Compression:     {mlp_compression:>8.2f}x                                      │
│                                                                      │
│  Full Model:                                                         │
│    Original:        {baseline_bytes/1e9:>8.2f} GB                                    │
│    Quantized:       {quantized_bytes/1e9:>8.2f} GB                                    │
│    Compression:     {full_compression:>8.2f}x                                      │
│                                                                      │
│  Theoretical (FP32 → int4_g8):                                       │
│    Per-weight:      {theoretical_compression:>8.1f}x                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Verdict
    if ppl_delta < 1.0:
        print("✓ SUCCESS: PPL delta under 1% target!")
    elif ppl_delta < 2.0:
        print("◐ CLOSE: PPL delta under 2%, acceptable for most use cases")
    else:
        print("✗ PPL delta exceeds 2% target")
    
    # Save results
    results = {
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_delta_percent": ppl_delta,
        "mlp_original_gb": quant_stats['original_mlp_bytes'] / 1e9,
        "mlp_quantized_gb": quant_stats['quantized_mlp_bytes'] / 1e9,
        "mlp_compression": mlp_compression,
        "full_model_original_gb": baseline_bytes / 1e9,
        "full_model_quantized_gb": quantized_bytes / 1e9,
        "full_model_compression": full_compression,
    }
    
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "tinyllama_v2_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/tinyllama_v2_eval.json")


if __name__ == "__main__":
    main()
