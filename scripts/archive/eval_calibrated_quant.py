#!/usr/bin/env python3
"""
Lightweight Calibrated Quantization

Strategy (inspired by AWQ):
1. Run ~128 calibration samples through the model
2. Measure activation magnitudes for each weight column
3. Scale weights by activation importance before quantization
4. This protects "salient" weights that have high activation × weight products

Key insight from AWQ paper:
- Not all weights are equally important
- Weights that multiply large activations matter more
- By scaling these weights up before quantization, we reduce their relative error

Storage is the same as regular INT4, but quality is much better.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

TENPAK_ROOT = Path(__file__).parent.parent


def get_calibration_data(tokenizer, num_samples=128, seq_len=512):
    """Get calibration samples from C4 dataset (like AWQ uses)."""
    print(f"Loading {num_samples} calibration samples...")
    
    try:
        # Try C4 first (what AWQ uses)
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            text = sample["text"]
            if len(text) > 100:  # Skip very short samples
                samples.append(text)
        
        if len(samples) < num_samples:
            raise ValueError("Not enough samples from C4")
            
    except Exception as e:
        print(f"C4 failed ({e}), falling back to WikiText...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])
        # Split into chunks
        samples = [text[i:i+2000] for i in range(0, len(text), 2000)][:num_samples]
    
    # Tokenize
    encodings = []
    for text in samples:
        enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
        if enc.input_ids.size(1) >= 64:  # Skip very short sequences
            encodings.append(enc.input_ids)
    
    print(f"Got {len(encodings)} calibration sequences")
    return encodings


def get_wikitext2_subset(tokenizer, num_tokens=20000):
    print("Loading WikiText-2 for evaluation...")
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
# Activation-Aware Weight Quantization (AWQ-style)
# ============================================================================

class ActivationCollector:
    """Collect activation statistics for calibration."""
    
    def __init__(self):
        self.activation_scales = {}  # layer_name -> scale per input channel
    
    def collect_hook(self, name):
        """Create a forward hook to collect activation magnitudes."""
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            
            # Compute mean absolute activation per channel
            # x shape: [batch, seq_len, hidden_dim]
            if x.dim() == 3:
                # Average over batch and sequence
                scale = x.abs().mean(dim=(0, 1))
            else:
                scale = x.abs().mean(dim=0)
            
            if name in self.activation_scales:
                # Running average
                self.activation_scales[name] = 0.9 * self.activation_scales[name] + 0.1 * scale
            else:
                self.activation_scales[name] = scale
        
        return hook


def calibrate_model(model, calibration_data: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Run calibration to collect activation statistics.
    
    Returns dict mapping layer names to activation scales.
    """
    print("Running calibration...")
    collector = ActivationCollector()
    hooks = []
    
    # Register hooks on all MLP linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
            hook = module.register_forward_hook(collector.collect_hook(name))
            hooks.append(hook)
    
    # Run calibration samples
    model.eval()
    with torch.no_grad():
        for i, input_ids in enumerate(calibration_data):
            if i % 20 == 0:
                print(f"  Calibration sample {i}/{len(calibration_data)}")
            try:
                model(input_ids)
            except Exception as e:
                print(f"  Warning: sample {i} failed: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Collected activation scales for {len(collector.activation_scales)} layers")
    return collector.activation_scales


def awq_quantize_weight(weight, activation_scale, group_size=128, scale_factor=1.0):
    """
    AWQ-style quantization with activation-aware scaling.
    
    Key idea: Scale up weights that correspond to high activations,
    then quantize, then scale back down during dequantization.
    
    This reduces relative quantization error for important weights.
    """
    out_features, in_features = weight.shape
    weight = weight.float()
    
    # Normalize activation scale
    act_scale = activation_scale.float()
    act_scale = act_scale / (act_scale.mean() + 1e-8)
    
    # Compute importance: weights × activation
    # Higher importance = protect more
    importance = (weight.abs() * act_scale.unsqueeze(0)).mean(dim=0)
    importance = importance / (importance.mean() + 1e-8)
    
    # Scale factor: how much to scale up important weights
    # AWQ uses a search to find optimal scale, we use a simpler heuristic
    scale = importance.pow(scale_factor).clamp(min=0.1, max=10.0)
    
    # Scale weights
    weight_scaled = weight * scale.unsqueeze(0)
    
    # Flatten for group quantization
    weight_flat = weight_scaled.flatten()
    
    # Pad
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    # Quantize to INT4
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return {
        'packed': packed,
        'scales': scales.half(),
        'offsets': offsets.half(),
        'inv_scale': (1.0 / scale).half(),  # For dequantization
        'shape': weight.shape,
        'pad_len': pad_len,
        'group_size': group_size,
    }


def awq_dequantize_weight(data):
    """Dequantize AWQ-style weights."""
    packed = data['packed']
    scales = data['scales']
    offsets = data['offsets']
    inv_scale = data['inv_scale']
    shape = data['shape']
    pad_len = data['pad_len']
    group_size = data['group_size']
    
    # Unpack INT4
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    
    # Dequantize
    weight_deq = weight_groups * scales.float().unsqueeze(1) + offsets.float().unsqueeze(1)
    
    # Flatten and remove padding
    weight_flat = weight_deq.flatten()
    n = shape[0] * shape[1]
    weight_flat = weight_flat[:n]
    weight = weight_flat.view(shape)
    
    # Apply inverse scale to undo the AWQ scaling
    weight = weight * inv_scale.float().unsqueeze(0)
    
    return weight


def compute_awq_bytes(data):
    """Compute storage size."""
    bytes_packed = data['packed'].numel() * 1
    bytes_scales = data['scales'].numel() * 2
    bytes_offsets = data['offsets'].numel() * 2
    bytes_inv_scale = data['inv_scale'].numel() * 2
    return bytes_packed + bytes_scales + bytes_offsets + bytes_inv_scale


class AWQLinear(nn.Module):
    """Linear layer with AWQ-style quantization."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data = None
        self.original_bytes = 0
        self.register_buffer('bias_data', None)
    
    @classmethod
    def from_linear(cls, linear, activation_scale, group_size=128, scale_factor=1.0):
        layer = cls(linear.in_features, linear.out_features)
        layer.original_bytes = linear.weight.numel() * linear.weight.element_size()
        layer.data = awq_quantize_weight(
            linear.weight.data, activation_scale, group_size, scale_factor
        )
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        
        return layer
    
    def quantized_bytes(self):
        return compute_awq_bytes(self.data)
    
    def forward(self, x):
        weight = awq_dequantize_weight(self.data).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model_awq(model, activation_scales: Dict[str, torch.Tensor], group_size=128, scale_factor=1.0):
    """Apply AWQ-style quantization using calibration data."""
    print(f"Applying AWQ-style quantization (g={group_size}, scale_factor={scale_factor})...")
    
    original_bytes = 0
    quantized_bytes = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if not hasattr(mlp, proj_name):
                continue
            
            original = getattr(mlp, proj_name)
            original_bytes += original.weight.numel() * original.weight.element_size()
            
            # Find activation scale for this layer
            full_name = None
            for name in activation_scales:
                if f"layers.{layer_idx}.mlp.{proj_name}" in name:
                    full_name = name
                    break
            
            if full_name is None:
                # Fallback: use uniform scale
                act_scale = torch.ones(original.in_features)
            else:
                act_scale = activation_scales[full_name]
            
            quantized = AWQLinear.from_linear(original, act_scale, group_size, scale_factor)
            setattr(mlp, proj_name, quantized)
            quantized_bytes += quantized.quantized_bytes()
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    compression = original_bytes / quantized_bytes
    print(f"\nCompression: {original_bytes/1e9:.3f} GB → {quantized_bytes/1e9:.3f} GB = {compression:.2f}x")
    
    return model, original_bytes, quantized_bytes


def run_experiment(model_name, calibration_data, eval_encodings, group_size, scale_factor, baseline_ppl):
    """Run AWQ-style quantization experiment."""
    print(f"\n{'='*70}")
    print(f"Group Size: {group_size}, Scale Factor: {scale_factor}")
    print('='*70)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Calibrate
    activation_scales = calibrate_model(model, calibration_data)
    
    # Quantize
    model, orig_bytes, quant_bytes = quantize_model_awq(
        model, activation_scales, group_size, scale_factor
    )
    
    # Evaluate
    quant_ppl = evaluate_ppl(model, eval_encodings)
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
        'group_size': group_size,
        'scale_factor': scale_factor,
        'baseline_ppl': baseline_ppl,
        'quantized_ppl': quant_ppl,
        'ppl_delta': ppl_delta,
        'compression_fp16': fp16_compression,
        'compression_fp32': fp32_compression,
    }


def main():
    print("="*70)
    print("LIGHTWEIGHT CALIBRATED QUANTIZATION (AWQ-style)")
    print("="*70)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get calibration data (lightweight: only 64 samples)
    calibration_data = get_calibration_data(tokenizer, num_samples=64, seq_len=256)
    
    # Get evaluation data
    eval_encodings = get_wikitext2_subset(tokenizer, num_tokens=20000)
    
    # Get baseline
    print("\nComputing baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    baseline_ppl = evaluate_ppl(model, eval_encodings)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    del model
    import gc
    gc.collect()
    
    # Experiments: (group_size, scale_factor)
    experiments = [
        # Small groups (higher quality, lower compression)
        (8, 0.5),
        (8, 1.0),
        
        # Medium groups
        (32, 0.5),
        (32, 1.0),
        
        # Large groups (lower quality, higher compression)
        (64, 0.5),
        (64, 1.0),
        
        (128, 0.5),
        (128, 1.0),
        
        # Very large groups (target 10x)
        (256, 0.5),
        (256, 1.0),
    ]
    
    results = []
    for group_size, scale_factor in experiments:
        try:
            result = run_experiment(
                model_name, calibration_data, eval_encodings,
                group_size, scale_factor, baseline_ppl
            )
            results.append(result)
            
            if result['compression_fp32'] >= 10 and result['ppl_delta'] < 1.0:
                print("\n*** FOUND TARGET: 10x compression with <1% PPL! ***")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Group':<8} {'Scale':<8} {'PPL Δ':>8} {'FP16→':>8} {'FP32→':>8} {'Target':>12}")
    print("-"*62)
    
    for r in results:
        target = "✓ 10x+<1%" if r['compression_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "◐ 10x" if r['compression_fp32'] >= 10 else \
                 "◐ <1%" if r['ppl_delta'] < 1.0 else \
                 "✗"
        print(f"{r['group_size']:<8} {r['scale_factor']:<8} {r['ppl_delta']:>+7.2f}% {r['compression_fp16']:>7.2f}x {r['compression_fp32']:>7.2f}x {target:>12}")
    
    # Save
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "calibrated_quant_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/calibrated_quant_experiments.json")


if __name__ == "__main__":
    main()
