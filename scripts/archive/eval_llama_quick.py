#!/usr/bin/env python3
"""
Quick Llama Evaluation - 100 steps only for fast validation.
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
    
    # Only tokenize first portion
    encodings = tokenizer(text, return_tensors="pt", max_length=num_tokens, truncation=True)
    print(f"Using {encodings.input_ids.size(1)} tokens for quick eval")
    return encodings


def evaluate_perplexity_quick(model, encodings, device, max_length=512, stride=256, max_steps=100):
    """Quick perplexity evaluation with limited steps."""
    print(f"Evaluating perplexity (max {max_steps} steps)...")
    
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
        
        if step % 20 == 0:
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  Step {step}/{max_steps}, PPL: {current_ppl:.2f}")
        
        if end_loc >= seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def quantize_linear_g8(weight, group_size=8):
    """Quantize weights using int4_g8_v1."""
    weight_flat = weight.flatten().float()
    
    pad_len = (group_size - len(weight_flat) % group_size) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len)])
    
    num_groups = len(weight_flat) // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    min_vals = weight_groups.min(dim=1).values
    max_vals = weight_groups.max(dim=1).values
    
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    weight_q_flat = weight_q.flatten()
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return packed, scales, offsets, weight.shape, pad_len


def dequantize_linear_g8(packed, scales, offsets, shape, pad_len, group_size=8):
    """Dequantize int4_g8_v1 weights."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    weight_deq = weight_groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    
    return weight_flat.view(shape)


class QuantizedLinear(nn.Module):
    """Linear layer with int4_g8 quantized weights."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('packed', None)
        self.register_buffer('scales', None)
        self.register_buffer('offsets', None)
        self.register_buffer('bias_data', None)
        self.pad_len = 0
    
    @classmethod
    def from_linear(cls, linear):
        layer = cls(linear.in_features, linear.out_features)
        packed, scales, offsets, shape, pad_len = quantize_linear_g8(linear.weight.data)
        layer.packed = packed
        layer.scales = scales
        layer.offsets = offsets
        layer.pad_len = pad_len
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        return layer
    
    def forward(self, x):
        weight = dequantize_linear_g8(
            self.packed, self.scales, self.offsets,
            (self.out_features, self.in_features), self.pad_len
        ).to(x.dtype).to(x.device)
        return torch.nn.functional.linear(x, weight, self.bias_data)


def quantize_model_mlp(model):
    """Quantize MLP layers only."""
    print("Quantizing MLP layers with int4_g8...")
    
    quantized_params = 0
    total_params = sum(p.numel() for p in model.parameters())
    
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                quantized = QuantizedLinear.from_linear(original)
                setattr(mlp, proj_name, quantized)
                quantized_params += original.weight.numel()
        
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)}")
    
    print(f"Quantized {quantized_params:,} / {total_params:,} params ({100*quantized_params/total_params:.1f}%)")
    return model


def measure_size(model):
    """Measure model size in bytes."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def main():
    print("="*60)
    print("TENPAK QUICK EVALUATION - TinyLlama 1.1B")
    print("="*60)
    
    device = "cpu"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading dataset...")
    encodings = get_wikitext2_subset(tokenizer, num_tokens=30000)
    
    # === BASELINE ===
    print(f"\n{'='*60}")
    print("BASELINE (FP16)")
    print("="*60)
    
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    baseline_size = measure_size(model)
    print(f"Model size: {baseline_size / 1e9:.2f} GB")
    
    start = time.time()
    baseline_ppl = evaluate_perplexity_quick(model, encodings, device, max_steps=50)
    baseline_time = time.time() - start
    print(f"\nBaseline PPL: {baseline_ppl:.2f} ({baseline_time:.1f}s)")
    
    # === QUANTIZED ===
    print(f"\n{'='*60}")
    print("TENPAK int4_g8_v1")
    print("="*60)
    
    model = quantize_model_mlp(model)
    model.eval()
    
    quantized_size = measure_size(model)
    print(f"Quantized size: {quantized_size / 1e9:.2f} GB")
    
    start = time.time()
    quantized_ppl = evaluate_perplexity_quick(model, encodings, device, max_steps=50)
    quantized_time = time.time() - start
    print(f"\nQuantized PPL: {quantized_ppl:.2f} ({quantized_time:.1f}s)")
    
    # === RESULTS ===
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    
    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    compression = baseline_size / quantized_size
    
    print(f"\n  Baseline PPL:   {baseline_ppl:.2f}")
    print(f"  Quantized PPL:  {quantized_ppl:.2f}")
    print(f"  PPL Delta:      {ppl_delta:+.2f}%")
    print(f"")
    print(f"  Baseline Size:  {baseline_size / 1e9:.2f} GB")
    print(f"  Quantized Size: {quantized_size / 1e9:.2f} GB")
    print(f"  Compression:    {compression:.2f}x")
    
    # Save results
    results = {
        "model": model_name,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_delta_percent": ppl_delta,
        "baseline_size_gb": baseline_size / 1e9,
        "quantized_size_gb": quantized_size / 1e9,
        "compression_ratio": compression,
    }
    
    results_path = TENPAK_ROOT / "results"
    results_path.mkdir(exist_ok=True)
    with open(results_path / "tinyllama_quick_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/tinyllama_quick_eval.json")
    
    if ppl_delta < 1.0:
        print(f"\n✓ SUCCESS: PPL delta {ppl_delta:+.2f}% is under 1% target!")
    else:
        print(f"\n✗ PPL delta {ppl_delta:+.2f}% exceeds 1% target")


if __name__ == "__main__":
    main()
