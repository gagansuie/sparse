#!/usr/bin/env python3
"""
Llama 2 7B Evaluation Script for Tenpak

Evaluates perplexity on WikiText-2 with:
1. Baseline FP32/FP16
2. Tenpak int4_g8_v1 quantization

Designed to run on 16GB RAM with CPU offloading.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np

# Tenpak project root
TENPAK_ROOT = Path(__file__).parent.parent


def get_wikitext2(tokenizer, max_length=2048):
    """Load and tokenize WikiText-2 test set."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    return encodings


def evaluate_perplexity(model, encodings, device, max_length=2048, stride=512):
    """
    Evaluate perplexity using sliding window.
    
    Args:
        model: The model to evaluate
        encodings: Tokenized text
        device: Device to run on
        max_length: Context window size
        stride: Sliding window stride
    
    Returns:
        Perplexity score
    """
    print(f"Evaluating perplexity (max_length={max_length}, stride={stride})...")
    
    seq_len = encodings.input_ids.size(1)
    print(f"Total tokens: {seq_len}")
    
    nlls = []
    prev_end_loc = 0
    
    total_steps = (seq_len - 1) // stride
    
    for begin_loc in range(0, seq_len - 1, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # Tokens to predict
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask already-seen tokens
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood.cpu())
        
        prev_end_loc = end_loc
        
        step = begin_loc // stride
        if step % 10 == 0:
            current_ppl = torch.exp(torch.stack(nlls).mean()).item()
            print(f"  Step {step}/{total_steps}, current PPL: {current_ppl:.2f}")
        
        if end_loc >= seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def quantize_linear_g8(weight, group_size=8):
    """
    Quantize a weight tensor using int4_g8_v1 codec.
    
    Args:
        weight: [out_features, in_features] tensor
        group_size: Number of weights per group (default 8)
    
    Returns:
        packed: Packed int4 weights
        scales: Per-group scales
        offsets: Per-group offsets
    """
    out_features, in_features = weight.shape
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
    
    # Compute scales and offsets
    scales = (max_vals - min_vals) / 15.0
    scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
    offsets = min_vals
    
    # Quantize to 0-15
    weight_q = ((weight_groups - offsets.unsqueeze(1)) / scales.unsqueeze(1)).round().clamp(0, 15).to(torch.uint8)
    
    # Pack two int4 values per byte
    weight_q_flat = weight_q.flatten()
    packed = torch.zeros(len(weight_q_flat) // 2, dtype=torch.uint8)
    packed = (weight_q_flat[0::2] & 0x0F) | ((weight_q_flat[1::2] & 0x0F) << 4)
    
    return packed, scales, offsets, (out_features, in_features), pad_len


def dequantize_linear_g8(packed, scales, offsets, shape, pad_len, group_size=8):
    """
    Dequantize int4_g8_v1 weights back to float.
    """
    out_features, in_features = shape
    
    # Unpack
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight_q = torch.zeros(len(packed) * 2, dtype=torch.uint8)
    weight_q[0::2] = low
    weight_q[1::2] = high
    
    # Reshape to groups
    num_groups = len(weight_q) // group_size
    weight_groups = weight_q.view(num_groups, group_size).float()
    
    # Dequantize
    weight_deq = weight_groups * scales.unsqueeze(1) + offsets.unsqueeze(1)
    
    # Flatten and remove padding
    weight_flat = weight_deq.flatten()
    if pad_len > 0:
        weight_flat = weight_flat[:-pad_len]
    
    return weight_flat.view(out_features, in_features)


class QuantizedLinear(nn.Module):
    """Linear layer with int4_g8 quantized weights."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantized storage
        self.register_buffer('packed', None)
        self.register_buffer('scales', None)
        self.register_buffer('offsets', None)
        self.register_buffer('bias_data', None)
        self.pad_len = 0
    
    @classmethod
    def from_linear(cls, linear):
        """Convert nn.Linear to QuantizedLinear."""
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None)
        
        # Quantize weights
        packed, scales, offsets, shape, pad_len = quantize_linear_g8(linear.weight.data)
        layer.packed = packed
        layer.scales = scales
        layer.offsets = offsets
        layer.pad_len = pad_len
        
        if linear.bias is not None:
            layer.bias_data = linear.bias.data.clone()
        
        return layer
    
    def forward(self, x):
        # Dequantize weights on-the-fly
        weight = dequantize_linear_g8(
            self.packed, self.scales, self.offsets,
            (self.out_features, self.in_features), self.pad_len
        ).to(x.dtype).to(x.device)
        
        output = torch.nn.functional.linear(x, weight, self.bias_data)
        return output


def quantize_model_mlp(model):
    """
    Quantize MLP layers in a Llama model using int4_g8.
    Only quantizes MLP (gate_proj, up_proj, down_proj), leaves attention in FP16.
    """
    print("Quantizing MLP layers with int4_g8...")
    
    total_params = 0
    quantized_params = 0
    
    for name, module in model.named_modules():
        total_params += sum(p.numel() for p in module.parameters(recurse=False))
    
    # Find and replace MLP linear layers
    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        
        # Quantize gate_proj, up_proj, down_proj
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(mlp, proj_name):
                original = getattr(mlp, proj_name)
                quantized = QuantizedLinear.from_linear(original)
                setattr(mlp, proj_name, quantized)
                quantized_params += original.weight.numel()
                
                if layer_idx == 0:
                    print(f"  Quantized {proj_name}: {original.weight.shape}")
        
        if layer_idx % 8 == 0:
            print(f"  Layer {layer_idx}/{len(model.model.layers)} done")
    
    print(f"Quantized {quantized_params:,} / {total_params:,} parameters ({100*quantized_params/total_params:.1f}%)")
    return model


def measure_model_size(model):
    """Estimate model size in bytes."""
    total_bytes = 0
    
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()
    
    for name, buffer in model.named_buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    return total_bytes


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tenpak on Llama 2")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name or path (default: TinyLlama-1.1B, no license required)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline evaluation")
    parser.add_argument("--quantized-only", action="store_true",
                        help="Only run quantized evaluation")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max sequence length for evaluation")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride for sliding window")
    parser.add_argument("--use-4bit-baseline", action="store_true",
                        help="Load baseline in 4-bit (for low memory)")
    args = parser.parse_args()
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("WARNING: No HF_TOKEN found. You may need to set it for gated models.")
        print("  export HF_TOKEN=your_token_here")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    
    # Load dataset
    encodings = get_wikitext2(tokenizer, args.max_length)
    
    results = {}
    
    # Baseline evaluation
    if not args.quantized_only:
        print(f"\n{'='*60}")
        print("BASELINE EVALUATION")
        print('='*60)
        
        print(f"\nLoading model {args.model}...")
        
        if args.use_4bit_baseline:
            # Load in 4-bit for memory-constrained systems
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token,
            )
            print("Loaded in 4-bit mode (bitsandbytes)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=hf_token,
            )
        
        model.eval()
        
        baseline_size = measure_model_size(model)
        print(f"Model size: {baseline_size / 1e9:.2f} GB")
        
        start_time = time.time()
        baseline_ppl = evaluate_perplexity(model, encodings, device, args.max_length, args.stride)
        baseline_time = time.time() - start_time
        
        print(f"\nBaseline PPL: {baseline_ppl:.2f}")
        print(f"Evaluation time: {baseline_time:.1f}s")
        
        results['baseline'] = {
            'ppl': baseline_ppl,
            'size_bytes': baseline_size,
            'time_seconds': baseline_time,
        }
        
        # Clean up for quantized eval
        if not args.baseline_only:
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
    
    # Quantized evaluation
    if not args.baseline_only:
        print(f"\n{'='*60}")
        print("TENPAK int4_g8_v1 EVALUATION")
        print('='*60)
        
        print(f"\nLoading model {args.model} for quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        
        # Quantize MLP layers
        model = quantize_model_mlp(model)
        model.eval()
        
        quantized_size = measure_model_size(model)
        print(f"Quantized model size: {quantized_size / 1e9:.2f} GB")
        
        start_time = time.time()
        quantized_ppl = evaluate_perplexity(model, encodings, device, args.max_length, args.stride)
        quantized_time = time.time() - start_time
        
        print(f"\nQuantized PPL: {quantized_ppl:.2f}")
        print(f"Evaluation time: {quantized_time:.1f}s")
        
        results['quantized'] = {
            'ppl': quantized_ppl,
            'size_bytes': quantized_size,
            'time_seconds': quantized_time,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)
    
    if 'baseline' in results and 'quantized' in results:
        baseline_ppl = results['baseline']['ppl']
        quantized_ppl = results['quantized']['ppl']
        baseline_size = results['baseline']['size_bytes']
        quantized_size = results['quantized']['size_bytes']
        
        ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
        compression = baseline_size / quantized_size
        
        print(f"\nBaseline PPL:    {baseline_ppl:.2f}")
        print(f"Quantized PPL:   {quantized_ppl:.2f}")
        print(f"PPL Delta:       {ppl_delta:+.2f}%")
        print(f"")
        print(f"Baseline Size:   {baseline_size / 1e9:.2f} GB")
        print(f"Quantized Size:  {quantized_size / 1e9:.2f} GB")
        print(f"Compression:     {compression:.1f}x")
        
        results['summary'] = {
            'ppl_delta_percent': ppl_delta,
            'compression_ratio': compression,
        }
        
        # Save results
        results_path = TENPAK_ROOT / "results" / "llama2_7b_eval.json"
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    elif 'baseline' in results:
        print(f"\nBaseline PPL: {results['baseline']['ppl']:.2f}")
    
    elif 'quantized' in results:
        print(f"\nQuantized PPL: {results['quantized']['ppl']:.2f}")


if __name__ == "__main__":
    main()
