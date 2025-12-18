#!/usr/bin/env python3
"""
Fast Calibration for TenPak-X

This script performs fast calibration to enable higher compression ratios.
It computes layer-wise sensitivity scores that guide quantization.

Key insight: Not all layers are equally sensitive to quantization.
By measuring sensitivity, we can use aggressive compression on insensitive
layers and preserve precision on sensitive ones.

Usage:
    python calibrate_fast.py --model gpt2 --output calibration.json

Time: ~5-10 minutes for a 7B model (vs hours for GPTQ/AQLM)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install transformers datasets")
    sys.exit(1)


def get_calibration_data(tokenizer, num_samples: int = 128, seq_len: int = 512) -> List[torch.Tensor]:
    """Load calibration data from WikiText-2."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    texts = []
    for item in dataset:
        if len(item["text"]) > 100:
            texts.append(item["text"])
        if len(texts) >= num_samples * 2:
            break
    
    calibration_data = []
    for text in texts[:num_samples]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens["input_ids"].shape[1] >= 64:
            calibration_data.append(tokens["input_ids"])
    
    return calibration_data[:num_samples]


def compute_layer_sensitivity(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Compute sensitivity score for each layer using Fisher information approximation.
    
    Sensitivity = gradient magnitude w.r.t. loss, which indicates how much
    changing this layer affects the output.
    
    High sensitivity = need more precision.
    Low sensitivity = can compress aggressively.
    """
    print("Computing layer sensitivity...")
    
    sensitivity = {}
    
    # Get all weight layers (Linear, Conv1D, etc.)
    # GPT-2 uses Conv1D, Llama uses Linear
    linear_layers = {}
    for name, module in model.named_modules():
        # Check if module has 2D weights (linear-like)
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                # Skip embeddings
                if 'wte' not in name and 'wpe' not in name and 'embed' not in name.lower():
                    linear_layers[name] = module
    
    print(f"Found {len(linear_layers)} weight layers")
    for name in list(linear_layers.keys())[:5]:
        print(f"  Example: {name}")
    
    model.eval()
    
    # Use gradient-based sensitivity (Fisher information approximation)
    # This is much faster than perturbation-based
    for name, layer in tqdm(linear_layers.items(), desc="Sensitivity"):
        grad_sum = 0.0
        num_samples = 0
        
        for tokens in calibration_data[:8]:
            tokens = tokens.to(device)
            
            # Forward pass with gradient
            model.zero_grad()
            try:
                outputs = model(tokens, labels=tokens)
                loss = outputs.loss
                if loss is not None:
                    loss.backward()
                    
                    if layer.weight.grad is not None:
                        # Fisher information ≈ E[grad^2]
                        grad_sum += (layer.weight.grad ** 2).mean().item()
                        num_samples += 1
            except Exception as e:
                continue
        
        sensitivity[name] = grad_sum / max(num_samples, 1)
    
    # Normalize sensitivity scores to [0, 1]
    if sensitivity:
        max_sens = max(sensitivity.values())
        min_sens = min(sensitivity.values())
        range_sens = max_sens - min_sens + 1e-8
        
        for name in sensitivity:
            sensitivity[name] = (sensitivity[name] - min_sens) / range_sens
    
    return sensitivity


def compute_activation_scales(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Compute activation scales for AWQ-style importance weighting.
    
    For each linear layer, compute the average magnitude of input activations
    per input channel. This tells us which input features are most important.
    """
    print("Computing activation scales...")
    
    activation_scales = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            if inp is not None and inp.dim() >= 2:
                # Compute mean absolute activation per channel
                # Shape: [batch, seq, hidden] -> [hidden]
                scales = inp.float().abs().mean(dim=(0, 1))
                
                if name not in activation_scales:
                    activation_scales[name] = scales.cpu()
                else:
                    activation_scales[name] = activation_scales[name] + scales.cpu()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run forward passes
    model.eval()
    num_samples = 0
    with torch.no_grad():
        for tokens in tqdm(calibration_data[:32], desc="Activations"):
            tokens = tokens.to(device)
            try:
                model(tokens)
                num_samples += 1
            except Exception as e:
                print(f"Warning: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average
    for name in activation_scales:
        activation_scales[name] = activation_scales[name] / max(num_samples, 1)
    
    return activation_scales


def main():
    parser = argparse.ArgumentParser(description="Fast calibration for TenPak-X")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--output", type=str, default="calibration.json", help="Output file")
    parser.add_argument("--num-samples", type=int, default=64, help="Calibration samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if args.device == "cpu":
        model = model.to(args.device)
    
    start_time = time.time()
    
    # Get calibration data
    print("Loading calibration data...")
    calibration_data = get_calibration_data(tokenizer, args.num_samples)
    print(f"Loaded {len(calibration_data)} calibration samples")
    
    # Compute sensitivity
    sensitivity = compute_layer_sensitivity(model, calibration_data, args.device)
    
    # Compute activation scales
    activation_scales = compute_activation_scales(model, calibration_data, args.device)
    
    elapsed = time.time() - start_time
    
    # Save results
    output = {
        "model": args.model,
        "num_samples": len(calibration_data),
        "elapsed_seconds": elapsed,
        "sensitivity": sensitivity,
        "activation_scales": {
            name: scales.tolist() 
            for name, scales in activation_scales.items()
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {len(calibration_data)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {args.output}")
    print(f"{'='*60}")
    
    # Print sensitivity summary
    print("\nLayer Sensitivity (top 10 most sensitive):")
    sorted_sens = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    for name, score in sorted_sens[:10]:
        print(f"  {name}: {score:.4f}")
    
    print("\nLayer Sensitivity (top 10 least sensitive):")
    for name, score in sorted_sens[-10:]:
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
