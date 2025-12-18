#!/usr/bin/env python3
"""
Calibration-Aware Compression Evaluation

Uses calibration data (sensitivity + activation scales) to achieve higher
compression with adaptive bit allocation per layer.

Strategy:
- High sensitivity layers: Use conservative compression (INT4 g=8)
- Low sensitivity layers: Use aggressive compression (INT4 g=64 or skip residual)
- Use activation scales for AWQ-style importance weighting

Target: 8-10x compression with <1% PPL delta
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install transformers datasets")
    sys.exit(1)


def load_calibration(path: str) -> Dict:
    """Load calibration data from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_compression_config(sensitivity: float) -> Dict:
    """
    Get compression configuration based on layer sensitivity.
    
    High sensitivity (>0.2): Conservative - preserve quality
    Medium sensitivity (0.02-0.2): Balanced
    Low sensitivity (<0.02): Aggressive - maximize compression
    
    Tuned for <1% PPL delta target.
    """
    if sensitivity > 0.5:
        # Very sensitive (lm_head) - minimal compression
        return {
            "strategy": "conservative",
            "group_size": 8,
            "use_residual": True,
            "residual_bits": 4,
        }
    elif sensitivity > 0.3:
        # High sensitivity - INT4 g=32 + INT2 residual
        return {
            "strategy": "high",
            "group_size": 32,
            "use_residual": True,
            "residual_bits": 2,
        }
    elif sensitivity > 0.1:
        # Medium sensitivity - INT4 g=128 + no residual
        return {
            "strategy": "balanced",
            "group_size": 128,
            "use_residual": False,
            "residual_bits": 0,
        }
    else:
        # Low sensitivity - INT4 g=256 + no residual
        return {
            "strategy": "aggressive",
            "group_size": 256,
            "use_residual": False,
            "residual_bits": 0,
        }


def apply_awq_scaling(weight: torch.Tensor, activation_scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply AWQ-style scaling to weights.
    
    Key insight from AWQ paper: scale = activation_scale^alpha where alpha=0.5
    This balances quantization error between weights and activations.
    
    Scale important input channels up before quantization,
    then scale them back down after dequantization.
    """
    # Normalize scales and apply AWQ alpha=0.5
    scales = activation_scales.float()
    scales = scales / (scales.mean() + 1e-8)
    scales = torch.pow(scales.clamp(0.01, 100.0), 0.5)  # AWQ alpha=0.5
    scales = scales.clamp(0.5, 2.0)
    
    # Scale weights: W_scaled = W * diag(scales)
    # For Conv1D: weight shape is [in, out], scale along dim 0
    # For Linear: weight shape is [out, in], scale along dim 1
    if weight.shape[0] == len(scales):
        scaled_weight = weight * scales.view(-1, 1)
    elif weight.shape[1] == len(scales):
        scaled_weight = weight * scales.view(1, -1)
    else:
        # Dimension mismatch, skip scaling
        return weight, torch.ones(1)
    
    return scaled_weight, scales


def quantize_layer_calibrated(
    weight: torch.Tensor,
    sensitivity: float,
    activation_scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Quantize a single layer using calibration-aware strategy.
    
    Returns: (quantized_weight, compression_ratio)
    """
    config = get_compression_config(sensitivity)
    original_size = weight.numel() * 4  # FP32 = 4 bytes
    
    # Apply AWQ scaling if available
    if activation_scales is not None and len(activation_scales) > 0:
        weight, scales = apply_awq_scaling(weight, activation_scales)
    else:
        scales = torch.ones(1)
    
    # Flatten for quantization
    flat = weight.flatten().float()
    n = len(flat)
    group_size = config["group_size"]
    
    # Pad to group size
    pad_len = (group_size - (n % group_size)) % group_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len)])
    
    # Group-wise quantization
    groups = flat.view(-1, group_size)
    
    if config["use_residual"]:
        # INT4 + residual quantization
        # First pass: INT4
        mins = groups.min(dim=1, keepdim=True).values
        maxs = groups.max(dim=1, keepdim=True).values
        scales_q = (maxs - mins) / 15.0
        scales_q = scales_q.clamp(min=1e-8)
        
        quantized = ((groups - mins) / scales_q).round().clamp(0, 15)
        dequantized = quantized * scales_q + mins
        
        # Compute residual
        residual = groups - dequantized
        
        if config["residual_bits"] == 4:
            # INT4 residual
            res_mins = residual.min(dim=1, keepdim=True).values
            res_maxs = residual.max(dim=1, keepdim=True).values
            res_scales = (res_maxs - res_mins) / 15.0
            res_scales = res_scales.clamp(min=1e-8)
            res_quantized = ((residual - res_mins) / res_scales).round().clamp(0, 15)
            res_dequantized = res_quantized * res_scales + res_mins
            
            final = dequantized + res_dequantized
            # Compression: 4 bits + 4 bits + scales = ~1 byte per weight
            compressed_size = n * 1.0
        else:
            # INT2 residual
            res_mins = residual.min(dim=1, keepdim=True).values
            res_maxs = residual.max(dim=1, keepdim=True).values
            res_scales = (res_maxs - res_mins) / 3.0
            res_scales = res_scales.clamp(min=1e-8)
            res_quantized = ((residual - res_mins) / res_scales).round().clamp(0, 3)
            res_dequantized = res_quantized * res_scales + res_mins
            
            final = dequantized + res_dequantized
            # Compression: 4 bits + 2 bits + scales = ~0.75 bytes per weight
            compressed_size = n * 0.75
    else:
        # INT4 only (aggressive)
        mins = groups.min(dim=1, keepdim=True).values
        maxs = groups.max(dim=1, keepdim=True).values
        scales_q = (maxs - mins) / 15.0
        scales_q = scales_q.clamp(min=1e-8)
        
        quantized = ((groups - mins) / scales_q).round().clamp(0, 15)
        final = quantized * scales_q + mins
        
        # Compression: 4 bits + scales = ~0.5 bytes per weight
        compressed_size = n * 0.5
    
    # Reshape back
    final = final.flatten()[:n].view(weight.shape)
    
    # Undo AWQ scaling
    if activation_scales is not None and len(activation_scales) > 0:
        if weight.shape[0] == len(scales):
            final = final / scales.view(-1, 1)
        elif weight.shape[1] == len(scales):
            final = final / scales.view(1, -1)
    
    compression_ratio = original_size / compressed_size
    return final, compression_ratio


def evaluate_calibrated(
    model_name: str,
    calibration_path: str,
    layers: str = "all",
    max_samples: int = 30,
    device: str = "cuda",
    offload: bool = False
) -> Dict:
    """
    Evaluate calibration-aware compression.
    """
    print(f"Loading calibration from: {calibration_path}")
    calibration = load_calibration(calibration_path)
    sensitivity = calibration.get("sensitivity", {})
    activation_scales = calibration.get("activation_scales", {})
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if offload:
        print("Using disk offloading for large model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload_tmp",
            offload_state_dict=True,
            low_cpu_mem_usage=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    model.eval()
    
    # Load evaluation data
    print("Loading evaluation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:max_samples]
    
    # Compute baseline PPL
    print("Computing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Get layers to quantize
    layers_to_quantize = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                # Skip embeddings
                if 'wte' not in name and 'wpe' not in name and 'embed' not in name.lower():
                    if layers == "all":
                        layers_to_quantize.append((name, module))
                    elif layers == "mlp" and "mlp" in name:
                        layers_to_quantize.append((name, module))
                    elif layers == "attn" and "attn" in name:
                        layers_to_quantize.append((name, module))
    
    print(f"Found {len(layers_to_quantize)} layers to quantize")
    
    # Quantize layers
    total_original = 0
    total_compressed = 0
    
    for name, module in tqdm(layers_to_quantize, desc="Quantizing"):
        weight = module.weight.data
        original_size = weight.numel() * 4
        total_original += original_size
        
        # Get sensitivity for this layer
        layer_sensitivity = sensitivity.get(name, 0.5)
        
        # Get activation scales for this layer
        layer_scales = activation_scales.get(name, [])
        if layer_scales:
            layer_scales = torch.tensor(layer_scales, dtype=torch.float32, device=device)
        else:
            layer_scales = None
        
        # Quantize
        quantized, compression = quantize_layer_calibrated(
            weight, layer_sensitivity, layer_scales
        )
        
        # Update weight
        module.weight.data = quantized
        
        compressed_size = original_size / compression
        total_compressed += compressed_size
    
    overall_compression = total_original / total_compressed
    print(f"\nCompression: {overall_compression:.2f}x")
    print(f"  Original: {total_original / 1e6:.2f} MB")
    print(f"  Compressed: {total_compressed / 1e6:.2f} MB")
    
    # Compute quantized PPL
    print("Computing quantized PPL...")
    quantized_ppl = compute_ppl(model, tokenizer, texts, device)
    print(f"Quantized PPL: {quantized_ppl:.4f}")
    
    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    status = "PASS" if abs(ppl_delta) < 1.0 else "FAIL"
    
    results = {
        "model": model_name,
        "calibration": calibration_path,
        "layers": layers,
        "compression": overall_compression,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_delta": ppl_delta,
        "status": status,
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Compression: {overall_compression:.2f}x")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Quantized PPL: {quantized_ppl:.4f}")
    print(f"PPL Delta: {ppl_delta:+.2f}%")
    print(f"Status: {'✅' if status == 'PASS' else '❌'} {status}")
    print(f"{'='*60}")
    
    return results


def compute_ppl(model, tokenizer, texts: List[str], device: str) -> float:
    """Compute perplexity on texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            
            if input_ids.shape[1] < 2:
                continue
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            if loss is not None:
                total_loss += loss.item() * input_ids.shape[1]
                total_tokens += input_ids.shape[1]
    
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = np.exp(avg_loss)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Calibration-aware compression evaluation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--calibration", type=str, required=True, help="Calibration JSON file")
    parser.add_argument("--layers", type=str, default="all", choices=["all", "mlp", "attn"])
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--offload", action="store_true", help="Use disk offloading for large models")
    args = parser.parse_args()
    
    results = evaluate_calibrated(
        args.model,
        args.calibration,
        args.layers,
        args.max_samples,
        args.device,
        args.offload
    )
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
