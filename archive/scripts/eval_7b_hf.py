#!/usr/bin/env python3
"""
7B Model Calibrated Compression Evaluation for HuggingFace Spaces

Run this on HF Spaces with GPU to evaluate 7B models.
Usage: python eval_7b_hf.py --model Qwen/Qwen2-7B
"""

import argparse
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_compression_config(sensitivity: float) -> Dict:
    """Get compression config based on sensitivity - tuned for 7B models."""
    if sensitivity > 0.5:
        # Very sensitive (lm_head) - minimal compression
        return {"strategy": "conservative", "group_size": 8, "use_residual": True, "residual_bits": 4}
    elif sensitivity > 0.3:
        # High sensitivity - INT4 g=32 + INT2 residual
        return {"strategy": "high", "group_size": 32, "use_residual": True, "residual_bits": 2}
    elif sensitivity > 0.1:
        # Medium sensitivity - INT4 g=128 + no residual
        return {"strategy": "balanced", "group_size": 128, "use_residual": False, "residual_bits": 0}
    else:
        # Low sensitivity - INT4 g=256 + no residual
        return {"strategy": "aggressive", "group_size": 256, "use_residual": False, "residual_bits": 0}


def compute_heuristic_sensitivity(model) -> Dict[str, float]:
    """Compute sensitivity using heuristics based on layer position and type."""
    sensitivity = {}
    
    # Get all weight layers
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                if 'embed' not in name.lower():
                    layer_names.append(name)
    
    # Count total transformer layers
    max_layer = 1
    for name in layer_names:
        for part in name.split('.'):
            if part.isdigit():
                max_layer = max(max_layer, int(part) + 1)
    
    for name in layer_names:
        # Extract layer number
        layer_num = 0
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break
        
        # Normalize layer position (0 = first, 1 = last)
        position = layer_num / max(max_layer - 1, 1)
        
        # Base sensitivity: early layers more sensitive
        base_sens = 1.0 - (position * 0.8)
        
        # Adjust by layer type
        if 'lm_head' in name or 'head' in name:
            sens = 1.0
        elif 'q_proj' in name or 'k_proj' in name or 'c_attn' in name:
            sens = base_sens * 0.8
        elif 'v_proj' in name or 'o_proj' in name or 'c_proj' in name:
            sens = base_sens * 0.6
        elif 'gate' in name or 'up' in name or 'c_fc' in name:
            sens = base_sens * 0.4
        elif 'down' in name:
            sens = base_sens * 0.3
        else:
            sens = base_sens * 0.5
        
        sensitivity[name] = sens
    
    # Normalize to [0, 1]
    if sensitivity:
        max_s = max(sensitivity.values())
        min_s = min(sensitivity.values())
        range_s = max_s - min_s + 1e-8
        for name in sensitivity:
            sensitivity[name] = (sensitivity[name] - min_s) / range_s
    
    return sensitivity


def apply_awq_scaling(weight: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Apply simple AWQ-style scaling based on weight magnitude."""
    # Use weight magnitude as proxy for importance
    scales = weight.abs().mean(dim=0)
    scales = scales / (scales.mean() + 1e-8)
    scales = torch.pow(scales.clamp(0.01, 100.0), alpha)
    scales = scales.clamp(0.5, 2.0)
    return weight * scales.unsqueeze(0), scales


def quantize_weight(weight: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, float]:
    """Quantize a single weight tensor with AWQ scaling."""
    original_size = weight.numel() * 4  # FP32
    group_size = config["group_size"]
    use_residual = config["use_residual"]
    residual_bits = config["residual_bits"]
    
    # Apply AWQ scaling
    scaled_weight, scales = apply_awq_scaling(weight.float())
    
    # Flatten and pad
    flat = scaled_weight.flatten()
    n = len(flat)
    pad_size = (group_size - (n % group_size)) % group_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
    
    # Reshape into groups
    groups = flat.view(-1, group_size)
    
    # INT4 quantization per group
    mins = groups.min(dim=1, keepdim=True).values
    maxs = groups.max(dim=1, keepdim=True).values
    q_scales = (maxs - mins) / 15.0
    q_scales = q_scales.clamp(min=1e-8)
    
    quantized = ((groups - mins) / q_scales).round().clamp(0, 15)
    dequantized = quantized * q_scales + mins
    
    # Compute compressed size
    compressed_size = (n * 4 / 8) + (n / group_size) * 4
    
    if use_residual and residual_bits > 0:
        residual = groups - dequantized
        r_mins = residual.min(dim=1, keepdim=True).values
        r_maxs = residual.max(dim=1, keepdim=True).values
        r_scales = (r_maxs - r_mins) / ((1 << residual_bits) - 1)
        r_scales = r_scales.clamp(min=1e-8)
        
        r_quantized = ((residual - r_mins) / r_scales).round().clamp(0, (1 << residual_bits) - 1)
        r_dequantized = r_quantized * r_scales + r_mins
        
        dequantized = dequantized + r_dequantized
        compressed_size += (n * residual_bits / 8) + (n / group_size) * 4
    
    # Reshape back and undo AWQ scaling
    final = dequantized.flatten()[:n].view(weight.shape)
    final = final / scales.unsqueeze(0)
    
    compression_ratio = original_size / compressed_size
    return final, compression_ratio


def compute_ppl(model, tokenizer, texts: List[str], device: str) -> float:
    """Compute perplexity on texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL"):
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
    return np.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Device: {args.device}")
    print(f"Loading model: {args.model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Compute heuristic sensitivity
    print("Computing layer sensitivity...")
    sensitivity = compute_heuristic_sensitivity(model)
    print(f"Found {len(sensitivity)} layers")
    
    # Load evaluation data
    print("Loading evaluation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:args.max_samples]
    
    # Compute baseline PPL
    print("Computing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts, args.device)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Get layers to quantize
    layers_to_quantize = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                if 'embed' not in name.lower():
                    layers_to_quantize.append((name, module))
    
    print(f"Quantizing {len(layers_to_quantize)} layers...")
    
    # Quantize layers
    total_original = 0
    total_compressed = 0
    
    for name, module in tqdm(layers_to_quantize, desc="Quantizing"):
        weight = module.weight.data.float()
        original_size = weight.numel() * 4
        total_original += original_size
        
        # Get sensitivity and config
        layer_sensitivity = sensitivity.get(name, 0.5)
        config = get_compression_config(layer_sensitivity)
        
        # Quantize
        quantized, compression = quantize_weight(weight, config)
        
        # Update weight
        module.weight.data = quantized.half().to(module.weight.device)
        
        compressed_size = original_size / compression
        total_compressed += compressed_size
        
        # Clear memory
        del weight, quantized
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    overall_compression = total_original / total_compressed
    print(f"\nCompression: {overall_compression:.2f}x")
    print(f"  Original: {total_original / 1e9:.2f} GB")
    print(f"  Compressed: {total_compressed / 1e9:.2f} GB")
    
    # Compute quantized PPL
    print("Computing quantized PPL...")
    quantized_ppl = compute_ppl(model, tokenizer, texts, args.device)
    print(f"Quantized PPL: {quantized_ppl:.4f}")
    
    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    status = "PASS" if abs(ppl_delta) < 1.0 else "FAIL"
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Compression: {overall_compression:.2f}x")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Quantized PPL: {quantized_ppl:.4f}")
    print(f"PPL Delta: {ppl_delta:+.2f}%")
    print(f"Status: {'✅' if status == 'PASS' else '❌'} {status}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "model": args.model,
        "compression": overall_compression,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_delta": ppl_delta,
        "status": status,
    }
    
    with open("results_7b.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results_7b.json")


if __name__ == "__main__":
    main()
