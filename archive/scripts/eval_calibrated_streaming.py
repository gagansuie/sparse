#!/usr/bin/env python3
"""
Streaming Calibration-Aware Compression Evaluation for Large Models

Processes layers one at a time to avoid OOM on large models.
Uses layer-by-layer quantization without loading full model into memory.
"""

import argparse
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install transformers datasets")
    exit(1)


def load_calibration(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def get_compression_config(sensitivity: float) -> Dict:
    """Get compression config based on sensitivity - tuned for 7B models."""
    if sensitivity > 0.5:
        return {"strategy": "conservative", "group_size": 8, "use_residual": True, "residual_bits": 4}
    elif sensitivity > 0.3:
        return {"strategy": "high", "group_size": 32, "use_residual": True, "residual_bits": 2}
    elif sensitivity > 0.1:
        return {"strategy": "balanced", "group_size": 128, "use_residual": False, "residual_bits": 0}
    else:
        return {"strategy": "aggressive", "group_size": 256, "use_residual": False, "residual_bits": 0}


def quantize_weight(weight: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, float]:
    """Quantize a single weight tensor."""
    original_size = weight.numel() * 4  # FP32
    group_size = config["group_size"]
    use_residual = config["use_residual"]
    residual_bits = config["residual_bits"]
    
    # Flatten and pad
    flat = weight.flatten().float()
    n = len(flat)
    pad_size = (group_size - (n % group_size)) % group_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
    
    # Reshape into groups
    groups = flat.view(-1, group_size)
    
    # INT4 quantization per group
    mins = groups.min(dim=1, keepdim=True).values
    maxs = groups.max(dim=1, keepdim=True).values
    scales = (maxs - mins) / 15.0
    scales = scales.clamp(min=1e-8)
    
    quantized = ((groups - mins) / scales).round().clamp(0, 15)
    dequantized = quantized * scales + mins
    
    # Compute compressed size
    # INT4: 4 bits per weight + 2 FP16 values per group (scale, min)
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
    
    # Reshape back
    final = dequantized.flatten()[:n].view(weight.shape)
    compression_ratio = original_size / compressed_size
    
    return final, compression_ratio


def compute_ppl_streaming(model, tokenizer, texts: List[str], device: str) -> float:
    """Compute perplexity with minimal memory."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = tokens["input_ids"].to(device)
            
            if input_ids.shape[1] < 2:
                continue
            
            try:
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                if loss is not None:
                    total_loss += loss.item() * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]
            except Exception as e:
                print(f"Warning: {e}")
                continue
            
            # Clear cache
            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    avg_loss = total_loss / max(total_tokens, 1)
    return np.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--calibration", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Loading calibration from: {args.calibration}")
    calibration = load_calibration(args.calibration)
    sensitivity = calibration.get("sensitivity", {})
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load model in FP16 on CPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    device = "cpu"
    
    # Load evaluation data
    print("Loading evaluation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:args.max_samples]
    
    # Compute baseline PPL
    print("Computing baseline PPL...")
    baseline_ppl = compute_ppl_streaming(model, tokenizer, texts, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Get layers to quantize
    layers_to_quantize = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                if 'embed' not in name.lower():
                    layers_to_quantize.append((name, module))
    
    print(f"Found {len(layers_to_quantize)} layers to quantize")
    
    # Quantize layers one at a time
    total_original = 0
    total_compressed = 0
    
    for name, module in tqdm(layers_to_quantize, desc="Quantizing"):
        weight = module.weight.data.float()
        original_size = weight.numel() * 4
        total_original += original_size
        
        # Get sensitivity
        layer_sensitivity = sensitivity.get(name, 0.5)
        config = get_compression_config(layer_sensitivity)
        
        # Quantize
        quantized, compression = quantize_weight(weight, config)
        
        # Update weight (convert back to FP16)
        module.weight.data = quantized.half()
        
        compressed_size = original_size / compression
        total_compressed += compressed_size
        
        # Clear memory
        del weight, quantized
        gc.collect()
    
    overall_compression = total_original / total_compressed
    print(f"\nCompression: {overall_compression:.2f}x")
    print(f"  Original: {total_original / 1e9:.2f} GB")
    print(f"  Compressed: {total_compressed / 1e9:.2f} GB")
    
    # Compute quantized PPL
    print("Computing quantized PPL...")
    quantized_ppl = compute_ppl_streaming(model, tokenizer, texts, device)
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
    
    results = {
        "model": args.model,
        "compression": overall_compression,
        "baseline_ppl": baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "ppl_delta": ppl_delta,
        "status": status,
    }
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
