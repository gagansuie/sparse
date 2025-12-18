#!/usr/bin/env python3
"""
Simple Fast Calibration - Activation Scales Only

Much faster than full sensitivity analysis. Just computes activation scales
for AWQ-style importance weighting. Uses heuristics for layer sensitivity.
"""

import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
except ImportError:
    print("Please install: pip install transformers datasets")
    exit(1)


def get_calibration_data(tokenizer, num_samples=32, seq_len=512):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:num_samples * 2]
    
    calibration_data = []
    for text in texts[:num_samples]:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if tokens["input_ids"].shape[1] >= 64:
            calibration_data.append(tokens["input_ids"])
    return calibration_data[:num_samples]


def compute_activation_scales(model, calibration_data, device):
    """Compute activation scales for each layer."""
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
                scales = inp.float().abs().mean(dim=(0, 1))
                if name not in activation_scales:
                    activation_scales[name] = scales.cpu()
                else:
                    activation_scales[name] = activation_scales[name] + scales.cpu()
        return hook
    
    # Register hooks on all weight layers
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                if 'embed' not in name.lower() and 'wte' not in name and 'wpe' not in name:
                    hooks.append(module.register_forward_hook(make_hook(name)))
    
    model.eval()
    num_samples = 0
    with torch.no_grad():
        for tokens in tqdm(calibration_data, desc="Activations"):
            tokens = tokens.to(device)
            try:
                model(tokens)
                num_samples += 1
            except:
                continue
    
    for hook in hooks:
        hook.remove()
    
    for name in activation_scales:
        activation_scales[name] = activation_scales[name] / max(num_samples, 1)
    
    return activation_scales


def compute_heuristic_sensitivity(model):
    """
    Compute sensitivity using heuristics based on layer position and type.
    
    Heuristics:
    - lm_head is most sensitive
    - Early layers are more sensitive than late layers
    - Attention projections are more sensitive than MLP
    - Q/K projections are more sensitive than V/O
    """
    sensitivity = {}
    
    # Count total layers
    layer_names = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2:
                if 'embed' not in name.lower() and 'wte' not in name and 'wpe' not in name:
                    layer_names.append(name)
    
    num_layers = len(layer_names)
    
    for i, name in enumerate(layer_names):
        # Base sensitivity decreases with layer depth
        # Extract layer number if present
        layer_num = 0
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break
        
        # Normalize layer position (0 = first, 1 = last)
        max_layer = 1
        for n in layer_names:
            for part in n.split('.'):
                if part.isdigit():
                    max_layer = max(max_layer, int(part) + 1)
        
        position = layer_num / max_layer  # 0 to 1
        
        # Base sensitivity: early layers more sensitive
        base_sens = 1.0 - (position * 0.8)  # 1.0 to 0.2
        
        # Adjust by layer type
        if 'lm_head' in name or 'head' in name:
            sens = 1.0  # Most sensitive
        elif 'q_proj' in name or 'k_proj' in name or 'c_attn' in name:
            sens = base_sens * 0.8  # Q/K attention
        elif 'v_proj' in name or 'o_proj' in name or 'c_proj' in name:
            sens = base_sens * 0.6  # V/O attention
        elif 'gate' in name or 'up' in name or 'c_fc' in name:
            sens = base_sens * 0.4  # MLP up
        elif 'down' in name or 'mlp' in name:
            sens = base_sens * 0.3  # MLP down
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--offload", action="store_true", help="Use disk offloading for large models")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Configure loading based on memory constraints
    if args.offload:
        print("Using CPU offloading for large model...")
        # Load in FP16 with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cpu",  # Keep on CPU to avoid OOM
            low_cpu_mem_usage=True
        )
    elif args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model = model.to(args.device)
    
    # Get calibration data
    print("Loading calibration data...")
    calibration_data = get_calibration_data(tokenizer, args.num_samples)
    print(f"Loaded {len(calibration_data)} samples")
    
    # Compute activation scales (fast)
    activation_scales = compute_activation_scales(model, calibration_data, args.device)
    
    # Compute heuristic sensitivity (instant)
    sensitivity = compute_heuristic_sensitivity(model)
    
    # Save
    output = {
        "model": args.model,
        "num_samples": len(calibration_data),
        "sensitivity": sensitivity,
        "activation_scales": {
            name: scales.tolist() for name, scales in activation_scales.items()
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nCalibration saved to: {args.output}")
    print(f"Layers: {len(sensitivity)}")
    
    # Show sensitivity distribution
    sens_values = list(sensitivity.values())
    print(f"Sensitivity range: {min(sens_values):.3f} - {max(sens_values):.3f}")


if __name__ == "__main__":
    main()
