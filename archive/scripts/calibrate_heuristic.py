#!/usr/bin/env python3
"""
Heuristic-Only Calibration for Large Models

No forward pass needed - just uses layer position/type heuristics.
Works for any model size without memory issues.
"""

import argparse
import json
import torch

try:
    from transformers import AutoModelForCausalLM, AutoConfig
except ImportError:
    print("Please install: pip install transformers")
    exit(1)


def compute_heuristic_sensitivity(config, model_type: str = "llama"):
    """
    Compute sensitivity using heuristics based on layer position and type.
    Works without loading the full model.
    """
    sensitivity = {}
    
    # Get number of layers from config
    if hasattr(config, 'num_hidden_layers'):
        num_layers = config.num_hidden_layers
    elif hasattr(config, 'n_layer'):
        num_layers = config.n_layer
    else:
        num_layers = 32  # Default
    
    # Generate layer names based on model type
    if model_type in ["llama", "qwen", "mistral"]:
        # Llama-style architecture
        layer_names = []
        for i in range(num_layers):
            layer_names.extend([
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
                f"model.layers.{i}.mlp.gate_proj",
                f"model.layers.{i}.mlp.up_proj",
                f"model.layers.{i}.mlp.down_proj",
            ])
        layer_names.append("lm_head")
    else:
        # GPT-2 style
        layer_names = []
        for i in range(num_layers):
            layer_names.extend([
                f"transformer.h.{i}.attn.c_attn",
                f"transformer.h.{i}.attn.c_proj",
                f"transformer.h.{i}.mlp.c_fc",
                f"transformer.h.{i}.mlp.c_proj",
            ])
        layer_names.append("lm_head")
    
    for name in layer_names:
        # Extract layer number
        layer_num = 0
        for part in name.split('.'):
            if part.isdigit():
                layer_num = int(part)
                break
        
        # Normalize layer position (0 = first, 1 = last)
        position = layer_num / max(num_layers - 1, 1)
        
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
        elif 'down' in name:
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
    parser.add_argument("--model-type", type=str, default="llama", 
                        choices=["llama", "qwen", "mistral", "gpt2"])
    args = parser.parse_args()
    
    print(f"Loading config for: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    
    print(f"Model has {getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 32))} layers")
    
    # Compute heuristic sensitivity (no model loading needed)
    sensitivity = compute_heuristic_sensitivity(config, args.model_type)
    
    # Save
    output = {
        "model": args.model,
        "num_samples": 0,  # No samples needed for heuristic
        "sensitivity": sensitivity,
        "activation_scales": {}  # Empty - not computed
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
