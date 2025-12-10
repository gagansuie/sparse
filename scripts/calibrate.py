#!/usr/bin/env python3
"""
Calibration Script for Tenpak 10x Compression

This script generates importance weights by running a small calibration
dataset through the model and measuring activation magnitudes.

Usage:
    python scripts/calibrate.py --model gpt2 --output calibrated_bundle.json
    
    # Then compress with calibration:
    ./target/release/tenpak compress \
        --input calibrated_bundle.json \
        --output model.tenpak \
        --codec mixed_cal_v1
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on text samples."""
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


class ActivationCollector:
    """Collects activation statistics during forward pass."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            
            if isinstance(x, torch.Tensor):
                # Compute importance as mean absolute activation per input feature
                # Shape: [batch, seq, features] -> [features]
                importance = x.abs().mean(dim=(0, 1)).detach().cpu()
                
                if name in self.activations:
                    # Running average
                    self.activations[name] = (self.activations[name] + importance) / 2
                else:
                    self.activations[name] = importance
        
        return hook
    
    def register_hooks(self, model):
        """Register hooks on all linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                hook = module.register_forward_hook(self.hook_fn(name + ".weight"))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_importance(self, weight_name):
        """Get importance weights for a given weight tensor."""
        # Map weight name to activation name
        # e.g., "transformer.h.0.mlp.c_fc.weight" -> "transformer.h.0.mlp.c_fc.weight"
        return self.activations.get(weight_name, None)


def calibrate_model(model_name: str, num_samples: int = 128, max_length: int = 512):
    """
    Run calibration and generate importance weights.
    
    Returns:
        bundle: dict with tensors and activation_stats
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading calibration data ({num_samples} samples)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"] for x in ds if x["text"].strip()][:num_samples]
    
    # Collect activations
    print("Running calibration forward passes...")
    collector = ActivationCollector()
    collector.register_hooks(model)
    
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            if (i + 1) % 32 == 0:
                print(f"  Processed {i + 1}/{len(texts)} samples")
            
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            model(**enc)
    
    collector.remove_hooks()
    print(f"Collected activations for {len(collector.activations)} layers")
    
    # Extract weights and build bundle
    print("Extracting weights and importance scores...")
    tensors = []
    activation_stats = {}
    
    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        
        weight = param.data.cpu().float()
        
        # Get importance for this weight
        importance = collector.get_importance(name)
        
        if importance is not None:
            # Expand importance to match weight shape
            # Weight shape is typically [out_features, in_features]
            # Importance is [in_features]
            if len(weight.shape) == 2:
                # Tile importance across output dimension
                imp_expanded = importance.unsqueeze(0).expand(weight.shape[0], -1).flatten()
            else:
                imp_expanded = importance
            
            activation_stats[name] = imp_expanded.tolist()
        else:
            # Default importance: weight magnitude
            activation_stats[name] = weight.abs().flatten().tolist()
        
        tensors.append({
            "name": name,
            "shape": list(weight.shape),
            "data": weight.flatten().tolist(),
        })
    
    print(f"Extracted {len(tensors)} tensors")
    
    return {
        "tensors": tensors,
        "activation_stats": activation_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate model for 10x compression")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--output", type=str, default="calibrated_bundle.json", help="Output path")
    parser.add_argument("--samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--eval", action="store_true", help="Evaluate PPL after calibration")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TENPAK CALIBRATION - Preparing for 10x Compression")
    print("=" * 70)
    
    bundle = calibrate_model(args.model, args.samples, args.max_length)
    
    # Save bundle
    print(f"\nSaving calibrated bundle to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(bundle, f)
    
    file_size = Path(args.output).stat().st_size / 1e6
    print(f"Bundle size: {file_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Build tenpak with calibration:")
    print(f"     cargo build --release --features calibration")
    print(f"  2. Compress with 10x codec:")
    print(f"     ./target/release/tenpak compress \\")
    print(f"         --input {args.output} \\")
    print(f"         --output model_10x.tenpak \\")
    print(f"         --codec mixed_cal_v1")
    print("=" * 70)


if __name__ == "__main__":
    main()
