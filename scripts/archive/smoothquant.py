#!/usr/bin/env python3
"""
SmoothQuant: Migrating Quantization Difficulty from Activations to Weights

The key insight: Activation outliers make quantization hard.
SmoothQuant multiplies weights by activation scale and divides activations by the same.

W_new = W * diag(s)
X_new = X / diag(s)

where s = max(|X|)^alpha / max(|W|)^(1-alpha)

This "smooths" the quantization difficulty between weights and activations.

For weight-only quantization (our case), we use a simpler approach:
Apply per-channel scaling to make weight distributions more uniform.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import json
import subprocess
import tempfile
import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TENPAK_BIN = ROOT / "target" / "release" / "tenpak"


def compute_perplexity(model, tokenizer, texts, max_length=512):
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


def collect_activation_scales(model, tokenizer, texts):
    """Collect per-channel activation scales for smoothing."""
    scales = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            
            if not isinstance(x, torch.Tensor) or not x.is_floating_point():
                return
            
            # Get per-channel max (last dimension)
            x_max = x.abs().view(-1, x.shape[-1]).max(dim=0).values.detach().cpu()
            
            if name in scales:
                scales[name] = torch.max(scales[name], x_max)
            else:
                scales[name] = x_max
        
        return hook
    
    from transformers.pytorch_utils import Conv1D
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, Conv1D)):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for text in texts[:32]:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            model(**enc)
    
    for hook in hooks:
        hook.remove()
    
    return scales


def apply_smoothquant(model, activation_scales, alpha=0.5):
    """Apply SmoothQuant scaling to weights."""
    from transformers.pytorch_utils import Conv1D
    
    smoothed = {}
    
    for name, module in model.named_modules():
        if name not in activation_scales:
            continue
        
        act_scale = activation_scales[name]
        
        if isinstance(module, Conv1D):
            # Conv1D weight: [in, out]
            weight = module.weight.data
            weight_scale = weight.abs().max(dim=1).values.clamp(min=1e-8)
            
            # Compute smoothing scale
            # s = act_scale^alpha / weight_scale^(1-alpha)
            smooth_scale = (act_scale.to(weight.device) ** alpha) / (weight_scale ** (1 - alpha))
            smooth_scale = smooth_scale.clamp(min=1e-8)
            
            # Apply: W_new = W * s
            module.weight.data = weight * smooth_scale.unsqueeze(1)
            smoothed[name] = smooth_scale.cpu()
            
        elif isinstance(module, nn.Linear):
            # Linear weight: [out, in]
            weight = module.weight.data
            weight_scale = weight.abs().max(dim=0).values.clamp(min=1e-8)
            
            smooth_scale = (act_scale.to(weight.device) ** alpha) / (weight_scale ** (1 - alpha))
            smooth_scale = smooth_scale.clamp(min=1e-8)
            
            module.weight.data = weight * smooth_scale.unsqueeze(0)
            smoothed[name] = smooth_scale.cpu()
    
    return smoothed


def quantize_and_eval(model, tokenizer, texts, baseline_ppl, codec, name):
    """Quantize model and evaluate using tenpak."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    # Build bundle from MLP layers
    tensors = []
    total_weights = 0
    
    from transformers.pytorch_utils import Conv1D
    
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ["c_fc", "c_proj"]:
            full_name = f"transformer.h.{block_idx}.mlp.{layer_name}.weight"
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            
            if isinstance(layer, Conv1D):
                weight = layer.weight.data.T.cpu().float()
            else:
                weight = layer.weight.data.cpu().float()
            
            total_weights += weight.numel()
            
            tensors.append({
                "name": full_name,
                "shape": list(weight.shape),
                "data": weight.flatten().tolist(),
            })
    
    bundle = {"tensors": tensors, "activation_stats": {}}
    
    # Save bundle
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
        bundle_path = f.name
    
    # Compress
    artifact_path = bundle_path.replace(".json", ".tenpak")
    result = subprocess.run([
        str(TENPAK_BIN), "compress",
        "--input", bundle_path,
        "--output", artifact_path,
        "--codec", codec,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Compression failed: {result.stderr}")
        os.unlink(bundle_path)
        return None
    
    artifact_size = os.path.getsize(artifact_path)
    
    # Decompress
    restored_path = bundle_path.replace(".json", "_restored.json")
    result = subprocess.run([
        str(TENPAK_BIN), "decompress",
        "--input", artifact_path,
        "--output", restored_path,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Decompression failed: {result.stderr}")
        os.unlink(bundle_path)
        os.unlink(artifact_path)
        return None
    
    # Load restored weights
    with open(restored_path) as f:
        restored = json.load(f)
    
    # Apply restored weights
    for t in restored["tensors"]:
        parts = t["name"].split(".")
        block_idx = int(parts[2])
        layer_name = parts[4]
        
        layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
        weight = torch.tensor(t["data"], dtype=torch.float32).view(*t["shape"])
        
        if isinstance(layer, Conv1D):
            layer.weight.data = weight.T
        else:
            layer.weight.data = weight
    
    # Evaluate
    print("Computing quantized PPL...", end=" ", flush=True)
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
    
    # Calculate compression
    original_size_fp32 = total_weights * 4
    compress_fp32 = original_size_fp32 / artifact_size
    bits_per_weight = (artifact_size * 8) / total_weights
    
    print(f"Compression vs FP32: {compress_fp32:.2f}x")
    print(f"Bits per weight: {bits_per_weight:.2f}")
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    return {
        'name': name,
        'ppl': ppl,
        'ppl_delta': delta,
        'compress_fp32': compress_fp32,
        'bits_per_weight': bits_per_weight,
    }


def main():
    print("=" * 70)
    print("SmoothQuant Experiment")
    print("Target: Better INT3 quality through activation-aware scaling")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:80]
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    results = []
    
    # Test 1: Standard INT4
    print("\n--- Test 1: Standard INT4 (baseline) ---")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    result = quantize_and_eval(model, tokenizer, texts, baseline_ppl, "int4_g8_fp16_v1", "INT4 g8 (no smooth)")
    if result:
        results.append(result)
    del model
    
    # Test 2: INT3 without SmoothQuant
    print("\n--- Test 2: INT3 without SmoothQuant ---")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    result = quantize_and_eval(model, tokenizer, texts, baseline_ppl, "int3_cal_v1", "INT3 g32 (no smooth)")
    if result:
        results.append(result)
    del model
    
    # Test 3: SmoothQuant + INT3
    print("\n--- Test 3: SmoothQuant + INT3 ---")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Collecting activation scales...")
    act_scales = collect_activation_scales(model, tokenizer, texts)
    print(f"Collected scales for {len(act_scales)} layers")
    
    print("Applying SmoothQuant (alpha=0.5)...")
    apply_smoothquant(model, act_scales, alpha=0.5)
    
    result = quantize_and_eval(model, tokenizer, texts, baseline_ppl, "int3_cal_v1", "INT3 g32 + SmoothQuant")
    if result:
        results.append(result)
    del model
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Method':<30} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "🎯 10x!" if r['compress_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "✓ Good" if r['ppl_delta'] < 1.0 else \
                 "~ OK" if r['ppl_delta'] < 5.0 else "✗"
        
        print(f"{r['name']:<30} {r['compress_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 70)
    print("=" * 80)


if __name__ == "__main__":
    main()
