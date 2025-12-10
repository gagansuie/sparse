#!/usr/bin/env python3
"""
Evaluate calibration-based codecs for 10x compression.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch
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


def collect_importance(model, tokenizer, texts, max_length=256):
    """Collect activation-based importance for calibration."""
    importance = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
            else:
                x = input
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                try:
                    if x.dim() >= 2:
                        imp = x.abs().mean(dim=tuple(range(x.dim() - 1))).detach().cpu()
                    else:
                        imp = x.abs().detach().cpu()
                    if name in importance:
                        importance[name] = (importance[name] + imp) / 2
                    else:
                        importance[name] = imp
                except:
                    pass  # Skip problematic tensors
        return hook
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            hook = module.register_forward_hook(make_hook(name + ".weight"))
            hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for text in texts[:64]:  # Use subset for speed
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            model(**enc)
    
    for hook in hooks:
        hook.remove()
    
    return importance


def test_calibrated_codec(model_name, tokenizer, texts, baseline_ppl, codec, use_calibration=True):
    """Test a calibrated codec."""
    print(f"\n{'='*60}")
    print(f"Testing: {codec}" + (" (with calibration)" if use_calibration else " (no calibration)"))
    print('='*60)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Collect importance if using calibration
    if use_calibration:
        print("Collecting activation importance (calibrating)...")
        importance = collect_importance(model, tokenizer, texts)
    else:
        importance = {}
    
    # Build bundle
    tensors = []
    activation_stats = {}
    total_weights = 0
    
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ["c_fc", "c_proj"]:
            name = f"transformer.h.{block_idx}.mlp.{layer_name}.weight"
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            weight = layer.weight.data.cpu().float()
            total_weights += weight.numel()
            
            tensors.append({
                "name": name,
                "shape": list(weight.shape),
                "data": weight.flatten().tolist(),
            })
            
            # Add importance weights
            imp_key = f"transformer.h.{block_idx}.mlp.{layer_name}.weight"
            if imp_key in importance:
                imp = importance[imp_key]
                if len(weight.shape) == 2:
                    imp_expanded = imp.unsqueeze(0).expand(weight.shape[0], -1).flatten()
                else:
                    imp_expanded = imp
                activation_stats[name] = imp_expanded.tolist()
            else:
                # Magnitude-based fallback
                activation_stats[name] = weight.abs().flatten().tolist()
    
    bundle = {"tensors": tensors, "activation_stats": activation_stats}
    
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
    model_quant = AutoModelForCausalLM.from_pretrained(model_name)
    
    for t in restored["tensors"]:
        name = t["name"]
        parts = name.split(".")
        block_idx = int(parts[2])
        layer_name = parts[4]
        
        layer = getattr(model_quant.transformer.h[block_idx].mlp, layer_name)
        weight = torch.tensor(t["data"], dtype=torch.float32).view(*t["shape"])
        layer.weight.data = weight
    
    # Evaluate
    print("Computing quantized PPL...", end=" ", flush=True)
    ppl = compute_perplexity(model_quant, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"{ppl:.4f} (Δ {delta:+.2f}%)")
    
    # Calculate compression
    original_size_fp32 = total_weights * 4
    compress_fp32 = original_size_fp32 / artifact_size
    bits_per_weight = (artifact_size * 8) / total_weights
    
    print(f"Weights: {total_weights:,}")
    print(f"Original (FP32): {original_size_fp32/1e6:.2f} MB")
    print(f"Compressed: {artifact_size/1e6:.2f} MB")
    print(f"Compression vs FP32: {compress_fp32:.2f}x")
    print(f"Bits per weight: {bits_per_weight:.2f}")
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    del model, model_quant
    
    return {
        'codec': codec,
        'calibrated': use_calibration,
        'ppl': ppl,
        'ppl_delta': delta,
        'compress_fp32': compress_fp32,
        'bits_per_weight': bits_per_weight,
    }


def main():
    print("=" * 70)
    print("CALIBRATION-BASED COMPRESSION - Target: 10x with <1% PPL")
    print("=" * 70)
    
    model_name = "gpt2"
    
    # Test calibration codecs (assumes built with --features calibration)
    codecs = ["gptq_v1", "int3_cal_v1", "int4_g8_fp16_v1"]
    
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
    
    # Test codecs
    results = []
    
    for codec in codecs:
        if codec in ["mixed_cal_v1", "int3_cal_v1", "gptq_v1"]:
            # Test with calibration
            result = test_calibrated_codec(model_name, tokenizer, texts, baseline_ppl, codec, use_calibration=True)
            if result:
                results.append(result)
        else:
            result = test_calibrated_codec(model_name, tokenizer, texts, baseline_ppl, codec, use_calibration=False)
            if result:
                results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Target: 10x compression, <1% PPL delta")
    print()
    print(f"{'Codec':<20} {'Calibrated':<12} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 75)
    
    for r in results:
        cal = "Yes" if r['calibrated'] else "No"
        status = "🎯 10x!" if r['compress_fp32'] >= 10 and r['ppl_delta'] < 1.0 else \
                 "✓ Good" if r['ppl_delta'] < 2.0 else \
                 "~ OK" if r['ppl_delta'] < 5.0 else "✗"
        
        print(f"{r['codec']:<20} {cal:<12} {r['compress_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 75)
    print("Target: 10.00x compression, <1% PPL delta")
    print("=" * 80)


if __name__ == "__main__":
    main()
