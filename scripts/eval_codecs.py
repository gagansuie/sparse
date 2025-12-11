#!/usr/bin/env python3
"""
Evaluate new high-compression INT4 codecs.
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


def test_codec(model, tokenizer, texts, baseline_ppl, codec):
    """Test a codec and return results."""
    from transformers.pytorch_utils import Conv1D
    
    # Build bundle from MLP layers
    tensors = []
    total_weights = 0
    
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
        print(f"ERROR: {result.stderr}")
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
        print(f"ERROR: {result.stderr}")
        os.unlink(bundle_path)
        os.unlink(artifact_path)
        return None
    
    # Load and apply restored weights
    with open(restored_path) as f:
        restored = json.load(f)
    
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
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    # Calculate compression
    original_size_fp32 = total_weights * 4
    compress_fp32 = original_size_fp32 / artifact_size
    bits_per_weight = (artifact_size * 8) / total_weights
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    return {
        'codec': codec,
        'ppl': ppl,
        'ppl_delta': delta,
        'compress_fp32': compress_fp32,
        'bits_per_weight': bits_per_weight,
    }


def main():
    print("=" * 70)
    print("New High-Compression Codec Evaluation")
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
    
    # Test codecs
    codecs = [
        "int4_g8_fp16_v1",   # Baseline
        "int4_g16_fp16_v1",  # Better compression
        "int4_opt_v1",       # NEW: Optimal with iterative refinement
        "int4_g32_fp16_v1",  # Even better compression
    ]
    
    results = []
    
    for codec in codecs:
        print(f"\n{'='*60}")
        print(f"Testing: {codec}")
        print('='*60)
        
        # Reload fresh model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        result = test_codec(model, tokenizer, texts, baseline_ppl, codec)
        if result:
            results.append(result)
            print(f"Compression: {result['compress_fp32']:.2f}x vs FP32")
            print(f"Bits/weight: {result['bits_per_weight']:.2f}")
            print(f"PPL: {result['ppl']:.4f} (Δ {result['ppl_delta']:+.2f}%)")
        
        del model
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Codec':<20} {'Compress':<12} {'Bits/W':<10} {'PPL Δ':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "🎯 BEST!" if r['compress_fp32'] >= 5 and r['ppl_delta'] < 2.0 else \
                 "✓ Good" if r['ppl_delta'] < 1.0 else \
                 "~ OK" if r['ppl_delta'] < 3.0 else "✗"
        
        print(f"{r['codec']:<20} {r['compress_fp32']:.2f}x        {r['bits_per_weight']:.2f}       {r['ppl_delta']:+.2f}%       {status}")
    
    print("-" * 70)
    print("=" * 80)


if __name__ == "__main__":
    main()
