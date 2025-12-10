#!/usr/bin/env python3
"""
Evaluate Rust codecs: int4_g8, int4_g16, int4_k

This script tests the actual Rust tenpak binary to measure
true compression and PPL delta.
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
    """Compute perplexity."""
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


def test_codec(model_name, tokenizer, texts, baseline_ppl, codec):
    """Test a codec using the Rust binary."""
    print(f"\n{'='*60}")
    print(f"Testing codec: {codec}")
    print('='*60)
    
    # Load fresh model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Extract MLP weights
    tensors = []
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
    
    bundle = {"tensors": tensors, "activation_stats": {}}
    
    # Save bundle
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
        bundle_path = f.name
    
    # Compress with codec
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
    original_size_fp16 = total_weights * 2
    compress_fp32 = original_size_fp32 / artifact_size
    compress_fp16 = original_size_fp16 / artifact_size
    bits_per_weight = (artifact_size * 8) / total_weights
    
    print(f"Weights: {total_weights:,}")
    print(f"Original (FP32): {original_size_fp32/1e6:.2f} MB")
    print(f"Original (FP16): {original_size_fp16/1e6:.2f} MB")
    print(f"Compressed: {artifact_size/1e6:.2f} MB")
    print(f"Compression vs FP32: {compress_fp32:.2f}x")
    print(f"Compression vs FP16: {compress_fp16:.2f}x")
    print(f"Bits per weight: {bits_per_weight:.2f}")
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    del model
    del model_quant
    
    return {
        'codec': codec,
        'ppl': ppl,
        'ppl_delta': delta,
        'compress_fp32': compress_fp32,
        'compress_fp16': compress_fp16,
        'bits_per_weight': bits_per_weight,
        'artifact_size': artifact_size,
    }


def main():
    print("=" * 70)
    print("RUST CODEC EVALUATION")
    print("=" * 70)
    
    model_name = "gpt2"
    
    # Build if needed
    if not TENPAK_BIN.exists():
        print("Building tenpak...")
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:128]
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    del model
    
    # Test codecs
    codecs = [
        "int4_g8_v1",       # Current best (g=8) with F32 scales
        "int4_g8_fp16_v1",  # NEW: g=8 with FP16 scales - should be best!
        "int4_g16_v1",      # Group 16
    ]
    
    results = []
    for codec in codecs:
        result = test_codec(model_name, tokenizer, texts, baseline_ppl, codec)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Codec':<15} {'vs FP32':<10} {'vs FP16':<10} {'Bits/W':<8} {'PPL Δ':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        if r['ppl_delta'] < 1.0:
            status = "✓ GREAT"
        elif r['ppl_delta'] < 2.0:
            status = "~ GOOD"
        elif r['ppl_delta'] < 5.0:
            status = "~ OK"
        else:
            status = "✗ FAIL"
        
        print(f"{r['codec']:<15} {r['compress_fp32']:.2f}x      {r['compress_fp16']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%     {status}")
    
    print("-" * 70)
    print("AWQ: ~4x vs FP32, <1% PPL | llama.cpp Q4_K_M: ~4x vs FP32, <1% PPL")
    print("=" * 80)
    
    # Best result
    if results:
        best = min(results, key=lambda x: abs(x['ppl_delta']))
        print(f"\n🎯 Best quality: {best['codec']}")
        print(f"   Compression: {best['compress_fp32']:.2f}x vs FP32, {best['compress_fp16']:.2f}x vs FP16")
        print(f"   PPL Delta: {best['ppl_delta']:+.2f}%")


if __name__ == "__main__":
    main()
