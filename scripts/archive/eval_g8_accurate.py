#!/usr/bin/env python3
"""
ACCURATE Evaluation of tenpak's int4_g8_v1 codec.

This script measures:
1. TRUE compression ratio (actual weight bytes, not JSON overhead)
2. PPL delta on the quantized model
3. Breakdown of storage costs
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


def compute_perplexity(model, tokenizer, texts, device, max_length=512):
    """Compute perplexity on a list of texts."""
    model.eval()
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def analyze_compression(weight_shape, group_size=8):
    """
    Calculate TRUE compression ratio for int4_g8 quantization.
    
    Storage format per weight tensor:
    - Packed int4: numel / 2 bytes (2 weights per byte)
    - Scales: (numel / group_size) * 4 bytes (f32 per group)
    - Offsets: (numel / group_size) * 4 bytes (f32 per group)
    """
    numel = weight_shape[0] * weight_shape[1]
    
    # Original size (FP16)
    original_fp16 = numel * 2
    
    # Original size (FP32)
    original_fp32 = numel * 4
    
    # Quantized size
    packed_bytes = numel // 2  # 2 int4 values per byte
    num_groups = numel // group_size
    scales_bytes = num_groups * 4  # f32 per group
    offsets_bytes = num_groups * 4  # f32 per group
    quantized_total = packed_bytes + scales_bytes + offsets_bytes
    
    return {
        'numel': numel,
        'original_fp16': original_fp16,
        'original_fp32': original_fp32,
        'packed_bytes': packed_bytes,
        'scales_bytes': scales_bytes,
        'offsets_bytes': offsets_bytes,
        'quantized_total': quantized_total,
        'compression_vs_fp16': original_fp16 / quantized_total,
        'compression_vs_fp32': original_fp32 / quantized_total,
        'bits_per_weight': (quantized_total * 8) / numel,
    }


def main():
    print("=" * 70)
    print("ACCURATE TENPAK int4_g8_v1 EVALUATION")
    print("=" * 70)
    
    device = "cpu"  # Use CPU for accurate comparison
    model_name = "gpt2"
    
    # Build tenpak if needed
    if not TENPAK_BIN.exists():
        print("Building tenpak...")
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT, check=True)
    
    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load eval data
    print("Loading WikiText-2 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:64]
    
    # Baseline PPL
    print("\nComputing baseline perplexity...")
    ppl_baseline = compute_perplexity(model, tokenizer, texts, device)
    print(f"Baseline PPL: {ppl_baseline:.4f}")
    
    # Analyze MLP weights
    print("\n" + "=" * 70)
    print("COMPRESSION ANALYSIS (MLP weights only)")
    print("=" * 70)
    
    total_original_fp16 = 0
    total_original_fp32 = 0
    total_quantized = 0
    
    mlp_weights = []
    
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ["c_fc", "c_proj"]:
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            weight = layer.weight.data
            shape = tuple(weight.shape)
            
            analysis = analyze_compression(shape)
            total_original_fp16 += analysis['original_fp16']
            total_original_fp32 += analysis['original_fp32']
            total_quantized += analysis['quantized_total']
            
            mlp_weights.append({
                'name': f"h.{block_idx}.mlp.{layer_name}",
                'shape': shape,
                'analysis': analysis,
            })
    
    # Print per-layer breakdown
    print(f"\n{'Layer':<25} {'Shape':<15} {'FP16':<10} {'Quant':<10} {'Ratio':<8}")
    print("-" * 70)
    for w in mlp_weights[:4]:  # First 4 layers
        a = w['analysis']
        print(f"{w['name']:<25} {str(w['shape']):<15} {a['original_fp16']/1e6:<10.2f}MB {a['quantized_total']/1e6:<10.2f}MB {a['compression_vs_fp16']:<8.2f}x")
    print("... (showing first 4 of {} layers)".format(len(mlp_weights)))
    
    # Total compression
    print("\n" + "=" * 70)
    print("TOTAL COMPRESSION (MLP weights)")
    print("=" * 70)
    
    compression_fp16 = total_original_fp16 / total_quantized
    compression_fp32 = total_original_fp32 / total_quantized
    bits_per_weight = (total_quantized * 8) / (total_original_fp16 / 2)
    
    print(f"Original (FP16):    {total_original_fp16 / 1e6:.2f} MB")
    print(f"Original (FP32):    {total_original_fp32 / 1e6:.2f} MB")
    print(f"Quantized:          {total_quantized / 1e6:.2f} MB")
    print(f"Compression vs FP16: {compression_fp16:.2f}x")
    print(f"Compression vs FP32: {compression_fp32:.2f}x")
    print(f"Bits per weight:    {bits_per_weight:.2f}")
    
    # Storage breakdown
    total_packed = sum(w['analysis']['packed_bytes'] for w in mlp_weights)
    total_scales = sum(w['analysis']['scales_bytes'] for w in mlp_weights)
    total_offsets = sum(w['analysis']['offsets_bytes'] for w in mlp_weights)
    
    print(f"\nStorage breakdown:")
    print(f"  Packed int4:  {total_packed / 1e6:.2f} MB ({100*total_packed/total_quantized:.1f}%)")
    print(f"  Scales (f32): {total_scales / 1e6:.2f} MB ({100*total_scales/total_quantized:.1f}%)")
    print(f"  Offsets (f32): {total_offsets / 1e6:.2f} MB ({100*total_offsets/total_quantized:.1f}%)")
    
    # Now actually compress and measure PPL
    print("\n" + "=" * 70)
    print("ACTUAL QUANTIZATION TEST")
    print("=" * 70)
    
    # Create bundle
    tensors = []
    for block_idx in range(len(model.transformer.h)):
        for layer_name in ["c_fc", "c_proj"]:
            name = f"transformer.h.{block_idx}.mlp.{layer_name}.weight"
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
            weight = layer.weight.data.cpu().float()
            tensors.append({
                "name": name,
                "shape": list(weight.shape),
                "data": weight.flatten().tolist(),
            })
    
    bundle = {"tensors": tensors, "activation_stats": {}}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
        bundle_path = f.name
    
    # Compress
    artifact_path = bundle_path.replace(".json", ".tenpak")
    subprocess.run([
        str(TENPAK_BIN), "compress",
        "--input", bundle_path,
        "--output", artifact_path,
        "--codec", "int4_g8_v1",
    ], check=True, capture_output=True)
    
    # Decompress
    restored_path = bundle_path.replace(".json", "_restored.json")
    subprocess.run([
        str(TENPAK_BIN), "decompress",
        "--input", artifact_path,
        "--output", restored_path,
    ], check=True, capture_output=True)
    
    # Load restored and apply
    with open(restored_path) as f:
        restored = json.load(f)
    
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
    print("Computing quantized perplexity...")
    ppl_quant = compute_perplexity(model_quant, tokenizer, texts, device)
    
    delta = ppl_quant - ppl_baseline
    pct_delta = (delta / ppl_baseline) * 100
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Model:              {model_name}")
    print(f"Baseline PPL:       {ppl_baseline:.4f}")
    print(f"Quantized PPL:      {ppl_quant:.4f}")
    print(f"PPL Delta:          {delta:+.4f} ({pct_delta:+.2f}%)")
    print()
    print(f"TRUE Compression (vs FP16): {compression_fp16:.2f}x")
    print(f"TRUE Compression (vs FP32): {compression_fp32:.2f}x")
    print(f"Bits per weight:    {bits_per_weight:.2f}")
    print("=" * 70)
    
    # Comparison
    print("\n⚠️  IMPORTANT NOTES:")
    print("-" * 70)
    print("1. The 14.4x compression in eval_g8.py was JSON size ratio, NOT weight bytes")
    print(f"2. TRUE compression vs FP16 is {compression_fp16:.2f}x")
    print(f"3. TRUE compression vs FP32 is {compression_fp32:.2f}x")
    print("4. Group size 8 has HIGH overhead: scales+offsets = 50% of storage")
    print("-" * 70)
    
    if compression_fp16 < 2.0:
        print("\n⚠️  WARNING: Compression is lower than expected!")
        print("   With g=8, each group needs 4+4=8 bytes overhead for 8 weights")
        print("   This limits compression to ~1.33x vs FP16")
        print("\n   To improve:")
        print("   - Use FP16 scales/offsets (2x less overhead)")
        print("   - Use larger group sizes for less-critical layers")
        print("   - Use mixed precision (g=8 for critical, g=32 for others)")


if __name__ == "__main__":
    main()
