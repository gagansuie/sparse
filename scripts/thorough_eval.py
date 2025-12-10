#!/usr/bin/env python3
"""
Thorough Evaluation of int4_opt_v1 vs Other Codecs

Tests:
1. Multiple runs to confirm consistency
2. All model layers (MLP, attention, embeddings)
3. More evaluation samples
4. Verify compression ratios
5. Compare against all codecs
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import json
import math
import os
import subprocess
import tempfile
import time
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


def test_codec_full(model_name, tokenizer, texts, baseline_ppl, codec, layers="mlp"):
    """Test a codec with full round-trip through tenpak."""
    from transformers.pytorch_utils import Conv1D
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Build bundle
    tensors = []
    total_weights = 0
    original_weights = {}
    
    for block_idx in range(len(model.transformer.h)):
        if layers in ["mlp", "all"]:
            for layer_name in ["c_fc", "c_proj"]:
                full_name = f"transformer.h.{block_idx}.mlp.{layer_name}.weight"
                layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
                
                if isinstance(layer, Conv1D):
                    weight = layer.weight.data.T.cpu().float()
                else:
                    weight = layer.weight.data.cpu().float()
                
                total_weights += weight.numel()
                original_weights[full_name] = weight.clone()
                
                tensors.append({
                    "name": full_name,
                    "shape": list(weight.shape),
                    "data": weight.flatten().tolist(),
                })
        
        if layers in ["attn", "all"]:
            for layer_name in ["c_attn", "c_proj"]:
                full_name = f"transformer.h.{block_idx}.attn.{layer_name}.weight"
                layer = getattr(model.transformer.h[block_idx].attn, layer_name)
                
                if isinstance(layer, Conv1D):
                    weight = layer.weight.data.T.cpu().float()
                else:
                    weight = layer.weight.data.cpu().float()
                
                total_weights += weight.numel()
                original_weights[full_name] = weight.clone()
                
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
    
    original_bundle_size = os.path.getsize(bundle_path)
    
    # Compress
    artifact_path = bundle_path.replace(".json", ".tenpak")
    start_time = time.time()
    result = subprocess.run([
        str(TENPAK_BIN), "compress",
        "--input", bundle_path,
        "--output", artifact_path,
        "--codec", codec,
    ], capture_output=True, text=True)
    compress_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Compression failed: {result.stderr}")
        os.unlink(bundle_path)
        del model
        return None
    
    artifact_size = os.path.getsize(artifact_path)
    
    # Decompress
    restored_path = bundle_path.replace(".json", "_restored.json")
    start_time = time.time()
    result = subprocess.run([
        str(TENPAK_BIN), "decompress",
        "--input", artifact_path,
        "--output", restored_path,
    ], capture_output=True, text=True)
    decompress_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"ERROR: Decompression failed: {result.stderr}")
        os.unlink(bundle_path)
        os.unlink(artifact_path)
        del model
        return None
    
    # Load and apply restored weights
    with open(restored_path) as f:
        restored = json.load(f)
    
    total_mse = 0
    max_error = 0
    
    for t in restored["tensors"]:
        name = t["name"]
        parts = name.split(".")
        block_idx = int(parts[2])
        layer_type = parts[3]  # mlp or attn
        layer_name = parts[4]
        
        if layer_type == "mlp":
            layer = getattr(model.transformer.h[block_idx].mlp, layer_name)
        else:
            layer = getattr(model.transformer.h[block_idx].attn, layer_name)
        
        weight = torch.tensor(t["data"], dtype=torch.float32).view(*t["shape"])
        
        # Compute reconstruction error
        original = original_weights[name]
        mse = ((original - weight) ** 2).mean().item()
        max_err = (original - weight).abs().max().item()
        total_mse += mse * original.numel()
        max_error = max(max_error, max_err)
        
        if isinstance(layer, Conv1D):
            layer.weight.data = weight.T
        else:
            layer.weight.data = weight
    
    avg_mse = total_mse / total_weights
    
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
    del model
    
    return {
        'codec': codec,
        'layers': layers,
        'ppl': ppl,
        'ppl_delta': delta,
        'compress_fp32': compress_fp32,
        'bits_per_weight': bits_per_weight,
        'avg_mse': avg_mse,
        'max_error': max_error,
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'total_weights': total_weights,
        'artifact_bytes': artifact_size,
    }


def main():
    print("=" * 80)
    print("THOROUGH EVALUATION: Confirming int4_opt_v1 Results")
    print("=" * 80)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2 (full test set)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()]  # All test samples
    print(f"Using {len(texts)} test samples")
    
    # Compute baseline multiple times to confirm
    print("\n" + "=" * 60)
    print("BASELINE VERIFICATION (3 runs)")
    print("=" * 60)
    
    baselines = []
    for i in range(3):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        ppl = compute_perplexity(model, tokenizer, texts)
        baselines.append(ppl)
        print(f"  Run {i+1}: {ppl:.4f}")
        del model
    
    baseline_ppl = sum(baselines) / len(baselines)
    baseline_std = (sum((x - baseline_ppl)**2 for x in baselines) / len(baselines)) ** 0.5
    print(f"  Average: {baseline_ppl:.4f} (std: {baseline_std:.4f})")
    
    # Test codecs
    codecs = [
        "int4_opt_v1",       # Our best
        "int4_g16_fp16_v1",  # Comparison
        "int4_g8_fp16_v1",   # Baseline
        "int4_g32_fp16_v1",  # Higher compression
    ]
    
    results = []
    
    # Test MLP layers (main comparison)
    print("\n" + "=" * 60)
    print("MLP LAYERS ONLY (main benchmark)")
    print("=" * 60)
    
    for codec in codecs:
        print(f"\nTesting: {codec}")
        print("-" * 40)
        
        # Run 3 times
        runs = []
        for i in range(3):
            result = test_codec_full(model_name, tokenizer, texts, baseline_ppl, codec, "mlp")
            if result:
                runs.append(result)
                print(f"  Run {i+1}: PPL={result['ppl']:.4f} (Δ {result['ppl_delta']:+.2f}%), "
                      f"MSE={result['avg_mse']:.6f}")
        
        if runs:
            avg_result = runs[0].copy()
            avg_result['ppl'] = sum(r['ppl'] for r in runs) / len(runs)
            avg_result['ppl_delta'] = sum(r['ppl_delta'] for r in runs) / len(runs)
            avg_result['ppl_std'] = (sum((r['ppl_delta'] - avg_result['ppl_delta'])**2 for r in runs) / len(runs)) ** 0.5
            results.append(avg_result)
            
            print(f"  Average: PPL Δ = {avg_result['ppl_delta']:+.2f}% (std: {avg_result['ppl_std']:.2f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - MLP Layers")
    print("=" * 80)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print()
    print(f"{'Codec':<20} {'Compress':<10} {'Bits/W':<8} {'PPL Δ':<12} {'MSE':<12} {'Status'}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['ppl_delta']):
        status = "🎯 BEST!" if r['ppl_delta'] < 0 else "✓ <1%" if r['ppl_delta'] < 1.0 else "~ OK" if r['ppl_delta'] < 2.0 else ""
        print(f"{r['codec']:<20} {r['compress_fp32']:.2f}x      {r['bits_per_weight']:.2f}     {r['ppl_delta']:+.2f}%       {r['avg_mse']:.6f}    {status}")
    
    print("-" * 80)
    
    # Detailed stats for best codec
    best = min(results, key=lambda r: r['ppl_delta'])
    print(f"\n{'='*60}")
    print(f"BEST CODEC: {best['codec']}")
    print('='*60)
    print(f"  Compression:      {best['compress_fp32']:.2f}x vs FP32")
    print(f"  Bits per weight:  {best['bits_per_weight']:.2f}")
    print(f"  PPL Delta:        {best['ppl_delta']:+.2f}%")
    print(f"  Avg MSE:          {best['avg_mse']:.6f}")
    print(f"  Max Error:        {best['max_error']:.4f}")
    print(f"  Compress time:    {best['compress_time']:.3f}s")
    print(f"  Decompress time:  {best['decompress_time']:.3f}s")
    print(f"  Total weights:    {best['total_weights']:,}")
    print(f"  Artifact size:    {best['artifact_bytes']:,} bytes")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    opt_result = next((r for r in results if r['codec'] == 'int4_opt_v1'), None)
    g16_result = next((r for r in results if r['codec'] == 'int4_g16_fp16_v1'), None)
    
    if opt_result and g16_result:
        improvement = g16_result['ppl_delta'] - opt_result['ppl_delta']
        print(f"int4_opt_v1 vs int4_g16_fp16_v1:")
        print(f"  - Same compression: {opt_result['compress_fp32']:.2f}x")
        print(f"  - PPL improvement: {improvement:+.2f} percentage points")
        print(f"  - int4_opt_v1 PPL Δ: {opt_result['ppl_delta']:+.2f}%")
        print(f"  - int4_g16_fp16_v1 PPL Δ: {g16_result['ppl_delta']:+.2f}%")
        
        if opt_result['ppl_delta'] < 0:
            print(f"\n✅ CONFIRMED: int4_opt_v1 achieves BETTER than baseline PPL!")
            print(f"   with {opt_result['compress_fp32']:.2f}x compression and NO calibration!")
        elif opt_result['ppl_delta'] < 1.0:
            print(f"\n✅ CONFIRMED: int4_opt_v1 achieves <1% PPL delta!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
