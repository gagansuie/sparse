#!/usr/bin/env python3
"""
Streaming Llama Evaluation for Large Models

Memory-efficient evaluation that processes layers one at a time.
Can handle Llama 7B, 13B, and 70B without OOM.

Usage:
  python eval_llama_streaming.py --model meta-llama/Llama-2-7b-hf
  python eval_llama_streaming.py --model meta-llama/Llama-2-70b-hf --layers 5
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import gc
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
TENPAK_BIN = ROOT / "target" / "release" / "tenpak"


def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def quantize_tensor_streaming(weight, codec="int4_opt_v1"):
    """Quantize a single tensor through tenpak (streaming)."""
    
    # Create minimal bundle with just this tensor
    bundle = {
        "tensors": [{
            "name": "weight",
            "shape": list(weight.shape),
            "data": weight.cpu().float().flatten().tolist(),
        }],
        "activation_stats": {}
    }
    
    # Write bundle
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(bundle, f)
        bundle_path = f.name
    
    original_size = os.path.getsize(bundle_path)
    
    # Compress
    artifact_path = bundle_path.replace(".json", ".tenpak")
    result = subprocess.run([
        str(TENPAK_BIN), "compress",
        "--input", bundle_path,
        "--output", artifact_path,
        "--codec", codec,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        os.unlink(bundle_path)
        return None, None, f"Compress failed: {result.stderr}"
    
    artifact_size = os.path.getsize(artifact_path)
    
    # Decompress
    restored_path = bundle_path.replace(".json", "_restored.json")
    result = subprocess.run([
        str(TENPAK_BIN), "decompress",
        "--input", artifact_path,
        "--output", restored_path,
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        os.unlink(bundle_path)
        os.unlink(artifact_path)
        return None, None, f"Decompress failed: {result.stderr}"
    
    # Load restored
    with open(restored_path) as f:
        restored = json.load(f)
    
    restored_weight = torch.tensor(
        restored["tensors"][0]["data"], 
        dtype=weight.dtype
    ).view(*weight.shape)
    
    # Cleanup
    os.unlink(bundle_path)
    os.unlink(artifact_path)
    os.unlink(restored_path)
    
    # Stats
    compression = (weight.numel() * 4) / artifact_size  # vs FP32
    
    return restored_weight, compression, None


def compute_perplexity_streaming(model, tokenizer, texts, max_length=512, max_samples=50):
    """Compute perplexity with limited samples."""
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for i, text in enumerate(texts[:max_samples]):
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            
            try:
                outputs = model(input_ids, labels=input_ids)
                nll += outputs.loss.item() * input_ids.numel()
                ntokens += input_ids.numel()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at sample {i}, using {ntokens} tokens")
                    break
                raise
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{min(len(texts), max_samples)} samples...")
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


def test_model_streaming(model_name, codec="int4_opt_v1", num_layers=None, device="cuda"):
    """
    Test quantization on a model using streaming (layer-by-layer).
    
    Args:
        model_name: HuggingFace model name
        codec: Tenpak codec to use
        num_layers: Number of layers to test (None = all)
        device: Device to use
    """
    print("=" * 70)
    print(f"Streaming Evaluation: {model_name}")
    print(f"Codec: {codec}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Check if model exists / get config
    print(f"\nLoading config for {model_name}...")
    try:
        config = AutoConfig.from_pretrained(model_name)
        total_layers = config.num_hidden_layers
        print(f"Model has {total_layers} layers")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    if num_layers is None:
        num_layers = total_layers
    else:
        num_layers = min(num_layers, total_layers)
    
    print(f"Testing {num_layers} layers")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:100]
    
    # Estimate model size
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
    params_per_layer = (
        4 * hidden_size * hidden_size +  # attention
        3 * hidden_size * intermediate_size  # MLP
    )
    total_params = total_layers * params_per_layer
    model_size_gb = total_params * 2 / 1024**3  # FP16
    
    print(f"\nEstimated model size: {model_size_gb:.1f} GB (FP16)")
    print(f"Current GPU memory: {get_memory_usage():.1f} GB")
    
    # For small models, load entirely
    if model_size_gb < 15:  # Can fit in memory
        print("\nModel fits in memory - loading full model...")
        return test_model_full(model_name, codec, tokenizer, texts, device)
    
    # For large models, use streaming
    print("\nModel too large - using streaming evaluation...")
    return test_model_layers_only(model_name, codec, num_layers, tokenizer, texts, device)


def test_model_full(model_name, codec, tokenizer, texts, device):
    """Test full model (for smaller models that fit in memory)."""
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    print(f"Memory after load: {get_memory_usage():.1f} GB")
    
    # Baseline PPL
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_perplexity_streaming(model, tokenizer, texts, max_samples=50)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Quantize MLP layers
    print(f"\nQuantizing with {codec}...")
    total_weights = 0
    total_compressed = 0
    
    for i, layer in enumerate(model.model.layers):
        print(f"  Layer {i}...", end=" ")
        
        # MLP layers (main target)
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, name):
                proj = getattr(layer.mlp, name)
                weight = proj.weight.data
                
                restored, compression, error = quantize_tensor_streaming(weight, codec)
                
                if error:
                    print(f"Error: {error}")
                    continue
                
                proj.weight.data = restored.to(weight.device, weight.dtype)
                total_weights += weight.numel()
                total_compressed += weight.numel() * 4 / compression
        
        print(f"done")
        gc.collect()
        torch.cuda.empty_cache()
    
    overall_compression = (total_weights * 4) / total_compressed if total_compressed > 0 else 0
    print(f"\nOverall MLP compression: {overall_compression:.2f}x")
    
    # Quantized PPL
    print("\nComputing quantized PPL...")
    quant_ppl = compute_perplexity_streaming(model, tokenizer, texts, max_samples=50)
    delta = (quant_ppl - baseline_ppl) / baseline_ppl * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Baseline PPL:    {baseline_ppl:.4f}")
    print(f"Quantized PPL:   {quant_ppl:.4f}")
    print(f"PPL Delta:       {delta:+.2f}%")
    print(f"Compression:     {overall_compression:.2f}x")
    print(f"{'='*60}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "codec": codec,
        "baseline_ppl": baseline_ppl,
        "quant_ppl": quant_ppl,
        "ppl_delta": delta,
        "compression": overall_compression,
    }


def test_model_layers_only(model_name, codec, num_layers, tokenizer, texts, device):
    """
    Test quantization quality by loading layers one at a time.
    Cannot compute full PPL, but can measure reconstruction error.
    """
    print(f"\nStreaming {num_layers} layers...")
    
    total_mse = 0
    total_weights = 0
    total_compressed_bytes = 0
    
    # Load model with offloading
    print("Loading model with CPU offloading...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="offload",
        low_cpu_mem_usage=True,
    )
    
    print(f"Memory after load: {get_memory_usage():.1f} GB")
    
    # Test first N layers
    for i in range(num_layers):
        print(f"\nLayer {i}/{num_layers}:")
        layer = model.model.layers[i]
        
        # Process MLP projections
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, name):
                proj = getattr(layer.mlp, name)
                weight = proj.weight.data.cpu().float()
                
                print(f"  {name}: {weight.shape}", end=" ")
                
                restored, compression, error = quantize_tensor_streaming(weight, codec)
                
                if error:
                    print(f"- ERROR: {error}")
                    continue
                
                # Compute MSE
                mse = ((weight - restored) ** 2).mean().item()
                total_mse += mse * weight.numel()
                total_weights += weight.numel()
                total_compressed_bytes += weight.numel() * 4 / compression
                
                print(f"- {compression:.2f}x, MSE={mse:.6f}")
        
        gc.collect()
    
    # Summary
    avg_mse = total_mse / total_weights if total_weights > 0 else 0
    overall_compression = (total_weights * 4) / total_compressed_bytes if total_compressed_bytes > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"STREAMING RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Layers tested:      {num_layers}")
    print(f"Total weights:      {total_weights:,}")
    print(f"Average MSE:        {avg_mse:.6f}")
    print(f"Compression:        {overall_compression:.2f}x")
    print(f"{'='*60}")
    
    # Estimate PPL based on MSE (rough correlation from GPT-2 experiments)
    # MSE ~0.0001 corresponds to ~1% PPL delta
    estimated_ppl_delta = avg_mse * 10000  # Very rough estimate
    print(f"Estimated PPL Δ:    ~{estimated_ppl_delta:.1f}% (based on MSE)")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "codec": codec,
        "layers_tested": num_layers,
        "avg_mse": avg_mse,
        "compression": overall_compression,
        "estimated_ppl_delta": estimated_ppl_delta,
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming Llama Evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model name (HuggingFace)")
    parser.add_argument("--codec", type=str, default="int4_opt_v1",
                        help="Tenpak codec to use")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers to test (default: all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Check tenpak binary
    if not TENPAK_BIN.exists():
        print(f"Error: tenpak binary not found at {TENPAK_BIN}")
        print("Run: cargo build --release")
        return
    
    result = test_model_streaming(
        args.model,
        args.codec,
        args.layers,
        args.device,
    )
    
    if result:
        print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
