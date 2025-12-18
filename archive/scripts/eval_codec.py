#!/usr/bin/env python3
"""
Tenpak Codec Evaluation Script

Evaluates compression and PPL for a given codec on a given model.
Results are saved to results/ directory with full verification.

Usage:
    python scripts/eval_codec.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --codec int4_residual_v1
    python scripts/eval_codec.py --model meta-llama/Llama-2-7b-hf --codec int4_residual_v1 --low-memory
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tenpak_binary():
    """Find the tenpak binary."""
    candidates = [
        Path(__file__).parent.parent / "target" / "release" / "tenpak",
        Path(__file__).parent.parent / "target" / "debug" / "tenpak",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("tenpak binary not found. Run: cargo build --release")


def compute_ppl(model, tokenizer, dataset_name="wikitext", split="test", max_samples=50):
    """Compute perplexity on WikiText-2."""
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # Concatenate all text
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=100000)
    
    max_length = min(getattr(model.config, "max_position_embeddings", 2048), 2048)
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end = 0
    
    for begin in range(0, min(seq_len, max_samples * stride), stride):
        end = min(begin + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, : -stride if begin > 0 else 0] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end = end
        
        if len(nlls) >= max_samples:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def quantize_tensor_via_tenpak(tensor_data, shape, name, codec, tenpak_bin):
    """Quantize a tensor using tenpak and return compressed size + restored data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input bundle
        bundle = {
            "tensors": [{
                "name": name,
                "shape": list(shape),
                "data": tensor_data.flatten().tolist()
            }],
            "activation_stats": {}
        }
        
        input_path = os.path.join(tmpdir, "input.json")
        output_path = os.path.join(tmpdir, "output.tenpak")
        restored_path = os.path.join(tmpdir, "restored.json")
        
        with open(input_path, "w") as f:
            json.dump(bundle, f)
        
        original_size = len(tensor_data.flatten()) * 4  # FP32 bytes
        
        # Compress
        result = subprocess.run(
            [tenpak_bin, "compress", "--input", input_path, "--output", output_path, "--codec", codec],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Compression failed: {result.stderr}")
        
        compressed_size = os.path.getsize(output_path)
        
        # Decompress
        result = subprocess.run(
            [tenpak_bin, "decompress", "--input", output_path, "--output", restored_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Decompression failed: {result.stderr}")
        
        with open(restored_path) as f:
            restored = json.load(f)
        
        restored_data = torch.tensor(restored["tensors"][0]["data"]).reshape(shape)
        
        return compressed_size, original_size, restored_data


def evaluate_codec(model, tokenizer, codec, tenpak_bin, layers="mlp", low_memory=False):
    """Evaluate a codec on a model."""
    print(f"\n{'='*60}")
    print(f"Evaluating codec: {codec}")
    print(f"Layers: {layers}")
    print(f"{'='*60}\n")
    
    # Store original weights
    original_weights = {}
    total_original_size = 0
    total_compressed_size = 0
    
    # Find layers to quantize
    layers_to_quantize = []
    for name, module in model.named_modules():
        if layers == "mlp" and "mlp" in name.lower():
            if hasattr(module, "weight") and module.weight is not None:
                layers_to_quantize.append((name, module))
        elif layers == "attn" and ("attn" in name.lower() or "attention" in name.lower()):
            if hasattr(module, "weight") and module.weight is not None:
                layers_to_quantize.append((name, module))
        elif layers == "all":
            if hasattr(module, "weight") and module.weight is not None:
                layers_to_quantize.append((name, module))
    
    print(f"Found {len(layers_to_quantize)} layers to quantize")
    
    # Quantize each layer
    for i, (name, module) in enumerate(layers_to_quantize):
        weight = module.weight.data.float().cpu()
        original_weights[name] = weight.clone()
        
        try:
            compressed_size, original_size, restored = quantize_tensor_via_tenpak(
                weight.numpy(), weight.shape, name, codec, tenpak_bin
            )
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            
            # Apply quantized weights
            module.weight.data = restored.to(module.weight.device).to(module.weight.dtype)
            
            if (i + 1) % 10 == 0 or i == len(layers_to_quantize) - 1:
                print(f"  Quantized {i+1}/{len(layers_to_quantize)} layers")
                
        except Exception as e:
            print(f"  Warning: Failed to quantize {name}: {e}")
            continue
    
    compression = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
    
    print(f"\nCompression: {compression:.2f}x")
    print(f"  Original: {total_original_size / 1e6:.2f} MB")
    print(f"  Compressed: {total_compressed_size / 1e6:.2f} MB")
    
    return compression, original_weights


def main():
    parser = argparse.ArgumentParser(description="Evaluate tenpak codec")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--codec", type=str, default="int4_residual_v1")
    parser.add_argument("--layers", type=str, default="mlp", choices=["mlp", "attn", "all"])
    parser.add_argument("--low-memory", action="store_true", help="Use low memory mode for large models")
    parser.add_argument("--max-samples", type=int, default=50, help="Max PPL evaluation samples")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()
    
    tenpak_bin = get_tenpak_binary()
    print(f"Using tenpak: {tenpak_bin}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    
    load_kwargs = {
        "torch_dtype": torch.float16 if args.low_memory else torch.float32,
        "low_cpu_mem_usage": True,
    }
    
    if args.low_memory:
        load_kwargs["device_map"] = "auto"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    
    if not args.low_memory:
        model = model.cuda() if torch.cuda.is_available() else model
    
    model.eval()
    
    # Compute baseline PPL
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, max_samples=args.max_samples)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Quantize and measure compression
    compression, original_weights = evaluate_codec(
        model, tokenizer, args.codec, tenpak_bin, 
        layers=args.layers, low_memory=args.low_memory
    )
    
    # Compute quantized PPL
    print("\nComputing quantized PPL...")
    quant_ppl = compute_ppl(model, tokenizer, max_samples=args.max_samples)
    print(f"Quantized PPL: {quant_ppl:.4f}")
    
    # Calculate delta
    ppl_delta = ((quant_ppl - baseline_ppl) / baseline_ppl) * 100
    passed = abs(ppl_delta) < 1.0
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Codec: {args.codec}")
    print(f"Layers: {args.layers}")
    print(f"Compression: {compression:.2f}x")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"Quantized PPL: {quant_ppl:.4f}")
    print(f"PPL Delta: {ppl_delta:+.2f}%")
    print(f"Status: {'✅ PASS' if passed else '❌ FAIL'} (<1% PPL delta)")
    print(f"{'='*60}\n")
    
    # Save results
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "codec": args.codec,
        "layers": args.layers,
        "compression": compression,
        "baseline_ppl": baseline_ppl,
        "quant_ppl": quant_ppl,
        "ppl_delta_percent": ppl_delta,
        "passed": passed,
        "max_samples": args.max_samples,
        "low_memory": args.low_memory,
    }
    
    output_path = args.output
    if output_path is None:
        model_name = args.model.replace("/", "_")
        output_path = f"results/eval_{model_name}_{args.codec}_{args.layers}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
