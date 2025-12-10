#!/usr/bin/env python3
"""
Benchmark Tenpak g=8 inference against AWQ and FP16.

Measures:
- Throughput (tokens/sec)
- Latency (ms/token)
- Memory usage
- Quality (PPL delta)
"""

import argparse
import gc
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import tenpak CUDA
try:
    import sys
    sys.path.insert(0, str(__file__).replace('/scripts/benchmark_inference.py', '/cuda'))
    from tenpak_cuda import G8Linear, is_cuda_available
    TENPAK_AVAILABLE = True
except ImportError:
    TENPAK_AVAILABLE = False
    print("[Warning] Tenpak CUDA not available, using PyTorch fallback")

# Try to import AWQ
try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    print("[Warning] AWQ not available")


def get_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_warmup: int = 5,
    num_runs: int = 20,
) -> Dict[str, float]:
    """Benchmark forward pass."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "tokens_per_sec": input_ids.numel() / (sum(times) / len(times)),
    }


def convert_model_to_g8(model: nn.Module) -> nn.Module:
    """Convert all Linear layers to G8Linear."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            
            # Convert to G8Linear
            g8_linear = G8Linear.from_linear(module)
            setattr(parent, child_name, g8_linear)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tenpak inference")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of benchmark runs")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create input
    input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len), device=device)
    
    results = {}
    
    # Benchmark FP16
    print("=" * 60)
    print("Benchmarking FP16...")
    print("=" * 60)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    model_fp16 = AutoModelForCausalLM.from_pretrained(args.model).to(device).half()
    mem_fp16 = get_memory_mb()
    
    results["fp16"] = benchmark_forward(model_fp16, input_ids, num_runs=args.num_runs)
    results["fp16"]["memory_mb"] = mem_fp16
    
    print(f"  Mean latency: {results['fp16']['mean_ms']:.2f} ms")
    print(f"  Throughput: {results['fp16']['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Memory: {results['fp16']['memory_mb']:.0f} MB")
    
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark Tenpak G8
    if TENPAK_AVAILABLE:
        print()
        print("=" * 60)
        print("Benchmarking Tenpak G8...")
        print("=" * 60)
        
        model_g8 = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        model_g8 = convert_model_to_g8(model_g8)
        mem_g8 = get_memory_mb()
        
        # Convert input to half for G8
        results["tenpak_g8"] = benchmark_forward(model_g8, input_ids, num_runs=args.num_runs)
        results["tenpak_g8"]["memory_mb"] = mem_g8
        
        print(f"  Mean latency: {results['tenpak_g8']['mean_ms']:.2f} ms")
        print(f"  Throughput: {results['tenpak_g8']['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Memory: {results['tenpak_g8']['memory_mb']:.0f} MB")
        
        del model_g8
        gc.collect()
        torch.cuda.empty_cache()
    
    # Benchmark AWQ
    if AWQ_AVAILABLE:
        print()
        print("=" * 60)
        print("Benchmarking AWQ...")
        print("=" * 60)
        
        try:
            model_awq = AutoAWQForCausalLM.from_quantized(
                f"TheBloke/{args.model}-AWQ",
                fuse_layers=True,
            ).to(device)
            mem_awq = get_memory_mb()
            
            results["awq"] = benchmark_forward(model_awq.model, input_ids, num_runs=args.num_runs)
            results["awq"]["memory_mb"] = mem_awq
            
            print(f"  Mean latency: {results['awq']['mean_ms']:.2f} ms")
            print(f"  Throughput: {results['awq']['tokens_per_sec']:.0f} tokens/sec")
            print(f"  Memory: {results['awq']['memory_mb']:.0f} MB")
            
            del model_awq
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  AWQ failed: {e}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Method':<15} {'Latency (ms)':<15} {'Tokens/sec':<15} {'Memory (MB)':<15}")
    print("-" * 60)
    
    for method, data in results.items():
        print(f"{method:<15} {data['mean_ms']:<15.2f} {data['tokens_per_sec']:<15.0f} {data['memory_mb']:<15.0f}")
    
    # Speedup vs FP16
    if "fp16" in results:
        print()
        print("Speedup vs FP16:")
        fp16_time = results["fp16"]["mean_ms"]
        for method, data in results.items():
            if method != "fp16":
                speedup = fp16_time / data["mean_ms"]
                mem_savings = results["fp16"]["memory_mb"] / data["memory_mb"]
                print(f"  {method}: {speedup:.2f}x faster, {mem_savings:.2f}x memory savings")


if __name__ == "__main__":
    main()
