"""
Benchmark Rust vs Python implementation for delta compression operations.

This script measures the performance improvement from Rust acceleration.
"""

import time
import numpy as np
import torch
from typing import Callable, Tuple

# Rust is required
from core.delta_rust import get_rust_info

from core.delta import (
    compress_delta_sparse,
    decompress_delta_sparse,
    compress_delta_int8,
    decompress_delta_int8,
)


def benchmark_function(
    func: Callable,
    *args,
    num_runs: int = 5,
    warmup: int = 1,
    **kwargs
) -> Tuple[float, float]:
    """Benchmark a function with multiple runs."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def benchmark_sparse_compression():
    """Benchmark sparse delta compression."""
    print("\n" + "="*60)
    print("SPARSE DELTA COMPRESSION BENCHMARK")
    print("="*60)
    
    sizes = [
        (1_000_000, "1M parameters (small layer)"),
        (10_000_000, "10M parameters (medium layer)"),
        (50_000_000, "50M parameters (large layer)"),
    ]
    
    for size, desc in sizes:
        print(f"\n{desc} ({size:,} elements):")
        
        # Create sparse delta (90% zeros)
        delta = torch.randn(size, dtype=torch.float32)
        delta[torch.abs(delta) < 1.5] = 0
        sparsity = (delta == 0).float().mean().item()
        print(f"  Sparsity: {sparsity*100:.1f}%")
        
        # Benchmark compression
        avg_time, std_time = benchmark_function(
            compress_delta_sparse,
            delta,
            threshold=1e-6,
            num_runs=3
        )
        
        print(f"  Time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Throughput: {size/avg_time/1e6:.2f} M elements/sec")


def benchmark_sparse_decompression():
    """Benchmark sparse delta decompression."""
    print("\n" + "="*60)
    print("SPARSE DELTA DECOMPRESSION BENCHMARK")
    print("="*60)
    
    sizes = [
        (1_000_000, "1M parameters"),
        (10_000_000, "10M parameters"),
        (50_000_000, "50M parameters"),
    ]
    
    for size, desc in sizes:
        print(f"\n{desc} ({size:,} elements):")
        
        # Create sparse data
        delta = torch.randn(size, dtype=torch.float32)
        delta[torch.abs(delta) < 1.5] = 0
        indices, values, _ = compress_delta_sparse(delta, threshold=1e-6)
        
        print(f"  Non-zero elements: {len(values):,}")
        
        # Benchmark decompression
        avg_time, std_time = benchmark_function(
            decompress_delta_sparse,
            indices,
            values,
            (size,),
            num_runs=3
        )
        
        print(f"  Time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Throughput: {size/avg_time/1e6:.2f} M elements/sec")


def benchmark_int8_quantization():
    """Benchmark INT8 quantization."""
    print("\n" + "="*60)
    print("INT8 QUANTIZATION BENCHMARK")
    print("="*60)
    
    sizes = [
        (1_000_000, "1M parameters"),
        (10_000_000, "10M parameters"),
        (50_000_000, "50M parameters"),
    ]
    
    for size, desc in sizes:
        print(f"\n{desc} ({size:,} elements):")
        
        # Create delta
        delta = torch.randn(size, dtype=torch.float32)
        
        # Benchmark quantization
        avg_time, std_time = benchmark_function(
            compress_delta_int8,
            delta,
            num_runs=3
        )
        
        print(f"  Quantization time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Throughput: {size/avg_time/1e6:.2f} M elements/sec")
        
        # Get quantized data for dequantization benchmark
        quantized_bytes, scale, _ = compress_delta_int8(delta)
        
        # Benchmark dequantization
        avg_time, std_time = benchmark_function(
            decompress_delta_int8,
            quantized_bytes,
            scale,
            (size,),
            num_runs=3
        )
        
        print(f"  Dequantization time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Throughput: {size/avg_time/1e6:.2f} M elements/sec")


def print_system_info():
    """Print system information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    info = get_rust_info()
    print(f"\n✓ Rust acceleration: REQUIRED AND ENABLED")
    print(f"  Version: {info['version']}")
    print(f"  Features: {', '.join(info['features'])}")


def main():
    """Run all benchmarks."""
    print_system_info()
    
    benchmark_sparse_compression()
    benchmark_sparse_decompression()
    benchmark_int8_quantization()
    
    print("\n" + "="*60)
    print("BENCHMARKS COMPLETE")
    print("="*60)
    
    print("\n✓ Rust acceleration is enabled!")


if __name__ == "__main__":
    main()
