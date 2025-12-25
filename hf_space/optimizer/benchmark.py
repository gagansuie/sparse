"""
Sparse Optimizer - Hardware Benchmarking

Measures latency, throughput, and memory usage for compression candidates.
"""

import time
import gc
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Results from benchmarking a compression candidate."""
    candidate_name: str
    hardware: str
    
    # Compression metrics
    compression_ratio: float
    ppl_baseline: float
    ppl_compressed: float
    ppl_delta_pct: float
    
    # Latency metrics (milliseconds)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    
    # Throughput metrics
    throughput_tps: float  # tokens per second
    
    # Memory metrics
    memory_peak_gb: float
    memory_model_gb: float
    
    # Cost estimate
    cost_per_1m_tokens: float
    
    # Metadata
    benchmark_time: datetime = field(default_factory=datetime.utcnow)
    num_samples: int = 100
    warmup_samples: int = 10
    
    def passes_constraints(
        self,
        max_ppl_delta: float = 2.0,
        max_latency_p99_ms: float = 100.0,
        min_throughput_tps: float = 1000.0,
    ) -> bool:
        """Check if this result passes the given constraints."""
        return (
            self.ppl_delta_pct <= max_ppl_delta and
            self.latency_p99_ms <= max_latency_p99_ms and
            self.throughput_tps >= min_throughput_tps
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "candidate_name": self.candidate_name,
            "hardware": self.hardware,
            "compression_ratio": self.compression_ratio,
            "ppl_baseline": self.ppl_baseline,
            "ppl_compressed": self.ppl_compressed,
            "ppl_delta_pct": self.ppl_delta_pct,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_mean_ms": self.latency_mean_ms,
            "throughput_tps": self.throughput_tps,
            "memory_peak_gb": self.memory_peak_gb,
            "memory_model_gb": self.memory_model_gb,
            "cost_per_1m_tokens": self.cost_per_1m_tokens,
            "benchmark_time": self.benchmark_time.isoformat(),
            "num_samples": self.num_samples,
        }


# Hardware cost estimates ($/hour for inference)
HARDWARE_COSTS = {
    "a10g": 1.00,      # AWS g5 instances
    "a100_40": 3.50,   # AWS p4d instances
    "a100_80": 5.00,   # AWS p4de instances
    "t4": 0.50,        # AWS g4dn instances
    "l4": 0.80,        # GCP L4 instances
    "h100": 8.00,      # H100 instances
    "cpu": 0.10,       # CPU-only inference
    "cuda": 1.00,      # Generic CUDA (assume A10G pricing)
}


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
    return info


def estimate_hardware_type() -> str:
    """Estimate the hardware type based on GPU name."""
    if not torch.cuda.is_available():
        return "cpu"
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    if "a100" in gpu_name:
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "a100_80" if mem > 50 else "a100_40"
    elif "a10" in gpu_name:
        return "a10g"
    elif "t4" in gpu_name:
        return "t4"
    elif "l4" in gpu_name:
        return "l4"
    elif "h100" in gpu_name:
        return "h100"
    else:
        return "cuda"  # Generic CUDA


def benchmark_inference(
    model: nn.Module,
    tokenizer,
    num_samples: int = 100,
    warmup_samples: int = 10,
    input_length: int = 128,
    output_length: int = 32,
    batch_size: int = 1,
) -> Tuple[List[float], float, float]:
    """Benchmark inference latency and throughput.
    
    Args:
        model: The model to benchmark
        tokenizer: Tokenizer for the model
        num_samples: Number of inference samples to run
        warmup_samples: Number of warmup samples (not counted)
        input_length: Input sequence length
        output_length: Number of tokens to generate
        batch_size: Batch size for inference
        
    Returns:
        Tuple of (latencies_ms, throughput_tps, memory_peak_gb)
    """
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randint(
        0, tokenizer.vocab_size,
        (batch_size, input_length),
        device=device
    )
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    latencies = []
    total_tokens = 0
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_samples):
            _ = model.generate(
                dummy_input,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    start_total = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_samples):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            outputs = model.generate(
                dummy_input,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # Convert to ms
            total_tokens += outputs.shape[1] * batch_size
    
    end_total = time.perf_counter()
    
    # Calculate throughput
    total_time_s = end_total - start_total
    throughput_tps = total_tokens / total_time_s
    
    # Get peak memory
    if torch.cuda.is_available():
        memory_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        memory_peak_gb = 0.0
    
    return latencies, throughput_tps, memory_peak_gb


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    import numpy as np
    
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def estimate_cost(
    throughput_tps: float,
    hardware: str,
) -> float:
    """Estimate cost per 1M tokens.
    
    Args:
        throughput_tps: Tokens per second
        hardware: Hardware type
        
    Returns:
        Cost in USD per 1M tokens
    """
    hourly_cost = HARDWARE_COSTS.get(hardware, 1.0)
    
    # Tokens per hour
    tokens_per_hour = throughput_tps * 3600
    
    # Cost per 1M tokens
    if tokens_per_hour > 0:
        cost_per_1m = (hourly_cost / tokens_per_hour) * 1_000_000
    else:
        cost_per_1m = float('inf')
    
    return cost_per_1m


def benchmark_candidate(
    model,
    tokenizer,
    candidate_name: str,
    compression_ratio: float,
    ppl_baseline: float,
    ppl_compressed: float,
    hardware: Optional[str] = None,
    num_samples: int = 50,
    warmup_samples: int = 5,
) -> BenchmarkResult:
    """Run full benchmark for a compression candidate.
    
    Args:
        model: Compressed model to benchmark
        tokenizer: Tokenizer
        candidate_name: Name of the candidate
        compression_ratio: Achieved compression ratio
        ppl_baseline: Baseline perplexity
        ppl_compressed: Compressed model perplexity
        hardware: Hardware type (auto-detected if None)
        num_samples: Number of benchmark samples
        warmup_samples: Number of warmup samples
        
    Returns:
        BenchmarkResult with all metrics
    """
    if hardware is None:
        hardware = estimate_hardware_type()
    
    # Run inference benchmark
    latencies, throughput_tps, memory_peak_gb = benchmark_inference(
        model=model,
        tokenizer=tokenizer,
        num_samples=num_samples,
        warmup_samples=warmup_samples,
    )
    
    # Calculate percentiles
    percentiles = calculate_percentiles(latencies)
    
    # Calculate PPL delta
    ppl_delta_pct = ((ppl_compressed - ppl_baseline) / ppl_baseline) * 100
    
    # Estimate cost
    cost_per_1m = estimate_cost(throughput_tps, hardware)
    
    # Get model memory
    model_params = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_model_gb = model_params / 1e9
    
    return BenchmarkResult(
        candidate_name=candidate_name,
        hardware=hardware,
        compression_ratio=compression_ratio,
        ppl_baseline=ppl_baseline,
        ppl_compressed=ppl_compressed,
        ppl_delta_pct=ppl_delta_pct,
        latency_p50_ms=percentiles["p50"],
        latency_p95_ms=percentiles["p95"],
        latency_p99_ms=percentiles["p99"],
        latency_mean_ms=percentiles["mean"],
        throughput_tps=throughput_tps,
        memory_peak_gb=memory_peak_gb,
        memory_model_gb=memory_model_gb,
        cost_per_1m_tokens=cost_per_1m,
        num_samples=num_samples,
    )


def quick_benchmark(
    model,
    tokenizer,
    candidate_name: str = "unknown",
    hardware: Optional[str] = None,
) -> Dict[str, float]:
    """Quick benchmark with fewer samples for fast iteration.
    
    Returns dict with latency_ms, throughput_tps, memory_gb
    """
    if hardware is None:
        hardware = estimate_hardware_type()
    
    latencies, throughput_tps, memory_peak_gb = benchmark_inference(
        model=model,
        tokenizer=tokenizer,
        num_samples=10,
        warmup_samples=2,
        output_length=16,
    )
    
    return {
        "latency_ms": sum(latencies) / len(latencies),
        "throughput_tps": throughput_tps,
        "memory_gb": memory_peak_gb,
        "hardware": hardware,
    }
