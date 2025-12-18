"""
TenPak Optimizer - Automatic Cost-per-Token Optimization

Auto-benchmark compression candidates and select the cheapest one
meeting quality/latency constraints.

Value: $300M-$1B/yr in inference cost savings

Usage:
    from optimizer import optimize_model, OptimizationResult
    
    result = optimize_model(
        model_id="mistralai/Mistral-7B-v0.1",
        hardware="a10g",
        constraints={
            "max_ppl_delta": 2.0,
            "max_latency_p99_ms": 100,
            "min_throughput_tps": 1000
        }
    )
    
    print(f"Winner: {result.winner.codec} @ {result.winner.cost_per_1m_tokens}")
"""

from .candidates import (
    CompressionCandidate,
    generate_candidates,
    CANDIDATE_PRESETS,
)

from .benchmark import (
    BenchmarkResult,
    benchmark_candidate,
    benchmark_inference,
)

from .selector import (
    OptimizationConstraints,
    OptimizationResult,
    select_optimal,
    optimize_model,
)

__all__ = [
    "CompressionCandidate",
    "generate_candidates",
    "CANDIDATE_PRESETS",
    "BenchmarkResult",
    "benchmark_candidate",
    "benchmark_inference",
    "OptimizationConstraints",
    "OptimizationResult",
    "select_optimal",
    "optimize_model",
]
