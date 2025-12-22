"""
TenPak Optimizer - Candidate Generation

Generates compression candidates for optimization.
Uses industry-standard quantization tools (AutoGPTQ, AutoAWQ, bitsandbytes).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class QuantizationMethod(str, Enum):
    """Available quantization methods (wrapping existing tools)."""
    GPTQ = "gptq"
    AWQ = "awq"
    BITSANDBYTES = "bitsandbytes"
    FP16 = "fp16"  # Baseline


@dataclass
class CompressionCandidate:
    """A compression candidate to benchmark."""
    name: str
    method: QuantizationMethod
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Expected characteristics (for filtering before benchmark)
    expected_compression: float = 1.0
    expected_ppl_delta: float = 0.0
    requires_calibration: bool = False
    
    # Results (populated after benchmark)
    actual_compression: Optional[float] = None
    actual_ppl_delta: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_tps: Optional[float] = None
    memory_gb: Optional[float] = None
    cost_per_1m_tokens: Optional[float] = None
    
    def __repr__(self):
        return f"Candidate({self.name}, {self.method.value}, comp={self.expected_compression}x)"


# Pre-defined candidate configurations using industry-standard tools
CANDIDATE_PRESETS: Dict[str, CompressionCandidate] = {
    # Baseline
    "fp16": CompressionCandidate(
        name="FP16 Baseline",
        method=QuantizationMethod.FP16,
        config={},
        expected_compression=1.0,
        expected_ppl_delta=0.0,
        requires_calibration=False,
    ),
    
    # GPTQ variants (AutoGPTQ)
    "gptq_quality": CompressionCandidate(
        name="GPTQ 4-bit g=128 (Quality)",
        method=QuantizationMethod.GPTQ,
        config={"bits": 4, "group_size": 128, "desc_act": False, "sym": True},
        expected_compression=7.5,
        expected_ppl_delta=1.0,
        requires_calibration=True,
    ),
    "gptq_balanced": CompressionCandidate(
        name="GPTQ 4-bit g=256 (Balanced)",
        method=QuantizationMethod.GPTQ,
        config={"bits": 4, "group_size": 256, "desc_act": False, "sym": True},
        expected_compression=7.8,
        expected_ppl_delta=1.5,
        requires_calibration=True,
    ),
    "gptq_size": CompressionCandidate(
        name="GPTQ 4-bit g=512 (Size)",
        method=QuantizationMethod.GPTQ,
        config={"bits": 4, "group_size": 512, "desc_act": False, "sym": True},
        expected_compression=8.0,
        expected_ppl_delta=2.5,
        requires_calibration=True,
    ),
    
    # AWQ variants (AutoAWQ)
    "awq_quality": CompressionCandidate(
        name="AWQ 4-bit g=128 (Quality)",
        method=QuantizationMethod.AWQ,
        config={"bits": 4, "group_size": 128, "zero_point": True},
        expected_compression=7.5,
        expected_ppl_delta=0.8,
        requires_calibration=True,
    ),
    "awq_balanced": CompressionCandidate(
        name="AWQ 4-bit g=256 (Balanced)",
        method=QuantizationMethod.AWQ,
        config={"bits": 4, "group_size": 256, "zero_point": True},
        expected_compression=7.8,
        expected_ppl_delta=1.2,
        requires_calibration=True,
    ),
    
    # bitsandbytes variants
    "bnb_int8": CompressionCandidate(
        name="bitsandbytes INT8",
        method=QuantizationMethod.BITSANDBYTES,
        config={"bits": 8, "llm_int8_threshold": 6.0},
        expected_compression=2.0,
        expected_ppl_delta=0.3,
        requires_calibration=False,
    ),
    "bnb_nf4": CompressionCandidate(
        name="bitsandbytes NF4",
        method=QuantizationMethod.BITSANDBYTES,
        config={"bits": 4},
        expected_compression=7.5,
        expected_ppl_delta=1.0,
        requires_calibration=False,
    ),
}


def generate_candidates(
    include_calibration: bool = True,
    max_expected_ppl_delta: float = 5.0,
    min_expected_compression: float = 1.0,
    specific_candidates: Optional[List[str]] = None,
) -> List[CompressionCandidate]:
    """Generate a list of compression candidates for benchmarking.
    
    Args:
        include_calibration: Whether to include candidates that require calibration
        max_expected_ppl_delta: Filter out candidates with expected PPL delta above this
        min_expected_compression: Filter out candidates with expected compression below this
        specific_candidates: If provided, only include these specific candidate names
        
    Returns:
        List of CompressionCandidate objects to benchmark
    """
    candidates = []
    
    for name, candidate in CANDIDATE_PRESETS.items():
        # Filter by specific list
        if specific_candidates and name not in specific_candidates:
            continue
            
        # Filter by calibration requirement
        if not include_calibration and candidate.requires_calibration:
            continue
            
        # Filter by expected PPL delta
        if candidate.expected_ppl_delta > max_expected_ppl_delta:
            continue
            
        # Filter by expected compression
        if candidate.expected_compression < min_expected_compression:
            continue
            
        # Create a copy to avoid mutating the preset
        candidates.append(CompressionCandidate(
            name=candidate.name,
            method=candidate.method,
            config=candidate.config.copy(),
            expected_compression=candidate.expected_compression,
            expected_ppl_delta=candidate.expected_ppl_delta,
            requires_calibration=candidate.requires_calibration,
        ))
    
    # Sort by expected compression (highest first)
    candidates.sort(key=lambda c: c.expected_compression, reverse=True)
    
    return candidates


def generate_custom_candidates(
    methods: List[str],
    group_sizes: List[int] = [128, 256, 512],
    include_baseline: bool = True,
) -> List[CompressionCandidate]:
    """Generate custom candidates from method + group size combinations.
    
    Args:
        methods: List of method names ("awq", "int4", "residual")
        group_sizes: List of group sizes to try
        include_baseline: Whether to include FP16 baseline
        
    Returns:
        List of CompressionCandidate objects
    """
    candidates = []
    
    if include_baseline:
        candidates.append(CANDIDATE_PRESETS["fp16"])
    
    for method in methods:
        for g in group_sizes:
            if method == "awq":
                # Estimate compression and PPL based on group size
                compression = 16 / (4 + 16/g)  # INT4 + scales overhead (vs FP16 baseline)
                ppl_delta = 0.5 + (g / 256)  # Larger groups = more PPL
                
                candidates.append(CompressionCandidate(
                    name=f"AWQ g={g}",
                    method=CompressionMethod.INT4_AWQ,
                    config={"group_size": g, "outlier_pct": 0.5, "iterations": 5},
                    expected_compression=compression,
                    expected_ppl_delta=ppl_delta,
                    requires_calibration=True,
                ))
            elif method == "int4":
                compression = 16 / (4 + 16/g)
                ppl_delta = 1.0 + (g / 128)
                
                candidates.append(CompressionCandidate(
                    name=f"INT4 g={g}",
                    method=CompressionMethod.INT4_OPT,
                    config={"group_size": g, "iterations": 5},
                    expected_compression=compression,
                    expected_ppl_delta=ppl_delta,
                    requires_calibration=False,
                ))
    
    return candidates
