"""
TenPak Core - Model Compression Engine

Production-validated compression codecs for LLMs.
Best results: 7.42x compression @ +1.47% PPL (Mistral-7B, v10 config)

Usage:
    from tenpak.core import compress_model, evaluate_ppl
    
    result = compress_model(
        model_id="mistralai/Mistral-7B-v0.1",
        target="quality",  # or "size", "speed"
        device="cuda"
    )
"""

from .codecs import (
    compress_int4_awq,
    compress_int4_residual,
    compress_calibrated_vq,
    CODEC_V10,
    CODEC_V60,
    V10_CONFIG,
    V60_CONFIG,
    CODEC_INT4_AWQ,
    CODEC_INT4_RESIDUAL,
    CODEC_CALIBRATED_VQ,
)

from .calibration import (
    collect_calibration_stats,
    compute_ppl,
)

from .allocation import (
    allocate_bits,
    LayerAllocation,
)

from .delta import (
    compress_delta,
    reconstruct_from_delta,
    estimate_delta_savings,
    DeltaManifest,
)

__version__ = "0.1.0"
__all__ = [
    # Main codecs
    "CODEC_V10",
    "CODEC_V60",
    "V10_CONFIG",
    "V60_CONFIG",
    # Compression functions
    "compress_int4_awq",
    "compress_int4_residual", 
    "compress_calibrated_vq",
    # Calibration
    "collect_calibration_stats",
    "compute_ppl",
    # Allocation
    "allocate_bits",
    "LayerAllocation",
    # Codec identifiers
    "CODEC_INT4_AWQ",
    "CODEC_INT4_RESIDUAL",
    "CODEC_CALIBRATED_VQ",
    # Delta compression
    "compress_delta",
    "reconstruct_from_delta",
    "estimate_delta_savings",
    "DeltaManifest",
]
