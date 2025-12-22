"""
TenPak Core Module

Quantization orchestration, calibration, delta compression, and optimization.

Usage:
    from core import QuantizationWrapper, QUANTIZATION_PRESETS
    
    # Quantize a model
    wrapper = QuantizationWrapper(method="gptq", bits=4)
    quantized_model = wrapper.quantize(model_id)
    
    # Batch compression to JSON
    artifact_json, ratio = compress_tensors_f32_json(tensors, codec, names=names)
"""

# Quantization wrapper (recommended approach)
from .quantization import (
    QuantizationWrapper,
    QuantizationConfig,
    QuantizationMethod,
    QUANTIZATION_PRESETS,
)

# Legacy Rust FFI (deprecated - use QuantizationWrapper instead)
# Kept for backward compatibility
from .native_ffi import (
    roundtrip_tensor_f32,
    compress_tensors_f32_json,
    decompress_artifact_to_json,
    decompress_artifact_to_tensors,
)

# Legacy codec constants (deprecated)
CODEC_INT4_RESIDUAL = "int4_residual_v1"
CODEC_INT4_OPT_LLAMA = "int4_opt_llama_v1"

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

__version__ = "0.2.0"
__all__ = [
    # Quantization wrapper (recommended)
    "QuantizationWrapper",
    "QuantizationConfig",
    "QuantizationMethod",
    "QUANTIZATION_PRESETS",
    # Legacy Rust FFI (deprecated)
    "roundtrip_tensor_f32",
    "compress_tensors_f32_json",
    "decompress_artifact_to_json",
    "decompress_artifact_to_tensors",
    "CODEC_INT4_RESIDUAL",
    "CODEC_INT4_OPT_LLAMA",
    # Calibration
    "collect_calibration_stats",
    "compute_ppl",
    # Allocation
    "allocate_bits",
    "LayerAllocation",
    # Delta compression
    "compress_delta",
    "reconstruct_from_delta",
    "estimate_delta_savings",
    "DeltaManifest",
]
