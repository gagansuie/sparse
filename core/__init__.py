"""
Sparse Core Module

Quantization orchestration, calibration, delta compression, and optimization.

Usage:
    from core import QuantizationWrapper, QUANTIZATION_PRESETS
    
    # Quantize a model
    wrapper = QuantizationWrapper.from_preset("gptq_quality")
    quantized_model = wrapper.quantize(model_id)
"""

# Quantization wrapper (recommended approach)
from .quantization import (
    QuantizationWrapper,
    QuantizationConfig,
    QuantizationMethod,
    QUANTIZATION_PRESETS,
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
    compress_adapter_delta,
    validate_int8_delta_quality,
    reconstruct_from_delta,
    estimate_delta_savings,
    DeltaManifest,
)

from .fast_reconstruct import (
    DeltaCache,
    from_pretrained_with_delta,
    get_global_cache,
)

__version__ = "0.2.0"
__all__ = [
    # Quantization wrapper
    "QuantizationWrapper",
    "QuantizationConfig",
    "QuantizationMethod",
    "QUANTIZATION_PRESETS",
    # Calibration
    "collect_calibration_stats",
    "compute_ppl",
    # Allocation
    "allocate_bits",
    "LayerAllocation",
    # Delta compression
    "compress_delta",
    "compress_adapter_delta",
    "validate_int8_delta_quality",
    "reconstruct_from_delta",
    "estimate_delta_savings",
    "DeltaManifest",
    # Fast reconstruction
    "DeltaCache",
    "from_pretrained_with_delta",
    "get_global_cache",
]
