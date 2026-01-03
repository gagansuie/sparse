"""
Sparse Core Module

Delta compression for fine-tuned models and datasets.

Usage:
    from core import compress_delta, reconstruct_from_delta
    
    # Compress a fine-tune as delta from base
    manifest = compress_delta(
        base_model_id="meta-llama/Llama-2-7b-hf",
        finetune_model_id="my-finetune",
        output_path="./delta"
    )
    
    # Reconstruct model from base + delta
    model = reconstruct_from_delta(
        base_model_id="meta-llama/Llama-2-7b-hf",
        delta_path="./delta"
    )
"""

from .delta import (
    compress_delta,
    compress_adapter_delta,
    validate_int8_delta_quality,
    reconstruct_from_delta,
    DeltaManifest,
    # SVD compression (LoRA-equivalent extraction)
    compress_delta_svd_full,
    reconstruct_from_svd_delta,
    SVDDeltaManifest,
)

from .dataset_delta import (
    compress_dataset_delta,
    reconstruct_from_dataset_delta,
    estimate_dataset_delta_savings,
    DatasetDeltaStats,
)

from .fast_reconstruct import (
    DeltaCache,
    from_pretrained_with_delta,
    get_global_cache,
)

# Import version from pyproject.toml (single source of truth)
try:
    from importlib.metadata import version
    __version__ = version("sparse-llm")
except Exception:
    __version__ = "0.0.0"  # Fallback for development
__all__ = [
    # Model delta compression (lossless)
    "compress_delta",
    "compress_adapter_delta",
    "validate_int8_delta_quality",
    "reconstruct_from_delta",
    "DeltaManifest",
    # SVD compression (lossy, LoRA-equivalent)
    "compress_delta_svd_full",
    "reconstruct_from_svd_delta",
    "SVDDeltaManifest",
    # Dataset delta compression
    "compress_dataset_delta",
    "reconstruct_from_dataset_delta",
    "estimate_dataset_delta_savings",
    "DatasetDeltaStats",
    # Fast reconstruction
    "DeltaCache",
    "from_pretrained_with_delta",
    "get_global_cache",
]
