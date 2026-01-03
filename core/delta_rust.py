"""
Sparse Core - Hybrid Rust/Python Delta Compression

Single source of truth for all delta compression operations.
Uses Rust for performance-critical operations, with minimal Python
for tensor handling. NO duplicate implementations.

Architecture:
- Rust: Heavy computation (sparse search, quantization, parallel ops)
- Python: Tensor conversion, shape handling, PyTorch integration
"""

import numpy as np
import torch
from typing import Tuple

# Rust implementation is REQUIRED - no Python fallback
try:
    import sparse_core
except ImportError as e:
    raise ImportError(
        "Rust acceleration (sparse_core) is required but not installed. "
        "Install with: pip install sparse-llm"
    ) from e


# =============================================================================
# HYBRID SPARSE COMPRESSION
# =============================================================================

def compress_delta_sparse(
    delta: torch.Tensor,
    threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compress delta using sparse representation.
    
    HYBRID: Uses Rust for finding non-zero elements (parallel),
    Python for tensor conversion.
    
    Args:
        delta: Delta tensor
        threshold: Threshold for considering values as zero
        
    Returns:
        Tuple of (indices, values, compression_ratio)
    """
    flat_delta = delta.flatten()
    original_size = delta.numel() * 2  # FP16 baseline
    
    # Rust: Fast parallel sparse search
    delta_np = flat_delta.cpu().numpy().astype(np.float32)
    indices_np, values_np = sparse_core.compress_sparse_delta(
        delta_np, threshold=threshold, parallel=True
    )
    indices = torch.from_numpy(indices_np).to(torch.int32)
    values = torch.from_numpy(values_np).to(delta.dtype)
    
    # Compression ratio calculation (shared)
    compressed_size = indices.numel() * 4 + values.numel() * 2
    compression_ratio = max(original_size / max(compressed_size, 1), 1.0)
    
    return indices, values, compression_ratio


def decompress_delta_sparse(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Decompress sparse delta back to full tensor.
    
    HYBRID: Uses Rust for scatter operation, Python for tensor handling.
    """
    flat_size = 1
    for dim in shape:
        flat_size *= dim
    
    # Rust: Fast scatter
    indices_np = indices.cpu().numpy().astype(np.uint32)
    values_np = values.cpu().numpy().astype(np.float32)
    delta_np = sparse_core.decompress_sparse_delta(indices_np, values_np, flat_size)
    delta = torch.from_numpy(delta_np).to(dtype).reshape(shape)
    
    return delta


# =============================================================================
# HYBRID INT8 QUANTIZATION
# =============================================================================

def compress_delta_int8(
    delta: torch.Tensor,
) -> Tuple[bytes, float, float]:
    """
    Compress delta using INT8 quantization.
    
    HYBRID: Uses Rust for parallel quantization, Python for tensor conversion.
    
    Returns:
        Tuple of (quantized_bytes, scale, compression_ratio)
    """
    original_size = delta.numel() * 2  # FP16 baseline
    
    # Rust: Fast parallel quantization
    delta_np = delta.cpu().numpy().astype(np.float32).flatten()
    quantized_np, scale = sparse_core.quantize_int8(delta_np)
    quantized_bytes = quantized_np.tobytes()
    
    compressed_size = len(quantized_bytes) + 4
    compression_ratio = original_size / compressed_size
    
    return quantized_bytes, scale, compression_ratio


def decompress_delta_int8(
    quantized_bytes: bytes,
    scale: float,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Decompress INT8 delta back to full tensor.
    
    HYBRID: Uses Rust for dequantization, Python for tensor handling.
    """
    quantized_np = np.frombuffer(quantized_bytes, dtype=np.int8).copy()
    
    # Rust: Fast dequantization
    delta_np = sparse_core.dequantize_int8(quantized_np, scale)
    delta = torch.from_numpy(delta_np).to(dtype).reshape(shape)
    
    return delta


# =============================================================================
# STATUS & INFO
# =============================================================================

def get_rust_info() -> dict:
    """Get information about Rust implementation."""
    return {
        "available": True,
        "version": "0.1.0",
        "features": [
            "sparse_compression",
            "int8_quantization", 
            "parallel_processing",
        ],
    }


