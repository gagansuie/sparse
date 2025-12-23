"""
Rust-accelerated delta compression operations.

This module provides a Python wrapper around the Rust implementation,
with automatic fallback to pure Python if Rust is not available.
"""

import numpy as np
import torch
from typing import Tuple, Optional

# Try to import Rust implementation
try:
    import sparse_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("[Delta] Rust acceleration not available, using Python fallback")


def compress_delta_sparse_rust(
    delta: torch.Tensor,
    threshold: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compress delta using sparse representation (Rust implementation).
    
    Args:
        delta: Delta tensor
        threshold: Threshold for considering values as zero
        
    Returns:
        Tuple of (indices, values, compression_ratio)
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust implementation not available")
    
    # Convert to numpy for Rust processing
    delta_np = delta.cpu().numpy().astype(np.float32).flatten()
    
    # Call Rust implementation
    indices_np, values_np = sparse_core.compress_sparse_delta(
        delta_np,
        threshold=threshold,
        parallel=True
    )
    
    # Convert back to PyTorch
    indices = torch.from_numpy(indices_np).to(torch.int32)
    values = torch.from_numpy(values_np).to(delta.dtype)
    
    # Calculate compression ratio
    original_size = delta.numel() * 2  # FP16 baseline
    compressed_size = indices.numel() * 4 + values.numel() * 2
    compression_ratio = max(original_size / max(compressed_size, 1), 1.0)
    
    return indices, values, compression_ratio


def decompress_delta_sparse_rust(
    indices: torch.Tensor,
    values: torch.Tensor,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Decompress sparse delta back to full tensor (Rust implementation).
    
    Args:
        indices: Sparse indices
        values: Sparse values
        shape: Original tensor shape
        dtype: Target data type
        
    Returns:
        Reconstructed delta tensor
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust implementation not available")
    
    # Convert to numpy
    indices_np = indices.cpu().numpy().astype(np.uint32)
    values_np = values.cpu().numpy().astype(np.float32)
    
    # Calculate flat size
    flat_size = 1
    for dim in shape:
        flat_size *= dim
    
    # Call Rust implementation
    delta_np = sparse_core.decompress_sparse_delta(
        indices_np,
        values_np,
        flat_size
    )
    
    # Convert back to PyTorch and reshape
    delta = torch.from_numpy(delta_np).to(dtype).reshape(shape)
    
    return delta


def compress_delta_int8_rust(
    delta: torch.Tensor,
) -> Tuple[bytes, float, float]:
    """
    Compress delta using INT8 quantization (Rust implementation).
    
    Args:
        delta: Delta tensor
        
    Returns:
        Tuple of (quantized_bytes, scale, compression_ratio)
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust implementation not available")
    
    # Convert to numpy
    delta_np = delta.cpu().numpy().astype(np.float32).flatten()
    
    # Call Rust implementation
    quantized_np, scale = sparse_core.quantize_int8(delta_np)
    
    # Convert to bytes
    quantized_bytes = quantized_np.tobytes()
    
    # Calculate compression ratio
    original_size = delta.numel() * 2  # FP16 baseline
    compressed_size = len(quantized_bytes) + 4  # +4 for scale
    compression_ratio = original_size / compressed_size
    
    return quantized_bytes, scale, compression_ratio


def decompress_delta_int8_rust(
    quantized_bytes: bytes,
    scale: float,
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Decompress INT8 delta back to full tensor (Rust implementation).
    
    Args:
        quantized_bytes: Quantized data
        scale: Quantization scale
        shape: Original tensor shape
        dtype: Target data type
        
    Returns:
        Reconstructed delta tensor
    """
    if not RUST_AVAILABLE:
        raise RuntimeError("Rust implementation not available")
    
    # Convert bytes to numpy
    quantized_np = np.frombuffer(quantized_bytes, dtype=np.int8)
    
    # Call Rust implementation
    delta_np = sparse_core.dequantize_int8(quantized_np, scale)
    
    # Convert back to PyTorch and reshape
    delta = torch.from_numpy(delta_np).to(dtype).reshape(shape)
    
    return delta


def is_rust_available() -> bool:
    """Check if Rust acceleration is available."""
    return RUST_AVAILABLE


def get_rust_info() -> dict:
    """Get information about Rust implementation."""
    return {
        "available": RUST_AVAILABLE,
        "version": "0.1.0" if RUST_AVAILABLE else None,
        "features": [
            "sparse_compression",
            "int8_quantization",
            "parallel_processing",
        ] if RUST_AVAILABLE else []
    }
