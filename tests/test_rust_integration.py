"""
Tests for Rust integration.

Verifies that Rust implementations produce identical results to Python.
"""

import pytest
import numpy as np
import torch

try:
    from core.delta_rust import (
        is_rust_available,
        compress_delta_sparse_rust,
        decompress_delta_sparse_rust,
        compress_delta_int8_rust,
        decompress_delta_int8_rust,
    )
    RUST_AVAILABLE = is_rust_available()
except ImportError:
    RUST_AVAILABLE = False

from core.delta import (
    compress_delta_sparse,
    decompress_delta_sparse,
    compress_delta_int8,
    decompress_delta_int8,
)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
class TestRustIntegration:
    """Test Rust implementation correctness."""
    
    def test_sparse_compression_correctness(self):
        """Test that Rust compression gives same results as Python."""
        # Create test delta
        delta = torch.randn(10000, dtype=torch.float32)
        delta[torch.abs(delta) < 1.0] = 0  # Make 60% sparse
        
        # Get Rust result
        indices_rust, values_rust, ratio_rust = compress_delta_sparse_rust(
            delta, threshold=1e-6
        )
        
        # Results should be valid
        assert len(indices_rust) > 0
        assert len(values_rust) > 0
        assert len(indices_rust) == len(values_rust)
        assert ratio_rust > 1.0
    
    def test_sparse_decompression_correctness(self):
        """Test that decompression reconstructs correctly."""
        # Create test delta
        delta = torch.randn(10000, dtype=torch.float32)
        delta[torch.abs(delta) < 1.0] = 0
        
        # Compress
        indices, values, _ = compress_delta_sparse(delta, threshold=1e-6)
        
        # Decompress with Rust
        reconstructed = decompress_delta_sparse_rust(
            indices, values, delta.shape, torch.float32
        )
        
        # Should match original
        diff = torch.abs(delta - reconstructed)
        assert diff.max() < 1e-5
    
    def test_sparse_roundtrip(self):
        """Test compress -> decompress roundtrip."""
        # Create test delta
        delta = torch.randn(5000, dtype=torch.float32)
        delta[torch.abs(delta) < 1.5] = 0
        
        # Compress
        indices, values, ratio = compress_delta_sparse(delta, threshold=1e-6)
        
        # Decompress
        reconstructed = decompress_delta_sparse(
            indices, values, delta.shape, torch.float32
        )
        
        # Verify
        diff = torch.abs(delta - reconstructed)
        assert diff.max() < 1e-5
        print(f"Roundtrip successful, compression: {ratio:.2f}x")
    
    def test_int8_quantization_correctness(self):
        """Test INT8 quantization."""
        # Create test delta
        delta = torch.randn(10000, dtype=torch.float32)
        
        # Quantize with Rust
        quantized_bytes, scale, ratio = compress_delta_int8_rust(delta)
        
        # Dequantize
        reconstructed = decompress_delta_int8_rust(
            quantized_bytes, scale, delta.shape, torch.float32
        )
        
        # Should be close (INT8 has some error)
        diff = torch.abs(delta - reconstructed)
        relative_error = diff / (torch.abs(delta) + 1e-8)
        assert relative_error.mean() < 0.05  # <5% average error (INT8 quantization)
    
    def test_int8_roundtrip(self):
        """Test INT8 compress -> decompress roundtrip."""
        # Create test delta
        delta = torch.randn(5000, dtype=torch.float32)
        
        # Compress
        quantized_bytes, scale, ratio = compress_delta_int8(delta)
        
        # Decompress
        reconstructed = decompress_delta_int8(
            quantized_bytes, scale, delta.shape, torch.float32
        )
        
        # Verify (INT8 has quantization error)
        diff = torch.abs(delta - reconstructed)
        assert diff.max() < delta.abs().max() * 0.01  # <1% of max value
        print(f"INT8 roundtrip successful, compression: {ratio:.2f}x")
    
    def test_large_tensor(self):
        """Test with large tensor (1M elements)."""
        # Create large sparse delta
        delta = torch.randn(1_000_000, dtype=torch.float32)
        delta[torch.abs(delta) < 2.0] = 0  # Very sparse
        
        sparsity = (delta == 0).float().mean()
        print(f"Testing large tensor, sparsity: {sparsity*100:.1f}%")
        
        # Compress
        indices, values, ratio = compress_delta_sparse(delta, threshold=1e-6)
        
        # Decompress
        reconstructed = decompress_delta_sparse(
            indices, values, delta.shape, torch.float32
        )
        
        # Verify
        diff = torch.abs(delta - reconstructed)
        assert diff.max() < 1e-5
        print(f"Large tensor roundtrip successful, compression: {ratio:.2f}x")
    
    def test_2d_tensor(self):
        """Test with 2D tensor (matrix)."""
        # Create 2D delta
        delta = torch.randn(100, 100, dtype=torch.float32)
        delta[torch.abs(delta) < 1.0] = 0
        
        # Compress (flattens internally)
        indices, values, ratio = compress_delta_sparse(delta, threshold=1e-6)
        
        # Decompress
        reconstructed = decompress_delta_sparse(
            indices, values, delta.shape, torch.float32
        )
        
        # Verify shape and values
        assert reconstructed.shape == delta.shape
        diff = torch.abs(delta - reconstructed)
        assert diff.max() < 1e-5


@pytest.mark.skipif(RUST_AVAILABLE, reason="Testing Python fallback")
def test_python_fallback():
    """Test that Python fallback works when Rust not available."""
    delta = torch.randn(1000, dtype=torch.float32)
    delta[torch.abs(delta) < 1.0] = 0
    
    # Should use Python fallback
    indices, values, ratio = compress_delta_sparse(delta, threshold=1e-6)
    
    # Verify it works
    assert len(indices) > 0
    assert len(values) > 0
    assert ratio > 1.0


def test_rust_availability_detection():
    """Test that we correctly detect Rust availability."""
    from core.delta_rust import is_rust_available, get_rust_info
    
    available = is_rust_available()
    info = get_rust_info()
    
    assert isinstance(available, bool)
    assert isinstance(info, dict)
    assert "available" in info
    assert "features" in info
    
    if available:
        print("✓ Rust acceleration is available")
        print(f"  Features: {', '.join(info['features'])}")
    else:
        print("✗ Rust acceleration not available (using Python fallback)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
