"""
Tests for Rust integration.

Rust acceleration is REQUIRED - these tests verify the Rust implementation works correctly.
"""

import pytest
import numpy as np
import torch

# Rust is required - import will fail if not available
from core.delta_rust import (
    compress_delta_sparse,
    decompress_delta_sparse,
    compress_delta_int8,
    decompress_delta_int8,
    get_rust_info,
)


class TestRustIntegration:
    """Test Rust implementation correctness."""
    
    def test_sparse_compression_correctness(self):
        """Test that sparse compression works correctly."""
        # Create test delta
        delta = torch.randn(10000, dtype=torch.float32)
        delta[torch.abs(delta) < 1.0] = 0  # Make 60% sparse
        
        # Compress
        indices, values, ratio = compress_delta_sparse(delta, threshold=1e-6)
        
        # Results should be valid
        assert len(indices) > 0
        assert len(values) > 0
        assert len(indices) == len(values)
        assert ratio > 1.0
    
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
        assert diff.max() < delta.abs().max() * 0.02  # <2% of max value
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


def test_rust_info():
    """Test that Rust info is available."""
    info = get_rust_info()
    
    assert isinstance(info, dict)
    assert info["available"] == True
    assert "features" in info
    assert len(info["features"]) > 0
    
    print("✓ Rust acceleration is available")
    print(f"  Features: {', '.join(info['features'])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
