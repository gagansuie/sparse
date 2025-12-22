"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(10, 10)


@pytest.fixture
def sample_model_config():
    """Sample model configuration."""
    return {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "vocab_size": 50257,
        "num_attention_heads": 12,
    }


@pytest.fixture
def mock_artifact_manifest():
    """Mock artifact manifest structure."""
    return {
        "version": "1.0",
        "model_id": "gpt2",
        "quantization": {
            "method": "awq",
            "bits": 4,
            "group_size": 128,
        },
        "compression_ratio": 7.5,
        "original_size_bytes": 500000000,
        "compressed_size_bytes": 66666666,
        "chunks": [],
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
