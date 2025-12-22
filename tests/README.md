# TenPak Tests

Comprehensive test suite for all TenPak features.

## Running Tests

### All Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=artifact --cov=inference --cov=optimizer --cov-report=html
```

### Specific Test Files

```bash
# Test quantization wrapper
pytest tests/test_quantization.py -v

# Test HTTP streaming
pytest tests/test_http_streaming.py -v

# Test vLLM integration
pytest tests/test_vllm_integration.py -v

# Test artifact format
pytest tests/test_artifact_format.py -v

# Test delta compression
pytest tests/test_delta_compression.py -v

# Test optimizer
pytest tests/test_optimizer.py -v
```

### Test Markers

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m integration

# Skip GPU-required tests
pytest tests/ -m "not requires_gpu"
```

## Test Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| `core/quantization.py` | 95%+ | ✅ Complete |
| `artifact/http_streaming.py` | 90%+ | ✅ Complete |
| `inference/vllm_integration.py` | 85%+ | ✅ Complete |
| `artifact/format.py` | 95%+ | ✅ Complete |
| `core/delta.py` | 85%+ | ✅ Complete |
| `optimizer/candidates.py` | 95%+ | ✅ Complete |

## Test Files

- **`test_quantization.py`** - Tests for quantization wrapper
  - QuantizationConfig
  - QuantizationWrapper methods
  - Presets and size estimation
  - GPTQ/AWQ/bitsandbytes integration

- **`test_http_streaming.py`** - Tests for HTTP streaming
  - HTTPArtifactStreamer
  - Manifest loading/caching
  - Chunk fetching and verification
  - Download functionality

- **`test_vllm_integration.py`** - Tests for inference integration
  - TenPakVLLMLoader
  - TenPakTGILoader
  - Configuration generation
  - Benchmarking

- **`test_artifact_format.py`** - Tests for artifact format
  - ArtifactManifest
  - ChunkInfo
  - Serialization/deserialization
  - Complete artifact structure

- **`test_delta_compression.py`** - Tests for delta compression
  - DeltaManifest
  - Savings estimation
  - Compress/reconstruct workflow

- **`test_optimizer.py`** - Tests for cost optimizer
  - CompressionCandidate
  - Candidate presets
  - Candidate generation
  - Filtering logic

## Mocking Strategy

Tests use `unittest.mock` to mock external dependencies:
- **AutoGPTQ/AutoAWQ/bitsandbytes** - Mocked to avoid CUDA requirements
- **HTTP requests** - Mocked to avoid network calls
- **Model loading** - Mocked to avoid downloading models
- **vLLM/TGI** - Mocked to avoid deployment dependencies

## CI/CD Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Writing New Tests

Follow this pattern:

```python
"""Tests for new_feature"""

import pytest
from unittest.mock import Mock, patch

from module import NewFeature


class TestNewFeature:
    """Test NewFeature class."""
    
    @pytest.fixture
    def feature(self):
        """Create feature instance."""
        return NewFeature()
    
    def test_basic_functionality(self, feature):
        """Test basic operation."""
        result = feature.do_something()
        assert result is not None
    
    @patch("module.external_dependency")
    def test_with_mock(self, mock_dep, feature):
        """Test with mocked dependency."""
        mock_dep.return_value = "mocked"
        result = feature.use_dependency()
        assert result == "mocked"
```

## Future Test Coverage

- [ ] End-to-end integration tests
- [ ] Performance benchmarks
- [ ] Stress tests for streaming
- [ ] GPU-specific tests (when available)
