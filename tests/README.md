# TenPak Tests

Test suite for TenPak's core features:
- Model Delta Compression
- Dataset Delta Compression (NEW)
- Smart Routing (NEW)
- Cost Optimizer

## Running Tests

### All Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov=optimizer --cov-report=html
```

### Specific Test Files

```bash
# Test model delta compression
pytest tests/test_delta_compression.py -v

# Test dataset delta compression (NEW)
pytest tests/test_dataset_delta.py -v

# Test smart routing (NEW)
pytest tests/test_routing.py -v

# Test cost optimizer
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
| `core/delta.py` | 85%+ | ✅ Complete |
| `core/dataset_delta.py` | 80%+ | ✅ Complete (NEW) |
| `optimizer/routing.py` | 85%+ | ✅ Complete (NEW) |
| `optimizer/candidates.py` | 90%+ | ✅ Complete |

## Test Files

- **`test_delta_compression.py`** - Tests for model delta compression
  - DeltaManifest
  - Savings estimation
  - Compress/reconstruct workflow
  - Integration tests

- **`test_dataset_delta.py`** - Tests for dataset delta compression (NEW)
  - DatasetDeltaStats
  - Savings estimation
  - Compress/reconstruct workflow
  - Mock dataset handling

- **`test_routing.py`** - Tests for smart routing (NEW)
  - Request complexity classification
  - Model recommendations
  - Hardware routing decisions
  - Savings calculations
  - Batching logic

- **`test_optimizer.py`** - Tests for cost optimizer
  - Candidate presets
  - Candidate generation
  - Constraint-based selection
  - Optimization workflow

## Mocking Strategy

Tests use `unittest.mock` to mock external dependencies:
- **Model loading** - Mocked to avoid downloading large models
- **Dataset loading** - Mocked to avoid network calls
- **File I/O** - Mocked for testing delta compression
- **AutoGPTQ/AWQ/bitsandbytes** - Mocked for cost optimizer tests

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

## Archived Tests

The following tests were removed when features were archived:
- ~~`test_quantization.py`~~ - Feature moved to wrapper-only (minimal testing needed)
- ~~`test_http_streaming.py`~~ - Archived to `archive/removed_features/`
- ~~`test_vllm_integration.py`~~ - Archived to `archive/removed_features/`
- ~~`test_artifact_format.py`~~ - Archived to `archive/removed_features/`

## Future Test Coverage

- [ ] End-to-end integration tests with real datasets
- [ ] Performance benchmarks for delta compression
- [ ] Large-scale routing decision tests
- [ ] Real model loading tests (when GPU available)
