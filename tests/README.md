# Sparse Tests

Test suite for Sparse delta compression:
- Model Delta Compression (lossless)
- SVD Compression (LoRA-equivalent)
- Dataset Delta Compression

## Running Tests

### All Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

### Specific Test Files

```bash
# Test model delta compression
pytest tests/test_delta_compression.py -v

# Test dataset delta compression
pytest tests/test_dataset_delta.py -v
```

### Test Markers

```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Skip GPU-required tests
pytest tests/ -m "not requires_gpu"
```

## Test Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| `core/delta.py` | 85%+ | ✅ Complete |
| `core/dataset_delta.py` | 80%+ | ✅ Complete |
| `core/delta_rust.py` | 90%+ | ✅ Complete |

## Test Files

- **`test_delta_compression.py`** - Tests for model delta compression
  - DeltaManifest / SVDDeltaManifest
  - Compress/reconstruct workflow
  - Lossless and SVD modes

- **`test_dataset_delta.py`** - Tests for dataset delta compression
  - DatasetDeltaStats
  - Compress/reconstruct workflow
  - Mock dataset handling

## Mocking Strategy

Tests use `unittest.mock` to mock external dependencies:
- **Model loading** - Mocked to avoid downloading large models
- **Dataset loading** - Mocked to avoid network calls
- **File I/O** - Mocked for testing delta compression

## CI/CD Integration

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
        run: pytest tests/ --cov=core --cov-report=xml
```

## Future Test Coverage

- [ ] SVD compression quality validation tests
- [ ] End-to-end integration tests with real models
- [ ] Performance benchmarks for Rust vs Python
