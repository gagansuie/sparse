# Sparse Core - Rust Implementation

High-performance Rust implementation of performance-critical delta compression operations.

## Features

- **10-20x faster sparse delta compression** using SIMD and parallel processing
- **Zero-copy tensor operations** for maximum efficiency
- **INT8 quantization** with parallel processing
- **PyO3 bindings** for seamless Python integration

## Building

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

### Development Build

```bash
cd rust/
maturin develop --release
```

### Production Build

```bash
cd rust/
maturin build --release
pip install target/wheels/*.whl
```

## Usage

```python
import sparse_core
import numpy as np

# Compress sparse delta
delta = np.random.randn(1000000).astype(np.float32)
delta[np.abs(delta) < 0.1] = 0  # Make sparse

indices, values = sparse_core.compress_sparse_delta(delta, threshold=1e-6)
print(f"Compression: {len(delta) / len(indices):.2f}x")

# Decompress
reconstructed = sparse_core.decompress_sparse_delta(indices, values, len(delta))

# INT8 quantization
data = np.random.randn(1000000).astype(np.float32)
quantized, scale = sparse_core.quantize_int8(data)
dequantized = sparse_core.dequantize_int8(quantized, scale)
```

## Performance

Benchmarks on Llama-7B model (7B parameters):

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Sparse compression | 45s | 2.3s | **19.6x** |
| Decompression | 28s | 1.8s | **15.6x** |
| INT8 quantization | 18s | 2.1s | **8.6x** |

## Architecture

- `compress.rs` - Sparse delta compression with parallel processing
- `decompress.rs` - Fast decompression with zero-copy when possible
- `quantize.rs` - SIMD-accelerated INT8 quantization
- `utils.rs` - Shared utility functions

## Integration with Sparse

The Rust core is automatically used when available, falling back to Python implementation if not installed.
