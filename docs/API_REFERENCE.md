# Sparse-LLM API Reference

Complete API documentation for `sparse-llm` - delta compression for LLM fine-tunes.

**Version 0.0.4** - Now with automatic performance optimizations!

⚡ **Performance Note:** Operations marked with ⚡ use built-in Rust acceleration (10-20x faster).

---

## Table of Contents

1. [Model Delta Compression (Lossless)](#model-delta-compression-lossless)
2. [Lossy Compression](#lossy-compression)
3. [Dataset Delta Compression](#dataset-delta-compression)
4. [Data Types](#data-types)

---

## Model Delta Compression (Lossless)

### `compress_delta()` ⚡

Compress a fine-tuned model as a delta from base model. **100% lossless reconstruction.**

**⚡ Rust-Accelerated:** This function uses built-in Rust implementation for 10-20x speedup.

**Signature:**
```python
def compress_delta(
    base_model_id: str,
    finetune_model_id: str,
    output_path: str,
    device: str = "cuda",
    progress_callback: Optional[callable] = None
) -> DeltaManifest
```

**Parameters:**
- `base_model_id` (str): Base model identifier (e.g., "meta-llama/Llama-2-7b-hf")
- `finetune_model_id` (str): Fine-tuned model identifier or local path
- `output_path` (str): Output directory for delta files
- `device` (str): Device for computation ("cuda" or "cpu")
- `progress_callback` (callable, optional): Progress callback(msg, progress)

**Returns:** `DeltaManifest` object with delta metadata

**Example:**
```python
from core import compress_delta

manifest = compress_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="my-org/llama-chat",
    output_path="./my-delta"
)

print(f"Compression: {manifest.compression_ratio:.1f}x")  # ~10x typical
print(f"Size: {manifest.total_params * 2 / 1e9 / manifest.compression_ratio:.2f} GB")
```

**CLI:**
```bash
sparse compress meta-llama/Llama-2-7b-hf my-org/llama-chat -o ./my-delta
```

---

### `reconstruct_from_delta()` ⚡

Reconstruct full model from base + delta. **100% identical to original.**

**⚡ Rust-Accelerated:** Decompression is 10-15x faster with built-in Rust acceleration.

**Signature:**
```python
def reconstruct_from_delta(
    base_model_id: str,
    delta_path: str,
    device: str = "cuda",
    progress_callback: Optional[callable] = None
) -> torch.nn.Module
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `delta_path` (str): Path to delta artifact directory
- `device` (str): Device for loading ("cpu" or "cuda")
- `progress_callback` (callable, optional): Progress callback(msg, progress)

**Returns:** Reconstructed model (identical to original fine-tune)

**Example:**
```python
from core import reconstruct_from_delta

model = reconstruct_from_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_path="./my-delta"
)

# Model is ready for inference - 100% identical output
outputs = model.generate(...)
```

**CLI:**
```bash
sparse reconstruct meta-llama/Llama-2-7b-hf ./my-delta -o ./reconstructed-model
```

---

### `validate_int8_delta_quality()` ⚡

Validate INT8 delta compression quality with real model inference.

**Signature:**
```python
def validate_int8_delta_quality(
    base_model_id: str,
    finetune_model_id: str,
    sample_layers: int = 2,
    prompts: Optional[List[str]] = None,
    max_length: int = 128,
) -> Dict[str, Any]
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `finetune_model_id` (str): Fine-tuned model identifier
- `sample_layers` (int): Number of large layers to sample (default: 2)
- `prompts` (List[str], optional): Test prompts for logits comparison
- `max_length` (int): Max tokenization length (default: 128)

**Returns:**
```python
{
    "status": str,                    # "✅ Completed" or "❌ Error: ..."
    "rust_acceleration": bool,
    "layer_metrics": [                # Per-layer reconstruction metrics
        {
            "name": str,
            "compression_ratio": float,
            "max_abs_error": float,
            "mean_abs_error": float,
        }
    ],
}
```

**Example:**
```python
from core import validate_int8_delta_quality

report = validate_int8_delta_quality(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="meta-llama/Llama-2-7b-chat-hf",
    sample_layers=2,
)

print(f"Status: {report['status']}")
for layer in report['layer_metrics']:
    print(f"  {layer['name']}: max_err={layer['max_abs_error']:.6f}")
```

---

## Lossy Compression

### `compress_delta_svd_full()`

Compress a fine-tune with lossy compression (LoRA-equivalent quality).

**This is lossy compression** (~95-99% quality) but achieves LoRA-sized files (~50MB for 7B models).

**Signature:**
```python
def compress_delta_svd_full(
    base_model_id: str,
    finetune_model_id: str,
    output_path: str,
    rank: int = 16,
    progress_callback: Optional[callable] = None,
) -> SVDDeltaManifest
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `finetune_model_id` (str): Fine-tuned model identifier or local path
- `output_path` (str): Output directory for SVD delta files
- `rank` (int): SVD rank (like LoRA rank, default 16). Higher = better quality, larger size.
- `progress_callback` (callable, optional): Progress callback(msg, progress)

**Returns:** `SVDDeltaManifest` object with compression statistics and quality metrics

**Example:**
```python
from core import compress_delta_svd_full

manifest = compress_delta_svd_full(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="my-org/llama-chat",
    output_path="./my-svd-delta",
    rank=16  # Like LoRA rank
)

print(f"Compression: {manifest.compression_ratio:.1f}x")  # ~280x
print(f"Size: {manifest.compressed_size_bytes / 1e6:.1f} MB")  # ~50 MB
print(f"Avg error: {manifest.avg_reconstruction_error:.6f}")
```

**CLI:**
```bash
sparse compress-lossy meta-llama/Llama-2-7b-hf my-org/llama-chat -o ./my-lossy-delta --rank 16
```

**Rank vs Quality:**

| Rank | Size (7B) | Quality |
|------|-----------|---------|
| 8 | ~25 MB | ~90-95% |
| 16 | ~50 MB | ~95-99% |
| 32 | ~100 MB | ~98-99% |
| 64 | ~200 MB | ~99%+ |

---

### `reconstruct_from_svd_delta()`

Reconstruct model from lossy-compressed delta. **This is lossy reconstruction.**

**Signature:**
```python
def reconstruct_from_svd_delta(
    base_model_id: str,
    delta_path: str,
    device: str = "cuda",
    progress_callback: Optional[callable] = None,
) -> torch.nn.Module
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `delta_path` (str): Path to SVD delta artifact directory
- `device` (str): Device for loading ("cpu" or "cuda")
- `progress_callback` (callable, optional): Progress callback(msg, progress)

**Returns:** Reconstructed model (approximate, ~95-99% quality)

**Example:**
```python
from core import reconstruct_from_svd_delta

model = reconstruct_from_svd_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_path="./my-svd-delta"
)

# Model is ready for inference - approximately same output
outputs = model.generate(...)
```

**CLI:**
```bash
sparse reconstruct-lossy meta-llama/Llama-2-7b-hf ./my-lossy-delta -o ./reconstructed-model
```

---

## Dataset Delta Compression

### `estimate_dataset_delta_savings()`

Estimate storage savings for dataset delta compression.

**Signature:**
```python
def estimate_dataset_delta_savings(
    base_dataset_id: str,
    derivative_dataset_id: str,
    split: str = "train",
    sample_size: int = 1000
) -> DatasetDeltaStats
```

**Parameters:**
- `base_dataset_id` (str): Base dataset identifier
- `derivative_dataset_id` (str): Derivative dataset identifier
- `split` (str): Dataset split to analyze
- `sample_size` (int): Number of samples to analyze

**Returns:** `DatasetDeltaStats` object with savings estimates

**Example:**
```python
from core import estimate_dataset_delta_savings

stats = estimate_dataset_delta_savings(
    base_dataset_id="squad",
    derivative_dataset_id="squad_v2",
    sample_size=1000
)

print(f"Savings: {stats.savings_pct:.1f}%")
```

---

### `compress_dataset_delta()`

Compress a derivative dataset as delta from base dataset.

**Signature:**
```python
def compress_dataset_delta(
    base_dataset_id: str,
    derivative_dataset_id: str,
    output_dir: str,
    splits: List[str] = None,
) -> Dict
```

**Parameters:**
- `base_dataset_id` (str): Base dataset identifier
- `derivative_dataset_id` (str): Derivative dataset identifier
- `output_dir` (str): Output directory for delta files
- `splits` (List[str], optional): Specific splits to compress (default: all)

**Returns:** Manifest dict with delta metadata

**Example:**
```python
from core import compress_dataset_delta

manifest = compress_dataset_delta(
    base_dataset_id="squad",
    derivative_dataset_id="squad_v2",
    output_dir="./squad_v2_delta"
)

print(f"Savings: {manifest['size_stats']['savings_pct']:.1f}%")
```

**CLI:**
```bash
sparse dataset-compress squad squad_v2 -o ./squad_v2_delta
```

---

### `reconstruct_from_dataset_delta()`

Reconstruct full dataset from base + delta.

**Signature:**
```python
def reconstruct_from_dataset_delta(
    delta_path: str,
) -> datasets.DatasetDict
```

**Parameters:**
- `delta_path` (str): Path to delta directory containing manifest

**Returns:** Reconstructed `datasets.DatasetDict`

**Example:**
```python
from core import reconstruct_from_dataset_delta

dataset = reconstruct_from_dataset_delta("./squad_v2_delta")
print(f"Rows: {len(dataset['train'])}")
```

**CLI:**
```bash
sparse dataset-reconstruct ./squad_v2_delta
```

---

## Data Types

### `DeltaManifest`

```python
@dataclass
class DeltaManifest:
    version: str = "1.0"
    delta_type: str = "model_delta"  # "model_delta" or "adapter"
    base_model_id: str = ""
    finetune_model_id: str = ""
    created_at: str = ""  # ISO timestamp
    num_layers: int = 0
    total_params: int = 0
    changed_params: int = 0
    compression_ratio: float = 1.0
```

### `SVDDeltaManifest`

```python
@dataclass
class SVDDeltaManifest:
    version: str = "1.0"
    delta_type: str = "svd_delta"
    base_model_id: str = ""
    finetune_model_id: str = ""
    created_at: str = ""
    rank: int = 16
    num_layers: int = 0
    total_params: int = 0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    avg_reconstruction_error: float = 0.0
    max_reconstruction_error: float = 0.0
```

### `DatasetDeltaStats`

```python
@dataclass
class DatasetDeltaStats:
    base_dataset_id: str
    derivative_dataset_id: str
    base_size_bytes: int
    derivative_size_bytes: int
    delta_size_bytes: int
    savings_pct: float
    num_shared_samples: int
    num_new_samples: int
```

---

## Compression Mode Comparison

| Feature | Lossless | Lossy |
|---------|----------|-----|
| **Function** | `compress_delta()` | `compress_delta_svd_full()` |
| **CLI** | `sparse compress` | `sparse compress-lossy` |
| **Size (7B)** | ~1.4 GB | ~50 MB |
| **Quality** | 100% | ~95-99% |
| **Use Case** | Production, quality-critical | Sharing, size-critical |

---

## Fast Reconstruction & Caching

### DeltaCache

Auto-caching system for reconstructed models. Reconstruct once, load instantly forever.

```python
from core.fast_reconstruct import DeltaCache

# Initialize cache
cache = DeltaCache(
    cache_dir="~/.cache/sparse",  # Optional, default location
    max_cache_size_gb=100.0,       # Optional, max cache size
    max_workers=2                  # Optional, concurrent jobs
)

# Get or reconstruct (waits for completion)
model_path = cache.get_or_reconstruct(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_path="./my-delta",
    background=False
)

# Load reconstructed model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path)

# Check cache stats
stats = cache.get_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Avg reconstruction time: {stats['avg_reconstruction_time_s']:.1f}s")
```

**Key Features:**
- **Smart caching**: Reconstructed models cached in `~/.cache/sparse/reconstructed/`
- **HF Hub integration**: Base models use HuggingFace's existing cache (no duplication)
- **Background reconstruction**: Prefetch multiple deltas in parallel
- **4-second reconstruction**: Rust-accelerated delta application

### Background Prefetching

Pre-load multiple deltas while you work:

```python
# Prefetch 10 deltas in background
job_ids = cache.prefetch_deltas(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_paths=[f"./delta_{i}" for i in range(10)]
)

# Check job status
for delta_path, job_id in job_ids.items():
    job = cache.get_job_status(job_id)
    print(f"{delta_path}: {job.status}")
```

### Drop-in Replacement for from_pretrained

```python
from core.fast_reconstruct import from_pretrained_with_delta

# Automatically detects deltas and reconstructs
model = from_pretrained_with_delta(
    "./my-delta",
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_mode="auto",  # "auto", "prefer_delta", "full_only"
    torch_dtype=torch.float16
)
```

### Benchmark Reconstruction Speed

```python
from core.fast_reconstruct import benchmark_reconstruction

# Test Rust acceleration speed
result = benchmark_reconstruction(tensor_size=1_000_000, iterations=10)
print(f"7B model reconstruction: {result['estimated_times']['7B_model']}")
# Output: ~4 seconds
```

---

## Rust Acceleration

**Rust acceleration is included automatically** when you install `sparse-llm`. No additional setup required.

```python
# Verify Rust is working
from core.delta_rust import get_rust_info

info = get_rust_info()
print("✓ Rust acceleration enabled")
print(f"Features: {', '.join(info['features'])}")
```

The package includes pre-compiled Rust extensions for all platforms (Linux, macOS, Windows), bundled in the Python wheel.

---

## Advanced Performance Optimizations

Sparse includes several advanced optimizations for maximum performance.

### Zstd Compression ⚡

Additional compression layer for delta files using Zstd. Achieves ~2x smaller files on top of INT8 quantization.

```python
from sparse_core import compress_zstd, decompress_zstd

# Compress any bytes data
data = b"delta_data..." * 10000
compressed = compress_zstd(data, level=3)  # level: 1 (fast) to 19 (best)
decompressed = decompress_zstd(compressed)

print(f"Ratio: {len(data) / len(compressed):.1f}x")  # Typically 2-10x
```

**Compression Levels:**
| Level | Speed | Ratio |
|-------|-------|-------|
| 1 | Fastest | ~2x |
| 3 | Default | ~3-4x |
| 19 | Best | ~5-10x |

---

### Streaming Reconstruction ⚡

Pipeline I/O and compute for large model reconstruction. Overlaps disk reads with delta application for maximum throughput.

```python
from sparse_core import StreamingReconstructor

# Create streaming reconstructor
reconstructor = StreamingReconstructor(
    num_workers=4,       # Parallel workers
    chunk_size=65536,    # Elements per chunk
    prefetch_count=2     # Prefetch buffer size
)

# Process layers in streaming pipeline
stats = reconstructor.process_layers_streaming(layer_count=100)
print(f"Throughput: {stats.throughput_mbs:.1f} MB/s")
print(f"Layers processed: {stats.layers_processed}")

# Benchmark sequential vs parallel
bench = reconstructor.benchmark(layer_count=10, elements_per_layer=1_000_000)
print(f"Sequential: {bench.sequential_ms}ms")
print(f"Parallel: {bench.parallel_ms}ms")
print(f"Speedup: {bench.speedup:.1f}x")
```

---

### GPU-Accelerated Reconstruction ⚡

**Automatic CUDA acceleration for INT8 delta reconstruction.**

GPU ops are **automatically used** when:
- Model is loaded on CUDA device (`device="cuda"`)
- INT8 compression method is used
- PyTorch with CUDA support is available

**No code changes needed** - just load model with CUDA:

```python
from core import compress_delta, reconstruct_from_delta

# Compress (uses INT8 automatically via smart heuristics)
compress_delta("gpt2", "./finetune", "./delta", device="cuda")

# Reconstruct - automatically uses GPU ops for INT8 deltas
model = reconstruct_from_delta("gpt2", "./delta", device="cuda")
```

**How it works:**
- Detects CUDA availability automatically
- Uses tiled processing with FMA instructions
- 2-3x faster reconstruction for INT8 deltas
- Falls back to standard Rust ops if GPU unavailable

**Requirements:**
- CUDA-capable GPU
- PyTorch installed with CUDA support (`torch.cuda.is_available() == True`)

**Performance:**
- **CPU (Rust):** ~5-8s for 7B model INT8 reconstruction
- **GPU (CUDA):** ~2-3s for 7B model INT8 reconstruction

---

### SIMD/AVX2 Acceleration

Native CPU SIMD instructions are automatically enabled for x86_64 (AVX2+FMA) and ARM64 (NEON) platforms.

**Enabled features by platform:**
| Platform | SIMD Features |
|----------|---------------|
| x86_64 Linux/macOS | AVX2, FMA |
| ARM64 Linux/macOS | NEON |

No configuration required - optimizations are built into the wheel.

---

---

## Performance Optimizations

### Automatic Optimizations (v0.0.4+)

**These are enabled by default** in all `compress_delta()` calls and CLI commands:

#### 1. Base Model Caching (✅ Automatic)

**Enabled automatically** - base models are cached to avoid repeated loading.

```python
# Automatically used in compress_delta()
from core import compress_delta

manifest = compress_delta("gpt2", "./finetune", "./output")
# Base model cached automatically!
```

**Performance:** ~20s saved per compression after first load.

**Manual Usage (Optional):**
```python
from core import get_cache

cache = get_cache(max_size_gb=50)
model = cache.get_or_load("meta-llama/Llama-2-7b-hf")

stats = cache.get_stats()
print(f"Utilization: {stats['utilization']*100:.1f}%")
```

---

#### 2. Rust SIMD Delta Computation (✅ Automatic)

**Enabled automatically** - hardware-accelerated delta computation.

```python
# Automatically used in compress_delta()
from core import compress_delta

manifest = compress_delta("gpt2", "./finetune", "./output")
# Rust SIMD acceleration used automatically!
```

**Performance:** 5-10x faster than pure Python.

**How it works:** Automatically detects float32 2D tensors and uses Rust SIMD, falls back to Python for other types.

---

#### 3. Smart Compression Heuristics (✅ Automatic)

**Enabled automatically** - layer-aware compression method selection.

```python
# Automatically used in compress_delta()
from core import compress_delta

manifest = compress_delta("gpt2", "./finetune", "./output")
# Smart heuristics analyze each layer type automatically!
```

**Performance:** 10-20% better compression ratios.

**How it works:** Analyzes layer names (attention, MLP, embedding) and delta distributions to choose optimal compression (sparse/int8/hybrid).

---

## Opt-in Utilities for Advanced Use Cases

These utilities are available for specialized scenarios:

### 1. LazyModelLoader - Stream Very Large Models

**Use when:** Processing 70B+ models with limited RAM.

**What it does:** Loads model layers one-by-one from safetensors instead of loading entire model into memory.

```python
from core import MmapDeltaStorage
from pathlib import Path

storage = MmapDeltaStorage(Path("./deltas"))

# Save delta with memory mapping
storage.save_delta("layer_0", delta_tensor, quantized=quantized_data, scale=0.01)

# Load with zero-copy
delta, scale = storage.load_delta("layer_0")

print(f"Total size: {storage.get_total_size_mb():.2f} MB")
```


---

**Memory savings:** 50-70% reduction.

**Usage:**

```python
from core import LazyModelLoader

loader = LazyModelLoader("meta-llama/Llama-2-7b-hf")

# Get layer names without loading weights
layer_names = loader.get_layer_names()

# Stream layers on-demand
for name, tensor in loader.iter_layers(layer_names[:10]):
    # Process one layer at a time
    delta = compute_delta(tensor, ...)
    
# Memory-efficient delta computation
from core import compute_deltas_streaming

for name, delta, stats in compute_deltas_streaming(base_loader, ft_loader):
    print(f"{name}: sparsity={stats['sparsity']:.2%}")
```

---

### 2. MmapDeltaStorage - Memory-Mapped File I/O

**Use when:** Processing many deltas, need zero-copy I/O.

**What it does:** Uses `numpy.memmap` for fast, zero-copy file operations.

**Performance:** 40% faster I/O operations.

**Usage:**

---

---

### 3. DifferentialCompressor - Model Family Compression

**Use when:** Compressing multiple related fine-tunes from the same base.

**What it does:** Stores incremental deltas between related models instead of full deltas.

**Storage savings:** 2-3x smaller for model families.

**Usage:**

```python
from core import DifferentialCompressor
from pathlib import Path

# Initialize family compressor
compressor = DifferentialCompressor("meta-llama/Llama-2-7b-hf", Path("./family"))

# Compress multiple related models
result1 = compressor.compress_to_family("my-model-v1", model1_params)
result2 = compressor.compress_to_family("my-model-v2", model2_params)
result3 = compressor.compress_to_family("my-model-v3", model3_params)

# Check savings
savings = compressor.estimate_savings()
print(f"Models: {savings['num_models']}")
print(f"Savings: {savings['savings_pct']:.1f}% vs full deltas")

# Get reconstruction chain
chain = compressor.get_reconstruction_chain("my-model-v3")
print(f"Reconstruction order: {chain}")
```


---

---

### 4. Parallel Layer Processing - Multi-core Compute

**Use when:** Production pipelines with multi-core CPUs and large batches.

**What it does:** Processes multiple layers in parallel using `ProcessPoolExecutor`.

**Performance:** 3-4x speedup on multi-core CPUs.

**Note:** Best for large models where layer processing time >> serialization overhead.

**Usage:**

```python
from core import batch_layer_processing

# Process layers in parallel batches
for name, delta, stats in batch_layer_processing(
    param_names=layer_names,
    base_params=base_model_params,
    finetune_params=ft_model_params,
    batch_size=10
):
    # Each batch processes 10 layers in parallel
    save_delta(name, delta)
```


---

## Combined Performance Impact

Using all optimizations together:

| Without Optimizations | With Optimizations | Speedup |
|----------------------|-------------------|---------|
| ~60s | ~8-12s | **5-8x faster** |

Additional benefits:
- 50-70% lower memory usage (lazy loading)
- 2-3x smaller storage for model families (differential compression)
- 10-20% better compression ratios (smart heuristics)

---

**Questions?** Open an issue on [GitHub](https://github.com/gagansuie/sparse)
