<div align="center">

# ∴ Sparse

**Delta Compression for Fine-tuned Models and Datasets**

> Compress your 14GB fine-tune to 1.4GB (lossless) or 50MB (LoRA-equivalent). Reconstruct in 4 seconds.

**Verified**: GPT-2 compression → reconstruction → **identical inference output** ✅

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://rustlang.org)

[Quick Start](#quick-start) • [How It Works](#how-it-works) • [CLI](#cli-reference) • [Python API](#python-api)

</div>

---

## What Sparse Does

**Sparse compresses fine-tuned models and datasets as deltas from their base versions.**

| Compression Mode | Size (7B) | Quality | Use Case |
|------------------|-----------|---------|----------|
| **Lossless** | ~1.4 GB | 100% | Production, quality-critical |
| **Lossy (SVD)** | ~50 MB | ~95-99% | Sharing, size-critical |
| **Dataset Delta** | 60-80% savings | 100% | Derivative datasets |

**Key benefit:** Works on models you've *already trained* - no LoRA required during training.

**Works with:** Full fine-tunes, RLHF, model merges, translated/augmented datasets

---

## Quick Start

```bash
pip install sparse-llm
```

### Compress a Fine-tune

```bash
# Lossless compression (~1.4GB for 7B model)
sparse compress meta-llama/Llama-2-7b-hf ./my-finetune -o ./my-delta

# OR: Lossy compression (~50MB, LoRA-equivalent quality)
sparse compress-lossy meta-llama/Llama-2-7b-hf ./my-finetune -o ./my-delta --rank 16
```

### Reconstruct from Delta

```bash
# From lossless delta
sparse reconstruct meta-llama/Llama-2-7b-hf ./my-delta -o ./reconstructed-model

# From lossy delta
sparse reconstruct-lossy meta-llama/Llama-2-7b-hf ./my-delta -o ./reconstructed-model
```

### Dataset Delta

```bash
# Compress derivative dataset
sparse dataset-compress squad squad_v2 -o ./squad_v2_delta

# Reconstruct
sparse dataset-reconstruct ./squad_v2_delta
```

---

## How It Works

```
Fine-tuned Model (14GB)  -  Base Model (14GB)  =  Delta (1.4GB or 50MB)
                                                        ↓
                                              Reconstruct: Base + Delta
```

- **Lossless:** Sparse + INT8 encoding → ~10% of original size, 100% quality
- **Lossy (SVD):** Low-rank approximation → ~0.4% of original, ~95-99% quality

---

## CLI Reference

```bash
# Lossless compression (100% quality)
sparse compress <base> <finetune> -o <output>
sparse reconstruct <base> <delta> [-o <output>]

# Lossy compression (~50MB, LoRA-equivalent quality)
sparse compress-lossy <base> <finetune> -o <output> [--rank 16]
sparse reconstruct-lossy <base> <delta> [-o <output>]

# Dataset commands
sparse dataset-compress <base> <derivative> -o <output>
sparse dataset-reconstruct <delta_dir>
sparse dataset-estimate <base> <derivative>

# Info
sparse info <path>
```

---

## Python API

```python
from core import compress_delta, reconstruct_from_delta
from core import compress_delta_svd_full, reconstruct_from_svd_delta

# Lossless compression
manifest = compress_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="./my-finetune",
    output_path="./my-delta"
)
print(f"Compression: {manifest.compression_ratio:.1f}x")  # ~10x

# Extract LoRA (lossy, LoRA-equivalent)
manifest = compress_delta_svd_full(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="./my-finetune",
    output_path="./my-svd-delta",
    rank=16  # Like LoRA rank
)
print(f"Compression: {manifest.compression_ratio:.1f}x")  # ~280x

# Reconstruct (lossless)
model = reconstruct_from_delta("meta-llama/Llama-2-7b-hf", "./my-delta")

# Reconstruct from extracted LoRA
model = reconstruct_from_svd_delta("meta-llama/Llama-2-7b-hf", "./my-lora-delta")
```

### Dataset API

```python
from core import compress_dataset_delta, reconstruct_from_dataset_delta

# Compress
manifest = compress_dataset_delta("squad", "squad_v2", "./squad_v2_delta")
print(f"Savings: {manifest['size_stats']['savings_pct']:.1f}%")

# Reconstruct
dataset = reconstruct_from_dataset_delta("./squad_v2_delta")
```

---

## Performance

**All optimizations are automatic** - no configuration needed:

- **Rust SIMD acceleration:** 5-10x faster compression
- **Base model caching:** ~20s saved per compression
- **Smart heuristics:** 10-20% better compression ratios
- **GPU reconstruction:** 2-3x faster on CUDA
- **Lazy loading:** 50-70% memory reduction for 30B+ models

**Typical speedup:** ~60s → ~8-12s (5-8x faster)

**📚 Advanced optimizations:** See [API_REFERENCE.md](docs/API_REFERENCE.md) for MmapDeltaStorage, DifferentialCompressor, and other utilities.

---

## Sparse vs LoRA

| | LoRA/PEFT | Sparse |
|--|-----------|--------|
| **When applied** | During training | After training |
| **Works on existing models** | ❌ | ✅ |
| **Lossless option** | ❌ | ✅ |

**Key insight:** `sparse compress-lossy` gives you LoRA-sized files (~50MB) from models that weren't trained with LoRA.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers
- Rust (included in wheel, no setup needed)

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

Free for personal and commercial use.
