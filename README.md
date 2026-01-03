<div align="center">

# ∴ Sparse

**Delta Compression for Fine-tuned Models and Datasets**

> Compress your 14GB fine-tune to 1.4GB (lossless) or 50MB (LoRA-equivalent). Reconstruct in 4 seconds.

**Verified**: GPT-2 compression → reconstruction → **identical inference output** ✅

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://rustlang.org)

[Quick Start](#quick-start) • [How It Works](#how-it-works) • [CLI](#cli) • [Python API](#python-api)

</div>

---

## What Sparse Does

**Sparse compresses fine-tuned models and derivative datasets as deltas from their base versions.**

### 📦 Model Delta Compression

| Mode | Size (7B) | Quality | Use Case |
|------|-----------|---------|----------|
| **Lossless** | ~1.4 GB | 100% | When quality matters |
| **SVD (LoRA-equiv)** | ~50 MB | ~95-99% | When size matters |

**Reconstruction:** 4 seconds • **Works on ANY existing fine-tune**

**Use cases:**
- Compress your existing full fine-tunes (trained without LoRA)
- Share smaller files with collaborators
- Save disk space storing multiple fine-tunes
- Works with ANY training method: full fine-tune, RLHF, merges

### 📊 Dataset Delta Compression

| Metric | Value |
|--------|-------|
| **Savings** | 60-80% typical |
| **Use case** | Derivative datasets (translations, versions, augmentations) |

---

## Quick Start

```bash
pip install sparse-llm
```

### Compress a Fine-tune

```bash
# Lossless compression (~1.4GB for 7B model)
sparse compress meta-llama/Llama-2-7b-hf ./my-finetune -o ./my-delta

# OR: SVD compression (~50MB, LoRA-equivalent quality)
sparse svd-compress meta-llama/Llama-2-7b-hf ./my-finetune -o ./my-delta --rank 16
```

### Reconstruct from Delta

```bash
# From lossless delta
sparse reconstruct meta-llama/Llama-2-7b-hf ./my-delta -o ./reconstructed-model

# From SVD delta
sparse svd-reconstruct meta-llama/Llama-2-7b-hf ./my-delta -o ./reconstructed-model
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
Fine-tuned Model (14GB)  -  Base Model (14GB)  =  Delta
                                    ↓
                    Lossless: 1.4GB  |  SVD: 50MB
                                    ↓
                         Reconstruct: Base + Delta
```

**Two compression modes:**

| Mode | How It Works | Size | Quality |
|------|--------------|------|--------|
| **Lossless** | Sparse + INT8 encoding | ~10% of original | 100% |
| **SVD** | Low-rank approximation (like LoRA) | ~0.4% of original | ~95-99% |

---

## CLI Reference

```bash
# Lossless compression (100% quality)
sparse compress <base> <finetune> -o <output>
sparse reconstruct <base> <delta> [-o <output>]

# SVD compression (LoRA-equivalent, ~50MB)
sparse svd-compress <base> <finetune> -o <output> [--rank 16]
sparse svd-reconstruct <base> <delta> [-o <output>]

# Adapter packaging
sparse compress-adapter <base> <adapter> -o <output>

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

# SVD compression (LoRA-equivalent)
manifest = compress_delta_svd_full(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="./my-finetune",
    output_path="./my-svd-delta",
    rank=16  # Like LoRA rank
)
print(f"Compression: {manifest.compression_ratio:.1f}x")  # ~280x

# Reconstruct (lossless)
model = reconstruct_from_delta("meta-llama/Llama-2-7b-hf", "./my-delta")

# Reconstruct (SVD)
model = reconstruct_from_svd_delta("meta-llama/Llama-2-7b-hf", "./my-svd-delta")
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

## Why Sparse?

**Post-hoc compression for ANY fine-tune.** Unlike LoRA (which requires training differently), Sparse works on models you've *already* trained.

| | LoRA/PEFT | Sparse Lossless | Sparse SVD |
|--|-----------|-----------------|------------|
| **When** | During training | After training | After training |
| **Size** | ~50 MB | ~1.4 GB | ~50 MB |
| **Quality** | ~95-99% | 100% | ~95-99% |
| **Works on existing models** | ❌ No | ✅ Yes | ✅ Yes |

**Key insight:** Sparse SVD gives you LoRA-sized files from models that weren't trained with LoRA.

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers
- Rust (required, included in package)

---

## Auto-Caching & Fast Reconstruction _(If integrated directly into HuggingFace)_

**Note:** This feature is available in the codebase but requires HuggingFace Hub integration to be fully functional.

```python
from core.fast_reconstruct import DeltaCache, from_pretrained_with_delta

# Create cache (reconstructed models stored in ~/.cache/sparse)
cache = DeltaCache()

# Reconstruct and cache - only takes time once!
model_path = cache.get_or_reconstruct(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_path="./my-delta",
    background=False  # Wait for completion
)

# Load model from cache
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_path)

# Or use drop-in replacement for from_pretrained
model = from_pretrained_with_delta(
    "./my-delta",
    base_model_id="meta-llama/Llama-2-7b-hf"
)

# Prefetch multiple deltas in background (10x faster workflow!)
cache.prefetch_deltas(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_paths=["./delta1", "./delta2", "./delta3"]
)
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

Free for personal and commercial use.
