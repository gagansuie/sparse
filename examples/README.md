# Sparse Examples

Example scripts demonstrating Sparse delta compression.

## Quick Start

### 1. Lossless Delta Compression

Store fine-tunes as deltas from base models (100% quality):

```bash
# CLI
sparse compress mistralai/Mistral-7B-v0.1 your-org/mistral-finetuned -o ./delta

# Python
python examples/delta_compression.py
```

Typical savings: **~10x compression** (14GB → 1.4GB)

### 2. SVD Compression (LoRA-equivalent)

Extract LoRA-sized files from ANY fine-tune (~95-99% quality):

```bash
# CLI
sparse svd-compress mistralai/Mistral-7B-v0.1 your-org/mistral-finetuned -o ./svd-delta --rank 16

# Python
from core import compress_delta_svd_full

manifest = compress_delta_svd_full(
    base_model_id="mistralai/Mistral-7B-v0.1",
    finetune_model_id="your-org/mistral-finetuned",
    output_path="./svd-delta",
    rank=16
)
```

Typical savings: **~280x compression** (14GB → 50MB)

### 3. Dataset Delta Compression

Compress derivative datasets as deltas:

```bash
sparse dataset-compress squad squad_v2 -o ./squad_v2_delta
```

## Python API Examples

### Lossless Compression

```python
from core import compress_delta, reconstruct_from_delta

# Compress fine-tune as delta
manifest = compress_delta(
    base_model_id="mistralai/Mistral-7B-v0.1",
    finetune_model_id="your-org/mistral-finetuned",
    output_path="./delta",
)
print(f"Compression: {manifest.compression_ratio:.1f}x")

# Reconstruct full model from base + delta
model = reconstruct_from_delta(
    base_model_id="mistralai/Mistral-7B-v0.1",
    delta_path="./delta",
)
```

### SVD Compression

```python
from core import compress_delta_svd_full, reconstruct_from_svd_delta

# Extract LoRA-equivalent (lossy)
manifest = compress_delta_svd_full(
    base_model_id="mistralai/Mistral-7B-v0.1",
    finetune_model_id="your-org/mistral-finetuned",
    output_path="./svd-delta",
    rank=16  # Like LoRA rank
)
print(f"Size: {manifest.compressed_size_bytes / 1e6:.1f} MB")

# Reconstruct
model = reconstruct_from_svd_delta(
    base_model_id="mistralai/Mistral-7B-v0.1",
    delta_path="./svd-delta",
)
```

## Compression Modes

| Mode | Size (7B) | Quality | Use Case |
|------|-----------|---------|----------|
| **Lossless** | ~1.4 GB | 100% | Production |
| **SVD rank 16** | ~50 MB | ~95-99% | Sharing |
| **SVD rank 64** | ~200 MB | ~99% | Balanced |

## Output Structure

```
delta/
├── manifest.json      # Metadata and stats
└── deltas/            # Compressed layer data
    ├── layer_0.pt
    └── layer_1.pt
```

## Installation

```bash
pip install -r requirements.txt

# Rust acceleration is included automatically in the package
```

## Next Steps

- See [../README.md](../README.md) for full documentation
- Deploy to HuggingFace Spaces (see `hf_space/`)
