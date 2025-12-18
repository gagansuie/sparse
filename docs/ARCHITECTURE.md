# TenPak Architecture

## Overview

TenPak is a model compression system for LLMs with three main components:

1. **TenPak Core** - Compression engine (codecs, calibration, allocation)
2. **TenPak Studio** - REST API and job management
3. **TenPak CLI** - Command-line interface

## Component Design

### Core (`tenpak/core/`)

The compression engine handles the actual quantization work.

```
core/
├── __init__.py      # Public API exports
├── codecs.py        # Compression algorithms (INT4, VQ, AWQ)
├── calibration.py   # Stats collection (Fisher, Hessian, activations)
└── allocation.py    # Bit allocation strategies per layer
```

**Key Functions:**
- `compress_int4_awq()` - Production codec (v10 config)
- `compress_int4_residual()` - Best quality without calibration
- `collect_calibration_stats()` - Fisher + Hessian + activation scales
- `allocate_bits()` - Per-layer compression strategy

### Studio (`tenpak/studio/`)

REST API for running compression jobs at scale.

```
studio/
├── __init__.py      # Public API exports
├── api.py           # FastAPI endpoints
├── jobs.py          # Async job runner
└── storage.py       # Artifact packaging and storage
```

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compress` | POST | Start compression job |
| `/status/{id}` | GET | Poll job progress |
| `/artifact/{id}` | GET | Download result |
| `/evaluate` | POST | Compute PPL |
| `/jobs` | GET | List recent jobs |

### CLI (`tenpak/cli/`)

Command-line tool for local compression.

```
cli/
├── __init__.py
└── main.py          # Entry point (tenpak pack/eval/info)
```

**Commands:**
```bash
tenpak pack <model_id> [--target quality|balanced|size]
tenpak eval <model_id> [--samples N]
tenpak info <artifact_path>
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      User Request                            │
│  (CLI: tenpak pack, API: POST /compress, Python: direct)    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    1. Load Model                             │
│  transformers.AutoModelForCausalLM.from_pretrained()        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                2. Collect Calibration Stats                  │
│  - Fisher information (gradient importance)                  │
│  - Activation scales (AWQ-style)                            │
│  - Hessian diagonal (GPTQ-style)                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. Allocate Bits Per Layer                  │
│  - Attention layers: smaller groups (g=256)                  │
│  - MLP layers: larger groups (g=2048)                       │
│  - Based on target: quality/balanced/size                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   4. Compress Each Layer                     │
│  For each Linear layer:                                      │
│    - Scale by activation importance                          │
│    - Extract outliers to FP16                               │
│    - INT4 quantize with iterative refinement                │
│    - Replace weight in-place                                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. Evaluate Quality                       │
│  - Compute baseline PPL (before compression)                 │
│  - Compute compressed PPL (after compression)               │
│  - Calculate PPL delta percentage                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    6. Save Artifact                          │
│  - manifest.json (metadata, allocations, metrics)           │
│  - weights/shard_0.bin (compressed tensors)                 │
└─────────────────────────────────────────────────────────────┘
```

## Compression Algorithm (v10 Config)

The production configuration (v10) uses INT4 + AWQ:

### 1. AWQ Scaling
```python
# Scale columns by activation importance
w_scaled = weight * act_scale.sqrt()
```

### 2. Outlier Extraction
```python
# Keep top 0.5% weights as FP16
threshold = torch.kthvalue(w.abs(), k=int(n * 0.995))
outliers = w * (w.abs() >= threshold)
w_to_quant = w * (w.abs() < threshold)
```

### 3. Iterative Scale Refinement
```python
for _ in range(5):  # 5 iterations
    scale = (g_max - g_min) / 15
    q = ((group - g_min) / scale).round().clamp(0, 15)
    deq = q * scale + g_min
    err = group - deq
    g_min += err.min() * 0.5
    g_max += err.max() * 0.5
```

### 4. Final Quantization
```python
q = ((group - g_min) / scale).round().clamp(0, 15)
dequantized = q * scale + g_min + outliers
```

## Artifact Format

```
artifact/
├── manifest.json      # Metadata
│   {
│     "version": "1.0",
│     "model_id": "mistralai/Mistral-7B-v0.1",
│     "codec": "int4_awq_v1",
│     "compression_ratio": 7.42,
│     "baseline_ppl": 5.234,
│     "compressed_ppl": 5.311,
│     "ppl_delta": 1.47,
│     "allocations": {...}
│   }
└── weights/
    └── shard_0.bin    # torch.save() of compressed weights
```

## Configuration Presets

| Target | Attention g | MLP g | Expected Compression | Expected PPL Δ |
|--------|-------------|-------|---------------------|----------------|
| `quality` | 128 | 512 | ~5x | <1% |
| `balanced` | 256 | 2048 | ~7x | <2% |
| `size` | 512 | 4096 | ~8x | <5% |

## Key Learnings

1. **4x compression is the no-calibration ceiling** - INT4 g=8 with iterative refinement
2. **Calibration is required for >6x** - AWQ activation scaling is essential
3. **Larger models compress better** - Llama 7B achieves negative PPL delta
4. **Custom GPTQ/AQLM implementations are fragile** - Stick to proven INT4+AWQ
5. **v10 config is battle-tested** - 7.42x @ +1.47% on Mistral-7B
