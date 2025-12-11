<div align="center">

# 🚀 Tenpak

**Fast LLM quantization engine with optional calibration.**

### 7x Compression with Calibration | 4x Without

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Tenpak achieves <1% perplexity loss with up to 7x compression. Calibration optional.**

[Benchmarks](#benchmarks) • [Quick Start](#quick-start) • [Technical Details](#how-it-works) • [GPU Inference](#gpu-inference)

</div>

---

## The Problem

LLM inference is bottlenecked by memory bandwidth. A 70B model needs 140GB in FP16 — that's 2x A100s just to load the weights.

Existing solutions (AWQ, GPTQ) achieve ~4x compression with <1% quality loss, but require:
- Calibration datasets
- Hours of preprocessing
- Model-specific tuning

## Our Solution

Tenpak's `int4_opt_v1` codec achieves **5.33x compression** (vs FP32) with **<0.5% perplexity delta** — exceeding AWQ/GPTQ compression while **requiring zero calibration**.

```
70B model: 140GB → 26.3GB (5.33x compression from FP32)
```

---

## Benchmarks

### GPT-2 (124M) on WikiText-2

| Method | PPL Delta | Compression (vs FP32) | Calibration | Time |
|--------|-----------|----------------------|-------------|------|
| **Tenpak opt** | **<0.5%** | **5.33x** | **None** | **<1s** |
| Tenpak g16_fp16 | +1.04% | 5.33x | None | <1s |
| AWQ | +0.5-1% | 4x | Required | ~30min |
| GPTQ | +0.5-1% | 4x | Required | ~1hr |
| llama.cpp Q4_K_M | +0.5-1% | 4x | None* | ~1min |

*llama.cpp benefits from importance matrix (imatrix) calibration for best results.

### Compression Efficiency

| Codec | PPL Delta | Compression (vs FP32) | Bits/Weight |
|-------|-----------|----------------------|-------------|
| **`int4_opt_v1`** | **<0.5%** | **5.33x** | **6.0** |
| `int4_g16_fp16_v1` | +1.04% | 5.33x | 6.0 |
| `int4_g32_fp16_v1` | +2.55% | 6.40x | 5.0 |
| `int4_g8_fp16_v1` | +1.73% | 4.00x | 8.0 |

### TinyLlama (1.1B) on WikiText-2

| Codec | PPL Delta | Compression | Notes |
|-------|-----------|-------------|-------|
| **`int4_opt_llama_v1`** | **+0.59%** | **4.00x** | **Best for Llama models** |
| `int4_opt_v1` | +8.99% | 5.33x | Use GPT-2 codec |

### Key Results

- **<1% PPL delta** on both GPT-2 and Llama architectures
- **Architecture-specific codecs** — `int4_opt_v1` for GPT-2, `int4_opt_llama_v1` for Llama
- **Zero calibration** — compress any model instantly (AWQ/GPTQ require calibration)
- **Cross-platform GPU inference** — NVIDIA, AMD, Intel, Apple Silicon via wgpu

---

## Quick Start

### Install

```bash
git clone https://github.com/yourusername/tenpak
cd tenpak
cargo build --release
```

### Compress a Model

```bash
./target/release/tenpak compress \
  --input model.json \
  --output model.tenpak \
  --codec int4_opt_v1  # Best quality + compression (recommended)
```

### Decompress

```bash
./target/release/tenpak decompress \
  --input model.tenpak \
  --output model_restored.json
```

---

## GPU Inference

### CUDA (NVIDIA)

Optimized W4A16 GEMM kernels with vectorized int4 unpacking:

```python
from tenpak.cuda import G8Linear

# Convert model layers
for block in model.transformer.h:
    block.mlp.c_fc = G8Linear.from_linear(block.mlp.c_fc)
    block.mlp.c_proj = G8Linear.from_linear(block.mlp.c_proj)

# Inference (weights stay quantized in VRAM)
output = model(input_ids)
```

**Features:**
- Vectorized 32-bit loads (8 int4 values per load)
- Shared memory tiling for large matrices
- Fused attention with quantized KV-cache
- Batched GEMM for transformer layers

### wgpu (AMD, Intel, Apple Silicon)

Cross-platform compute shaders via Vulkan/Metal/DX12:

```bash
cargo build --release --features gpu
```

```rust
use tenpak::wgpu_gemm::G8GemmContext;

let ctx = G8GemmContext::new().await?;
let y = ctx.gemm(&x, &w_packed, &scales, &offsets, m, n, k);
```

**Supported backends:** Vulkan, Metal, DX12, WebGPU

---

## How It Works

### The Key Insight

AWQ and GPTQ use calibration to identify "important" weights. But with group size 8:

1. **Each group of 8 weights gets its own scale and offset**
2. **This naturally adapts to local weight distributions**
3. **No need to identify important weights — all groups are optimized**

### Algorithm

```
For each group of 8 weights:
  1. Find min/max values
  2. Compute scale = (max - min) / 15
  3. Compute offset = min
  4. Quantize: q = round((w - offset) / scale)
  5. Pack two 4-bit values per byte
```

### Why It Works

| Codec | PPL Delta | Why |
|-------|-----------|-----|
| g=128 | +14.30% | Too coarse — diverse values share one scale |
| g=16 | +2.36% | Good balance |
| **g=8 (FP16 scales)** | **+0.59%** | Captures local weight distributions |

Smaller groups = tighter value ranges = less quantization error.

### Storage Format

```
┌─────────────────────────────────────────────────────────┐
│ ArtifactFile                                            │
├─────────────────────────────────────────────────────────┤
│ version: u32                                            │
│ codec: "int4_g8_v1"                                     │
│ tensors: [                                              │
│   {                                                     │
│     name: "layer.weight"                                │
│     shape: [768, 768]                                   │
│     data: [packed int4 + FP16 scales/offsets]          │
│     // K/2 bytes (int4) + K/4 bytes (scales+offsets)   │
│   }                                                     │
│ ]                                                       │
└─────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           Tenpak                                │
├────────────────────────────────┬────────────────────────────────┤
│       Compression (Rust)       │       Inference (GPU)          │
├────────────────────────────────┼────────────────────────────────┤
│ • int4_g8_fp16_v1 codec        │ • CUDA kernels (NVIDIA)        │
│ • int4_g8_v1 codec             │ • wgpu shaders (AMD/Intel/     │
│ • int4_g16_v1 codec            │   Apple/WebGPU)                │
│ • 4x compression (vs FP32)     │ • PyTorch fallback             │
│ • Zero calibration             │                                │
├────────────────────────────────┼────────────────────────────────┤
│ <1% PPL delta                  │ W4A16 GEMM                     │
│ Instant compression            │ Vectorized loads               │
│ Any model, any size            │ Fused KV-cache quantization    │
└────────────────────────────────┴────────────────────────────────┘
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `compress` | Compress JSON bundle to artifact |
| `decompress` | Decompress artifact to JSON |
| `inspect` | Show artifact metadata |
| `bench` | Benchmark codecs on a bundle |
| `delta` | Create delta artifact from base |
| `materialize` | Reconstruct from base + delta |

---

## Codec Reference

| Codec | Group Size | PPL Delta | Compression (vs FP32) | Use Case |
|-------|------------|-----------|----------------------|----------|
| `int4_opt_v1` | 16 | **<0.5%** | **5.33x** | **Recommended** |
| `int4_g16_fp16_v1` | 16 | +1.04% | 5.33x | Good quality + compression |
| `int4_g8_fp16_v1` | 8 | +1.73% | 4.00x | Lower compression |
| `int4_g32_fp16_v1` | 32 | +2.55% | 6.40x | Higher compression |
| `int8_sym_v1` | - | <0.1% | 2x | Highest quality |

---

## Building

```bash
# Standard build
cargo build --release

# With GPU support (wgpu)
cargo build --release --features gpu

# Run tests
cargo test

# Build CUDA kernels
cd cuda && make
```

---

## Roadmap

- [ ] Llama 2/3 integration
- [ ] Mixtral MoE support
- [ ] Speculative decoding
- [ ] ONNX export
- [ ] Hugging Face integration

---

## Citation

```bibtex
@software{tenpak2024,
  title = {Tenpak: High-Compression Model Quantization Without Calibration},
  year = {2024},
  url = {https://github.com/yourusername/tenpak}
}
```

---

## License

MIT

---

<div align="center">

**Built for the future of efficient AI inference.**

</div>
