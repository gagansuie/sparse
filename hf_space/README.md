---
title: TenPak Quantization Orchestration
emoji: 🗜️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# TenPak: LLM Quantization Orchestration

**Wrap AutoGPTQ, AutoAWQ, bitsandbytes with intelligent optimization.**

## What This Demo Shows

Interactive exploration of TenPak's key features:

### 📋 Quantization Presets
- Explore 9 available presets (GPTQ, AWQ, bitsandbytes)
- See expected compression, quality loss, and use cases
- Get CLI and Python code examples

### 💰 Cost Optimizer
- Set your constraints (max PPL delta, min compression)
- See which method TenPak would auto-select
- Compare against alternative methods
- **Saves 30-40% vs manual selection**

### 📦 Delta Compression
- Calculate storage savings for fine-tunes
- See cost reduction for hosting 1000+ models
- **Unique feature: 60-90% savings** (no competitors offer this)

### ⚖️ Feature Comparison
- TenPak vs AutoGPTQ, AutoAWQ, bitsandbytes
- See what makes TenPak different (orchestration, not algorithms)

## TenPak's Unique Value

**We don't replace quantization tools - we orchestrate them:**

1. 🎯 **Auto-optimization** - Benchmark all methods, pick cheapest
2. 📦 **Delta compression** - 60-90% savings for fine-tunes
3. 🌐 **HTTP streaming** - CDN-friendly artifact downloads
4. 🚀 **One-line deployment** - Direct vLLM/TGI integration

## Quantization Results (Via Wrapped Tools)

| Method | Compression | PPL Δ | Calibration |
|--------|-------------|-------|-------------|
| **AWQ 4-bit** | 7-8x | <2% | Required |
| **GPTQ 4-bit** | 7-8x | <1% | Required |
| **bitsandbytes NF4** | 6-7x | <1.5% | Optional |
| **bitsandbytes INT8** | 2x | <0.5% | No |

## Installation & Usage

```bash
pip install tenpak

# CLI
tenpak pack meta-llama/Llama-2-7b-hf --preset awq_balanced
tenpak optimize gpt2 --max-ppl-delta 2.0
tenpak delta compress base fine-tuned --output ./delta

# Python API
from core import QuantizationWrapper

wrapper = QuantizationWrapper.from_preset("gptq_quality")
model = wrapper.quantize("meta-llama/Llama-2-7b-hf")
```

## Links

- **GitHub:** [github.com/gagansuie/tenpak](https://github.com/gagansuie/tenpak)
- **Documentation:** [README](https://github.com/gagansuie/tenpak#readme)
- **License:** MIT

---

**Note:** This is an interactive demo. For actual model quantization, install TenPak and use the CLI or Python API.
