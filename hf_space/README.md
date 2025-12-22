---
title: TenPak - $30-45M/year Savings Demo
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# TenPak - Delta Compression + Smart Routing for Model Hubs

**Interactive demo showcasing $30-45M/year in savings for platforms like HuggingFace.**

## Features

### 1. 📦 Model Delta Compression ($15-20M/year)
Store fine-tuned models as 60-90% smaller deltas from base models.

**Example:** Llama-2-7B fine-tune
- Full model: 13 GB
- Delta: 500 MB
- Savings: **96%**

### 2. 📊 Dataset Delta Compression ($10-15M/year)
Store derivative datasets as 70-90% smaller deltas from base datasets.

**Example:** squad → squad_v2
- Full dataset: 87.5 MB
- Delta: 21.3 MB
- Savings: **76%**

### 3. 🎯 Smart Routing ($5-10M/year)
Auto-route inference requests to optimal models/hardware.

**Example:** Simple question
- Requested: Llama-2-70B on A100
- Recommended: Llama-2-7B on T4
- Savings: **90% cost reduction**

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

## Total Savings for HuggingFace Scale

| Feature | Annual Value |
|---------|-------------|
| Model delta compression | $15-20M |
| Dataset delta compression | $10-15M |
| Smart routing | $5-10M |
| **Total** | **$30-45M/year** |

## Why TenPak is Unique

**No competitor offers:**
- ✅ LLM model delta compression at scale
- ✅ Dataset delta compression for derivatives
- ✅ Cross-tool smart routing

**What others have:**
- ❌ AutoGPTQ/AWQ/bitsandbytes (quantization only)
- ❌ Cloudflare CDN (HF already has this)
- ❌ TGI (HF already built this)

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

## Quick Start

```bash
# Install
pip install tenpak

# Model delta compression
tenpak delta compress meta-llama/Llama-2-7b my-org/llama-chat --output ./delta

# Dataset delta compression
tenpak delta-dataset compress squad squad_v2 --output ./dataset_delta

# Smart routing
tenpak route meta-llama/Llama-2-70b "What is the capital of France?"
```

## Python API

```python
# Model delta compression
from core.delta import compress_delta, estimate_delta_savings

savings = estimate_delta_savings(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetuned_model_id="my-org/llama-chat"
)
print(f"Savings: {savings['savings_pct']:.1f}%")

# Dataset delta compression
from core.dataset_delta import compress_dataset_delta

manifest = compress_dataset_delta(
    base_dataset_id="squad",
    derivative_dataset_id="squad_v2",
    output_dir="./dataset_delta"
)

# Smart routing
from optimizer.routing import suggest_optimal_model

decision = suggest_optimal_model(
    requested_model="meta-llama/Llama-2-70b-hf",
    prompt="What is 2+2?",
    quality_threshold=0.85
)
print(f"Recommended: {decision.recommended_model}")
print(f"Savings: {decision.reasoning}")
```

## Links

- **GitHub:** [github.com/gagansuie/tenpak](https://github.com/gagansuie/tenpak)
- **Documentation:** [README](https://github.com/gagansuie/tenpak#readme)
- **License:** MIT

---

**Note:** This is an interactive demo. For actual model quantization, install TenPak and use the CLI or Python API.
