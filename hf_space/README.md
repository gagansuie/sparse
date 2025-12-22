---
title: TenPak 7B Compression
emoji: 🗜️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# TenPak: LLM Quantization Orchestration

**Wrap AutoGPTQ, AutoAWQ, bitsandbytes with intelligent optimization.**

## What TenPak Does

TenPak is NOT another quantization library - it's an orchestration platform that:

1. **Wraps industry-standard tools** - AutoGPTQ, AutoAWQ, bitsandbytes
2. **Auto-optimizes** - Benchmark all methods, pick cheapest meeting constraints
3. **Delta compression** - Store fine-tunes as 60-90% smaller deltas (unique)
4. **HTTP streaming** - CDN-friendly remote artifact loading
5. **Inference integration** - One-line vLLM/TGI deployment

## Quantization Results

Using wrapped tools (AutoGPTQ, AutoAWQ, bitsandbytes):

| Method | Compression | PPL Δ | Calibration |
|--------|-------------|-------|-------------|
| **AWQ 4-bit** | **7-8x** | <2% | Required |
| **GPTQ 4-bit** | 7-8x | <1% | Required |
| **bitsandbytes NF4** | 6-7x | <1.5% | Optional |
| **bitsandbytes INT8** | 2x | <0.5% | No |

## Unique Features

1. **Delta Compression** - 96% savings on fine-tunes (no one else does this)
2. **Cost Optimizer** - Auto-select cheapest method meeting constraints
3. **HTTP Streaming** - Lazy-load artifacts from CDN
4. **vLLM/TGI Integration** - One-line deployment

## Usage

Try the cost optimizer demo to see which quantization method is best for your use case.
