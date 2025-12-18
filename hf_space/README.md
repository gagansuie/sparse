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

# TenPak-10X: Calibration-Guided Hierarchical Compression

**Novel approach for 10x+ compression with <2% PPL delta.**

## Target: Meta AI Research

Achieve **10x+ compression** on 7B+ models with **<2% perplexity degradation**.

## Key Innovations

| Innovation | Description |
|------------|-------------|
| **Hessian-Weighted VQ** | Importance-aware codebook learning using activation statistics |
| **Calibrated K-Means** | Weight clustering guided by E[x²] Hessian proxy |
| **Adaptive Vec Dim** | Smaller vectors for attention (quality), larger for MLP (compression) |
| **GPU-Accelerated** | torch.compile + vectorized scatter_add for 5-10x speedup |

## Current Results (Mistral-7B)

| Version | Method | Compression | PPL Δ | Status |
|---------|--------|-------------|-------|--------|
| v32 | INT4 g=512 | 7.26x | +2.25% | ⚠️ Best so far |
| v22 | Selective 3:4 sparsity | 7.35x | +1.62% | ⚠️ Best quality |
| v38 | Calibrated VQ | 26.8x | +371521% | ❌ Too aggressive |

## Approach

1. **Calibration Phase** - Collect activation statistics (AWQ-style)
2. **Hessian Proxy** - Use E[x²] to estimate weight importance
3. **Adaptive Compression** - More protection for attention, aggressive on MLP
4. **Vector Quantization** - Codebook learning with importance weighting

## Usage

Select a 7B model and click "Run Evaluation" to test the compression.
