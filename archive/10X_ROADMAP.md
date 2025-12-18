# TenPak 10x Compression Roadmap

**Goal: 10x+ compression with <1% PPL delta on 7B+ models**

---

## Executive Summary

### Current State (Dec 2024)

| Codec | Compression | PPL Δ | Calibration | Status |
|-------|-------------|-------|-------------|--------|
| `int4_residual_v1` | 3.2x | -0.35% | ❌ None | ✅ Production |
| `int4_opt_llama_v1` | 4.0x | +0.88% | ❌ None | ✅ Production |
| `int4_calibrated_v1` | 4.9x | +1.46% | ✅ Required | ✅ Production |
| `int4_g256` | 7.76x | +7.94% | ❌ None | ❌ PPL too high |

### Key Insight

**Calibration-free ceiling is ~4x compression.** To achieve 10x, we MUST use calibration.

**Larger models compress better:**
- GPT-2 (124M): 4.82x @ +0.68% PPL
- TinyLlama (1.1B): 4.58x @ +0.15% PPL  
- GPT-2 XL (1.5B): **6.03x @ -0.21% PPL** ← Model improves!

**Extrapolation:** 7B+ models should achieve **8-10x compression with <1% PPL** (possibly negative).

---

## The 10x Challenge: Math

To achieve 10x compression vs FP32, we need **~3.2 bits per weight**.

| Method | Bits/Weight | Compression vs FP32 |
|--------|-------------|---------------------|
| FP32 | 32 | 1x |
| FP16 | 16 | 2x |
| INT8 | 8 | 4x |
| INT4 g=8 | 8 | 4x |
| INT4 g=128 | 4.25 | 7.5x |
| INT4 g=256 | 4.06 | 7.9x |
| **INT3 g=16** | **~4.5** | **~7x** |
| **INT2 g=16** | **~3.5** | **~9x** |
| **Target** | **~3.2** | **10x** |

**Problem:** INT2/INT3 alone destroys quality. Need smarter approaches.

---

## Approaches to 10x

### Tier 1: Most Promising (Calibration-Based)

#### 1. GPTQ + Outlier Extraction 🔥
**Status:** Not fully implemented
**Target:** 10x compression, <1% PPL

**How it works:**
1. Extract top 0.1% outlier weights → store at FP16 (negligible overhead)
2. Apply GPTQ to remaining weights with large groups (g=256)
3. GPTQ's Hessian-weighted reconstruction minimizes output perturbation

**Why it should work:**
- GPTQ already achieves 8x with <1% PPL on 7B models (published results)
- Outlier extraction handles the tail that ruins large group quantization
- Calibration data provides importance signal for optimal reconstruction

**Implementation needed:**
- [ ] `scripts/calibrate_gptq.py` - Hessian computation (requires forward/backward passes)
- [ ] `src/lib.rs` - GPTQ codec with outlier extraction
- [ ] HF Space integration with calibration data upload

**Reference:** [GPTQ Paper](https://arxiv.org/abs/2210.17323)

---

#### 2. AWQ + Adaptive Groups 🔥
**Status:** Partially implemented (`int4_awq_10x_v1`)
**Target:** 8-10x compression, <1% PPL

**How it works:**
1. Compute activation statistics on calibration data
2. Scale weights by `activation_scale^0.5` before quantization (preserves important channels)
3. Use adaptive group sizes based on layer sensitivity
4. Large groups (g=128-256) for insensitive layers, small (g=8-16) for sensitive

**Why it should work:**
- AWQ is state-of-the-art for calibration-based quantization
- Activation scaling protects salient weights from quantization error
- Adaptive groups maximize compression while preserving quality

**Current results (failed without proper calibration):**
| Model | Compression | PPL Δ | Status |
|-------|-------------|-------|--------|
| TinyLlama | 7.53x | +8.6% | ❌ |

**What's missing:**
- [ ] Proper activation statistics collection (current implementation is incomplete)
- [ ] Per-channel importance scores
- [ ] Gradient-based scale optimization

**Reference:** [AWQ Paper](https://arxiv.org/abs/2306.00978)

---

#### 3. CALDERA: Low-Rank + Quantization Hybrid 🔥
**Status:** Implemented (`caldera_v1`), needs calibration
**Target:** 8-10x compression, <1% PPL

**How it works:**
```
W ≈ Q + L @ R

Where:
- Q = INT2 quantized backbone (2 bits/weight)
- L = Low-rank left factor (rank 8-32), INT8 quantized
- R = Low-rank right factor, INT8 quantized
```

**Why it should work:**
- Low-rank captures structured redundancy in weight matrices
- INT2 backbone is sufficient when low-rank handles important components
- Published results show 10x on Llama-2-7B with <1% PPL

**Storage math:**
```
For 4096x4096 matrix with rank=32:
- INT2 backbone: 4096*4096 * 0.25 bytes = 4MB
- L factor: 4096*32 * 1 byte = 128KB
- R factor: 32*4096 * 1 byte = 128KB
- Total: 4.25MB vs 64MB FP32 = 15x compression!
```

**What's missing:**
- [ ] Calibration-aware rank selection
- [ ] Importance-weighted SVD (preserve important directions)
- [ ] Joint optimization of Q, L, R

**Reference:** [CALDERA Paper](https://arxiv.org/abs/2312.09852)

---

#### 4. AQLM: Additive Quantization with Learned Codebooks 🔥
**Status:** Implemented (`aqlm_v1`), needs calibration
**Target:** 8-10x compression, <1% PPL

**How it works:**
```
w ≈ codebook1[i1] + codebook2[i2]

Where:
- codebook1, codebook2 are learned via k-means on calibration data
- i1, i2 are low-bit indices (2-4 bits each)
```

**Why it should work:**
- Additive structure dramatically increases representable values
- Single 4-bit: 16 values
- Two 2-bit additive: 16 combinations × 2 scales = much more precision
- Published results show 2 bits/weight with <1% PPL

**What's missing:**
- [ ] Gradient-based codebook learning (current uses k-means only)
- [ ] Beam search for optimal index assignment
- [ ] Calibration loss minimization

**Reference:** [AQLM Paper](https://arxiv.org/abs/2401.06118)

---

#### 5. PocketLLM: Vector Quantization 
**Status:** Implemented (`pocketllm_v1`, `pocketllm_v2`)
**Target:** 10x+ compression, <1% PPL

**How it works:**
1. Group weights into vectors (e.g., 8 consecutive weights)
2. Learn codebook via k-means clustering
3. Replace each vector with nearest codebook entry (8-bit index)

**Storage math:**
```
For vectors of size 8 with 256 codebook entries:
- Indices: 1 byte per 8 weights = 0.125 bytes/weight
- Codebook: 256 * 8 * 2 bytes = 4KB (negligible)
- Total: ~0.13 bytes/weight = ~30x compression!
```

**Current results:**
| Variant | Compression | PPL Δ | Status |
|---------|-------------|-------|--------|
| v1 (vec=8, cb=256) | 5.32x | +1.39% | ❌ |
| v2 (vec=4, cb=256) | 4.00x | +0.23% | ✅ |

**What's missing:**
- [ ] Calibration-aware codebook learning
- [ ] Product quantization (multiple codebooks)
- [ ] Residual vector quantization

**Reference:** [PocketLLM Paper](https://arxiv.org/abs/2311.04066)

---

### Tier 2: Hybrid Approaches

#### 6. TenPak-X v3: Ultimate Hybrid 
**Status:** Design phase
**Target:** 10x compression, <1% PPL

**Combines ALL techniques:**
```
W ≈ L @ R + Codebook[indices] + Outliers + Residual_INT2

Layer 1: Low-rank decomposition (rank=32)
Layer 2: Vector quantization of residual (vec=4, cb=512)
Layer 3: Outlier extraction (top 0.1% at FP16)
Layer 4: INT2 residual for remaining error
```

**Storage estimate:**
```
For 4096x4096 matrix:
- Low-rank (rank=32): 256KB
- VQ indices: 2MB
- Outliers (0.1%): 32KB
- INT2 residual: 4MB
- Total: ~6.3MB vs 64MB = ~10x
```

**Implementation plan:**
1. Implement calibration infrastructure
2. Test each component independently
3. Joint optimization with calibration loss

---

#### 7. SpinQuant + GPTQ
**Status:** Partially implemented
**Target:** 8x compression, <1% PPL

**How it works:**
1. Apply learned rotation matrix to spread outliers
2. GPTQ quantization in rotated space
3. Store rotation for dequantization

**Why it might work:**
- Rotation eliminates outliers (main cause of quantization error)
- GPTQ optimizes reconstruction in rotated space
- Published SpinQuant achieves 4-bit with <1% PPL

**Current results (without GPTQ):**
| Model | Compression | PPL Δ | Status |
|-------|-------------|-------|--------|
| TinyLlama | 7.76x | +10.3% | ❌ |

**What's missing:**
- [ ] Calibration-based rotation learning
- [ ] GPTQ integration
- [ ] Per-layer optimal rotation

**Reference:** [SpinQuant Paper](https://arxiv.org/abs/2405.16406)

---

### Tier 3: Novel Research Directions

#### 8. Learned Quantization Levels (NF4-style)
Instead of uniform INT4 levels, learn optimal levels from data.

```python
# Instead of: levels = [0, 1, 2, ..., 15] / 15 * (max - min)
# Learn: levels = model.learned_levels  # optimized via gradient descent
```

#### 9. Mixed-Precision by Layer Type
```
- Embedding: FP16 (small, critical)
- Q/K projections: INT4 g=8 (attention sensitive)
- V/O projections: INT4 g=32
- MLP gate/up: INT4 g=64
- MLP down: INT4 g=128 (most robust)
- LM head: FP16 (critical for output)
```

#### 10. Importance-Based Weight Pruning + Quantization
```
1. Prune 50% of weights (importance-based)
2. Quantize remaining to INT4
3. Store sparse indices + INT4 values
4. Effective: 2 bits/weight average
```

---

## Implementation Roadmap

### Phase 1: Calibration Infrastructure (1 week)
- [ ] `scripts/calibrate.py` - Unified calibration data collection
  - Forward pass on calibration set (WikiText-2, C4)
  - Collect activation statistics per layer
  - Compute Fisher information / Hessian diagonal
- [ ] `src/calibration.rs` - Rust calibration data structures
- [ ] Update `FloatBundle` format to include calibration data

### Phase 2: GPTQ Implementation (1 week)
- [ ] Hessian computation (diagonal approximation)
- [ ] Iterative weight update (GPTQ algorithm)
- [ ] Integration with outlier extraction
- [ ] Benchmarks on TinyLlama, Mistral-7B

### Phase 3: AWQ Enhancement (1 week)
- [ ] Per-channel activation scales
- [ ] Optimal scale search (grid or gradient)
- [ ] Adaptive group size selection
- [ ] Benchmarks

### Phase 4: CALDERA/AQLM (2 weeks)
- [ ] Importance-weighted SVD
- [ ] Calibration-aware codebook learning
- [ ] Gradient-based optimization
- [ ] Benchmarks

### Phase 5: Integration & Testing (1 week)
- [ ] HF Space with calibration upload
- [ ] Comprehensive benchmark suite
- [ ] Documentation

---

## Quick Wins (Can Implement Today)

### 1. Test int4_opt_llama on Mistral-7B
Expected: 4x compression, <1% PPL

### 2. Test int4_residual on Mistral-7B  
Expected: 3.2x compression, possibly negative PPL delta

### 3. Add simple outlier extraction to existing codecs
```python
def extract_outliers(weight, threshold=0.001):
    """Extract top 0.1% of weights by magnitude."""
    flat = weight.flatten()
    k = max(1, int(len(flat) * threshold))
    top_k_idx = torch.topk(flat.abs(), k).indices
    outliers = flat[top_k_idx]
    flat[top_k_idx] = 0  # Zero out outliers
    return weight.view_as(weight), top_k_idx, outliers
```

---

## References

1. **GPTQ** - [Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
2. **AWQ** - [Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
3. **CALDERA** - [Low-Rank Quantization](https://arxiv.org/abs/2312.09852)
4. **AQLM** - [Additive Quantization of Language Models](https://arxiv.org/abs/2401.06118)
5. **PocketLLM** - [Optimizing GPU Memory for LLM Inference](https://arxiv.org/abs/2311.04066)
6. **SpinQuant** - [LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
7. **QuIP#** - [Even Better LLM Quantization with Hadamard Incoherence](https://arxiv.org/abs/2402.04396)

---

## Success Criteria

| Model | Target Compression | Max PPL Delta | Calibration Time |
|-------|-------------------|---------------|------------------|
| TinyLlama (1.1B) | 8x | <2% | <5 min |
| Mistral-7B | 10x | <1% | <30 min |
| Llama-2-13B | 12x | <1% | <60 min |

---

## Appendix: Failed Approaches

| Approach | Why It Failed |
|----------|---------------|
| Large groups without calibration | PPL explodes (7-20%) |
| INT3/INT2 alone | Too few levels, massive error |
| Uniform sparsity | Removes important weights |
| Hadamard rotation without calibration | Doesn't adapt to weight distribution |
| k-means codebook without calibration | Codebook not optimized for reconstruction loss |

**Lesson:** Achieving 10x requires calibration to identify and preserve important weights.
