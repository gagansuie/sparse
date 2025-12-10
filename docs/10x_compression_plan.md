# 10x Compression Plan: The Path to 3.2 Bits Per Weight

## Executive Summary

**UPDATE: Experiments completed. Key finding: 10x with <1% PPL requires calibration.**

### Experimental Results (Dec 2024)

| Approach | Compression | PPL Δ | Verdict |
|----------|-------------|-------|---------|
| Pure PQ | 31.56x | +5648% | Compression works, quality destroyed |
| PQ + residual | 4-4.5x | +1.6-9% | Not better than INT4 |
| NF3 (3-bit) | 6-9x | +14-54% | Catastrophic |
| **INT3 packed (g=32)** | **8.00x** | +20.91% | Compression works, PPL needs optimization |
| **INT4 opt (iterative)** | **5.33x** | **-0.42%** | **🎯 BETTER THAN BASELINE!** |
| INT4 g16_fp16 | 5.33x | +1.04% | Good |
| INT4 g32_fp16 | 6.40x | +2.55% | Higher compression |
| INT4 g8_fp16 | 4.00x | +1.73% | Baseline |

### Implemented (with --features calibration)

- `int3_cal_v1` - **8x compression**, but +21% PPL (needs GPTQ-style optimization)
- `mixed_cal_v1` - Mixed precision (needs packing fix)
- Calibration scripts: `scripts/calibrate.py`, `scripts/eval_calibrated.py`

### What We Learned About GPTQ

Attempted GPTQ-style weight updates with diagonal Hessian approximation - **failed catastrophically** (+57588% PPL).

**Proper GPTQ requires:**
1. Full Hessian matrix H = X^T * X (not just diagonal)
2. Cholesky decomposition for efficient H^{-1}
3. Column-wise weight updates across entire matrix
4. Python preprocessing step with real forward passes

This is why AWQ/GPTQ implementations require a **calibration dataset** - they need real activations to compute the Hessian.

### Python GPTQ Experiments (Dec 2024)

We implemented full GPTQ in Python (`scripts/gptq_calibrate.py`):

| Config | Compression | PPL Δ | Result |
|--------|-------------|-------|--------|
| Full INT3 all layers | ~10x | +250,000% | Catastrophic |
| Mixed INT4/INT3 | ~10x | +60,000% | Catastrophic |
| INT3 MLP only | 8x | +21% | Best INT3 result |
| **INT4 g8_fp16** | **4x** | **+0.59%** | **Production ready** |

**Conclusion:** Even with proper Hessian collection, quantizing attention layers below INT4 causes catastrophic quality loss. The GPTQ paper's results likely rely on:
1. Much larger models (7B+) with more redundancy
2. More sophisticated block-diagonal Hessian handling
3. Careful numerical precision in weight updates

### The Hard Truth

**Without calibration, ~4x compression with <1% PPL is the fundamental limit.**

To achieve 10x compression:
1. **Option A**: Accept lightweight calibration (128 samples, 5 min)
2. **Option B**: Accept 2-5% PPL degradation
3. **Option C**: Use training-aware quantization (requires fine-tuning)

---

## Original Analysis

To achieve 10x compression (3.2 bits/weight) with <1% PPL delta, we must move beyond scalar quantization. This document outlines 5 approaches ranked by feasibility.

---

## Approach 1: Product Quantization (PQ) + Residual
**Feasibility: HIGH | Expected: 8-12x compression | PPL: <1%**

### The Idea
Instead of quantizing weights individually, treat groups as vectors and use a learned codebook.

```
Traditional INT4:
  weight[0..7] → 8 × 4 bits = 32 bits

Product Quantization:
  weight[0..7] → lookup codebook[index] → 8 bits (index only!)
  + 2-bit residual correction → 16 bits
  = 24 bits for 8 weights = 3 bits/weight = 10.67x compression
```

### Why It Could Work
- Codebooks capture common weight patterns
- Neural network weights are NOT uniformly distributed
- Similar technique used in FAISS (Facebook's vector search) at massive scale
- ResQ paper shows 2-3 bit PQ works on transformers

### Implementation Plan
1. Cluster weight vectors (k-means on 8-weight vectors)
2. Create codebook of 256 representative vectors
3. For each weight group, find nearest codebook entry
4. Store 8-bit index + optional residual
5. Codebook shared per layer or per tensor

### Storage Math
```
Per tensor (N weights):
- Indices: N/8 bytes
- Codebook: 256 × 8 × 2 bytes = 4KB (amortized ~0)
- Optional residual: N/8 bytes (if 2-bit per weight)

Without residual: 1 bit/weight = 32x compression (but ~5% PPL)
With 2-bit residual: 3 bits/weight = 10.67x compression
```

### Research References
- [Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)
- [Compressing BERT](https://arxiv.org/abs/1910.01108)
- [ResQ: Residual Quantization](https://arxiv.org/abs/2303.08424)

---

## Approach 2: Mixed Precision by Layer Sensitivity
**Feasibility: HIGH | Expected: 6-8x compression | PPL: <1%**

### The Idea
Not all layers need the same precision. Quantize aggressively where it's safe.

### Layer Sensitivity Analysis (from our experiments)
```
MOST SENSITIVE (keep high precision):
- Attention Q/K projections
- First/last transformer blocks
- Layer norms (never quantize)

MODERATELY SENSITIVE:
- Attention V/O projections
- MLP up projection

LEAST SENSITIVE (compress aggressively):
- MLP down projection
- Middle transformer blocks
- Embeddings (surprisingly robust)
```

### Proposed Configuration
```
Layer Type          | Precision | Bits/Weight | % of Model
--------------------|-----------|-------------|------------
Embeddings          | INT4 g=32 | 4.5         | 15%
Attention Q/K       | INT4 g=8  | 8.0         | 10%
Attention V/O       | INT4 g=16 | 5.0         | 10%
MLP (first/last 2)  | INT4 g=8  | 8.0         | 10%
MLP (middle)        | INT3 g=8  | 5.0         | 45%
Other               | INT4 g=32 | 4.5         | 10%
--------------------|-----------|-------------|------------
Weighted Average    |           | ~5.2        | = 6.2x
```

### With More Aggressive INT3
If we can make INT3 work (see Approach 4), middle MLP at 4 bits/weight:
```
Weighted average: ~4.0 bits/weight = 8x compression
```

---

## Approach 3: Sparse-Quantized Hybrid
**Feasibility: MEDIUM | Expected: 8-10x compression | PPL: <2%**

### The Idea
Combine structured sparsity with quantization.

```
Step 1: Identify 50% of weights to keep (by magnitude or gradient)
Step 2: Quantize remaining weights to INT4
Step 3: Store sparse indices + quantized values

Effective compression:
- 50% sparsity = 2x
- INT4 on remaining = 4x
- Combined = 8x
```

### The Challenge
Pure magnitude pruning doesn't work well without fine-tuning.
BUT: We can use activation-based importance (lightweight calibration).

### Implementation
1. Run 128 samples through model (5 minutes)
2. Track which weights contribute most to output
3. Keep top 50% by importance
4. INT4 quantize the kept weights
5. Store as sparse tensor (CSR format)

### Storage Math
```
For N weights at 50% sparsity:
- Values: N/2 × 0.5 bytes (INT4) = N/4 bytes
- Indices: N/2 × 2 bytes (INT16) = N bytes
- Total: 1.25N bytes vs 4N original = 3.2x

Wait, that's worse! Indices kill us.
```

### Better: Block Sparsity (2:4)
```
2:4 sparsity: Keep 2 of every 4 weights
- No indices needed (pattern is fixed)
- 50% weights × 4 bits = 2 bits/weight
- Plus 1-bit mask per 4 weights = 0.25 bits
- Total: 2.25 bits/weight = 14x compression!

But: Requires specific hardware support or custom kernels
```

---

## Approach 4: Learned Optimal Quantization (LOQ)
**Feasibility: MEDIUM | Expected: 6-10x compression | PPL: <1%**

### The Idea
INT3 failed because we used uniform quantization. What if we learned optimal quantization points?

### The Problem with Uniform INT3
```
Uniform INT3: -3, -2, -1, 0, 1, 2, 3 (7 levels centered at 0)

But weight distributions are NOT uniform!
They're typically:
- Concentrated around 0
- With long tails
- Often asymmetric
```

### Solution: Non-Uniform Quantization (NF3/NF4 style)
```
Learn quantization points from the actual distribution:

For Gaussian-like weights:
  levels = [-1.5, -0.7, -0.3, 0, 0.3, 0.7, 1.5] × scale

This puts more levels where weights are dense!
```

### Implementation
1. Analyze weight distribution per tensor
2. Compute optimal 8 levels (k-means or Lloyd's algorithm)
3. Store 8 levels (8 × 4 bytes = 32 bytes) per tensor
4. Quantize to 3 bits using these levels

### Storage Math
```
Per tensor (N weights):
- Weights: N × 3 bits = 0.375N bytes
- Levels: 32 bytes (negligible)
- Scale: 4 bytes (negligible)
- Group overhead (g=8): ~0.5 bits/weight

Total: ~3.5 bits/weight = 9.1x compression
```

### NF4 Reference
bitsandbytes uses NF4 (4-bit normalized float) which is similar:
- 16 levels optimized for Gaussian distribution
- Achieves better quality than uniform INT4
- We could create NF3 (8 levels)

---

## Approach 5: Low-Rank + Quantization (with Lightweight Calibration)
**Feasibility: LOW-MEDIUM | Expected: 10-15x compression | PPL: 1-3%**

### The Idea
Decompose weight matrices, then quantize the factors.

```
W (4096 × 4096) ≈ U (4096 × r) × V (r × 4096)

If r = 256 (rank 256):
- Original: 16M weights
- Factored: 2 × 4096 × 256 = 2M weights
- Already 8x compression before quantization!
- Quantize U, V to INT4 → 32x total!
```

### Why It Failed Before
Our SVD experiment failed because:
1. We used mathematical SVD (minimizes Frobenius norm)
2. But we need OUTPUT-aware decomposition (minimizes task loss)

### Solution: Calibration-Guided Rank Selection
```python
# Instead of pure SVD, use reconstruction error on real data
for rank in [512, 256, 128, 64]:
    U, S, V = svd(W)
    W_approx = U[:, :rank] @ diag(S[:rank]) @ V[:rank, :]
    
    # Test on 128 calibration samples
    error = measure_output_error(W_approx, calibration_data)
    
    if error < threshold:
        break  # Found optimal rank
```

### The Catch
This requires calibration data. But it's lightweight:
- Only 128 samples
- Only forward passes (no gradients)
- ~5 minutes total

### Storage Math
```
Rank-128 factorization + INT4:
- U: 4096 × 128 × 0.5 bytes = 256 KB
- V: 128 × 4096 × 0.5 bytes = 256 KB  
- Total: 512 KB vs 32 MB original = 64x compression

Even rank-512 + INT4 = 16x compression
```

---

## Recommended Path: Phased Implementation

### Phase 1: Quick Wins (This Week)
**Target: 6x compression**

1. **Implement Mixed Precision** - 2 days
   - Profile layer sensitivity on GPT-2
   - Apply g=8 to sensitive, g=32 to robust layers
   - Expected: 5-6x compression, <1% PPL

2. **Implement NF4-style quantization** - 2 days
   - Use pre-computed optimal levels for Gaussian
   - Apply to INT4 (will help INT3 later)
   - Expected: Same compression, -0.1% PPL improvement

### Phase 2: Product Quantization (Next Week)
**Target: 8-10x compression**

1. **Build PQ infrastructure** - 3 days
   - K-means clustering on weight vectors
   - Codebook storage format
   - Fast lookup implementation

2. **PQ + Residual quantization** - 2 days
   - 8-bit codebook index + 2-bit residual
   - Expected: 10x compression, <2% PPL

3. **Tune for quality** - 2 days
   - Per-layer codebook vs global
   - Optimal vector size (4, 8, 16)
   - Residual precision (1, 2, 3 bits)

### Phase 3: Hybrid Systems (Week 3)
**Target: 10x+ compression with <1% PPL**

1. **Combine best approaches**
   - PQ for MLP layers (most weights)
   - INT4 g=8 for attention (quality-critical)
   - Mixed precision across blocks

2. **Add lightweight calibration option**
   - 128 samples, 5 minutes
   - For users who want extra quality
   - Enables low-rank factorization path

---

## Concrete First Experiment: PQ Baseline

```python
# Test Product Quantization feasibility

def product_quantize(weight, codebook_size=256, vector_size=8):
    """
    Quantize weight matrix using product quantization.
    
    Args:
        weight: [out_features, in_features] tensor
        codebook_size: number of centroids (256 = 8-bit index)
        vector_size: weights per vector (8 = process in groups)
    
    Returns:
        indices: [out_features, in_features // vector_size] uint8
        codebook: [codebook_size, vector_size] float16
    """
    # Reshape to vectors
    flat = weight.reshape(-1, vector_size)  # [N/8, 8]
    
    # K-means clustering
    codebook, indices = kmeans(flat, codebook_size)
    
    return indices.astype(np.uint8), codebook.astype(np.float16)

def product_dequantize(indices, codebook):
    """Reconstruct weights from indices and codebook."""
    return codebook[indices].reshape(original_shape)

# Storage: indices = N/8 bytes, codebook = 256*8*2 = 4KB
# Total for 1M weights: 125KB + 4KB = 129KB
# Original FP32: 4MB
# Compression: 31x (but quality?)
```

---

## Success Metrics

| Milestone | Compression | PPL Delta | Timeline |
|-----------|-------------|-----------|----------|
| Phase 1 Complete | 6x | <1% | Week 1 |
| Phase 2 Complete | 8x | <2% | Week 2 |
| Phase 3 Complete | **10x** | **<1%** | Week 3 |

---

## Key Insight

**The path to 10x is NOT about finding one magic technique.**

It's about:
1. Understanding which weights matter
2. Applying the right technique to each layer
3. Combining complementary approaches
4. Accepting that <1% PPL at 10x may require SOME form of data-aware optimization

**The goal is achievable. Let's execute.**
