# TenPak Strategy V2: Path to 10x @ <2% PPL

**Date:** December 2024  
**Goal:** 10x+ compression with <2% PPL delta on 7B-70B models  
**Status:** Refining approach after extensive experimentation

---

## Part 1: What We've Learned

### Verified Results Summary (Mistral-7B)

| Version | Method | Compression | PPL Δ | Status | Key Insight |
|---------|--------|-------------|-------|--------|-------------|
| v10 | INT4 layer-aware | 7.42x | +1.47% | ⚠️ | Best simple approach |
| v16 | Fisher-weighted groups | 7.48x | +1.87% | ⚠️ | Dynamic allocation helps |
| v22 | Selective 3:4 sparsity | 7.35x | +1.62% | ✅ | **Sparsity on bottom 25% MLP works** |
| v32 | INT4 g=512 | 7.26x | +2.25% | ⚠️ | Large groups hurt quality |
| v38 | Calibrated VQ (aggressive) | 26.8x | +371521% | ❌ | **VQ alone destroys quality** |

### Calibration-Free Ceiling: ~4-5x

| Model | Method | Compression | PPL Δ |
|-------|--------|-------------|-------|
| GPT-2 | TenPak-X | 4.08x | +0.03% |
| TinyLlama | TenPak-X | 4.36x | **-0.02%** |
| GPT-2 XL | Calibrated | 6.03x | **-0.21%** |

**Key Finding:** Larger models compress better. 7B should exceed 1.5B results.

### What Works

1. **INT4 with small groups (g=64-256)** for attention layers
2. **AWQ-style activation scaling** (`scale^0.5` before quantization)
3. **Outlier extraction (0.5%)** - protects important weights
4. **Fisher/importance-guided allocation** - more bits for sensitive layers
5. **Selective 3:4 sparsity on MLP** - bottom 25% by importance
6. **Low-rank + residual** - TenPak-X approach

### What Doesn't Work

1. **Aggressive VQ** (vec_dim > 2) - destroys quality without calibration
2. **INT2/INT3 alone** - too few quantization levels
3. **Large groups for attention** (g > 512) - attention is sensitive
4. **Full 2:4 sparsity** - removes too much information
5. **Physics-inspired approaches** - all failed catastrophically
6. **No outlier extraction** - v17 showed +32.72% PPL

---

## Part 2: Published Methods That Achieved 10x @ <2%

### 1. AQLM (Additive Quantization of Language Models)
**Compression:** 2 bits/weight → 16x  
**PPL Delta:** <1% on Llama-2-7B  
**How it works:**
- Two learned codebooks (additive): `w ≈ cb1[i1] + cb2[i2]`
- Gradient-based codebook optimization (not k-means)
- Per-layer codebook learning
- **Requires:** Days of GPU time, calibration data

**What we tried:** Simple k-means VQ → failed  
**What's missing:** Gradient-based optimization, additive structure

### 2. CALDERA (Low-Rank + Quantization)
**Compression:** 10-15x  
**PPL Delta:** <1% on Llama-2-7B  
**How it works:**
- `W ≈ Q + L @ R`
- Q = INT2 quantized backbone
- L, R = Low-rank factors (INT8)
- Calibration-aware rank selection

**What we tried:** Basic low-rank + INT4 residual → 4.36x  
**What's missing:** INT2 backbone, calibration-aware rank

### 3. GPTQ (Gradient Post-Training Quantization)
**Compression:** 8x (4-bit with large groups)  
**PPL Delta:** <1% on most 7B+ models  
**How it works:**
- Hessian-weighted error compensation
- Iterative weight updates during quantization
- Per-row processing with column updates

**What we tried:** GPTQ-Lite (simplified) → similar to INT4  
**What's missing:** Full Hessian computation, iterative updates

### 4. QuIP# (Quantization with Incoherence Processing)
**Compression:** 8-10x (2-bit)  
**PPL Delta:** <1%  
**How it works:**
- Hadamard rotation to spread outliers
- Lattice codebook quantization
- Incoherence processing for uniform quantization

**What we tried:** Hadamard rotation alone → failed without GPTQ  
**What's missing:** Lattice codebooks, proper incoherence

---

## Part 3: Analysis - Why 10x is Hard

### The Math Problem

To achieve **10x compression vs FP32**, we need **~3.2 bits per weight**.

| Bits/Weight | Compression | Method |
|-------------|-------------|--------|
| 4.0 | 8x | INT4 only |
| 3.5 | 9.1x | INT4 + some INT3 |
| 3.2 | 10x | **TARGET** |
| 2.5 | 12.8x | INT2 + overhead |
| 2.0 | 16x | Pure INT2 |

### The Quality Problem

| Method | Bits | Quality Without Calibration | Quality With Calibration |
|--------|------|----------------------------|-------------------------|
| INT4 | 4.0-4.5 | ✅ <1% PPL | ✅ <0.5% PPL |
| INT3 | 3.0-3.5 | ❌ 5-30% PPL | ⚠️ 1-3% PPL |
| INT2 | 2.0-2.5 | ❌ 100%+ PPL | ✅ <1% PPL (AQLM) |

**Key Insight:** The gap between INT4 and INT2 quality is enormous without proper calibration. AQLM bridges this gap with learned codebooks.

### Why Our VQ Failed

Our v38 Calibrated VQ used **simple k-means clustering**:
- K-means minimizes L2 distance to centroids
- Does NOT minimize reconstruction loss on model outputs
- Important weights get same treatment as unimportant ones

AQLM uses **gradient-based optimization**:
- Directly minimizes output perturbation
- Codebooks trained end-to-end
- Takes days but produces quality codebooks

---

## Part 4: Refined Strategies for 10x

### Strategy A: GPTQ-Style Hessian Compensation (Most Promising)

**Target:** 8-10x @ <2% PPL  
**Complexity:** Medium  
**Time:** Minutes

**Approach:**
1. Compute Hessian diagonal (activation² as proxy)
2. Quantize in importance order (low Hessian first)
3. Compensate remaining weights for each quantized weight
4. Use INT4 with large groups (g=256-512)

**Key Difference from Our Current:**
- We quantize all at once, then try to compensate
- GPTQ quantizes ONE weight, compensates REMAINING, then next weight
- This iterative approach is critical

**Implementation:**
```python
def gptq_quantize(W, H):  # H = Hessian diagonal
    Q = W.clone()
    for i in range(cols):
        # Quantize column i
        w = Q[:, i]
        q = quantize(w)
        Q[:, i] = q
        
        # Compensate remaining columns
        error = (w - q) / H[i]
        Q[:, i+1:] -= error.outer(H[i+1:])
    return Q
```

### Strategy B: Hybrid Low-Rank + Conservative VQ (Novel)

**Target:** 8-10x @ <2% PPL  
**Complexity:** Medium  
**Time:** Minutes

**Approach:**
1. Low-rank decomposition (rank=32-64) captures ~30% of information
2. Conservative VQ (vec_dim=2, 256 entries) on residual → 4 bits
3. Sparse INT4 residual on important outliers
4. Total: ~3-4 bits average

**Math:**
```
For 4096x4096 matrix:
- Low-rank (rank=32): 4096*32*2 * 2 = 512KB → 0.25 bits/weight
- VQ indices: 4096*4096/2 * 1 = 8MB → 4 bits/weight  
- But VQ is on residual (70% of variance) → effective 2.8 bits
- Total: ~3.05 bits → 10.5x compression
```

**Key Innovation:** Apply VQ to LOW-RANK RESIDUAL, not raw weights. The residual is more uniform and compresses better.

### Strategy C: Importance-Guided Mixed Precision (Practical)

**Target:** 8-9x @ <2% PPL  
**Complexity:** Low  
**Time:** Minutes

**Approach:**
```
Layer Allocation by Fisher Importance:
- Top 10% (critical): INT4 g=32 → 5.0 bits
- Next 20% (high): INT4 g=64 → 4.5 bits
- Next 30% (medium): INT4 g=256 + 25% sparse → 4.0 bits
- Bottom 40% (low): INT4 g=512 + 50% sparse → 3.0 bits

Weighted average: 0.1*5 + 0.2*4.5 + 0.3*4 + 0.4*3 = 3.8 bits → 8.4x
```

**Enhancement:** Combine with 3:4 sparsity on bottom layers:
- 3:4 sparsity = keep 75% → 1.33x compression bonus
- Bottom 40% becomes: 3.0 bits * 0.75 = 2.25 effective bits
- New average: ~3.4 bits → 9.4x

### Strategy D: AQLM-Lite with Additive Codebooks (Research)

**Target:** 10-12x @ <2% PPL  
**Complexity:** High  
**Time:** Hours

**Approach:**
1. Learn two small codebooks (cb1: 16 entries, cb2: 16 entries)
2. Use additive structure: `w ≈ cb1[i1] + cb2[i2] * scale`
3. Train codebooks on calibration data with MSE loss
4. 4+4 bits for indices = 8 bits per 4-8 weights = 1-2 bits/weight

**Why This Might Work:**
- Additive structure gives 256 unique values from 32 codebook entries
- Much more expressive than single codebook
- Still faster than full AQLM (fewer optimization steps)

---

## Part 5: 70B Testing Strategy

### The Problem
- 70B model = 140GB in FP16
- Full compression takes hours
- HF Space has ~14GB VRAM

### Solution: Partial Compression Testing

**Approach 1: First N Layers Only**
```python
# Only compress first 10 of 80 layers
for i, (name, module) in enumerate(model.named_modules()):
    if i > 10:  # Skip remaining
        break
    compress(module)

# Extrapolate: if 10 layers give X% PPL delta, 80 layers give ~8X%
```

**Approach 2: Sample Layers**
```python
# Compress every 8th layer (10 of 80)
layers_to_compress = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72]
```

**Approach 3: Layer-Type Sampling**
```python
# Compress 1 layer of each type from each block
# MLP: layer 0, 10, 20, ...
# Attention: layer 0, 10, 20, ...
# Total: ~20 layers (25%)
```

### Extrapolation Formula

```
PPL_delta_estimated = PPL_delta_measured * (total_layers / compressed_layers) * correction_factor

Where correction_factor ≈ 0.8-1.2 depending on layer correlation
```

### 70B Expected Results

Based on scaling trend (larger models compress better):
| Model | Size | Expected Compression | Expected PPL Δ |
|-------|------|---------------------|----------------|
| Mistral-7B | 7B | 7.5x | +1.5% |
| Llama-70B | 70B | **9-10x** | **<1%** |

---

## Part 6: Recommended Next Steps

### Immediate (Test Today)

1. **Strategy C: Mixed Precision + Selective Sparsity**
   - Build on v22 (our best result)
   - Add INT4 g=512 for bottom 40% MLP
   - Add 50% selective sparsity on bottom 20%
   - Expected: 8.5x @ <2%

2. **70B Partial Test**
   - Compress 10% of layers (8 of 80)
   - Use Strategy C config
   - Measure PPL delta
   - Extrapolate full result

### This Week

3. **Strategy B: Low-Rank + Conservative VQ**
   - Apply low-rank FIRST (rank=32)
   - Apply VQ to RESIDUAL only (vec_dim=2)
   - Should avoid v38's quality disaster

4. **True GPTQ Implementation**
   - Implement iterative weight updates
   - One weight at a time with compensation
   - This is the proven path to 8x @ <1%

### Research Track

5. **AQLM-Lite: Additive Codebooks**
   - Two 16-entry codebooks
   - Train on calibration data
   - Target: 2 bits/weight

---

## Part 7: Questions for Cross-Validation

Please validate with other LLMs:

1. **Is iterative GPTQ-style compensation worth the complexity?**
   - Our current approach: quantize all, then compensate
   - GPTQ approach: quantize one, compensate remaining, repeat
   - How much does this improve quality?

2. **Low-rank before VQ vs VQ on raw weights?**
   - Hypothesis: Low-rank residual is more uniform, compresses better
   - Is there evidence for this?

3. **Additive codebooks: 16+16 vs 256?**
   - 16+16 additive = 256 values with 8 bits
   - 256 single = 256 values with 8 bits
   - Why does additive work better? (AQLM claims it does)

4. **70B scaling: linear or sublinear?**
   - If 7B gives 7.5x @ 1.5% PPL
   - Does 70B give 10x @ 0.5% PPL (sublinear - better)?
   - Or 10x @ 2% PPL (linear)?

5. **Importance metrics: Fisher vs Hessian vs Activation?**
   - Fisher: gradient² 
   - Hessian diagonal: second derivative
   - Activation: E[x²] (AWQ style)
   - Which is best for bit allocation?

---

## Appendix: Configuration for Next Test (v40)

```python
# v40: Mixed Precision + Selective Sparsity for 8.5x target

def allocate_bits_v40(model, fisher_scores):
    allocations = {}
    
    # Sort layers by Fisher importance
    layers = sorted(fisher_scores.items(), key=lambda x: x[1], reverse=True)
    n = len(layers)
    
    for i, (name, importance) in enumerate(layers):
        percentile = i / n
        
        if 'q_proj' in name or 'k_proj' in name:
            # Q/K always protected
            method = 'int4'
            group_size = 64
            sparsity = 0.0
        elif percentile < 0.10:
            # Top 10%: Protected
            method = 'int4'
            group_size = 32
            sparsity = 0.0
        elif percentile < 0.30:
            # 10-30%: High importance
            method = 'int4'
            group_size = 64
            sparsity = 0.0
        elif percentile < 0.60:
            # 30-60%: Medium - add light sparsity
            method = 'int4'
            group_size = 256
            sparsity = 0.25  # 3:4 sparse
        else:
            # Bottom 40%: Aggressive
            method = 'int4'
            group_size = 512
            sparsity = 0.50  # 2:4 sparse
        
        allocations[name] = {
            'method': method,
            'group_size': group_size,
            'sparsity': sparsity
        }
    
    return allocations
```

**Expected bits/weight:**
- Top 10%: 5.0 bits
- 10-30%: 4.5 bits  
- 30-60%: 4.0 * 0.75 = 3.0 effective bits
- Bottom 40%: 4.0 * 0.50 = 2.0 effective bits

**Weighted average:** 0.1×5 + 0.2×4.5 + 0.3×3 + 0.4×2 = **3.1 bits → 10.3x**

---

*Document created for cross-validation with other LLMs*
