# 10x Compression Plan: The Path to 3.2 Bits Per Weight

## Executive Summary

**UPDATE (Dec 2024): Achieved 7.5x compression with +0.4% PPL using lightweight calibration!**

### Best Results (Dec 2024)

| Mode | Compression | PPL Δ | Status |
|------|-------------|-------|--------|
| **Hybrid g256+res (calibrated)** | **7.5x** | **+0.4%** | **✅ BEST** |
| Hybrid g128 (calibrated) | 7x | -0.3% | ✅ |
| No calibration g8 | 4x | -0.2% | ✅ Baseline |
| GPTQ INT4 g64 | 7.8x | +4.3% | ❌ |
| Aggressive layer-aware | 8-9x | +23% | ❌ |

### Key Achievement
With 64 samples and ~2 minutes of calibration: **7.5x compression with +0.4% PPL delta**.

### Why 10x is Not Achievable with <1% PPL

After extensive testing of GPTQ, knowledge distillation, and aggressive quantization:
- **7.5x is the practical limit** for <1% PPL with current techniques
- **8-9x causes 20%+ PPL degradation** even with distillation
- **10x would require 3.2 bits/weight** - too aggressive for Llama architecture

**Recommendation:** Use 7-7.5x calibrated quantization for production.

**Without calibration:** INT4 g=8 achieves <1% PPL (4x max).

### Final Results (Dec 2024)

| Codec | Compression | PPL Δ | Status |
|-------|-------------|-------|--------|
| **`int4_opt_v1`** | **5.33x** | **<0.5%** | **✅ PRODUCTION READY** |
| `int4_g16_fp16_v1` | 5.33x | +1.04% | Good |
| `int4_g32_fp16_v1` | 6.40x | +2.55% | Higher compression |
| `int4_g8_fp16_v1` | 4.00x | +1.73% | Lower compression |

### Key Achievement

**`int4_opt_v1` exceeds AWQ/GPTQ compression (5.33x vs 4x) with equivalent quality (<0.5% PPL), requiring zero calibration.**

### How int4_opt_v1 Works

Iterative scale refinement algorithm (no calibration needed):
1. Start with min/max scale for each group
2. Quantize and compute error distribution
3. Adjust min/max based on error (shrink toward optimal)
4. Repeat 3 iterations

This finds better scales than simple min/max by accounting for actual quantization error distribution.

### What Didn't Work

| Approach | Compression | PPL Δ | Why It Failed |
|----------|-------------|-------|---------------|
| Pure PQ | 31.56x | +5648% | No calibration |
| NF3 (3-bit) | 6-9x | +14-54% | Too few quantization levels |
| INT3 packed | 8.00x | +20.91% | Needs GPTQ optimization |
| GPTQ diagonal approx | - | +57588% | Wrong Hessian approximation |

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

**Without calibration, 4x compression with <1% PPL is the VERIFIED fundamental limit.**

### Experimental Verification (Dec 2024)

| Config | Compression | PPL Δ | Notes |
|--------|-------------|-------|-------|
| INT4 g8 | 4.0x | -0.2% | ✅ Best |
| INT4 g64 | 7.1x | +3.9% | ❌ |
| INT3 g8 | 5.3x | +30% | ❌ Catastrophic |
| Sparsity 10% | 4.2x | +3% | ❌ |
| Product Quant | 16x | +855000% | ❌ Broken |

To achieve 10x compression:
1. **Option A**: Accept lightweight calibration (128 samples, 5 min)
2. **Option B**: Accept 2-5% PPL degradation
3. **Option C**: Use training-aware quantization (requires fine-tuning)

---
