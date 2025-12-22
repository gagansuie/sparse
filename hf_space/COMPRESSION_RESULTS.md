# TenPak Quantization Results

**Note**: This document contains historical results from custom codec experiments (v0.1.0). TenPak v0.2.0 now wraps industry-standard tools instead.

## Current Approach (v0.2.0)

TenPak wraps AutoGPTQ, AutoAWQ, and bitsandbytes:

| Method | Compression | PPL Δ | Notes |
|--------|-------------|-------|-------|
| **AutoGPTQ 4-bit** | 7-8x | <1% | Best quality, requires calibration |
| **AutoAWQ 4-bit** | 7-8x | <2% | Best balance, requires calibration |
| **bitsandbytes NF4** | 6-7x | <1.5% | Fast, optional calibration |
| **bitsandbytes INT8** | 2x | <0.5% | Conservative |

## Historical Custom Codec Experiments (v0.1.0 - Deprecated)

These experiments led to the decision to wrap existing tools rather than maintain custom codecs:

| Version | Compression | PPL Delta | Status | Strategy |
|---------|-------------|-----------|--------|----------|
| **v10** | **7.42x** | **+1.47%** | ✅ BEST | Custom INT4 - attn g=256, MLP g=2048, 0.5% outliers |
| **v60** | **7.48x** | **+1.54%** | ✅ BEST | Custom INT4 - attn g=512, MLP g=4096 |
| v67 | 7.86x | +50737% | ❌ FAIL | Custom GPTQ-lite - error accumulation bug |
| v62 | 8.00x | +9092% | ❌ FAIL | Hadamard rotation - broken |

---

## Best Configurations

### Best PPL (lowest delta)
- **v10**: 7.42x, **+1.47%** - attn g=256, MLP g=2048, 0.5% outliers

### Best Compression (with <2% PPL)
- **v60**: **7.48x**, +1.54% - attn g=512, MLP g=4096, 0.5% outliers

### Conclusion
**v10 and v60 are both excellent.** Choose based on priority:
- Want best quality? → **v10** (+1.47% PPL)
- Want best compression? → **v60** (7.48x)

---

## Key Learnings

### What Works
1. **AWQ-style activation scaling** - Essential for quality
2. **Small groups for attention (g=64-256)** - Attention is sensitive
3. **Outlier extraction (0.5%)** - Protects important weights
4. **Fisher-guided allocation** - Helps optimize per-layer

### What Doesn't Work
1. **Large groups for attention (g=512+)** - v13 showed +2.67% PPL vs +1.48%
2. **No outlier extraction** - v17 showed +32.72% PPL
3. **1% outliers** - v14, v65 showed worse quality (too much in FP16)
4. **Custom GPTQ/AQLM implementations** - All broke with catastrophic PPL
5. **Hadamard rotation** - Needs forward pass modification (not demo-time feasible)
6. **Structured sparsity (2:4, 3:4)** - v20, v21, v63 all hurt PPL significantly

### Theoretical Limits
- **INT4 max**: ~8x compression
- **INT4 + 0.5% outliers**: ~7.5x practical limit
- **To hit 14x**: Need sparsity or lower bit-width

---

## Configuration Details

### v10 (Best Quality)
```
Attention: g=256, INT4, 0.5% outliers
MLP: g=2048, INT4, 0.5% outliers
```

### v60 (Best Compression)
```
Attention: g=512, INT4, 0.5% outliers
MLP: g=4096, INT4, 0.5% outliers
Result: 7.48x, +1.54% PPL
```

---

## Final Results

| Goal | Best Version | Compression | PPL Delta |
|------|--------------|-------------|-----------||
| Best Quality | **v10** | 7.42x | **+1.47%** |
| Best Compression | **v60** | **7.48x** | +1.54% |

**Ceiling reached:** ~7.5x compression at <2% PPL is the limit for demo-time INT4+AWQ.

To achieve 10x+ compression with <1% PPL would require:
1. Pre-trained quantization (offline calibration, hours of compute)
2. Production GPTQ/AWQ libraries (AutoGPTQ, llm-awq)
3. Hardware-specific kernels
