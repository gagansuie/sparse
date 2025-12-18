# TenPak-10X Compression Results Log

## Mistral-7B-v0.1 Experiments

| Version | Compression | PPL Delta | Status | Strategy |
|---------|-------------|-----------|--------|----------|
| **v10** | **7.42x** | **+1.47%** | ✅ PASS | **BEST PPL** - attn g=256, MLP g=2048, 0.5% outliers |
| v11 | 7.70x | +3.59% | ⚠️ MARGINAL | Push harder: attn g=512, MLP g=4096, 0.25% outliers |
| v12 | 7.37x | +1.48% | ✅ PASS | Per-sublayer: q=128, kv/o=256, gate/up=2048, down=1024 |
| v13 | 7.42x | +2.67% | ⚠️ MARGINAL | Push k/v/o to g=512 - HURT PPL |
| v14 | 7.01x | +1.94% | ⚠️ MARGINAL | 1% outliers - hurt both metrics |
| v16 | 7.48x | +1.87% | ⚠️ MARGINAL | Fisher-weighted dynamic groups |
| v17 | 7.95x | +32.72% | ❌ FAIL | Ultra aggressive INT4 g=8192, no outliers |
| v20 | 10.09x | +63.68% | ❌ FAIL | 2:4 sparsity (50% pruning) too aggressive |
| v21 | 8.04x | +8.30% | ❌ FAIL | 3:4 on ALL MLP still too aggressive |
| v22 | 7.35x | +1.62% | ✅ PASS | Selective 3:4 on 25% MLP |
| v23 | 7.57x | +2.42% | ⚠️ MARGINAL | Selective 3:4 on 50% MLP |
| **v60** | **7.48x** | **+1.54%** | ✅ PASS | **BEST COMPRESSION** - attn g=512, MLP g=4096 |
| v61 | 7.51x | +3.46% | ⚠️ MARGINAL | Larger groups hurt PPL |
| v62 | 8.00x | +9092% | ❌ FAIL | Hadamard rotation - BROKEN |
| v63 | 7.67x | +5.43% | ❌ FAIL | 3:4 sparsity hurt too much |
| v64 | ~8x | N/A | ❌ SKIP | 5% magnitude pruning (not tested) |
| v65 | 6.95x | +2.15% | ⚠️ MARGINAL | 1% outliers HURT quality |
| v67 | 7.86x | +50737% | ❌ FAIL | GPTQ-lite - BROKEN (error accumulation bug) |

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
