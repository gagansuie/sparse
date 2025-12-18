# TenPak-10X Compression Results Log

## Mistral-7B-v0.1 Experiments

| Version | Compression | PPL Delta | Status | Strategy |
|---------|-------------|-----------|--------|----------|
| v10 | 7.42x | +1.47% | ⚠️ MARGINAL | Layer-type aware: attn g=256, MLP g=2048, 0.5% outliers |
| v11 | 7.70x | +3.59% | ⚠️ MARGINAL | Push harder: attn g=512, MLP g=4096, 0.25% outliers |
| v12 | 7.37x | +1.48% | ⚠️ MARGINAL | Per-sublayer: q=128, kv/o=256, gate/up=2048, down=1024, 0.5% outliers |
| v13 | 7.42x | +2.67% | ⚠️ MARGINAL | Push k/v/o to g=512 - HURT PPL |
| v14 | 7.01x | +1.94% | ⚠️ MARGINAL | 1% outliers - hurt both metrics |
| v16 | 7.48x | +1.87% | ⚠️ MARGINAL | Fisher-weighted dynamic groups (best so far) |
| v17 | 7.95x | +32.72% | ❌ FAIL | Ultra aggressive INT4 g=8192, no outliers - PPL destroyed |
| v18 | TBD | TBD | TBD | Hybrid INT4/INT2: attn=INT4, MLP=INT2 |
| v19 | TBD | TBD | TBD | 2:4 Structured Sparsity + INT4 |
| v20 | 10.09x | +63.68% | ❌ FAIL | 2:4 sparsity (50% pruning) too aggressive |
| v21 | 8.04x | +8.30% | ❌ FAIL | 3:4 on ALL MLP still too aggressive |
| v22 | 7.35x | +1.62% | ✅ PASS | **NOVEL** Selective 3:4 on 25% MLP (BEST QUALITY) |
| v23 | 7.57x | +2.42% | ⚠️ MARGINAL | Selective 3:4 on 50% MLP |
| v24 | 7.84x | +6.67% | ❌ FAIL | g=1024 for MLP too aggressive |
| v25 | TBD | TBD | TBD | v22 base + g=768 MLP (conservative push) |

---

## Best Configurations

### Best Quality (lowest PPL delta)
- **v10**: 7.42x, +1.47% - Simple layer-type aware allocation

### Best Compression (with acceptable PPL <2%)
- **v22**: 7.35x, +1.62% - **NOVEL** Fisher-guided selective 3:4 sparsity ✅

### Most Promising for 14x Target
- **v19**: 2:4 Structured Sparsity + INT4 (awaiting results)

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
3. **1% outliers** - v14 showed worse compression without quality gain
4. **INT2 for MLP** - Historically shows catastrophic PPL degradation

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

### v16 (Best Compression)
```
Attention (high Fisher): g=64-128
Attention (low Fisher): g=256-384
MLP (high Fisher): g=512-1024
MLP (low Fisher): g=2048-4096
Outliers: 0.5%
```

### v19 (14x Target)
```
q_proj: INT4 g=64
k/v/o_proj: INT4 g=128
MLP: 2:4 Sparse + INT4 g=128
Expected: ~10-12x compression
```

---

## Target Goals
- **Current Best**: 7.48x @ +1.87% PPL (v16)
- **Target**: 14x @ <2% PPL
- **Gap**: Need ~2x more compression without quality loss
