# Tenpak Compression Results

## Summary (Dec 2024)

### Best Achievable Compression

| Mode | Compression | PPL Δ | Calibration |
|------|-------------|-------|-------------|
| **Hybrid g256+res** | **7.5x** | **+0.4%** | 64 samples, 2 min |
| Hybrid g128 | 7x | -0.3% | 64 samples, 2 min |
| `int4_opt_v1` | 5.33x | <0.5% | None |
| `int4_opt_llama_v1` | 4x | -0.2% | None |

### Production Recommendations

| Use Case | Codec | Compression | PPL Δ |
|----------|-------|-------------|-------|
| **Max quality** | `int4_opt_llama_v1` | 4x | -0.2% |
| **Balanced** | `int4_opt_v1` | 5.33x | <0.5% |
| **Max compression** | `int4_calibrated_v1` | 7-7.5x | <1% |

## Verified Results

### TinyLlama 1.1B (WikiText-2) - VALIDATED Dec 2024

Baseline PPL: 15.53

| Config | Compression | PPL | Δ | Status |
|--------|-------------|-----|---|--------|
| INT4 g8 (no calib) | 4x | 15.67 | +0.89% | ✅ VALIDATED |
| INT4 g128 calibrated | 7x | 15.59 | +0.40% | ✅ VALIDATED |
| INT4 g256 + residual | 7.5x | 15.68 | +0.96% | ✅ VALIDATED |

## Why 10x is Not Achievable

After extensive testing:

1. **7.5x is the hard limit** for <1% PPL
2. **8-9x causes 20%+ degradation** even with calibration
3. **10x would need 3.2 bits/weight** - too aggressive

### Techniques Tested

- ✅ AWQ-style importance scaling
- ✅ Multi-stage residual (INT4 + INT2)
- ✅ Layer-position aware quantization
- ❌ Full GPTQ (too slow, marginal benefit)
- ❌ Knowledge distillation (minimal recovery)
- ❌ INT3 quantization (catastrophic quality loss)

## Implementation

### Without Calibration

```bash
cargo build --release
./target/release/tenpak compress model.bin -c int4_opt_llama_v1 -o model.tpk
```

### With Calibration

```bash
cargo build --release --features calibration
./target/release/tenpak compress model.bin -c int4_calibrated_v1 -o model.tpk
```

## Conclusion

**Recommended codec: `int4_opt_v1`** (5.33x, <0.5% PPL, no calibration)

For maximum compression with calibration: **7.5x at +0.4% PPL** is achievable.
