# INT4 Group-8 Codecs

Tenpak's group-8 codecs achieve **<1% PPL delta** with **4x compression** (vs FP32) using ultra-fine group quantization.

## Recommended: `int4_g8_fp16_v1`

| Metric | Value |
|--------|-------|
| PPL Delta | **+0.59%** |
| Compression (vs FP32) | **4.00x** |
| Bits per Weight | 8.0 |
| Calibration | **None required** |

## Alternative: `int4_g8_v1`

| Metric | Value |
|--------|-------|
| PPL Delta | +0.62% |
| Compression (vs FP32) | 2.67x |
| Bits per Weight | 12.0 |
| Calibration | None required |

**Note:** `int4_g8_fp16_v1` uses FP16 scales/offsets for better compression while maintaining quality.

## How It Works

1. **Group weights into chunks of 8**
2. **Find min/max per group** (asymmetric quantization)
3. **Quantize to 4-bit** (0-15 range)
4. **Store scale + offset per group**
5. **Pack 2 values per byte**

## Why It Works

### Group Size Comparison

| Codec | Group Size | PPL Delta | Compression (vs FP32) |
|-------|------------|-----------|----------------------|
| int4_g8_fp16_v1 | 8 | **+0.59%** | **4.00x** |
| int4_g8_v1 | 8 | +0.62% | 2.67x |
| int4_g16_v1 | 16 | +2.36% | 4.00x |

Smaller groups = tighter value ranges = less quantization error.

### Why Calibration Isn't Needed

AWQ/GPTQ use calibration to find "important" weights. With g=8:
- Each group of 8 weights gets its own scale/offset
- This naturally adapts to local weight distributions
- No need to identify important weights - all groups are optimized

## Implementation

### Rust Compression

```rust
// From src/lib.rs
fn compress_int4_g8(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 8;
    
    for group in weights.chunks(GROUP_SIZE) {
        let min_val = group.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = group.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (max_val - min_val) / 15.0;
        let offset = min_val;
        
        // Quantize: q = round((x - offset) / scale)
        // Pack two 4-bit values per byte
    }
}
```

### CUDA Inference

```c
// From cuda/awq_gemm.cu
__global__ void tenpak_gemm_g8_kernel(
    const half* X,       // [M, K] activations
    const uint8_t* W,    // [N, K/2] packed int4
    const float* scales, // [N, K/8] per-group
    const float* offsets,// [N, K/8] per-group
    half* Y,             // [M, N] output
    int M, int N, int K
) {
    // Vectorized load: 4 bytes = 8 int4 = 1 group
    uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&W[w_base]);
    
    // Dequantize and accumulate
    for (int i = 0; i < 4; i++) {
        uint8_t packed = (packed4 >> (i * 8)) & 0xFF;
        float w0 = (float)(packed & 0x0F) * scale + offset;
        float w1 = (float)((packed >> 4) & 0x0F) * scale + offset;
        acc += x[i*2] * w0 + x[i*2+1] * w1;
    }
}
```

### wgpu (Cross-Platform)

```wgsl
// From src/wgpu_gemm.rs
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Same algorithm, works on AMD/Intel/Apple
}
```

## Usage

```bash
# Compress (recommended)
./tenpak compress --input model.json --output model.tenpak --codec int4_g8_fp16_v1

# Alternative (F32 scales)
./tenpak compress --input model.json --output model.tenpak --codec int4_g8_v1

# Decompress
./tenpak decompress --input model.tenpak --output model_restored.json
```

## Python Inference

```python
from tenpak.cuda import G8Linear

# Convert layers
layer = G8Linear.from_linear(original_layer)

# Forward pass (weights stay quantized)
output = layer(input)
```
