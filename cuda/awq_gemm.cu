/*
 * AWQ GEMM Kernels for Tenpak
 * 
 * High-performance CUDA kernels for int4 weight-only quantization inference.
 * Supports:
 * - W4A16 GEMM (4-bit weights, 16-bit activations)
 * - Group quantization (g=32, 64, 128)
 * - Per-channel scaling with AWQ-style importance weighting
 * - KV-cache quantization
 * 
 * Based on techniques from:
 * - AWQ (MIT-HAN-LAB)
 * - GPTQ
 * - llama.cpp
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// Configuration
#define WARP_SIZE 32
#define MAX_GROUP_SIZE 128
#define TILE_M 16
#define TILE_N 16
#define TILE_K 64

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


/*
 * Pack two int4 values into a single byte
 */
__device__ __forceinline__ uint8_t pack_int4(int8_t a, int8_t b) {
    return ((uint8_t)(a & 0x0F)) | (((uint8_t)(b & 0x0F)) << 4);
}

/*
 * Unpack two int4 values from a single byte
 */
__device__ __forceinline__ void unpack_int4(uint8_t packed, int8_t* a, int8_t* b) {
    *a = (int8_t)((packed & 0x0F) | ((packed & 0x08) ? 0xF0 : 0x00));
    *b = (int8_t)(((packed >> 4) & 0x0F) | ((packed & 0x80) ? 0xF0 : 0x00));
}

/*
 * Unpack int4 to float with scale
 */
__device__ __forceinline__ float dequant_int4(uint8_t packed, int idx, float scale, float zero_point) {
    int8_t val;
    if (idx == 0) {
        val = (int8_t)((packed & 0x0F) | ((packed & 0x08) ? 0xF0 : 0x00));
    } else {
        val = (int8_t)(((packed >> 4) & 0x0F) | ((packed & 0x80) ? 0xF0 : 0x00));
    }
    return ((float)val - zero_point) * scale;
}

/*
 * Asymmetric int4 dequantization (0-15 range)
 */
__device__ __forceinline__ float dequant_int4_asym(uint8_t packed, int idx, float scale, float offset) {
    uint8_t val;
    if (idx == 0) {
        val = packed & 0x0F;
    } else {
        val = (packed >> 4) & 0x0F;
    }
    return (float)val * scale + offset;
}


/*
 * W4A16 GEMM Kernel - Basic Version
 * 
 * Computes: Y = X @ W^T where W is int4 quantized
 * 
 * X: [M, K] float16
 * W: [N, K/2] uint8 (packed int4)
 * scales: [N, K/group_size] float
 * zeros: [N, K/group_size] float (optional)
 * Y: [M, N] float16
 */
__global__ void awq_gemm_w4a16_kernel(
    const half* __restrict__ X,      // [M, K]
    const uint8_t* __restrict__ W,   // [N, K/2] packed int4
    const float* __restrict__ scales, // [N, num_groups]
    const float* __restrict__ zeros,  // [N, num_groups] or nullptr
    half* __restrict__ Y,             // [M, N]
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    
    if (row >= M || col >= N) return;
    
    int num_groups = (K + group_size - 1) / group_size;
    
    float acc = 0.0f;
    
    // Process K dimension
    for (int k = 0; k < K; k += 2) {
        int group_idx = k / group_size;
        float scale = scales[col * num_groups + group_idx];
        float zero = zeros ? zeros[col * num_groups + group_idx] : 0.0f;
        
        // Load packed int4 weights
        uint8_t packed = W[col * (K / 2) + k / 2];
        
        // Dequantize
        float w0 = dequant_int4(packed, 0, scale, zero);
        float w1 = dequant_int4(packed, 1, scale, zero);
        
        // Load activations
        float x0 = __half2float(X[row * K + k]);
        float x1 = (k + 1 < K) ? __half2float(X[row * K + k + 1]) : 0.0f;
        
        // Accumulate
        acc += x0 * w0 + x1 * w1;
    }
    
    Y[row * N + col] = __float2half(acc);
}


/*
 * W4A16 GEMM Kernel - Optimized with Shared Memory
 * 
 * Uses tiled approach with shared memory for better memory access patterns.
 */
__global__ void awq_gemm_w4a16_tiled_kernel(
    const half* __restrict__ X,
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    half* __restrict__ Y,
    int M, int N, int K,
    int group_size
) {
    __shared__ float Xs[TILE_M][TILE_K];
    __shared__ float Ws[TILE_N][TILE_K];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;
    
    int num_groups = (K + group_size - 1) / group_size;
    
    float acc = 0.0f;
    
    // Tile over K dimension
    for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load X tile into shared memory
        for (int i = 0; i < TILE_K; i += blockDim.x) {
            int k_idx = tile_k + tx + i;
            if (row < M && k_idx < K) {
                Xs[ty][tx + i] = __half2float(X[row * K + k_idx]);
            } else {
                Xs[ty][tx + i] = 0.0f;
            }
        }
        
        // Load and dequantize W tile into shared memory
        for (int i = 0; i < TILE_K; i += blockDim.y * 2) {
            int k_idx = tile_k + ty * 2 + i;
            if (col < N && k_idx < K) {
                int group_idx = k_idx / group_size;
                float scale = scales[col * num_groups + group_idx];
                float zero = zeros ? zeros[col * num_groups + group_idx] : 0.0f;
                
                uint8_t packed = W[col * (K / 2) + k_idx / 2];
                Ws[tx][ty * 2 + i] = dequant_int4(packed, 0, scale, zero);
                if (k_idx + 1 < K) {
                    Ws[tx][ty * 2 + i + 1] = dequant_int4(packed, 1, scale, zero);
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_K && tile_k + k < K; k++) {
            acc += Xs[ty][k] * Ws[tx][k];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}


/*
 * AWQ-style GEMM with per-channel importance scaling
 * 
 * Applies learned channel scales during dequantization:
 * W_dequant = (W_int4 * scale) / channel_scale
 */
__global__ void awq_gemm_with_channel_scales_kernel(
    const half* __restrict__ X,
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const float* __restrict__ channel_scales,  // [K] per-input-channel scales
    half* __restrict__ Y,
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    int num_groups = (K + group_size - 1) / group_size;
    
    float acc = 0.0f;
    
    for (int k = 0; k < K; k += 2) {
        int group_idx = k / group_size;
        float scale = scales[col * num_groups + group_idx];
        float zero = zeros ? zeros[col * num_groups + group_idx] : 0.0f;
        
        uint8_t packed = W[col * (K / 2) + k / 2];
        
        // Dequantize with channel scale correction
        float cs0 = channel_scales[k];
        float cs1 = (k + 1 < K) ? channel_scales[k + 1] : 1.0f;
        
        float w0 = dequant_int4(packed, 0, scale, zero) / cs0;
        float w1 = dequant_int4(packed, 1, scale, zero) / cs1;
        
        float x0 = __half2float(X[row * K + k]);
        float x1 = (k + 1 < K) ? __half2float(X[row * K + k + 1]) : 0.0f;
        
        acc += x0 * w0 + x1 * w1;
    }
    
    Y[row * N + col] = __float2half(acc);
}


/*
 * KV-Cache Quantization Kernel
 * 
 * Quantizes KV cache to int4/int8 for memory efficiency.
 */
__global__ void quantize_kv_cache_kernel(
    const half* __restrict__ kv_fp16,  // [batch, heads, seq, head_dim]
    uint8_t* __restrict__ kv_int4,     // [batch, heads, seq, head_dim/2]
    float* __restrict__ scales,         // [batch, heads, seq]
    float* __restrict__ zeros,          // [batch, heads, seq]
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vectors = batch_size * num_heads * seq_len;
    
    if (idx >= total_vectors) return;
    
    int batch = idx / (num_heads * seq_len);
    int head = (idx / seq_len) % num_heads;
    int seq = idx % seq_len;
    
    // Find min/max for this vector
    float min_val = 1e10f;
    float max_val = -1e10f;
    
    int base_idx = ((batch * num_heads + head) * seq_len + seq) * head_dim;
    
    for (int d = 0; d < head_dim; d++) {
        float val = __half2float(kv_fp16[base_idx + d]);
        min_val = fminf(min_val, val);
        max_val = fmaxf(max_val, val);
    }
    
    // Compute scale and zero point for asymmetric quantization
    float range = max_val - min_val;
    float scale = range / 15.0f;
    if (scale < 1e-8f) scale = 1.0f;
    float zero = min_val;
    
    scales[idx] = scale;
    zeros[idx] = zero;
    
    // Quantize and pack
    int out_base = ((batch * num_heads + head) * seq_len + seq) * (head_dim / 2);
    
    for (int d = 0; d < head_dim; d += 2) {
        float v0 = __half2float(kv_fp16[base_idx + d]);
        float v1 = (d + 1 < head_dim) ? __half2float(kv_fp16[base_idx + d + 1]) : 0.0f;
        
        uint8_t q0 = (uint8_t)fminf(15.0f, fmaxf(0.0f, roundf((v0 - zero) / scale)));
        uint8_t q1 = (uint8_t)fminf(15.0f, fmaxf(0.0f, roundf((v1 - zero) / scale)));
        
        kv_int4[out_base + d / 2] = q0 | (q1 << 4);
    }
}


/*
 * KV-Cache Dequantization Kernel
 */
__global__ void dequantize_kv_cache_kernel(
    const uint8_t* __restrict__ kv_int4,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    half* __restrict__ kv_fp16,
    int batch_size, int num_heads, int seq_len, int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vectors = batch_size * num_heads * seq_len;
    
    if (idx >= total_vectors) return;
    
    int batch = idx / (num_heads * seq_len);
    int head = (idx / seq_len) % num_heads;
    int seq = idx % seq_len;
    
    float scale = scales[idx];
    float zero = zeros[idx];
    
    int in_base = ((batch * num_heads + head) * seq_len + seq) * (head_dim / 2);
    int out_base = ((batch * num_heads + head) * seq_len + seq) * head_dim;
    
    for (int d = 0; d < head_dim; d += 2) {
        uint8_t packed = kv_int4[in_base + d / 2];
        
        float v0 = (float)(packed & 0x0F) * scale + zero;
        float v1 = (float)((packed >> 4) & 0x0F) * scale + zero;
        
        kv_fp16[out_base + d] = __float2half(v0);
        if (d + 1 < head_dim) {
            kv_fp16[out_base + d + 1] = __float2half(v1);
        }
    }
}


/*
 * Fused Attention with Quantized KV Cache
 * 
 * Computes attention directly on quantized KV cache without full dequantization.
 */
__global__ void attention_with_quant_kv_kernel(
    const half* __restrict__ Q,           // [batch, heads, 1, head_dim]
    const uint8_t* __restrict__ K_int4,   // [batch, heads, seq, head_dim/2]
    const uint8_t* __restrict__ V_int4,   // [batch, heads, seq, head_dim/2]
    const float* __restrict__ K_scales,   // [batch, heads, seq]
    const float* __restrict__ K_zeros,
    const float* __restrict__ V_scales,
    const float* __restrict__ V_zeros,
    half* __restrict__ output,            // [batch, heads, 1, head_dim]
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale_factor  // 1/sqrt(head_dim)
) {
    extern __shared__ float smem[];
    float* scores = smem;  // [seq_len]
    
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= num_heads) return;
    
    int q_base = ((batch * num_heads + head) * 1) * head_dim;
    
    // Step 1: Compute attention scores Q @ K^T
    for (int s = tid; s < seq_len; s += blockDim.x) {
        int k_scale_idx = (batch * num_heads + head) * seq_len + s;
        float k_scale = K_scales[k_scale_idx];
        float k_zero = K_zeros[k_scale_idx];
        
        int k_base = ((batch * num_heads + head) * seq_len + s) * (head_dim / 2);
        
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d += 2) {
            float q0 = __half2float(Q[q_base + d]);
            float q1 = (d + 1 < head_dim) ? __half2float(Q[q_base + d + 1]) : 0.0f;
            
            uint8_t packed = K_int4[k_base + d / 2];
            float k0 = (float)(packed & 0x0F) * k_scale + k_zero;
            float k1 = (float)((packed >> 4) & 0x0F) * k_scale + k_zero;
            
            dot += q0 * k0 + q1 * k1;
        }
        
        scores[s] = dot * scale_factor;
    }
    
    __syncthreads();
    
    // Step 2: Softmax
    // Find max for numerical stability
    float max_score = -1e10f;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        max_score = fmaxf(max_score, scores[s]);
    }
    
    // Reduce max across threads
    __shared__ float shared_max;
    if (tid == 0) shared_max = -1e10f;
    __syncthreads();
    atomicMax((int*)&shared_max, __float_as_int(max_score));
    __syncthreads();
    max_score = shared_max;
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int s = tid; s < seq_len; s += blockDim.x) {
        scores[s] = expf(scores[s] - max_score);
        sum_exp += scores[s];
    }
    
    // Reduce sum
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum_exp);
    __syncthreads();
    
    // Normalize
    for (int s = tid; s < seq_len; s += blockDim.x) {
        scores[s] /= shared_sum;
    }
    
    __syncthreads();
    
    // Step 3: Weighted sum of V
    int out_base = ((batch * num_heads + head) * 1) * head_dim;
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        
        for (int s = 0; s < seq_len; s++) {
            int v_scale_idx = (batch * num_heads + head) * seq_len + s;
            float v_scale = V_scales[v_scale_idx];
            float v_zero = V_zeros[v_scale_idx];
            
            int v_base = ((batch * num_heads + head) * seq_len + s) * (head_dim / 2);
            
            uint8_t packed = V_int4[v_base + d / 2];
            float v;
            if (d % 2 == 0) {
                v = (float)(packed & 0x0F) * v_scale + v_zero;
            } else {
                v = (float)((packed >> 4) & 0x0F) * v_scale + v_zero;
            }
            
            acc += scores[s] * v;
        }
        
        output[out_base + d] = __float2half(acc);
    }
}


// C API for integration with Python/Rust

extern "C" {

/*
 * Initialize CUDA device
 */
int tenpak_cuda_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}

/*
 * W4A16 GEMM
 */
int tenpak_awq_gemm_w4a16(
    const void* X,
    const void* W,
    const void* scales,
    const void* zeros,
    void* Y,
    int M, int N, int K,
    int group_size,
    void* stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    awq_gemm_w4a16_kernel<<<grid, block, 0, cuda_stream>>>(
        (const half*)X,
        (const uint8_t*)W,
        (const float*)scales,
        (const float*)zeros,
        (half*)Y,
        M, N, K, group_size
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

/*
 * W4A16 GEMM with channel scales (AWQ-style)
 */
int tenpak_awq_gemm_with_scales(
    const void* X,
    const void* W,
    const void* scales,
    const void* zeros,
    const void* channel_scales,
    void* Y,
    int M, int N, int K,
    int group_size,
    void* stream
) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    awq_gemm_with_channel_scales_kernel<<<grid, block, 0, cuda_stream>>>(
        (const half*)X,
        (const uint8_t*)W,
        (const float*)scales,
        (const float*)zeros,
        (const float*)channel_scales,
        (half*)Y,
        M, N, K, group_size
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

/*
 * Quantize KV cache
 */
int tenpak_quantize_kv_cache(
    const void* kv_fp16,
    void* kv_int4,
    void* scales,
    void* zeros,
    int batch_size, int num_heads, int seq_len, int head_dim,
    void* stream
) {
    int total = batch_size * num_heads * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    quantize_kv_cache_kernel<<<grid, block, 0, cuda_stream>>>(
        (const half*)kv_fp16,
        (uint8_t*)kv_int4,
        (float*)scales,
        (float*)zeros,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

/*
 * Dequantize KV cache
 */
int tenpak_dequantize_kv_cache(
    const void* kv_int4,
    const void* scales,
    const void* zeros,
    void* kv_fp16,
    int batch_size, int num_heads, int seq_len, int head_dim,
    void* stream
) {
    int total = batch_size * num_heads * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    dequantize_kv_cache_kernel<<<grid, block, 0, cuda_stream>>>(
        (const uint8_t*)kv_int4,
        (const float*)scales,
        (const float*)zeros,
        (half*)kv_fp16,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

/*
 * Fused attention with quantized KV cache
 */
int tenpak_attention_quant_kv(
    const void* Q,
    const void* K_int4,
    const void* V_int4,
    const void* K_scales,
    const void* K_zeros,
    const void* V_scales,
    const void* V_zeros,
    void* output,
    int batch_size, int num_heads, int seq_len, int head_dim,
    void* stream
) {
    dim3 grid(batch_size, num_heads);
    int block = 256;
    size_t smem_size = seq_len * sizeof(float);
    
    float scale_factor = 1.0f / sqrtf((float)head_dim);
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    attention_with_quant_kv_kernel<<<grid, block, smem_size, cuda_stream>>>(
        (const half*)Q,
        (const uint8_t*)K_int4,
        (const uint8_t*)V_int4,
        (const float*)K_scales,
        (const float*)K_zeros,
        (const float*)V_scales,
        (const float*)V_zeros,
        (half*)output,
        batch_size, num_heads, seq_len, head_dim,
        scale_factor
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

/*
 * Optimized W4A16 GEMM for g=8 format
 * 
 * This kernel is specifically optimized for group_size=8:
 * - Each group of 8 weights shares one scale and one offset
 * - Uses vectorized loads (4 bytes = 8 int4 values = 1 group)
 * - Processes 8 weights per iteration with single scale/offset load
 * - Uses FP16 accumulation for speed, FP32 for final reduction
 */
__global__ void tenpak_gemm_g8_kernel(
    const half* __restrict__ X,       // [M, K] activations
    const uint8_t* __restrict__ W,    // [N, K/2] packed int4 weights
    const float* __restrict__ scales, // [N, K/8] per-group scales
    const float* __restrict__ offsets,// [N, K/8] per-group offsets
    half* __restrict__ Y,             // [M, N] output
    int M, int N, int K
) {
    // Each thread computes one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension
    
    if (row >= M || col >= N) return;
    
    const int GROUP_SIZE = 8;
    int num_groups = K / GROUP_SIZE;
    
    float acc = 0.0f;
    
    // Process K dimension in groups of 8
    for (int g = 0; g < num_groups; g++) {
        // Load scale and offset for this group (single memory access)
        float scale = scales[col * num_groups + g];
        float offset = offsets[col * num_groups + g];
        
        int k_base = g * GROUP_SIZE;
        int w_base = col * (K / 2) + k_base / 2;
        
        // Load 4 bytes = 8 int4 values (one full group)
        // Use vectorized load for better memory bandwidth
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&W[w_base]);
        
        // Unpack and dequantize 8 weights
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t packed = (packed4 >> (i * 8)) & 0xFF;
            
            // Low nibble
            float w0 = (float)(packed & 0x0F) * scale + offset;
            float x0 = __half2float(X[row * K + k_base + i * 2]);
            acc += x0 * w0;
            
            // High nibble
            float w1 = (float)((packed >> 4) & 0x0F) * scale + offset;
            float x1 = __half2float(X[row * K + k_base + i * 2 + 1]);
            acc += x1 * w1;
        }
    }
    
    // Handle remaining elements if K is not divisible by 8
    int remaining = K % GROUP_SIZE;
    if (remaining > 0) {
        int g = num_groups;
        float scale = scales[col * (num_groups + 1) + g];
        float offset = offsets[col * (num_groups + 1) + g];
        int k_base = g * GROUP_SIZE;
        
        for (int i = 0; i < remaining; i += 2) {
            uint8_t packed = W[col * (K / 2) + (k_base + i) / 2];
            
            float w0 = (float)(packed & 0x0F) * scale + offset;
            float x0 = __half2float(X[row * K + k_base + i]);
            acc += x0 * w0;
            
            if (i + 1 < remaining) {
                float w1 = (float)((packed >> 4) & 0x0F) * scale + offset;
                float x1 = __half2float(X[row * K + k_base + i + 1]);
                acc += x1 * w1;
            }
        }
    }
    
    Y[row * N + col] = __float2half(acc);
}


/*
 * Optimized W4A16 GEMM for g=8 with shared memory tiling
 * 
 * Uses shared memory to reduce global memory bandwidth:
 * - Tiles X into shared memory
 * - Dequantizes W on-the-fly (weights are small due to int4)
 * - Uses warp-level primitives for reduction
 */
#define G8_TILE_M 64
#define G8_TILE_N 64
#define G8_TILE_K 32  // Must be multiple of 8

__global__ void tenpak_gemm_g8_tiled_kernel(
    const half* __restrict__ X,
    const uint8_t* __restrict__ W,
    const float* __restrict__ scales,
    const float* __restrict__ offsets,
    half* __restrict__ Y,
    int M, int N, int K
) {
    __shared__ half Xs[G8_TILE_M][G8_TILE_K + 1];  // +1 to avoid bank conflicts
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * G8_TILE_M + ty;
    int col = bx * G8_TILE_N + tx;
    
    const int GROUP_SIZE = 8;
    int num_groups = K / GROUP_SIZE;
    
    float acc = 0.0f;
    
    // Tile over K dimension
    for (int tile_k = 0; tile_k < K; tile_k += G8_TILE_K) {
        // Collaboratively load X tile into shared memory
        // Each thread loads multiple elements
        for (int i = ty; i < G8_TILE_M; i += blockDim.y) {
            for (int j = tx; j < G8_TILE_K; j += blockDim.x) {
                int global_row = by * G8_TILE_M + i;
                int global_k = tile_k + j;
                if (global_row < M && global_k < K) {
                    Xs[i][j] = X[global_row * K + global_k];
                } else {
                    Xs[i][j] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        if (row < M && col < N) {
            // Process this tile in groups of 8
            for (int k = 0; k < G8_TILE_K && tile_k + k < K; k += GROUP_SIZE) {
                int global_k = tile_k + k;
                int g = global_k / GROUP_SIZE;
                
                float scale = scales[col * num_groups + g];
                float offset = offsets[col * num_groups + g];
                
                int w_base = col * (K / 2) + global_k / 2;
                uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&W[w_base]);
                
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    uint8_t packed = (packed4 >> (i * 8)) & 0xFF;
                    
                    float w0 = (float)(packed & 0x0F) * scale + offset;
                    float x0 = __half2float(Xs[ty][k + i * 2]);
                    acc += x0 * w0;
                    
                    float w1 = (float)((packed >> 4) & 0x0F) * scale + offset;
                    float x1 = __half2float(Xs[ty][k + i * 2 + 1]);
                    acc += x1 * w1;
                }
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        Y[row * N + col] = __float2half(acc);
    }
}


/*
 * Batched GEMM for transformer layers
 * Processes multiple GEMM operations in parallel
 */
__global__ void tenpak_gemm_g8_batched_kernel(
    const half* __restrict__ X,       // [B, M, K]
    const uint8_t* __restrict__ W,    // [N, K/2]
    const float* __restrict__ scales, // [N, K/8]
    const float* __restrict__ offsets,// [N, K/8]
    half* __restrict__ Y,             // [B, M, N]
    int B, int M, int N, int K
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    const int GROUP_SIZE = 8;
    int num_groups = K / GROUP_SIZE;
    
    const half* X_batch = X + batch * M * K;
    half* Y_batch = Y + batch * M * N;
    
    float acc = 0.0f;
    
    for (int g = 0; g < num_groups; g++) {
        float scale = scales[col * num_groups + g];
        float offset = offsets[col * num_groups + g];
        
        int k_base = g * GROUP_SIZE;
        int w_base = col * (K / 2) + k_base / 2;
        
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&W[w_base]);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint8_t packed = (packed4 >> (i * 8)) & 0xFF;
            
            float w0 = (float)(packed & 0x0F) * scale + offset;
            float x0 = __half2float(X_batch[row * K + k_base + i * 2]);
            acc += x0 * w0;
            
            float w1 = (float)((packed >> 4) & 0x0F) * scale + offset;
            float x1 = __half2float(X_batch[row * K + k_base + i * 2 + 1]);
            acc += x1 * w1;
        }
    }
    
    Y_batch[row * N + col] = __float2half(acc);
}


/*
 * C API for g=8 GEMM
 */
int tenpak_gemm_g8(
    const void* X,
    const void* W,
    const void* scales,
    const void* offsets,
    void* Y,
    int M, int N, int K,
    void* stream
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    // Use tiled kernel for larger matrices
    if (M >= 64 && N >= 64 && K >= 64) {
        dim3 tiled_block(16, 16);
        dim3 tiled_grid((N + G8_TILE_N - 1) / G8_TILE_N, (M + G8_TILE_M - 1) / G8_TILE_M);
        
        tenpak_gemm_g8_tiled_kernel<<<tiled_grid, tiled_block, 0, cuda_stream>>>(
            (const half*)X,
            (const uint8_t*)W,
            (const float*)scales,
            (const float*)offsets,
            (half*)Y,
            M, N, K
        );
    } else {
        tenpak_gemm_g8_kernel<<<grid, block, 0, cuda_stream>>>(
            (const half*)X,
            (const uint8_t*)W,
            (const float*)scales,
            (const float*)offsets,
            (half*)Y,
            M, N, K
        );
    }
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}


/*
 * Batched g=8 GEMM for transformer inference
 */
int tenpak_gemm_g8_batched(
    const void* X,
    const void* W,
    const void* scales,
    const void* offsets,
    void* Y,
    int B, int M, int N, int K,
    void* stream
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, B);
    
    cudaStream_t cuda_stream = stream ? (cudaStream_t)stream : 0;
    
    tenpak_gemm_g8_batched_kernel<<<grid, block, 0, cuda_stream>>>(
        (const half*)X,
        (const uint8_t*)W,
        (const float*)scales,
        (const float*)offsets,
        (half*)Y,
        B, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? 0 : -1;
}

}  // extern "C"
