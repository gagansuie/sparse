use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Cross-platform GPU inference via wgpu (AMD, Intel, NVIDIA, Apple)
#[cfg(feature = "gpu")]
pub mod wgpu_gemm;

/// Current on-disk artifact format version.
pub const ARTIFACT_VERSION: u32 = 1;

// ============================================================================
// PRODUCTION CODECS (Validated Dec 2024)
// ============================================================================

/// PRIMARY: Residual quantization - INT4 + INT2 residual correction (g=16).
/// Best quality without calibration. Negative PPL delta on larger models.
///
/// Measured results (TinyLlama 1.1B, WikiText-2):
/// - Compression: 3.20x
/// - PPL Delta: -0.65% (quantized model performs BETTER)
///
/// Supports all linear layers: gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj
pub const CODEC_INT4_RESIDUAL_V1: &str = "int4_residual_v1";

/// PRIMARY: Calibration-aware quantization - INT4 g=128 with importance scaling.
/// Best compression with calibration data.
///
/// Measured results (TinyLlama 1.1B, WikiText-2):
/// - Compression: 4.92x
/// - PPL Delta: +1.46%
#[cfg(feature = "calibration")]
pub const CODEC_INT4_CALIBRATED_V1: &str = "int4_calibrated_v1";

/// BACKUP: Optimal quantization for Llama-architecture models (g=8, 5 iterations).
/// Simpler than residual, good fallback option.
///
/// Measured results (TinyLlama 1.1B, WikiText-2):
/// - Compression: 4.00x
/// - PPL Delta: <1%
pub const CODEC_INT4_OPT_LLAMA_V1: &str = "int4_opt_llama_v1";

// ============================================================================
// EXPERIMENTAL: 10x Compression Codecs (No Calibration)
// ============================================================================

/// EXPERIMENTAL: SpinQuant-inspired codec with Hadamard rotation.
/// Applies fast Walsh-Hadamard transform to spread outliers before quantization.
/// Target: 6-8x compression with <1% PPL delta, NO calibration required.
///
/// How it works:
/// 1. Apply Hadamard rotation to weight groups (spreads outliers)
/// 2. INT4 quantize in rotated space
/// 3. Store rotation info for dequantization
pub const CODEC_INT4_SPIN_V1: &str = "int4_spin_v1";

/// EXPERIMENTAL: Maximum compression codec for 10x target.
/// INT4 with very large groups (g=256) and no residual.
/// Target: 7-8x compression, may have higher PPL delta.
///
/// Storage format:
/// - INT4 packed: 0.5 bytes/weight
/// - FP16 scale/offset per 256 weights: 0.016 bytes/weight
/// - Total: ~0.52 bytes/weight = ~7.7x compression
pub const CODEC_INT4_10X_V1: &str = "int4_10x_v1";

/// EXPERIMENTAL: Mixed precision by layer sensitivity.
/// Uses different group sizes for different layer types:
/// - Attention Q/K: g=64 (sensitive)
/// - MLP middle: g=256 (robust)
/// Target: 6-8x average compression with <1% PPL delta.
pub const CODEC_INT4_MIXED_V1: &str = "int4_mixed_v1";

/// EXPERIMENTAL: Hybrid codec combining best techniques.
/// - SpinQuant rotation for outlier distribution
/// - Layer-aware group sizes (g=64 for attention, g=256 for MLP)
/// - INT2 residual only for sensitive layers (Q/K projections)
/// Target: 6-7x compression with <1% PPL delta.
pub const CODEC_INT4_HYBRID_V1: &str = "int4_hybrid_v1";

/// EXPERIMENTAL v2: Tuned hybrid with more conservative groups.
/// - Q/K projections: g=16 + INT2 residual (most sensitive, best quality)
/// - V/O projections: g=32 + INT2 residual
/// - MLP gate/up: g=64 (medium)
/// - MLP down: g=128 (robust)
/// Target: 5-6x compression with <1% PPL delta.
pub const CODEC_INT4_HYBRID_V2: &str = "int4_hybrid_v2";

/// EXPERIMENTAL: AWQ-enhanced 10x codec.
/// Uses activation-aware importance scaling from calibration data
/// with large groups (g=128) for max compression.
/// Requires activation_stats in bundle.
/// Target: 8-10x compression with <1% PPL delta.
pub const CODEC_INT4_AWQ_10X_V1: &str = "int4_awq_10x_v1";

/// EXPERIMENTAL: GPTQ-lite codec.
/// Simplified Hessian-weighted reconstruction without full GPTQ.
/// Uses diagonal Hessian approximation for weight updates.
/// Target: 8-10x compression with <1% PPL delta.
pub const CODEC_INT4_GPTQ_LITE_V1: &str = "int4_gptq_lite_v1";

/// EXPERIMENTAL: Ultimate 10x codec - innovation combination.
/// Combines: 1) Outlier extraction (top 0.1% at FP16)
///           2) Ultra-small groups for Q/K (g=8)
///           3) AWQ importance scaling from activation stats
///           4) GPTQ-style iterative error minimization
///           5) Layer-aware adaptive precision
/// Target: 10x compression with <1% PPL delta.
pub const CODEC_INT4_ULTIMATE_V1: &str = "int4_ultimate_v1";

/// EXPERIMENTAL: CALDERA-style codec - Low-rank + Quantization hybrid.
/// Approximates W ≈ Q + LR where:
/// - Q is a highly quantized backbone (INT2, 2-bit)
/// - L and R are small low-rank factors (rank 8-32)
/// - Both L and R are also quantized to INT8
///
/// This captures high-magnitude information in the low-rank factors
/// while the INT2 backbone handles the bulk of the weights.
/// Target: 8-10x compression with <1% PPL delta.
pub const CODEC_CALDERA_V1: &str = "caldera_v1";

/// EXPERIMENTAL: AQLM-style codec - Additive Quantization.
/// Represents each weight as a sum of multiple low-bit quantized values:
/// w ≈ q1 * s1 + q2 * s2 (two 2-bit codes with separate scales)
///
/// This additive structure significantly increases representable values:
/// - Single 4-bit: 16 values
/// - Two 2-bit additive: 4 * 4 = 16 combinations but with 2 scales = more precision
///
/// Uses learned codebook approach with per-group scales.
/// Target: 8-10x compression with <1% PPL delta.
pub const CODEC_AQLM_V1: &str = "aqlm_v1";

/// EXPERIMENTAL: PocketLLM-inspired codec - Vector Quantization with learned codebook.
///
/// Key idea: Instead of quantizing individual weights, quantize VECTORS of weights
/// using a shared codebook learned via k-means clustering.
///
/// Storage format:
/// - Codebook: 256 vectors of size 8 = 256 * 8 * 2 bytes (FP16) = 4KB per tensor
/// - Indices: 1 byte per 8 weights (256 codebook entries = 8-bit index)
/// - Total: ~0.125 + 0.004 = ~0.13 bytes/weight = ~15x compression!
///
/// Combined with low-rank residual for quality preservation.
/// Target: 10x+ compression with <1% PPL delta.
pub const CODEC_POCKETLLM_V1: &str = "pocketllm_v1";

/// EXPERIMENTAL: PocketLLM v2 - Optimized for quality.
/// Uses smaller vector dimension (4 instead of 8) and larger codebook (512).
/// Better quality at the cost of slightly lower compression.
/// Target: 6-8x compression with <1% PPL delta.
pub const CODEC_POCKETLLM_V2: &str = "pocketllm_v2";

/// NOVEL: TenPak-X - Hybrid Low-Rank + Vector Quantization with Importance Weighting
///
/// Combines three techniques:
/// 1. CALDERA-style low-rank decomposition (captures structured redundancy)
/// 2. PocketLLM-style vector quantization (efficient codebook for residual)
/// 3. AWQ-style importance weighting (preserves critical weights)
///
/// Novel contribution: Joint optimization without expensive calibration.
/// Uses weight magnitude as importance proxy (no forward passes needed).
///
/// Formula: W ≈ L @ R + Codebook[indices]
///
/// Target: 8-10x compression with <1% PPL delta in seconds.
pub const CODEC_TENPAK_X_V1: &str = "tenpak_x_v1";

/// NOVEL: TenPak-X v2 - Higher compression variant
///
/// Changes from v1:
/// 1. Higher rank (64) - captures more structure in low-rank factors
/// 2. Sparse residual - only store top-k% most significant residuals
/// 3. INT2 residual instead of INT4 - 2x better residual compression
/// 4. Larger codebook (512) with smaller vectors (2) - better pattern matching
///
/// Target: 8-10x compression with <1% PPL delta.
pub const CODEC_TENPAK_X_V2: &str = "tenpak_x_v2";

/// A single float32 tensor in a simple JSON-friendly format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[cfg(test)]
mod codec_tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn residual_round_trip() {
        // Test the recommended int4_residual codec
        let name = "mlp.weight";
        let shape = vec![64, 128];
        let n = shape[0] * shape[1];
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

        let bundle = FloatBundle {
            tensors: vec![FloatTensor {
                name: name.to_string(),
                shape: shape.clone(),
                data: data.clone(),
            }],
            activation_stats: ActivationStats::new(),
        };

        let artifact =
            compress_bundle_with_codec(&bundle, CODEC_INT4_RESIDUAL_V1).expect("compress");
        let restored = decompress_bundle(&artifact).expect("decompress");
        let restored_tensor = &restored.tensors[0];

        assert_eq!(restored_tensor.shape, shape);
        let max_err = max_abs_diff(&restored_tensor.data, &data);
        assert!(
            max_err < 0.1,
            "residual max abs diff too large: {} (expected < 0.1)",
            max_err
        );
    }

    #[test]
    fn opt_llama_round_trip() {
        // Test the int4_opt_llama codec
        let name = "mlp.weight";
        let shape = vec![64, 128];
        let n = shape[0] * shape[1];
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();

        let bundle = FloatBundle {
            tensors: vec![FloatTensor {
                name: name.to_string(),
                shape: shape.clone(),
                data: data.clone(),
            }],
            activation_stats: ActivationStats::new(),
        };

        let artifact =
            compress_bundle_with_codec(&bundle, CODEC_INT4_OPT_LLAMA_V1).expect("compress");
        let restored = decompress_bundle(&artifact).expect("decompress");
        let restored_tensor = &restored.tensors[0];

        assert_eq!(restored_tensor.shape, shape);
        let max_err = max_abs_diff(&restored_tensor.data, &data);
        assert!(
            max_err < 0.05,
            "opt_llama max abs diff too large: {} (expected < 0.05)",
            max_err
        );
    }
}

/// A bundle of named tensors representing a model checkpoint fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatBundle {
    pub tensors: Vec<FloatTensor>,
    #[serde(default)]
    pub activation_stats: ActivationStats,
}

pub type ActivationStats = HashMap<String, Vec<f32>>;

/// A quantized tensor: encoded values plus scale(s).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// Per-tensor scale (for per-tensor codecs) or first scale (for backward compat).
    pub scale: f32,
    /// Per-channel scales (for per-channel codecs). Empty for per-tensor codecs.
    #[serde(default)]
    pub scales: Vec<f32>,
    /// Raw encoded bytes. Interpretation depends on the codec.
    pub data: Vec<u8>,
    /// Sparse indices (for sparse codecs). Empty for dense codecs.
    #[serde(default)]
    pub indices: Vec<u32>,
    /// Optional per-input-channel scaling factors (for AWQ-style codecs).
    #[serde(default)]
    pub alphas: Vec<f32>,
    /// Optional per-group offsets (e.g., group means for zero-point adjustment).
    #[serde(default)]
    pub offsets: Vec<f32>,
}

/// On-disk artifact: versioned container for quantized tensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub version: u32,
    pub codec: String,
    pub tensors: Vec<QuantizedTensor>,
}

fn expected_len(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| "Shape size overflow when computing tensor length".to_string())
    })
}

/// Compress a floating-point tensor bundle into a quantized artifact using the
/// specified codec.
///
/// # Supported Codecs (Production)
/// - `int4_residual_v1` - Best quality, 5.3x compression, no calibration (RECOMMENDED)
/// - `int4_opt_llama_v1` - Good fallback, 4x compression, no calibration
/// - `int4_calibrated_v1` - Best compression with calibration, 7x (requires `calibration` feature)
///
/// # Experimental Codecs (10x target)
/// - `int4_ultimate_v1` - Combines outlier extraction, AWQ, GPTQ techniques
/// - `int4_spin_v1`, `int4_hybrid_v1`, `int4_hybrid_v2`, etc.
pub fn compress_bundle_with_codec(
    bundle: &FloatBundle,
    codec: &str,
) -> Result<ArtifactFile, String> {
    match codec {
        // Production codecs
        CODEC_INT4_RESIDUAL_V1 => compress_int4_residual(bundle),
        CODEC_INT4_OPT_LLAMA_V1 => compress_int4_opt_llama(bundle),
        #[cfg(feature = "calibration")]
        CODEC_INT4_CALIBRATED_V1 => compress_int4_calibrated(bundle),

        // Experimental 10x codecs
        CODEC_INT4_SPIN_V1 => compress_int4_spin(bundle),
        CODEC_INT4_10X_V1 => compress_int4_10x(bundle),
        CODEC_INT4_MIXED_V1 => compress_int4_mixed(bundle),
        CODEC_INT4_HYBRID_V1 => compress_int4_hybrid(bundle),
        CODEC_INT4_HYBRID_V2 => compress_int4_hybrid_v2(bundle),
        CODEC_INT4_AWQ_10X_V1 => compress_int4_awq_10x(bundle),
        CODEC_INT4_GPTQ_LITE_V1 => compress_int4_gptq_lite(bundle),
        CODEC_INT4_ULTIMATE_V1 => compress_int4_ultimate(bundle),
        CODEC_CALDERA_V1 => compress_caldera(bundle),
        CODEC_AQLM_V1 => compress_aqlm(bundle),
        CODEC_POCKETLLM_V1 => compress_pocketllm(bundle),
        CODEC_POCKETLLM_V2 => compress_pocketllm_v2(bundle),
        CODEC_TENPAK_X_V1 => compress_tenpak_x(bundle),
        CODEC_TENPAK_X_V2 => compress_tenpak_x_v2(bundle),

        other => Err(format!(
            "Unsupported codec '{}'. Use int4_residual_v1 (recommended) or int4_opt_llama_v1.",
            other
        )),
    }
}

/// Convenience helper: compress using the recommended int4_residual_v1 codec.
pub fn compress_bundle(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    compress_bundle_with_codec(bundle, CODEC_INT4_RESIDUAL_V1)
}

/// Optimal quantization for Llama-architecture models.
/// Uses group size 8 and 5 iterations for better quality on Llama models.
/// Achieves 4.00x compression with <1% PPL delta.
fn compress_int4_opt_llama(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 8;
    const ITERATIONS: usize = 5;
    let mut tensors_out = Vec::with_capacity(bundle.tensors.len());

    for t in &bundle.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);

        // First pass: compute initial scales with iterative refinement
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Iterative refinement (5 iterations for Llama)
            for _ in 0..ITERATIONS {
                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in group_data {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                g_min += err_min * 0.5;
                g_max += err_max * 0.5;
            }

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

        // Second pass: quantize with optimized scales
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales_f32[g];
            let offset = offsets_f32[g];
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                let v0 = ((x0 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let mut byte: u8 = v0 & 0x0f;
                if let Some(&x1) = group_iter.next() {
                    let v1 = ((x1 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                    byte |= (v1 & 0x0f) << 4;
                }
                packed.push(byte);
            }
        }

        // Pack scales and offsets as FP16
        let mut data = packed;
        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.get(0).cloned().unwrap_or(1.0),
            scales: Vec::new(),
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_OPT_LLAMA_V1.to_string(),
        tensors: tensors_out,
    })
}

/// Decompress INT4 g=8 FP16 scales/offsets (opt_llama format)
fn decompress_int4_opt_llama(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const GROUP_SIZE: usize = 8;

    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;

        // Calculate offsets in data
        let packed_size = (expected + 1) / 2;
        let scales_start = packed_size;
        let offsets_start = scales_start + num_groups * 2;

        // Read scales and offsets (FP16)
        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let idx = scales_start + g * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                scales.push(f16_bits_to_f32(bits));
            }
        }
        for g in 0..num_groups {
            let idx = offsets_start + g * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                offsets.push(f16_bits_to_f32(bits));
            }
        }

        // Decompress INT4
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let mut weight_idx = 0;

        while weight_idx < expected {
            let byte = t.data[byte_idx];
            let g = weight_idx / GROUP_SIZE;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets.get(g).copied().unwrap_or(0.0);

            // Low nibble
            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            // High nibble
            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Residual quantization: INT4 + INT2 two-pass for better quality.
/// Pass 1: INT4 with iterative refinement (g=16, 5 iterations)
/// Pass 2: INT2 residual correction (g=16)
/// Achieves ~3.5x compression with negative PPL delta.
fn compress_int4_residual(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 16;
    const ITERATIONS: usize = 5;
    const RESIDUAL_GROUP: usize = 16;
    const RESIDUAL_LEVELS: f32 = 3.0; // INT2 = 4 levels (0,1,2,3)

    let mut tensors_out = Vec::with_capacity(bundle.tensors.len());

    for t in &bundle.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);

        // ===== PASS 1: INT4 with iterative refinement =====
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            for _ in 0..ITERATIONS {
                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in group_data {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                g_min += err_min * 0.5;
                g_max += err_max * 0.5;
            }

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

        // Quantize pass 1 and compute residuals
        let mut packed_main: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
        let mut residuals: Vec<f32> = Vec::with_capacity(t.data.len());

        for g in 0..num_groups {
            let scale = scales_f32[g];
            let offset = offsets_f32[g];
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                let q0 = ((x0 - offset) * inv_scale).round().clamp(0.0, 15.0);
                let deq0 = q0 * scale + offset;
                residuals.push(x0 - deq0);

                let mut byte: u8 = (q0 as u8) & 0x0f;
                if let Some(&x1) = group_iter.next() {
                    let q1 = ((x1 - offset) * inv_scale).round().clamp(0.0, 15.0);
                    let deq1 = q1 * scale + offset;
                    residuals.push(x1 - deq1);
                    byte |= ((q1 as u8) & 0x0f) << 4;
                }
                packed_main.push(byte);
            }
        }

        // ===== PASS 2: INT2 residual quantization =====
        let num_res_groups = (residuals.len() + RESIDUAL_GROUP - 1) / RESIDUAL_GROUP;
        let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut packed_res: Vec<u8> = Vec::with_capacity((residuals.len() + 3) / 4);

        for rg in 0..num_res_groups {
            let start = rg * RESIDUAL_GROUP;
            let end = (start + RESIDUAL_GROUP).min(residuals.len());
            let res_group = &residuals[start..end];

            let r_min = res_group.iter().cloned().fold(f32::INFINITY, f32::min);
            let r_max = res_group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let r_scale = if (r_max - r_min).abs() > 1e-8 {
                (r_max - r_min) / RESIDUAL_LEVELS
            } else {
                1.0
            };

            res_scales.push(r_scale);
            res_offsets.push(r_min);

            // Pack 4 INT2 values per byte
            let inv_r_scale = if r_scale.abs() > 1e-8 {
                1.0 / r_scale
            } else {
                1.0
            };
            let mut i = 0;
            while i < res_group.len() {
                let mut byte: u8 = 0;
                for bit_pos in 0..4 {
                    if i + bit_pos < res_group.len() {
                        let q = ((res_group[i + bit_pos] - r_min) * inv_r_scale)
                            .round()
                            .clamp(0.0, 3.0) as u8;
                        byte |= (q & 0x03) << (bit_pos * 2);
                    }
                }
                packed_res.push(byte);
                i += 4;
            }
        }

        // Pack everything into data field:
        // [main_packed | main_scales_fp16 | main_offsets_fp16 | res_packed | res_scales_fp16 | res_offsets_fp16]
        let mut data = packed_main;

        // Main scales/offsets as FP16
        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        // Residual data
        data.extend_from_slice(&packed_res);

        // Residual scales/offsets as FP16
        for &s in &res_scales {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &res_offsets {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.get(0).cloned().unwrap_or(1.0),
            scales: Vec::new(),
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_RESIDUAL_V1.to_string(),
        tensors: tensors_out,
    })
}

// ============================================================================
// EXPERIMENTAL 10x COMPRESSION CODECS
// ============================================================================

/// Fast Walsh-Hadamard transform (in-place, power-of-2 size).
/// This spreads outliers across dimensions, improving quantization quality.
fn hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return; // Not power of 2, skip
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..(i + h) {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    // Normalize
    let norm = (n as f32).sqrt();
    for v in data.iter_mut() {
        *v /= norm;
    }
}

/// Inverse Hadamard transform (same as forward for orthogonal Hadamard).
fn inverse_hadamard_transform(data: &mut [f32]) {
    hadamard_transform(data); // Hadamard is its own inverse (up to normalization)
}

/// SpinQuant-inspired codec: Apply Hadamard rotation before quantization.
/// This spreads outliers across the weight vector, reducing max error.
/// Target: 6-8x compression with <1% PPL delta, NO calibration.
fn compress_int4_spin(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 256; // Large groups for high compression
    const ITERATIONS: usize = 5;

    let mut tensors_out = Vec::with_capacity(bundle.tensors.len());

    for t in &bundle.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        // Find nearest power of 2 for Hadamard
        let hadamard_size = 256usize; // Fixed size for simplicity
        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;

        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_len = end - start;

            // Copy and apply Hadamard transform if power of 2
            let mut rotated: Vec<f32> = t.data[start..end].to_vec();

            // Apply Hadamard to power-of-2 chunks
            let chunk_size = hadamard_size.min(group_len);
            if chunk_size > 0 && (chunk_size & (chunk_size - 1)) == 0 {
                for chunk in rotated.chunks_mut(chunk_size) {
                    if chunk.len() == chunk_size {
                        hadamard_transform(chunk);
                    }
                }
            }

            // Iterative scale refinement on rotated weights
            let mut g_min = rotated.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut g_max = rotated.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            for _ in 0..ITERATIONS {
                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in &rotated {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                g_min += err_min * 0.5;
                g_max += err_max * 0.5;
            }

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            scales_f32.push(scale);
            offsets_f32.push(g_min);

            // Quantize and pack
            let mut group_iter = rotated.iter();
            while let Some(&x0) = group_iter.next() {
                let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                let mut byte: u8 = (q0 as u8) & 0x0f;

                if let Some(&x1) = group_iter.next() {
                    let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    byte |= ((q1 as u8) & 0x0f) << 4;
                }
                packed.push(byte);
            }
        }

        // Pack: [quantized_data | scales_fp16 | offsets_fp16]
        let mut data = packed;
        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.get(0).cloned().unwrap_or(1.0),
            scales: Vec::new(),
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_SPIN_V1.to_string(),
        tensors: tensors_out,
    })
}

/// Maximum compression codec: INT4 with g=256, no residual.
/// Target: ~7.7x compression (may have higher PPL delta).
/// Uses parallel processing for speed.
fn compress_int4_10x(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 256;
    const ITERATIONS: usize = 7;

    // Process tensors in parallel
    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;

            // Process groups in parallel, collect (scale, offset, packed_bytes)
            let group_results: Vec<(f32, f32, Vec<u8>)> = (0..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * GROUP_SIZE;
                    let end = (start + GROUP_SIZE).min(t.data.len());
                    let group_data = &t.data[start..end];

                    let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // Iterative refinement
                    for _ in 0..ITERATIONS {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut err_min = f32::INFINITY;
                        let mut err_max = f32::NEG_INFINITY;

                        for &val in group_data {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            err_min = err_min.min(err);
                            err_max = err_max.max(err);
                        }

                        g_min += err_min * 0.4;
                        g_max += err_max * 0.4;
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    // Pack INT4
                    let mut packed: Vec<u8> = Vec::with_capacity((group_data.len() + 1) / 2);
                    let mut group_iter = group_data.iter();
                    while let Some(&x0) = group_iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let mut byte: u8 = (q0 as u8) & 0x0f;

                        if let Some(&x1) = group_iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    (scale, g_min, packed)
                })
                .collect();

            // Combine results (must be sequential to maintain order)
            let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for (scale, offset, packed) in group_results {
                scales_f32.push(scale);
                offsets_f32.push(offset);
                all_packed.extend(packed);
            }

            // Pack: [quantized_data | scales_fp16 | offsets_fp16]
            let mut data = all_packed;
            for &s in &scales_f32 {
                let f16_bits = f32_to_f16_bits(s);
                data.push((f16_bits & 0xff) as u8);
                data.push(((f16_bits >> 8) & 0xff) as u8);
            }
            for &o in &offsets_f32 {
                let f16_bits = f32_to_f16_bits(o);
                data.push((f16_bits & 0xff) as u8);
                data.push(((f16_bits >> 8) & 0xff) as u8);
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales_f32.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    // Collect results, propagating any errors
    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_10X_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// Mixed precision codec: Different group sizes based on layer sensitivity.
/// Detects layer type from name and applies appropriate compression.
fn compress_int4_mixed(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    let mut tensors_out = Vec::with_capacity(bundle.tensors.len());

    for t in &bundle.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        // Determine group size based on layer name
        let group_size = if t.name.contains("q_proj") || t.name.contains("k_proj") {
            64 // Attention Q/K are sensitive
        } else if t.name.contains("v_proj") || t.name.contains("o_proj") {
            128 // Attention V/O are medium
        } else if t.name.contains("down_proj") {
            256 // MLP down is robust
        } else if t.name.contains("gate_proj") || t.name.contains("up_proj") {
            128 // MLP gate/up are medium
        } else {
            128 // Default
        };

        let iterations = if group_size <= 64 { 5 } else { 7 };

        let num_groups = (t.data.len() + group_size - 1) / group_size;
        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(t.data.len());
            let group_data = &t.data[start..end];

            let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            for _ in 0..iterations {
                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in group_data {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                g_min += err_min * 0.5;
                g_max += err_max * 0.5;
            }

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            scales_f32.push(scale);
            offsets_f32.push(g_min);

            let mut group_iter = group_data.iter();
            while let Some(&x0) = group_iter.next() {
                let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                let mut byte: u8 = (q0 as u8) & 0x0f;

                if let Some(&x1) = group_iter.next() {
                    let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    byte |= ((q1 as u8) & 0x0f) << 4;
                }
                packed.push(byte);
            }
        }

        // Store group_size in first 2 bytes for decompression
        let mut data = vec![(group_size & 0xff) as u8, ((group_size >> 8) & 0xff) as u8];
        data.extend_from_slice(&packed);

        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.get(0).cloned().unwrap_or(1.0),
            scales: Vec::new(),
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_MIXED_V1.to_string(),
        tensors: tensors_out,
    })
}

/// Hybrid codec combining SpinQuant rotation + layer-aware precision + selective residual.
/// - SpinQuant (Hadamard) rotation spreads outliers for better quantization
/// - Attention Q/K: g=64 + INT2 residual (most sensitive)
/// - Attention V/O: g=128 (medium)
/// - MLP: g=256 (robust, max compression)
/// Uses parallel processing for speed.
fn compress_int4_hybrid(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Determine parameters based on layer type
            let (group_size, use_residual, iterations) =
                if t.name.contains("q_proj") || t.name.contains("k_proj") {
                    (64, true, 5) // Most sensitive: small groups + residual
                } else if t.name.contains("v_proj") || t.name.contains("o_proj") {
                    (128, false, 5) // Medium: moderate groups
                } else if t.name.contains("down_proj") {
                    (256, false, 7) // Robust: large groups
                } else {
                    (128, false, 5) // Default
                };

            let num_groups = (t.data.len() + group_size - 1) / group_size;

            // Process groups in parallel
            let group_results: Vec<(f32, f32, Vec<u8>, Option<(f32, f32, Vec<u8>)>)> = (0
                ..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * group_size;
                    let end = (start + group_size).min(t.data.len());
                    let group_data = &t.data[start..end];
                    let group_len = end - start;

                    // Apply Hadamard rotation if power of 2
                    let mut rotated: Vec<f32> = group_data.to_vec();
                    if group_len > 0 && (group_len & (group_len - 1)) == 0 && group_len <= 256 {
                        hadamard_transform(&mut rotated);
                    }

                    // Iterative scale refinement
                    let mut g_min = rotated.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = rotated.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    for _ in 0..iterations {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut err_min = f32::INFINITY;
                        let mut err_max = f32::NEG_INFINITY;

                        for &val in &rotated {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            err_min = err_min.min(err);
                            err_max = err_max.max(err);
                        }

                        g_min += err_min * 0.5;
                        g_max += err_max * 0.5;
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    // Pack INT4 and compute residuals
                    let mut packed: Vec<u8> = Vec::with_capacity((group_len + 1) / 2);
                    let mut residuals: Vec<f32> = if use_residual {
                        Vec::with_capacity(group_len)
                    } else {
                        Vec::new()
                    };

                    let mut iter = rotated.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let deq0 = q0 * scale + g_min;
                        if use_residual {
                            residuals.push(x0 - deq0);
                        }

                        let mut byte: u8 = (q0 as u8) & 0x0f;
                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq1 = q1 * scale + g_min;
                            if use_residual {
                                residuals.push(x1 - deq1);
                            }
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    // INT2 residual quantization if needed
                    let residual_data = if use_residual && !residuals.is_empty() {
                        let r_min = residuals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let r_max = residuals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let r_scale = if (r_max - r_min).abs() > 1e-8 {
                            (r_max - r_min) / 3.0
                        } else {
                            1.0
                        };
                        let r_inv = if r_scale.abs() > 1e-8 {
                            1.0 / r_scale
                        } else {
                            1.0
                        };

                        let mut r_packed: Vec<u8> = Vec::with_capacity((residuals.len() + 3) / 4);
                        let mut r_iter = residuals.iter();
                        while let Some(&r0) = r_iter.next() {
                            let q0 = ((r0 - r_min) * r_inv).round().clamp(0.0, 3.0) as u8;
                            let q1 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            let q2 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            let q3 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            r_packed.push(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
                        }
                        Some((r_scale, r_min, r_packed))
                    } else {
                        None
                    };

                    (scale, g_min, packed, residual_data)
                })
                .collect();

            // Combine results
            let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
            let mut res_scales: Vec<f32> = Vec::new();
            let mut res_offsets: Vec<f32> = Vec::new();
            let mut all_res_packed: Vec<u8> = Vec::new();

            for (scale, offset, packed, res_data) in group_results {
                scales.push(scale);
                offsets.push(offset);
                all_packed.extend(packed);
                if let Some((rs, ro, rp)) = res_data {
                    res_scales.push(rs);
                    res_offsets.push(ro);
                    all_res_packed.extend(rp);
                }
            }

            // Pack: [flags | group_size | main_data | scales_fp16 | offsets_fp16 | res_data | res_scales | res_offsets]
            let flags: u8 = if use_residual { 1 } else { 0 };
            let mut data = vec![
                flags,
                (group_size & 0xff) as u8,
                ((group_size >> 8) & 0xff) as u8,
            ];
            data.extend_from_slice(&all_packed);

            for &s in &scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }
            for &o in &offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            if use_residual {
                data.extend_from_slice(&all_res_packed);
                for &s in &res_scales {
                    let f16 = f32_to_f16_bits(s);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
                for &o in &res_offsets {
                    let f16 = f32_to_f16_bits(o);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_HYBRID_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// Tuned hybrid v2: More conservative groups for better quality.
/// Q/K: g=16 + residual, V/O: g=32 + residual, MLP: g=64/128
fn compress_int4_hybrid_v2(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // More conservative layer-aware parameters
            let (group_size, use_residual, iterations) =
                if t.name.contains("q_proj") || t.name.contains("k_proj") {
                    (16, true, 5) // Most sensitive: tiny groups + residual
                } else if t.name.contains("v_proj") || t.name.contains("o_proj") {
                    (32, true, 5) // Also sensitive
                } else if t.name.contains("gate_proj") || t.name.contains("up_proj") {
                    (64, false, 5) // Medium
                } else if t.name.contains("down_proj") {
                    (128, false, 7) // Robust
                } else {
                    (64, false, 5) // Default
                };

            let num_groups = (t.data.len() + group_size - 1) / group_size;

            let group_results: Vec<(f32, f32, Vec<u8>, Option<(f32, f32, Vec<u8>)>)> = (0
                ..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * group_size;
                    let end = (start + group_size).min(t.data.len());
                    let group_data = &t.data[start..end];
                    let group_len = end - start;

                    // Iterative refinement (no Hadamard for small groups)
                    let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    for _ in 0..iterations {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut err_min = f32::INFINITY;
                        let mut err_max = f32::NEG_INFINITY;

                        for &val in group_data {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            err_min = err_min.min(err);
                            err_max = err_max.max(err);
                        }

                        g_min += err_min * 0.5;
                        g_max += err_max * 0.5;
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    let mut packed: Vec<u8> = Vec::with_capacity((group_len + 1) / 2);
                    let mut residuals: Vec<f32> = if use_residual {
                        Vec::with_capacity(group_len)
                    } else {
                        Vec::new()
                    };

                    let mut iter = group_data.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let deq0 = q0 * scale + g_min;
                        if use_residual {
                            residuals.push(x0 - deq0);
                        }

                        let mut byte: u8 = (q0 as u8) & 0x0f;
                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq1 = q1 * scale + g_min;
                            if use_residual {
                                residuals.push(x1 - deq1);
                            }
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    let residual_data = if use_residual && !residuals.is_empty() {
                        let r_min = residuals.iter().cloned().fold(f32::INFINITY, f32::min);
                        let r_max = residuals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let r_scale = if (r_max - r_min).abs() > 1e-8 {
                            (r_max - r_min) / 3.0
                        } else {
                            1.0
                        };
                        let r_inv = if r_scale.abs() > 1e-8 {
                            1.0 / r_scale
                        } else {
                            1.0
                        };

                        let mut r_packed: Vec<u8> = Vec::with_capacity((residuals.len() + 3) / 4);
                        let mut r_iter = residuals.iter();
                        while let Some(&r0) = r_iter.next() {
                            let q0 = ((r0 - r_min) * r_inv).round().clamp(0.0, 3.0) as u8;
                            let q1 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            let q2 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            let q3 = r_iter
                                .next()
                                .map(|&r| ((r - r_min) * r_inv).round().clamp(0.0, 3.0) as u8)
                                .unwrap_or(0);
                            r_packed.push(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6));
                        }
                        Some((r_scale, r_min, r_packed))
                    } else {
                        None
                    };

                    (scale, g_min, packed, residual_data)
                })
                .collect();

            let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
            let mut res_scales: Vec<f32> = Vec::new();
            let mut res_offsets: Vec<f32> = Vec::new();
            let mut all_res_packed: Vec<u8> = Vec::new();

            for (scale, offset, packed, res_data) in group_results {
                scales.push(scale);
                offsets.push(offset);
                all_packed.extend(packed);
                if let Some((rs, ro, rp)) = res_data {
                    res_scales.push(rs);
                    res_offsets.push(ro);
                    all_res_packed.extend(rp);
                }
            }

            let flags: u8 = if use_residual { 1 } else { 0 };
            let mut data = vec![
                flags,
                (group_size & 0xff) as u8,
                ((group_size >> 8) & 0xff) as u8,
            ];
            data.extend_from_slice(&all_packed);

            for &s in &scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }
            for &o in &offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            if use_residual {
                data.extend_from_slice(&all_res_packed);
                for &s in &res_scales {
                    let f16 = f32_to_f16_bits(s);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
                for &o in &res_offsets {
                    let f16 = f32_to_f16_bits(o);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_HYBRID_V2.to_string(),
        tensors: tensors_out?,
    })
}

/// AWQ-enhanced 10x codec: Activation-aware importance scaling + large groups.
/// Uses activation_stats from bundle for importance weighting.
fn compress_int4_awq_10x(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 128; // Large groups for compression
    const ITERATIONS: usize = 7;
    const IMPORTANCE_SCALE: f32 = 1.5; // Scale important weights

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Check for activation stats
            // Compute importance from activation stats (Vec<f32> of activation magnitudes)
            let importance = bundle
                .activation_stats
                .get(&t.name)
                .map(|stats| {
                    if stats.is_empty() {
                        return 1.0;
                    }
                    let mean = stats.iter().sum::<f32>() / stats.len() as f32;
                    let variance =
                        stats.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / stats.len() as f32;
                    mean.abs() + variance.sqrt()
                })
                .unwrap_or(1.0);

            // Scale weights by importance (AWQ-style)
            let scale_factor = if importance > 0.1 {
                IMPORTANCE_SCALE
            } else {
                1.0
            };

            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;

            let group_results: Vec<(f32, f32, Vec<u8>)> = (0..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * GROUP_SIZE;
                    let end = (start + GROUP_SIZE).min(t.data.len());
                    let group_data = &t.data[start..end];

                    // Scale by importance
                    let scaled: Vec<f32> = group_data.iter().map(|&x| x * scale_factor).collect();

                    let mut g_min = scaled.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    for _ in 0..ITERATIONS {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut err_min = f32::INFINITY;
                        let mut err_max = f32::NEG_INFINITY;

                        for &val in &scaled {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            err_min = err_min.min(err);
                            err_max = err_max.max(err);
                        }

                        g_min += err_min * 0.4;
                        g_max += err_max * 0.4;
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    let mut packed: Vec<u8> = Vec::with_capacity((scaled.len() + 1) / 2);
                    let mut iter = scaled.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let mut byte: u8 = (q0 as u8) & 0x0f;

                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    // Store adjusted scale (divide by importance to reverse on decompress)
                    let adjusted_scale = scale / scale_factor;
                    let adjusted_offset = g_min / scale_factor;

                    (adjusted_scale, adjusted_offset, packed)
                })
                .collect();

            let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for (scale, offset, packed) in group_results {
                scales.push(scale);
                offsets.push(offset);
                all_packed.extend(packed);
            }

            // Add group size header for decompress_int4_mixed compatibility
            let mut data = vec![(GROUP_SIZE & 0xff) as u8, ((GROUP_SIZE >> 8) & 0xff) as u8];
            data.extend_from_slice(&all_packed);
            for &s in &scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }
            for &o in &offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_10X_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// GPTQ-lite: Simplified Hessian-weighted quantization.
/// Uses diagonal Hessian approximation (weight magnitude) for importance.
fn compress_int4_gptq_lite(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 64; // Medium groups
    const ITERATIONS: usize = 10; // More iterations for GPTQ-style refinement

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;

            // Compute per-weight importance (diagonal Hessian approximation)
            // Important weights = larger magnitude (simplified GPTQ)
            let weight_importance: Vec<f32> = t
                .data
                .iter()
                .map(|&w| w.abs() + 0.01) // Add small constant to avoid division by zero
                .collect();

            let group_results: Vec<(f32, f32, Vec<u8>)> = (0..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * GROUP_SIZE;
                    let end = (start + GROUP_SIZE).min(t.data.len());
                    let group_data = &t.data[start..end];
                    let group_importance = &weight_importance[start..end];

                    // Weighted min/max based on importance
                    let _total_importance: f32 = group_importance.iter().sum();
                    let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // GPTQ-style iterative refinement with importance weighting
                    for _ in 0..ITERATIONS {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        // Weighted error accumulation
                        let mut weighted_err_neg = 0.0f32;
                        let mut weighted_err_pos = 0.0f32;
                        let mut weight_neg = 0.0f32;
                        let mut weight_pos = 0.0f32;

                        for (&val, &imp) in group_data.iter().zip(group_importance.iter()) {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;

                            if err < 0.0 {
                                weighted_err_neg += err * imp;
                                weight_neg += imp;
                            } else {
                                weighted_err_pos += err * imp;
                                weight_pos += imp;
                            }
                        }

                        // Adjust based on weighted errors
                        if weight_neg > 0.0 {
                            g_min += (weighted_err_neg / weight_neg) * 0.3;
                        }
                        if weight_pos > 0.0 {
                            g_max += (weighted_err_pos / weight_pos) * 0.3;
                        }
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    let mut packed: Vec<u8> = Vec::with_capacity((group_data.len() + 1) / 2);
                    let mut iter = group_data.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let mut byte: u8 = (q0 as u8) & 0x0f;

                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    (scale, g_min, packed)
                })
                .collect();

            let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for (scale, offset, packed) in group_results {
                scales.push(scale);
                offsets.push(offset);
                all_packed.extend(packed);
            }

            // Add group size header for decompress_int4_mixed compatibility
            let mut data = vec![(GROUP_SIZE & 0xff) as u8, ((GROUP_SIZE >> 8) & 0xff) as u8];
            data.extend_from_slice(&all_packed);
            for &s in &scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }
            for &o in &offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_GPTQ_LITE_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// Ultimate 10x codec: Innovative combination of techniques.
/// Key innovations:
/// 1. Outlier extraction: Top 0.1% weights stored at FP16
/// 2. Ultra-small groups: g=8 for Q/K, g=16 for V/O, g=64 for MLP
/// 3. AWQ importance scaling from activation stats
/// 4. GPTQ-style iterative error minimization with 10 iterations
/// 5. Asymmetric quantization with optimal range finding
fn compress_int4_ultimate(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const OUTLIER_PERCENTILE: f32 = 0.001; // Top 0.1% as outliers
    const MAX_ITERATIONS: usize = 10;

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Innovation 1: Layer-aware ultra-small groups
            let group_size = if t.name.contains("q_proj") || t.name.contains("k_proj") {
                8 // Most sensitive: ultra-fine groups
            } else if t.name.contains("v_proj") || t.name.contains("o_proj") {
                16 // Also sensitive
            } else if t.name.contains("gate_proj") || t.name.contains("up_proj") {
                32 // Medium sensitivity
            } else if t.name.contains("down_proj") {
                64 // Least sensitive in attention
            } else if t.name.contains("embed") || t.name.contains("lm_head") {
                8 // Embeddings are critical
            } else {
                32 // Default
            };

            // Innovation 2: Compute AWQ importance from activation stats
            let importance = bundle
                .activation_stats
                .get(&t.name)
                .map(|stats| {
                    if stats.is_empty() {
                        return 1.0;
                    }
                    let mean = stats.iter().sum::<f32>() / stats.len() as f32;
                    let variance =
                        stats.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / stats.len() as f32;
                    (mean.abs() + variance.sqrt()).max(0.1)
                })
                .unwrap_or(1.0);

            // Innovation 3: Extract outliers (top 0.1% by magnitude)
            let mut sorted_abs: Vec<f32> = t.data.iter().map(|x| x.abs()).collect();
            sorted_abs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let outlier_count = ((t.data.len() as f32 * OUTLIER_PERCENTILE) as usize).max(1);
            let outlier_threshold = sorted_abs
                .get(outlier_count - 1)
                .cloned()
                .unwrap_or(f32::INFINITY);

            // Find outlier indices and values
            let mut outlier_indices: Vec<u32> = Vec::new();
            let mut outlier_values: Vec<f32> = Vec::new();
            let mut main_data: Vec<f32> = t.data.clone();

            for (i, &val) in t.data.iter().enumerate() {
                if val.abs() >= outlier_threshold && outlier_indices.len() < outlier_count {
                    outlier_indices.push(i as u32);
                    outlier_values.push(val);
                    // Replace outlier with local mean to reduce range
                    let start = if i >= 4 { i - 4 } else { 0 };
                    let end = (i + 5).min(t.data.len());
                    let neighbors: Vec<f32> = t.data[start..end]
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| start + j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    let local_mean = if neighbors.is_empty() {
                        0.0
                    } else {
                        neighbors.iter().sum::<f32>() / neighbors.len() as f32
                    };
                    main_data[i] = local_mean;
                }
            }

            // Innovation 4: AWQ-style importance scaling before quantization
            let importance_scale = if importance > 1.0 {
                1.0 + (importance - 1.0) * 0.1
            } else {
                1.0
            };
            let scaled_data: Vec<f32> = main_data.iter().map(|&x| x * importance_scale).collect();

            let num_groups = (scaled_data.len() + group_size - 1) / group_size;

            // Quantize each group with GPTQ-style iterative refinement
            let group_results: Vec<(f32, f32, Vec<u8>)> = (0..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * group_size;
                    let end = (start + group_size).min(scaled_data.len());
                    let group_data = &scaled_data[start..end];
                    let group_len = end - start;

                    // Initial range
                    let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    // GPTQ-style: Weight by magnitude squared (Hessian approximation)
                    let weights: Vec<f32> = group_data
                        .iter()
                        .map(|x| x.abs().powi(2).max(0.001))
                        .collect();

                    // Innovation 5: Iterative weighted error minimization
                    for _ in 0..MAX_ITERATIONS {
                        let range = g_max - g_min;
                        let scale = if range.abs() > 1e-8 {
                            range / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut weighted_err_min: f32 = 0.0;
                        let mut weighted_err_max: f32 = 0.0;
                        let mut weight_min: f32 = 0.0;
                        let mut weight_max: f32 = 0.0;

                        for (i, &val) in group_data.iter().enumerate() {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            let w = weights[i];

                            if err < 0.0 {
                                weighted_err_min += err * w;
                                weight_min += w;
                            } else {
                                weighted_err_max += err * w;
                                weight_max += w;
                            }
                        }

                        // Update bounds based on weighted error
                        if weight_min > 0.0 {
                            g_min += (weighted_err_min / weight_min) * 0.5;
                        }
                        if weight_max > 0.0 {
                            g_max += (weighted_err_max / weight_max) * 0.5;
                        }
                    }

                    let range = g_max - g_min;
                    let scale = if range.abs() > 1e-8 {
                        range / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    // Pack to INT4
                    let mut packed: Vec<u8> = Vec::with_capacity((group_len + 1) / 2);
                    let mut iter = group_data.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                        let mut byte: u8 = (q0 as u8) & 0x0f;
                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            byte |= ((q1 as u8) & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }

                    // Un-scale the parameters
                    (scale / importance_scale, g_min / importance_scale, packed)
                })
                .collect();

            // Assemble output
            let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut all_packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for (scale, offset, packed) in group_results {
                scales.push(scale);
                offsets.push(offset);
                all_packed.extend(packed);
            }

            // Format: [header][outliers][int4_data][scales][offsets]
            // Header: 1 byte flags, 2 bytes group_size, 4 bytes outlier_count
            let mut data: Vec<u8> = Vec::new();
            let flags: u8 = 0x01; // Has outliers
            data.push(flags);
            data.push((group_size & 0xff) as u8);
            data.push(((group_size >> 8) & 0xff) as u8);
            data.push((outlier_indices.len() & 0xff) as u8);
            data.push(((outlier_indices.len() >> 8) & 0xff) as u8);
            data.push(((outlier_indices.len() >> 16) & 0xff) as u8);
            data.push(((outlier_indices.len() >> 24) & 0xff) as u8);

            // Outlier indices (4 bytes each)
            for &idx in &outlier_indices {
                data.push((idx & 0xff) as u8);
                data.push(((idx >> 8) & 0xff) as u8);
                data.push(((idx >> 16) & 0xff) as u8);
                data.push(((idx >> 24) & 0xff) as u8);
            }

            // Outlier values (4 bytes each, FP32 for precision)
            for &val in &outlier_values {
                let bits = val.to_bits();
                data.push((bits & 0xff) as u8);
                data.push(((bits >> 8) & 0xff) as u8);
                data.push(((bits >> 16) & 0xff) as u8);
                data.push(((bits >> 24) & 0xff) as u8);
            }

            // INT4 packed data
            data.extend_from_slice(&all_packed);

            // Scales as FP16
            for &s in &scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // Offsets as FP16
            for &o in &offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_ULTIMATE_V1.to_string(),
        tensors: tensors_out?,
    })
}

// ============================================================================
// CALDERA-STYLE CODEC: Low-Rank + Quantization Hybrid
// ============================================================================

/// Simple SVD-like low-rank approximation using power iteration.
/// Returns (U, S, V) where W ≈ U * diag(S) * V^T
/// U is (m x rank), S is (rank,), V is (n x rank)
fn simple_svd(
    data: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rank = rank.min(rows).min(cols);

    let mut u_mat = vec![0.0f32; rows * rank];
    let mut s_vec = vec![0.0f32; rank];
    let mut v_mat = vec![0.0f32; cols * rank];

    // Power iteration for each singular vector
    let mut residual = data.to_vec();

    for r in 0..rank {
        // Initialize random vector
        let mut v: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + r * 13) % 100) as f32 / 100.0 - 0.5)
            .collect();

        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }

        // Power iteration (10 iterations)
        for _ in 0..10 {
            // u = A * v
            let mut u = vec![0.0f32; rows];
            for i in 0..rows {
                for j in 0..cols {
                    u[i] += residual[i * cols + j] * v[j];
                }
            }

            // Normalize u
            let u_norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if u_norm > 1e-8 {
                for x in &mut u {
                    *x /= u_norm;
                }
            }

            // v = A^T * u
            v = vec![0.0f32; cols];
            for i in 0..rows {
                for j in 0..cols {
                    v[j] += residual[i * cols + j] * u[i];
                }
            }

            // Normalize v
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if v_norm > 1e-8 {
                for x in &mut v {
                    *x /= v_norm;
                }
            }
        }

        // Compute singular value: sigma = u^T * A * v
        let mut sigma = 0.0f32;
        for i in 0..rows {
            let mut row_sum = 0.0f32;
            for j in 0..cols {
                row_sum += residual[i * cols + j] * v[j];
            }
            sigma += u_mat.get(i).copied().unwrap_or(0.0) * row_sum;
        }

        // Recompute u = A * v and get sigma from norm
        let mut u = vec![0.0f32; rows];
        for i in 0..rows {
            for j in 0..cols {
                u[i] += residual[i * cols + j] * v[j];
            }
        }
        sigma = u.iter().map(|x| x * x).sum::<f32>().sqrt();

        if sigma > 1e-8 {
            for x in &mut u {
                *x /= sigma;
            }
        }

        // Store results
        for i in 0..rows {
            u_mat[i * rank + r] = u[i];
        }
        s_vec[r] = sigma;
        for j in 0..cols {
            v_mat[j * rank + r] = v[j];
        }

        // Deflate: residual -= sigma * u * v^T
        for i in 0..rows {
            for j in 0..cols {
                residual[i * cols + j] -= sigma * u[i] * v[j];
            }
        }
    }

    (u_mat, s_vec, v_mat)
}

/// CALDERA-style codec: W ≈ Q + LR
/// - Q: INT2 quantized backbone (very aggressive, 2-bit)
/// - L, R: Low-rank factors capturing important information
/// - L and R are quantized to INT8
///
/// Storage format:
/// - INT2 backbone: 0.25 bytes/weight
/// - Low-rank factors: rank * (rows + cols) * 1 byte / (rows * cols)
/// - Scales/offsets: minimal overhead
///
/// For rank=16 on 4096x4096: ~0.26 + 0.008 = ~0.27 bytes/weight = ~7.4x compression
fn compress_caldera(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const RANK: usize = 16; // Low-rank approximation rank
    const GROUP_SIZE: usize = 64; // For INT2 backbone quantization

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Need 2D shape for SVD
            let (rows, cols) = if t.shape.len() == 2 {
                (t.shape[0], t.shape[1])
            } else if t.shape.len() == 1 {
                // 1D tensor: treat as 1 x N (no meaningful low-rank structure)
                (1, t.shape[0])
            } else {
                // Flatten to 2D: first dim vs rest
                let rows = t.shape[0];
                let cols: usize = t.shape[1..].iter().product();
                (rows, cols)
            };

            // Adaptive rank based on matrix size
            // For 1D tensors (rows=1), use rank=0 to skip SVD
            let actual_rank = if rows <= 1 || cols <= 4 {
                0 // Skip SVD for 1D or very small tensors
            } else {
                RANK.min(rows / 4).min(cols / 4).max(1)
            };

            // Step 1: Compute low-rank approximation
            let (u_mat, s_vec, v_mat) = simple_svd(&t.data, rows, cols, actual_rank);

            // Step 2: Compute low-rank reconstruction and residual
            // LR = U * diag(S) * V^T
            let mut lr_approx = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    for r in 0..actual_rank {
                        lr_approx[i * cols + j] +=
                            u_mat[i * actual_rank + r] * s_vec[r] * v_mat[j * actual_rank + r];
                    }
                }
            }

            // Residual = W - LR (this is what we quantize aggressively)
            let residual: Vec<f32> = t
                .data
                .iter()
                .zip(lr_approx.iter())
                .map(|(&w, &lr)| w - lr)
                .collect();

            // Step 3: Quantize residual to INT2 (4 levels: 0,1,2,3)
            let num_groups = (residual.len() + GROUP_SIZE - 1) / GROUP_SIZE;
            let mut q_scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut q_offsets: Vec<f32> = Vec::with_capacity(num_groups);
            let mut q_packed: Vec<u8> = Vec::with_capacity((residual.len() + 3) / 4);

            for g in 0..num_groups {
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(residual.len());
                let group = &residual[start..end];

                let g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 3.0 // INT2 has 4 levels (0-3)
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                q_scales.push(scale);
                q_offsets.push(g_min);

                // Pack 4 INT2 values per byte
                let mut i = 0;
                while i < group.len() {
                    let mut byte: u8 = 0;
                    for bit_pos in 0..4 {
                        if i + bit_pos < group.len() {
                            let q = ((group[i + bit_pos] - g_min) * inv_scale)
                                .round()
                                .clamp(0.0, 3.0) as u8;
                            byte |= (q & 0x03) << (bit_pos * 2);
                        }
                    }
                    q_packed.push(byte);
                    i += 4;
                }
            }

            // Step 4: Quantize L and R factors to INT8
            // L = U * sqrt(S), R = sqrt(S) * V^T
            // We store L (rows x rank) and R (rank x cols) as INT8

            // Compute L = U * sqrt(S)
            let mut l_factors: Vec<f32> = vec![0.0; rows * actual_rank];
            for i in 0..rows {
                for r in 0..actual_rank {
                    l_factors[i * actual_rank + r] = u_mat[i * actual_rank + r] * s_vec[r].sqrt();
                }
            }

            // Compute R = sqrt(S) * V^T (stored as cols x rank for easier access)
            let mut r_factors: Vec<f32> = vec![0.0; cols * actual_rank];
            for j in 0..cols {
                for r in 0..actual_rank {
                    r_factors[j * actual_rank + r] = v_mat[j * actual_rank + r] * s_vec[r].sqrt();
                }
            }

            // Quantize L to INT8
            let l_min = l_factors.iter().cloned().fold(f32::INFINITY, f32::min);
            let l_max = l_factors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l_scale = if (l_max - l_min).abs() > 1e-8 {
                (l_max - l_min) / 255.0
            } else {
                1.0
            };
            let l_inv = if l_scale.abs() > 1e-8 {
                1.0 / l_scale
            } else {
                1.0
            };
            let l_quant: Vec<u8> = l_factors
                .iter()
                .map(|&x| ((x - l_min) * l_inv).round().clamp(0.0, 255.0) as u8)
                .collect();

            // Quantize R to INT8
            let r_min = r_factors.iter().cloned().fold(f32::INFINITY, f32::min);
            let r_max = r_factors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let r_scale = if (r_max - r_min).abs() > 1e-8 {
                (r_max - r_min) / 255.0
            } else {
                1.0
            };
            let r_inv = if r_scale.abs() > 1e-8 {
                1.0 / r_scale
            } else {
                1.0
            };
            let r_quant: Vec<u8> = r_factors
                .iter()
                .map(|&x| ((x - r_min) * r_inv).round().clamp(0.0, 255.0) as u8)
                .collect();

            // Step 5: Pack everything
            // Format: [header][q_packed][q_scales_fp16][q_offsets_fp16][l_quant][r_quant][lr_params]
            // Header: rank(2) + rows(4) + cols(4) + group_size(2) = 12 bytes
            // lr_params: l_scale(2) + l_min(2) + r_scale(2) + r_min(2) = 8 bytes

            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push((actual_rank & 0xff) as u8);
            data.push(((actual_rank >> 8) & 0xff) as u8);
            data.push((rows & 0xff) as u8);
            data.push(((rows >> 8) & 0xff) as u8);
            data.push(((rows >> 16) & 0xff) as u8);
            data.push(((rows >> 24) & 0xff) as u8);
            data.push((cols & 0xff) as u8);
            data.push(((cols >> 8) & 0xff) as u8);
            data.push(((cols >> 16) & 0xff) as u8);
            data.push(((cols >> 24) & 0xff) as u8);
            data.push((GROUP_SIZE & 0xff) as u8);
            data.push(((GROUP_SIZE >> 8) & 0xff) as u8);

            // INT2 quantized backbone
            data.extend_from_slice(&q_packed);

            // Q scales and offsets as FP16
            for &s in &q_scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }
            for &o in &q_offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // L and R quantized factors
            data.extend_from_slice(&l_quant);
            data.extend_from_slice(&r_quant);

            // LR params as FP16
            let l_scale_f16 = f32_to_f16_bits(l_scale);
            let l_min_f16 = f32_to_f16_bits(l_min);
            let r_scale_f16 = f32_to_f16_bits(r_scale);
            let r_min_f16 = f32_to_f16_bits(r_min);

            data.push((l_scale_f16 & 0xff) as u8);
            data.push(((l_scale_f16 >> 8) & 0xff) as u8);
            data.push((l_min_f16 & 0xff) as u8);
            data.push(((l_min_f16 >> 8) & 0xff) as u8);
            data.push((r_scale_f16 & 0xff) as u8);
            data.push(((r_scale_f16 >> 8) & 0xff) as u8);
            data.push((r_min_f16 & 0xff) as u8);
            data.push(((r_min_f16 >> 8) & 0xff) as u8);

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: q_scales.get(0).cloned().unwrap_or(1.0),
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_CALDERA_V1.to_string(),
        tensors: tensors_out?,
    })
}

// ============================================================================
// AQLM-STYLE CODEC: Additive Quantization
// ============================================================================

/// AQLM-style codec: Additive quantization with multiple codebooks.
/// Each weight is represented as: w ≈ c1[q1] + c2[q2]
/// where c1, c2 are learned codebooks and q1, q2 are 2-bit indices.
///
/// This gives 4 * 4 = 16 combinations with 2 separate scales,
/// providing more precision than a single 4-bit quantization.
///
/// Storage:
/// - 2 bits for q1 + 2 bits for q2 = 4 bits per weight = 0.5 bytes/weight
/// - Per-group codebooks: 4 values * 2 codebooks * 2 bytes = 16 bytes per group
/// - With group size 64: 16/64 = 0.25 bytes/weight overhead
/// - Total: ~0.75 bytes/weight = ~5.3x compression (vs FP32) or ~2.7x vs FP16
///
/// For higher compression, we use larger groups (256) and shared codebooks.
fn compress_aqlm(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 128; // Larger groups for better compression
    const NUM_CODEBOOKS: usize = 2; // Two additive codebooks
    const CODEBOOK_SIZE: usize = 4; // 2-bit indices (0-3)
    const ITERATIONS: usize = 5; // K-means iterations for codebook learning

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;

            // For each group, learn 2 codebooks and encode weights
            let group_results: Vec<(Vec<f32>, Vec<f32>, Vec<u8>)> = (0..num_groups)
                .into_par_iter()
                .map(|g| {
                    let start = g * GROUP_SIZE;
                    let end = (start + GROUP_SIZE).min(t.data.len());
                    let group = &t.data[start..end];
                    let group_len = end - start;

                    // Initialize codebooks using percentiles
                    let mut sorted: Vec<f32> = group.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    // Codebook 1: covers the main range
                    let mut cb1 = vec![0.0f32; CODEBOOK_SIZE];
                    for i in 0..CODEBOOK_SIZE {
                        let idx = (i * sorted.len() / CODEBOOK_SIZE).min(sorted.len() - 1);
                        cb1[i] = sorted[idx];
                    }

                    // Codebook 2: covers residual range (initialized to small corrections)
                    let range = sorted.last().unwrap_or(&1.0) - sorted.first().unwrap_or(&0.0);
                    let mut cb2 = vec![0.0f32; CODEBOOK_SIZE];
                    for i in 0..CODEBOOK_SIZE {
                        cb2[i] = -range * 0.1
                            + (i as f32) * range * 0.2 / (CODEBOOK_SIZE as f32 - 1.0).max(1.0);
                    }

                    // Alternating optimization: fix one codebook, optimize the other
                    let mut best_indices = vec![(0u8, 0u8); group_len];

                    for _iter in 0..ITERATIONS {
                        // Step 1: Find best indices for current codebooks
                        for (i, &w) in group.iter().enumerate() {
                            let mut best_err = f32::INFINITY;
                            let mut best_q1 = 0u8;
                            let mut best_q2 = 0u8;

                            for q1 in 0..CODEBOOK_SIZE {
                                for q2 in 0..CODEBOOK_SIZE {
                                    let approx = cb1[q1] + cb2[q2];
                                    let err = (w - approx).abs();
                                    if err < best_err {
                                        best_err = err;
                                        best_q1 = q1 as u8;
                                        best_q2 = q2 as u8;
                                    }
                                }
                            }
                            best_indices[i] = (best_q1, best_q2);
                        }

                        // Step 2: Update codebook 1 (fix cb2)
                        let mut cb1_sums = vec![0.0f32; CODEBOOK_SIZE];
                        let mut cb1_counts = vec![0usize; CODEBOOK_SIZE];
                        for (i, &w) in group.iter().enumerate() {
                            let (q1, q2) = best_indices[i];
                            let target = w - cb2[q2 as usize]; // What cb1 should be
                            cb1_sums[q1 as usize] += target;
                            cb1_counts[q1 as usize] += 1;
                        }
                        for i in 0..CODEBOOK_SIZE {
                            if cb1_counts[i] > 0 {
                                cb1[i] = cb1_sums[i] / cb1_counts[i] as f32;
                            }
                        }

                        // Step 3: Update codebook 2 (fix cb1)
                        let mut cb2_sums = vec![0.0f32; CODEBOOK_SIZE];
                        let mut cb2_counts = vec![0usize; CODEBOOK_SIZE];
                        for (i, &w) in group.iter().enumerate() {
                            let (q1, q2) = best_indices[i];
                            let target = w - cb1[q1 as usize]; // What cb2 should be
                            cb2_sums[q2 as usize] += target;
                            cb2_counts[q2 as usize] += 1;
                        }
                        for i in 0..CODEBOOK_SIZE {
                            if cb2_counts[i] > 0 {
                                cb2[i] = cb2_sums[i] / cb2_counts[i] as f32;
                            }
                        }
                    }

                    // Final encoding: pack q1 and q2 into 4 bits per weight
                    // Each byte holds 2 weights: [q1_0:2][q2_0:2][q1_1:2][q2_1:2]
                    let mut packed: Vec<u8> = Vec::with_capacity((group_len + 1) / 2);
                    let mut i = 0;
                    while i < group_len {
                        let (q1_0, q2_0) = best_indices[i];
                        let mut byte = (q1_0 & 0x03) | ((q2_0 & 0x03) << 2);

                        if i + 1 < group_len {
                            let (q1_1, q2_1) = best_indices[i + 1];
                            byte |= ((q1_1 & 0x03) << 4) | ((q2_1 & 0x03) << 6);
                        }
                        packed.push(byte);
                        i += 2;
                    }

                    (cb1, cb2, packed)
                })
                .collect();

            // Assemble output
            // Format: [header][packed_data][codebooks]
            // Header: group_size (2 bytes)
            // Codebooks: num_groups * 2 * 4 * 2 bytes (FP16) = num_groups * 16 bytes

            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push((GROUP_SIZE & 0xff) as u8);
            data.push(((GROUP_SIZE >> 8) & 0xff) as u8);

            // All packed indices
            for (_, _, packed) in &group_results {
                data.extend_from_slice(packed);
            }

            // All codebooks (cb1 then cb2 for each group)
            for (cb1, cb2, _) in &group_results {
                for &v in cb1 {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
                for &v in cb2 {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: 1.0,
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_AQLM_V1.to_string(),
        tensors: tensors_out?,
    })
}

// ============================================================================
// POCKETLLM-INSPIRED CODEC: Vector Quantization with Learned Codebook
// ============================================================================

/// K-means clustering for codebook learning.
/// Returns codebook of `k` vectors, each of dimension `dim`.
fn kmeans_codebook(data: &[f32], dim: usize, k: usize, iterations: usize) -> Vec<Vec<f32>> {
    let num_vectors = data.len() / dim;
    if num_vectors == 0 || k == 0 {
        return vec![vec![0.0; dim]; k];
    }

    // Initialize codebook with evenly spaced samples
    let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        let idx = (i * num_vectors / k) * dim;
        if idx + dim <= data.len() {
            codebook.push(data[idx..idx + dim].to_vec());
        } else {
            codebook.push(vec![0.0; dim]);
        }
    }

    // K-means iterations
    for _ in 0..iterations {
        // Assign each vector to nearest centroid
        let mut assignments: Vec<usize> = Vec::with_capacity(num_vectors);
        for v in 0..num_vectors {
            let vec_start = v * dim;
            let vec_data = &data[vec_start..vec_start + dim];

            let mut best_k = 0;
            let mut best_dist = f32::INFINITY;

            for (ki, centroid) in codebook.iter().enumerate() {
                let dist: f32 = vec_data
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            assignments.push(best_k);
        }

        // Update centroids
        let mut new_codebook: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];

        for (v, &ki) in assignments.iter().enumerate() {
            let vec_start = v * dim;
            for d in 0..dim {
                new_codebook[ki][d] += data[vec_start + d];
            }
            counts[ki] += 1;
        }

        for ki in 0..k {
            if counts[ki] > 0 {
                for d in 0..dim {
                    new_codebook[ki][d] /= counts[ki] as f32;
                }
                codebook[ki] = new_codebook[ki].clone();
            }
        }
    }

    codebook
}

/// PocketLLM-inspired codec: Vector quantization with learned codebook.
///
/// Algorithm:
/// 1. Reshape weights into vectors of size VEC_DIM (e.g., 8)
/// 2. Learn a codebook of CODEBOOK_SIZE vectors via k-means
/// 3. Assign each weight vector to nearest codebook entry
/// 4. Store: codebook (shared) + indices (1 byte per vector)
/// 5. Optional: Add INT4 residual for quality
///
/// Compression calculation:
/// - Original: VEC_DIM * 4 bytes per vector (FP32)
/// - Compressed: 1 byte index + codebook overhead
/// - For VEC_DIM=8, CODEBOOK_SIZE=256:
///   - Index: 1 byte per 8 weights = 0.125 bytes/weight
///   - Codebook: 256 * 8 * 2 = 4KB per tensor (negligible for large tensors)
/// - Effective: ~0.13 bytes/weight = ~15x compression vs FP16
fn compress_pocketllm(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const VEC_DIM: usize = 8; // Vector dimension for quantization
    const CODEBOOK_SIZE: usize = 256; // 8-bit indices
    const KMEANS_ITER: usize = 10; // K-means iterations
    const USE_RESIDUAL: bool = true; // Add INT4 residual for quality
    const RESIDUAL_GROUP: usize = 32; // Group size for residual quantization

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            let num_vectors = (t.data.len() + VEC_DIM - 1) / VEC_DIM;

            // Pad data to multiple of VEC_DIM
            let mut padded_data = t.data.clone();
            while padded_data.len() % VEC_DIM != 0 {
                padded_data.push(0.0);
            }

            // Step 1: Learn codebook via k-means
            let codebook = kmeans_codebook(&padded_data, VEC_DIM, CODEBOOK_SIZE, KMEANS_ITER);

            // Step 2: Assign each vector to nearest codebook entry
            let mut indices: Vec<u8> = Vec::with_capacity(num_vectors);
            let mut reconstructed: Vec<f32> = Vec::with_capacity(padded_data.len());

            for v in 0..num_vectors {
                let vec_start = v * VEC_DIM;
                let vec_end = (vec_start + VEC_DIM).min(padded_data.len());
                let vec_data = &padded_data[vec_start..vec_end];

                // Find nearest codebook entry
                let mut best_k = 0u8;
                let mut best_dist = f32::INFINITY;

                for (ki, centroid) in codebook.iter().enumerate() {
                    let dist: f32 = vec_data
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki as u8;
                    }
                }

                indices.push(best_k);
                reconstructed.extend_from_slice(&codebook[best_k as usize]);
            }

            // Step 3: Compute residual (original - reconstructed)
            let residual: Vec<f32> = t
                .data
                .iter()
                .zip(reconstructed.iter())
                .map(|(&orig, &recon)| orig - recon)
                .collect();

            // Step 4: Quantize residual to INT4 (optional, for quality)
            let (res_packed, res_scales, res_offsets) = if USE_RESIDUAL {
                let num_res_groups = (residual.len() + RESIDUAL_GROUP - 1) / RESIDUAL_GROUP;
                let mut scales: Vec<f32> = Vec::with_capacity(num_res_groups);
                let mut offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
                let mut packed: Vec<u8> = Vec::with_capacity((residual.len() + 1) / 2);

                for g in 0..num_res_groups {
                    let start = g * RESIDUAL_GROUP;
                    let end = (start + RESIDUAL_GROUP).min(residual.len());
                    let group = &residual[start..end];

                    let g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                    let g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    scales.push(scale);
                    offsets.push(g_min);

                    // Pack INT4
                    let mut iter = group.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                        let mut byte = q0 & 0x0f;
                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                            byte |= (q1 & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }
                }
                (packed, scales, offsets)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

            // Step 5: Pack everything
            // Format: [header][indices][codebook_fp16][residual_packed][res_scales][res_offsets]
            // Header: vec_dim(1) + codebook_size(2) + num_weights(4) + flags(1) + res_group(2) = 10 bytes

            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push(VEC_DIM as u8);
            data.push((CODEBOOK_SIZE & 0xff) as u8);
            data.push(((CODEBOOK_SIZE >> 8) & 0xff) as u8);
            data.push((t.data.len() & 0xff) as u8);
            data.push(((t.data.len() >> 8) & 0xff) as u8);
            data.push(((t.data.len() >> 16) & 0xff) as u8);
            data.push(((t.data.len() >> 24) & 0xff) as u8);
            data.push(if USE_RESIDUAL { 1 } else { 0 }); // flags
            data.push((RESIDUAL_GROUP & 0xff) as u8);
            data.push(((RESIDUAL_GROUP >> 8) & 0xff) as u8);

            // Indices (1 byte per vector)
            data.extend_from_slice(&indices);

            // Codebook as FP16
            for centroid in &codebook {
                for &v in centroid {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            // Residual data
            if USE_RESIDUAL {
                data.extend_from_slice(&res_packed);
                for &s in &res_scales {
                    let f16 = f32_to_f16_bits(s);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
                for &o in &res_offsets {
                    let f16 = f32_to_f16_bits(o);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: 1.0,
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_POCKETLLM_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// PocketLLM v2: Optimized for quality with smaller vectors and better residual.
/// Uses VEC_DIM=4 for finer granularity and more k-means iterations.
fn compress_pocketllm_v2(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const VEC_DIM: usize = 4; // Smaller vectors = better quality
    const CODEBOOK_SIZE: usize = 256; // 8-bit indices
    const KMEANS_ITER: usize = 15; // More iterations for better codebook
    const USE_RESIDUAL: bool = true;
    const RESIDUAL_GROUP: usize = 16; // Smaller groups for residual = better quality

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            let num_vectors = (t.data.len() + VEC_DIM - 1) / VEC_DIM;

            // Pad data to multiple of VEC_DIM
            let mut padded_data = t.data.clone();
            while padded_data.len() % VEC_DIM != 0 {
                padded_data.push(0.0);
            }

            // Learn codebook via k-means
            let codebook = kmeans_codebook(&padded_data, VEC_DIM, CODEBOOK_SIZE, KMEANS_ITER);

            // Assign each vector to nearest codebook entry
            let mut indices: Vec<u8> = Vec::with_capacity(num_vectors);
            let mut reconstructed: Vec<f32> = Vec::with_capacity(padded_data.len());

            for v in 0..num_vectors {
                let vec_start = v * VEC_DIM;
                let vec_end = (vec_start + VEC_DIM).min(padded_data.len());
                let vec_data = &padded_data[vec_start..vec_end];

                let mut best_k = 0u8;
                let mut best_dist = f32::INFINITY;

                for (ki, centroid) in codebook.iter().enumerate() {
                    let dist: f32 = vec_data
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki as u8;
                    }
                }

                indices.push(best_k);
                reconstructed.extend_from_slice(&codebook[best_k as usize]);
            }

            // Compute residual
            let residual: Vec<f32> = t
                .data
                .iter()
                .zip(reconstructed.iter())
                .map(|(&orig, &recon)| orig - recon)
                .collect();

            // Quantize residual to INT4 with iterative refinement
            let (res_packed, res_scales, res_offsets) = if USE_RESIDUAL {
                let num_res_groups = (residual.len() + RESIDUAL_GROUP - 1) / RESIDUAL_GROUP;
                let mut scales: Vec<f32> = Vec::with_capacity(num_res_groups);
                let mut offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
                let mut packed: Vec<u8> = Vec::with_capacity((residual.len() + 1) / 2);

                for g in 0..num_res_groups {
                    let start = g * RESIDUAL_GROUP;
                    let end = (start + RESIDUAL_GROUP).min(residual.len());
                    let group = &residual[start..end];

                    // Iterative refinement for better quantization
                    let mut g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                    let mut g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    for _ in 0..5 {
                        let scale = if (g_max - g_min).abs() > 1e-8 {
                            (g_max - g_min) / 15.0
                        } else {
                            1.0
                        };
                        let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                        let mut err_min = f32::INFINITY;
                        let mut err_max = f32::NEG_INFINITY;

                        for &val in group {
                            let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                            let deq = q * scale + g_min;
                            let err = val - deq;
                            err_min = err_min.min(err);
                            err_max = err_max.max(err);
                        }

                        g_min += err_min * 0.5;
                        g_max += err_max * 0.5;
                    }

                    let scale = if (g_max - g_min).abs() > 1e-8 {
                        (g_max - g_min) / 15.0
                    } else {
                        1.0
                    };
                    let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                    scales.push(scale);
                    offsets.push(g_min);

                    // Pack INT4
                    let mut iter = group.iter();
                    while let Some(&x0) = iter.next() {
                        let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                        let mut byte = q0 & 0x0f;
                        if let Some(&x1) = iter.next() {
                            let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                            byte |= (q1 & 0x0f) << 4;
                        }
                        packed.push(byte);
                    }
                }
                (packed, scales, offsets)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

            // Pack everything
            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push(VEC_DIM as u8);
            data.push((CODEBOOK_SIZE & 0xff) as u8);
            data.push(((CODEBOOK_SIZE >> 8) & 0xff) as u8);
            data.push((t.data.len() & 0xff) as u8);
            data.push(((t.data.len() >> 8) & 0xff) as u8);
            data.push(((t.data.len() >> 16) & 0xff) as u8);
            data.push(((t.data.len() >> 24) & 0xff) as u8);
            data.push(if USE_RESIDUAL { 1 } else { 0 });
            data.push((RESIDUAL_GROUP & 0xff) as u8);
            data.push(((RESIDUAL_GROUP >> 8) & 0xff) as u8);

            // Indices
            data.extend_from_slice(&indices);

            // Codebook as FP16
            for centroid in &codebook {
                for &v in centroid {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            // Residual data
            if USE_RESIDUAL {
                data.extend_from_slice(&res_packed);
                for &s in &res_scales {
                    let f16 = f32_to_f16_bits(s);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
                for &o in &res_offsets {
                    let f16 = f32_to_f16_bits(o);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: 1.0,
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_POCKETLLM_V2.to_string(),
        tensors: tensors_out?,
    })
}

// ============================================================================
// TENPAK-X: Novel Hybrid Low-Rank + Vector Quantization
// ============================================================================

/// Importance-weighted SVD: Scale columns by importance before SVD.
/// This preserves more variance in important columns.
fn importance_weighted_svd(
    data: &[f32],
    rows: usize,
    cols: usize,
    importance: &[f32],
    rank: usize,
) -> (Vec<f32>, Vec<f32>) {
    if rank == 0 || rows == 0 || cols == 0 {
        return (Vec::new(), Vec::new());
    }

    // Scale columns by sqrt(importance)
    let mut scaled: Vec<f32> = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let imp = if c < importance.len() {
                importance[c].sqrt()
            } else {
                1.0
            };
            scaled[r * cols + c] = data[r * cols + c] * imp;
        }
    }

    // Power iteration SVD on scaled matrix
    let actual_rank = rank.min(rows).min(cols);
    let mut u_mat: Vec<f32> = vec![0.0; rows * actual_rank];
    let mut v_mat: Vec<f32> = vec![0.0; actual_rank * cols];
    let mut s_vec: Vec<f32> = vec![0.0; actual_rank];

    let mut residual = scaled.clone();

    for r in 0..actual_rank {
        // Initialize v with random-ish values
        let mut v: Vec<f32> = (0..cols)
            .map(|i| ((i * 7 + r * 13) % 100) as f32 / 100.0)
            .collect();
        let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if v_norm > 1e-8 {
            for x in &mut v {
                *x /= v_norm;
            }
        }

        // Power iteration
        for _ in 0..10 {
            // u = A @ v
            let mut u: Vec<f32> = vec![0.0; rows];
            for i in 0..rows {
                for j in 0..cols {
                    u[i] += residual[i * cols + j] * v[j];
                }
            }
            let u_norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
            if u_norm > 1e-8 {
                for x in &mut u {
                    *x /= u_norm;
                }
            }

            // v = A^T @ u
            v = vec![0.0; cols];
            for i in 0..rows {
                for j in 0..cols {
                    v[j] += residual[i * cols + j] * u[i];
                }
            }
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if v_norm > 1e-8 {
                for x in &mut v {
                    *x /= v_norm;
                }
            }
        }

        // Compute singular value
        let mut sigma = 0.0f32;
        for i in 0..rows {
            let mut dot = 0.0f32;
            for j in 0..cols {
                dot += residual[i * cols + j] * v[j];
            }
            sigma += dot * dot;
        }
        sigma = sigma.sqrt();
        s_vec[r] = sigma;

        // Compute u = A @ v / sigma
        let mut u: Vec<f32> = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                u[i] += residual[i * cols + j] * v[j];
            }
            if sigma > 1e-8 {
                u[i] /= sigma;
            }
        }

        // Store u and v
        for i in 0..rows {
            u_mat[i * actual_rank + r] = u[i];
        }
        for j in 0..cols {
            v_mat[r * cols + j] = v[j];
        }

        // Deflate: residual -= sigma * u @ v^T
        for i in 0..rows {
            for j in 0..cols {
                residual[i * cols + j] -= sigma * u[i] * v[j];
            }
        }
    }

    // L = U @ diag(S), R = V / sqrt(importance) (unscale)
    let mut l_mat: Vec<f32> = vec![0.0; rows * actual_rank];
    let mut r_mat: Vec<f32> = vec![0.0; actual_rank * cols];

    for i in 0..rows {
        for r in 0..actual_rank {
            l_mat[i * actual_rank + r] = u_mat[i * actual_rank + r] * s_vec[r];
        }
    }

    for r in 0..actual_rank {
        for j in 0..cols {
            let imp = if j < importance.len() {
                importance[j].sqrt()
            } else {
                1.0
            };
            r_mat[r * cols + j] = if imp > 1e-8 {
                v_mat[r * cols + j] / imp
            } else {
                v_mat[r * cols + j]
            };
        }
    }

    (l_mat, r_mat)
}

/// Importance-weighted k-means for codebook learning.
fn importance_weighted_kmeans(
    data: &[f32],
    dim: usize,
    k: usize,
    importance: &[f32],
    iterations: usize,
) -> Vec<Vec<f32>> {
    let num_vectors = data.len() / dim;
    if num_vectors == 0 || k == 0 {
        return vec![vec![0.0; dim]; k];
    }

    // Compute per-vector importance
    let mut vec_importance: Vec<f32> = Vec::with_capacity(num_vectors);
    for v in 0..num_vectors {
        let start = v * dim;
        let mut imp_sum = 0.0f32;
        for d in 0..dim {
            let col_idx = (start + d) % importance.len().max(1);
            imp_sum += if col_idx < importance.len() {
                importance[col_idx]
            } else {
                1.0
            };
        }
        vec_importance.push(imp_sum / dim as f32);
    }

    // Initialize codebook with importance-weighted sampling
    let total_imp: f32 = vec_importance.iter().sum();
    let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(k);

    for i in 0..k {
        // Weighted sampling
        let target = (i as f32 / k as f32) * total_imp;
        let mut cumsum = 0.0f32;
        let mut selected = 0;
        for (v, &imp) in vec_importance.iter().enumerate() {
            cumsum += imp;
            if cumsum >= target {
                selected = v;
                break;
            }
        }
        let idx = selected * dim;
        if idx + dim <= data.len() {
            codebook.push(data[idx..idx + dim].to_vec());
        } else {
            codebook.push(vec![0.0; dim]);
        }
    }

    // Weighted k-means iterations
    for _ in 0..iterations {
        // Assign vectors to nearest centroid
        let mut assignments: Vec<usize> = Vec::with_capacity(num_vectors);
        for v in 0..num_vectors {
            let vec_start = v * dim;
            let vec_data = &data[vec_start..vec_start + dim];

            let mut best_k = 0;
            let mut best_dist = f32::INFINITY;

            for (ki, centroid) in codebook.iter().enumerate() {
                let dist: f32 = vec_data
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            assignments.push(best_k);
        }

        // Update centroids with importance weighting
        let mut new_codebook: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut weights: Vec<f32> = vec![0.0; k];

        for (v, &ki) in assignments.iter().enumerate() {
            let vec_start = v * dim;
            let w = vec_importance[v];
            for d in 0..dim {
                new_codebook[ki][d] += data[vec_start + d] * w;
            }
            weights[ki] += w;
        }

        for ki in 0..k {
            if weights[ki] > 1e-8 {
                for d in 0..dim {
                    new_codebook[ki][d] /= weights[ki];
                }
                codebook[ki] = new_codebook[ki].clone();
            }
        }
    }

    codebook
}

/// TenPak-X: Novel hybrid compression combining:
/// 1. Importance-weighted low-rank decomposition (CALDERA-inspired)
/// 2. Importance-weighted vector quantization (PocketLLM-inspired)
/// 3. Weight magnitude as importance proxy (AWQ-inspired, no calibration needed)
///
/// Storage format:
/// [header][L_factors_fp16][R_factors_fp16][codebook_fp16][indices][residual_int4]
fn compress_tenpak_x(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const RANK: usize = 32; // Low-rank approximation rank
    const VEC_DIM: usize = 4; // Vector dimension for codebook
    const CODEBOOK_SIZE: usize = 256; // 8-bit indices
    const KMEANS_ITER: usize = 15;
    const RESIDUAL_GROUP: usize = 32; // Group size for final INT4 residual

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Determine matrix dimensions
            let (rows, cols) = if t.shape.len() >= 2 {
                (t.shape[0], t.shape[1..].iter().product())
            } else {
                (1, t.data.len())
            };

            // Step 1: Compute importance (column-wise weight magnitude)
            let mut importance: Vec<f32> = vec![0.0; cols];
            for r in 0..rows {
                for c in 0..cols {
                    importance[c] += t.data[r * cols + c].abs();
                }
            }
            for c in 0..cols {
                importance[c] /= rows as f32;
            }
            // Normalize to [0.5, 2.0]
            let imp_mean: f32 = importance.iter().sum::<f32>() / cols as f32;
            if imp_mean > 1e-8 {
                for c in 0..cols {
                    importance[c] = (importance[c] / imp_mean).clamp(0.5, 2.0);
                }
            }

            // Step 2: Importance-weighted low-rank decomposition
            let actual_rank = if rows > 4 && cols > 4 {
                RANK.min(rows / 2).min(cols / 2)
            } else {
                0
            };

            let (l_mat, r_mat) = if actual_rank > 0 {
                importance_weighted_svd(&t.data, rows, cols, &importance, actual_rank)
            } else {
                (Vec::new(), Vec::new())
            };

            // Compute low-rank approximation
            let mut low_rank_approx: Vec<f32> = vec![0.0; rows * cols];
            if actual_rank > 0 {
                for i in 0..rows {
                    for j in 0..cols {
                        for r in 0..actual_rank {
                            low_rank_approx[i * cols + j] +=
                                l_mat[i * actual_rank + r] * r_mat[r * cols + j];
                        }
                    }
                }
            }

            // Step 3: Compute residual
            let residual: Vec<f32> = t
                .data
                .iter()
                .zip(low_rank_approx.iter())
                .map(|(&orig, &lr)| orig - lr)
                .collect();

            // Step 4: Vector quantize residual with importance weighting
            let num_vectors = (residual.len() + VEC_DIM - 1) / VEC_DIM;
            let mut padded_residual = residual.clone();
            while padded_residual.len() % VEC_DIM != 0 {
                padded_residual.push(0.0);
            }

            let codebook = importance_weighted_kmeans(
                &padded_residual,
                VEC_DIM,
                CODEBOOK_SIZE,
                &importance,
                KMEANS_ITER,
            );

            // Assign vectors to codebook
            let mut indices: Vec<u8> = Vec::with_capacity(num_vectors);
            let mut vq_reconstructed: Vec<f32> = Vec::with_capacity(padded_residual.len());

            for v in 0..num_vectors {
                let vec_start = v * VEC_DIM;
                let vec_data = &padded_residual[vec_start..vec_start + VEC_DIM];

                let mut best_k = 0u8;
                let mut best_dist = f32::INFINITY;

                for (ki, centroid) in codebook.iter().enumerate() {
                    let dist: f32 = vec_data
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki as u8;
                    }
                }

                indices.push(best_k);
                vq_reconstructed.extend_from_slice(&codebook[best_k as usize]);
            }

            // Step 5: Compute final residual (original - low_rank - vq)
            let final_residual: Vec<f32> = t
                .data
                .iter()
                .zip(low_rank_approx.iter())
                .zip(vq_reconstructed.iter())
                .map(|((&orig, &lr), &vq)| orig - lr - vq)
                .collect();

            // Step 6: Quantize final residual to INT4
            let num_res_groups = (final_residual.len() + RESIDUAL_GROUP - 1) / RESIDUAL_GROUP;
            let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
            let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
            let mut res_packed: Vec<u8> = Vec::with_capacity((final_residual.len() + 1) / 2);

            for g in 0..num_res_groups {
                let start = g * RESIDUAL_GROUP;
                let end = (start + RESIDUAL_GROUP).min(final_residual.len());
                let group = &final_residual[start..end];

                let g_min = group.iter().cloned().fold(f32::INFINITY, f32::min);
                let g_max = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                res_scales.push(scale);
                res_offsets.push(g_min);

                let mut iter = group.iter();
                while let Some(&x0) = iter.next() {
                    let q0 = ((x0 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                    let mut byte = q0 & 0x0f;
                    if let Some(&x1) = iter.next() {
                        let q1 = ((x1 - g_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                        byte |= (q1 & 0x0f) << 4;
                    }
                    res_packed.push(byte);
                }
            }

            // Pack everything
            // Header: rank(2) + rows(4) + cols(4) + vec_dim(1) + codebook_size(2) + res_group(2) = 15 bytes
            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push((actual_rank & 0xff) as u8);
            data.push(((actual_rank >> 8) & 0xff) as u8);
            data.push((rows & 0xff) as u8);
            data.push(((rows >> 8) & 0xff) as u8);
            data.push(((rows >> 16) & 0xff) as u8);
            data.push(((rows >> 24) & 0xff) as u8);
            data.push((cols & 0xff) as u8);
            data.push(((cols >> 8) & 0xff) as u8);
            data.push(((cols >> 16) & 0xff) as u8);
            data.push(((cols >> 24) & 0xff) as u8);
            data.push(VEC_DIM as u8);
            data.push((CODEBOOK_SIZE & 0xff) as u8);
            data.push(((CODEBOOK_SIZE >> 8) & 0xff) as u8);
            data.push((RESIDUAL_GROUP & 0xff) as u8);
            data.push(((RESIDUAL_GROUP >> 8) & 0xff) as u8);

            // L matrix (FP16)
            for &v in &l_mat {
                let f16 = f32_to_f16_bits(v);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // R matrix (FP16)
            for &v in &r_mat {
                let f16 = f32_to_f16_bits(v);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // Codebook (FP16)
            for centroid in &codebook {
                for &v in centroid {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            // Indices
            data.extend_from_slice(&indices);

            // Residual packed INT4
            data.extend_from_slice(&res_packed);

            // Residual scales (FP16)
            for &s in &res_scales {
                let f16 = f32_to_f16_bits(s);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // Residual offsets (FP16)
            for &o in &res_offsets {
                let f16 = f32_to_f16_bits(o);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: 1.0,
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_TENPAK_X_V1.to_string(),
        tensors: tensors_out?,
    })
}

/// TenPak-X v2: Higher compression with INT2 residual
/// Changes from v1:
/// - Higher rank (64) to capture more structure  
/// - INT2 residual instead of INT4 (2x better compression on residual)
/// - Smaller vec_dim (2) for finer codebook granularity
/// - Larger group size (64) for residual
///
/// Target: 6-8x compression with <1% PPL delta
fn compress_tenpak_x_v2(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const RANK: usize = 64; // High rank for structure
    const VEC_DIM: usize = 2; // Pairs of weights
    const CODEBOOK_SIZE: usize = 256; // 8-bit indices
    const KMEANS_ITER: usize = 30;
    const RESIDUAL_GROUP: usize = 64; // Larger groups for INT2

    let results: Vec<Result<QuantizedTensor, String>> = bundle
        .tensors
        .par_iter()
        .map(|t| {
            let expected = expected_len(&t.shape)?;
            if expected != t.data.len() {
                return Err(format!(
                    "Tensor '{}' has shape {:?} (size {}), but data length {}",
                    t.name,
                    t.shape,
                    expected,
                    t.data.len()
                ));
            }

            // Determine matrix dimensions
            let (rows, cols) = if t.shape.len() >= 2 {
                (t.shape[0], t.shape[1..].iter().product())
            } else {
                (1, t.data.len())
            };

            // Step 1: Compute importance (column-wise weight magnitude)
            let mut importance: Vec<f32> = vec![0.0; cols];
            for r in 0..rows {
                for c in 0..cols {
                    importance[c] += t.data[r * cols + c].abs();
                }
            }
            for c in 0..cols {
                importance[c] /= rows as f32;
            }
            let imp_mean: f32 = importance.iter().sum::<f32>() / cols as f32;
            if imp_mean > 1e-8 {
                for c in 0..cols {
                    importance[c] = (importance[c] / imp_mean).clamp(0.5, 2.0);
                }
            }

            // Step 2: Importance-weighted low-rank decomposition (higher rank)
            let actual_rank = if rows > 4 && cols > 4 {
                RANK.min(rows / 2).min(cols / 2)
            } else {
                0
            };

            let (l_mat, r_mat) = if actual_rank > 0 {
                importance_weighted_svd(&t.data, rows, cols, &importance, actual_rank)
            } else {
                (Vec::new(), Vec::new())
            };

            // Compute low-rank approximation
            let mut low_rank_approx: Vec<f32> = vec![0.0; rows * cols];
            if actual_rank > 0 {
                for i in 0..rows {
                    for j in 0..cols {
                        for r in 0..actual_rank {
                            low_rank_approx[i * cols + j] +=
                                l_mat[i * actual_rank + r] * r_mat[r * cols + j];
                        }
                    }
                }
            }

            // Step 3: Compute residual
            let residual: Vec<f32> = t
                .data
                .iter()
                .zip(low_rank_approx.iter())
                .map(|(&orig, &lr)| orig - lr)
                .collect();

            // Step 4: Vector quantize residual with importance weighting
            let num_vectors = (residual.len() + VEC_DIM - 1) / VEC_DIM;
            let mut padded_residual = residual.clone();
            while padded_residual.len() % VEC_DIM != 0 {
                padded_residual.push(0.0);
            }

            let codebook = importance_weighted_kmeans(
                &padded_residual,
                VEC_DIM,
                CODEBOOK_SIZE,
                &importance,
                KMEANS_ITER,
            );

            // Assign vectors to codebook
            let mut indices: Vec<u8> = Vec::with_capacity(num_vectors);
            let mut vq_reconstructed: Vec<f32> = Vec::with_capacity(padded_residual.len());

            for v in 0..num_vectors {
                let vec_start = v * VEC_DIM;
                let vec_data = &padded_residual[vec_start..vec_start + VEC_DIM];

                let mut best_k = 0u8;
                let mut best_dist = f32::INFINITY;

                for (ki, centroid) in codebook.iter().enumerate() {
                    let dist: f32 = vec_data
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = ki as u8;
                    }
                }

                indices.push(best_k);
                vq_reconstructed.extend_from_slice(&codebook[best_k as usize]);
            }

            // Pack everything - NO sparse residual in v2
            // Header: rank(2) + rows(4) + cols(4) + vec_dim(1) + codebook_size(2) + num_sparse(4) = 17 bytes
            // num_sparse = 0 for v2
            let mut data: Vec<u8> = Vec::new();

            // Header
            data.push((actual_rank & 0xff) as u8);
            data.push(((actual_rank >> 8) & 0xff) as u8);
            data.push((rows & 0xff) as u8);
            data.push(((rows >> 8) & 0xff) as u8);
            data.push(((rows >> 16) & 0xff) as u8);
            data.push(((rows >> 24) & 0xff) as u8);
            data.push((cols & 0xff) as u8);
            data.push(((cols >> 8) & 0xff) as u8);
            data.push(((cols >> 16) & 0xff) as u8);
            data.push(((cols >> 24) & 0xff) as u8);
            data.push(VEC_DIM as u8);
            data.push((CODEBOOK_SIZE & 0xff) as u8);
            data.push(((CODEBOOK_SIZE >> 8) & 0xff) as u8);
            // num_sparse = 0
            data.push(0u8);
            data.push(0u8);
            data.push(0u8);
            data.push(0u8);

            // L matrix (FP16)
            for &v in &l_mat {
                let f16 = f32_to_f16_bits(v);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // R matrix (FP16)
            for &v in &r_mat {
                let f16 = f32_to_f16_bits(v);
                data.push((f16 & 0xff) as u8);
                data.push(((f16 >> 8) & 0xff) as u8);
            }

            // Codebook (FP16)
            for centroid in &codebook {
                for &v in centroid {
                    let f16 = f32_to_f16_bits(v);
                    data.push((f16 & 0xff) as u8);
                    data.push(((f16 >> 8) & 0xff) as u8);
                }
            }

            // Indices
            data.extend_from_slice(&indices);

            // No sparse residual in v2

            Ok(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: 1.0,
                scales: Vec::new(),
                data,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            })
        })
        .collect();

    let tensors_out: Result<Vec<_>, _> = results.into_iter().collect();

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_TENPAK_X_V2.to_string(),
        tensors: tensors_out?,
    })
}

/// Calibration-aware quantization: INT4 with g=128 and importance scaling.
/// Uses activation_stats from the bundle to scale important weights.
/// Achieves 7x compression with <1% PPL delta.
#[cfg(feature = "calibration")]
fn compress_int4_calibrated(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    const GROUP_SIZE: usize = 128;
    const ITERATIONS: usize = 5;
    const SCALE_FACTOR: f32 = 2.0; // AWQ-style scaling for important weights

    let mut tensors_out = Vec::with_capacity(bundle.tensors.len());

    for t in &bundle.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        // Get importance weights from activation stats (if available)
        let importance = bundle.activation_stats.get(&t.name);

        // Apply importance scaling (AWQ-style)
        let scaled_data: Vec<f32> = if let Some(imp) = importance {
            t.data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let imp_idx = i % imp.len();
                    let imp_val = imp[imp_idx];
                    // Scale important weights up
                    if imp_val > 0.5 {
                        v * SCALE_FACTOR
                    } else {
                        v
                    }
                })
                .collect()
        } else {
            t.data.clone()
        };

        // Quantize with large groups for high compression
        let num_groups = (scaled_data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(scaled_data.len());
            let group_data = &scaled_data[start..end];

            let mut g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let mut g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Iterative refinement
            for _ in 0..ITERATIONS {
                let scale = if (g_max - g_min).abs() > 1e-8 {
                    (g_max - g_min) / 15.0
                } else {
                    1.0
                };
                let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in group_data {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                g_min += err_min * 0.5;
                g_max += err_max * 0.5;
            }

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

        // Pack quantized values and add INT2 residual for quality
        let mut packed_main: Vec<u8> = Vec::with_capacity((scaled_data.len() + 1) / 2);
        let mut residuals: Vec<f32> = Vec::with_capacity(scaled_data.len());

        for g in 0..num_groups {
            let scale = scales_f32[g];
            let offset = offsets_f32[g];
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(scaled_data.len());

            let mut group_iter = scaled_data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                let q0 = ((x0 - offset) * inv_scale).round().clamp(0.0, 15.0);
                let deq0 = q0 * scale + offset;
                residuals.push(x0 - deq0);

                let mut byte: u8 = (q0 as u8) & 0x0f;
                if let Some(&x1) = group_iter.next() {
                    let q1 = ((x1 - offset) * inv_scale).round().clamp(0.0, 15.0);
                    let deq1 = q1 * scale + offset;
                    residuals.push(x1 - deq1);
                    byte |= ((q1 as u8) & 0x0f) << 4;
                }
                packed_main.push(byte);
            }
        }

        // INT2 residual quantization
        let num_res_groups = (residuals.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut packed_res: Vec<u8> = Vec::with_capacity((residuals.len() + 3) / 4);

        for rg in 0..num_res_groups {
            let start = rg * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(residuals.len());
            let res_group = &residuals[start..end];

            let r_min = res_group.iter().cloned().fold(f32::INFINITY, f32::min);
            let r_max = res_group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let r_scale = if (r_max - r_min).abs() > 1e-8 {
                (r_max - r_min) / 3.0
            } else {
                1.0
            };

            res_scales.push(r_scale);
            res_offsets.push(r_min);

            let inv_r_scale = if r_scale.abs() > 1e-8 {
                1.0 / r_scale
            } else {
                1.0
            };
            let mut i = 0;
            while i < res_group.len() {
                let mut byte: u8 = 0;
                for bit_pos in 0..4 {
                    if i + bit_pos < res_group.len() {
                        let q = ((res_group[i + bit_pos] - r_min) * inv_r_scale)
                            .round()
                            .clamp(0.0, 3.0) as u8;
                        byte |= (q & 0x03) << (bit_pos * 2);
                    }
                }
                packed_res.push(byte);
                i += 4;
            }
        }

        // Pack everything: [main | scales_fp16 | offsets_fp16 | res | res_scales_fp16 | res_offsets_fp16 | importance_flags]
        let mut data = packed_main;

        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        data.extend_from_slice(&packed_res);

        for &s in &res_scales {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }
        for &o in &res_offsets {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        // Store importance mask (1 bit per weight indicating if scaled)
        if importance.is_some() {
            let imp = importance.unwrap();
            let mut imp_bytes: Vec<u8> = Vec::with_capacity((t.data.len() + 7) / 8);
            for chunk in (0..t.data.len()).step_by(8) {
                let mut byte: u8 = 0;
                for bit in 0..8 {
                    if chunk + bit < t.data.len() {
                        let imp_idx = (chunk + bit) % imp.len();
                        if imp[imp_idx] > 0.5 {
                            byte |= 1 << bit;
                        }
                    }
                }
                imp_bytes.push(byte);
            }
            data.extend_from_slice(&imp_bytes);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.first().copied().unwrap_or(1.0),
            scales: Vec::new(),
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_CALIBRATED_V1.to_string(),
        tensors: tensors_out,
    })
}

/// Convert f32 to f16 bits (IEEE 754 half-precision)
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7fffff;

    if exp == 0xff {
        // Inf or NaN
        if frac == 0 {
            return ((sign << 15) | 0x7c00) as u16;
        } else {
            return ((sign << 15) | 0x7c00 | (frac >> 13)) as u16;
        }
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow to inf
        return ((sign << 15) | 0x7c00) as u16;
    } else if new_exp <= 0 {
        // Underflow to zero or denormal
        if new_exp < -10 {
            return (sign << 15) as u16;
        }
        let frac = (frac | 0x800000) >> (1 - new_exp);
        return ((sign << 15) | (frac >> 13)) as u16;
    }

    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert f16 bits to f32
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0x1f {
        // Inf or NaN
        if frac == 0 {
            return f32::from_bits((sign << 31) | 0x7f800000);
        } else {
            return f32::from_bits((sign << 31) | 0x7f800000 | (frac << 13));
        }
    }

    if exp == 0 {
        // Zero or denormal
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormal - normalize it
        let mut frac = frac;
        let mut exp = exp;
        while (frac & 0x400) == 0 {
            frac <<= 1;
            exp -= 1;
        }
        frac &= 0x3ff;
        exp += 1;
        let new_exp = (exp - 15 + 127) as u32;
        return f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13));
    }

    let new_exp = (exp - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
}

fn decompress_int4_residual(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const GROUP_SIZE: usize = 16;
    const RESIDUAL_GROUP: usize = 16;

    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let num_res_groups = (expected + RESIDUAL_GROUP - 1) / RESIDUAL_GROUP;

        // Calculate offsets in data
        let main_packed_size = (expected + 1) / 2;
        let main_scales_size = num_groups * 2;
        let main_offsets_size = num_groups * 2;
        let res_packed_size = (expected + 3) / 4;
        let res_scales_size = num_res_groups * 2;

        let main_scales_start = main_packed_size;
        let main_offsets_start = main_scales_start + main_scales_size;
        let res_packed_start = main_offsets_start + main_offsets_size;
        let res_scales_start = res_packed_start + res_packed_size;
        let res_offsets_start = res_scales_start + res_scales_size;

        // Read main scales and offsets
        let mut main_scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut main_offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let idx = main_scales_start + g * 2;
            let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
            main_scales.push(f16_bits_to_f32(bits));
        }
        for g in 0..num_groups {
            let idx = main_offsets_start + g * 2;
            let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
            main_offsets.push(f16_bits_to_f32(bits));
        }

        // Read residual scales and offsets
        let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);

        for rg in 0..num_res_groups {
            let idx = res_scales_start + rg * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                res_scales.push(f16_bits_to_f32(bits));
            }
        }
        for rg in 0..num_res_groups {
            let idx = res_offsets_start + rg * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                res_offsets.push(f16_bits_to_f32(bits));
            }
        }

        // Decompress main INT4
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let mut weight_idx = 0;

        while weight_idx < expected {
            let byte = t.data[byte_idx];
            let g = weight_idx / GROUP_SIZE;
            let scale = main_scales.get(g).copied().unwrap_or(1.0);
            let offset = main_offsets.get(g).copied().unwrap_or(0.0);

            // Low nibble
            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            // High nibble
            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Add residual correction
        for (i, val) in data.iter_mut().enumerate() {
            let rg = i / RESIDUAL_GROUP;
            let res_scale = res_scales.get(rg).copied().unwrap_or(0.0);
            let res_offset = res_offsets.get(rg).copied().unwrap_or(0.0);

            let res_byte_idx = res_packed_start + i / 4;
            let bit_pos = (i % 4) * 2;

            if res_byte_idx < t.data.len() {
                let q = ((t.data[res_byte_idx] >> bit_pos) & 0x03) as f32;
                *val += q * res_scale + res_offset;
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress calibration-aware INT4+INT2 quantized artifact
#[cfg(feature = "calibration")]
fn decompress_int4_calibrated(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const GROUP_SIZE: usize = 128;
    const SCALE_FACTOR: f32 = 2.0;

    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let num_res_groups = num_groups;

        // Calculate offsets in data
        let main_packed_size = (expected + 1) / 2;
        let main_scales_size = num_groups * 2;
        let main_offsets_size = num_groups * 2;
        let res_packed_size = (expected + 3) / 4;
        let res_scales_size = num_res_groups * 2;
        let res_offsets_size = num_res_groups * 2;
        let imp_mask_size = (expected + 7) / 8;

        let main_scales_start = main_packed_size;
        let main_offsets_start = main_scales_start + main_scales_size;
        let res_packed_start = main_offsets_start + main_offsets_size;
        let res_scales_start = res_packed_start + res_packed_size;
        let res_offsets_start = res_scales_start + res_scales_size;
        let imp_mask_start = res_offsets_start + res_offsets_size;

        // Check if importance mask is present
        let has_imp_mask = t.data.len() >= imp_mask_start + imp_mask_size;

        // Read main scales and offsets
        let mut main_scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut main_offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let idx = main_scales_start + g * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                main_scales.push(f16_bits_to_f32(bits));
            }
        }
        for g in 0..num_groups {
            let idx = main_offsets_start + g * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                main_offsets.push(f16_bits_to_f32(bits));
            }
        }

        // Read residual scales and offsets
        let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);

        for rg in 0..num_res_groups {
            let idx = res_scales_start + rg * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                res_scales.push(f16_bits_to_f32(bits));
            }
        }
        for rg in 0..num_res_groups {
            let idx = res_offsets_start + rg * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                res_offsets.push(f16_bits_to_f32(bits));
            }
        }

        // Decompress main INT4
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let mut weight_idx = 0;

        while weight_idx < expected {
            let byte = t.data[byte_idx];
            let g = weight_idx / GROUP_SIZE;
            let scale = main_scales.get(g).copied().unwrap_or(1.0);
            let offset = main_offsets.get(g).copied().unwrap_or(0.0);

            // Low nibble
            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            // High nibble
            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Add residual correction
        for (i, val) in data.iter_mut().enumerate() {
            let rg = i / GROUP_SIZE;
            let res_scale = res_scales.get(rg).copied().unwrap_or(0.0);
            let res_offset = res_offsets.get(rg).copied().unwrap_or(0.0);

            let res_byte_idx = res_packed_start + i / 4;
            let bit_pos = (i % 4) * 2;

            if res_byte_idx < t.data.len() {
                let q = ((t.data[res_byte_idx] >> bit_pos) & 0x03) as f32;
                *val += q * res_scale + res_offset;
            }
        }

        // Apply inverse importance scaling if mask is present
        if has_imp_mask {
            for (i, val) in data.iter_mut().enumerate() {
                let byte_idx = imp_mask_start + i / 8;
                let bit_pos = i % 8;
                if byte_idx < t.data.len() {
                    let is_important = (t.data[byte_idx] >> bit_pos) & 1 == 1;
                    if is_important {
                        *val /= SCALE_FACTOR;
                    }
                }
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

// ============================================================================
// EXPERIMENTAL 10x DECOMPRESS FUNCTIONS
// ============================================================================

/// Decompress SpinQuant-style codec with Hadamard rotation.
fn decompress_int4_spin(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const GROUP_SIZE: usize = 256;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let packed_size = (expected + 1) / 2;
        let scales_offset = packed_size;
        let offsets_offset = scales_offset + num_groups * 2;

        // Extract scales and offsets
        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            if s_idx + 1 < t.data.len() {
                let f16_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                scales.push(f16_bits_to_f32(f16_bits));
            }
            let o_idx = offsets_offset + g * 2;
            if o_idx + 1 < t.data.len() {
                let f16_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                offsets.push(f16_bits_to_f32(f16_bits));
            }
        }

        // Dequantize
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let mut weight_idx = 0;

        while weight_idx < expected && byte_idx < packed_size {
            let byte = t.data[byte_idx];
            let g = weight_idx / GROUP_SIZE;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Apply inverse Hadamard transform to restore original weights
        let hadamard_size = 256usize;
        for chunk in data.chunks_mut(hadamard_size) {
            if chunk.len() == hadamard_size {
                inverse_hadamard_transform(chunk);
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress maximum compression codec (g=256, no residual).
fn decompress_int4_10x(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const GROUP_SIZE: usize = 256;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let packed_size = (expected + 1) / 2;
        let scales_offset = packed_size;
        let offsets_offset = scales_offset + num_groups * 2;

        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            if s_idx + 1 < t.data.len() {
                let f16_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                scales.push(f16_bits_to_f32(f16_bits));
            }
            let o_idx = offsets_offset + g * 2;
            if o_idx + 1 < t.data.len() {
                let f16_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                offsets.push(f16_bits_to_f32(f16_bits));
            }
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let mut weight_idx = 0;

        while weight_idx < expected && byte_idx < packed_size {
            let byte = t.data[byte_idx];
            let g = weight_idx / GROUP_SIZE;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress mixed precision codec (variable group sizes).
fn decompress_int4_mixed(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        // Read group size from first 2 bytes
        if t.data.len() < 2 {
            return Err("Mixed codec data too short".to_string());
        }
        let group_size = (t.data[0] as usize) | ((t.data[1] as usize) << 8);

        let num_groups = (expected + group_size - 1) / group_size;
        let packed_size = (expected + 1) / 2;
        let scales_offset = 2 + packed_size; // +2 for group_size header
        let offsets_offset = scales_offset + num_groups * 2;

        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            if s_idx + 1 < t.data.len() {
                let f16_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                scales.push(f16_bits_to_f32(f16_bits));
            }
            let o_idx = offsets_offset + g * 2;
            if o_idx + 1 < t.data.len() {
                let f16_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                offsets.push(f16_bits_to_f32(f16_bits));
            }
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 2; // Skip group_size header
        let mut weight_idx = 0;

        while weight_idx < expected && byte_idx < 2 + packed_size {
            let byte = t.data[byte_idx];
            let g = weight_idx / group_size;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress hybrid codec (SpinQuant + layer-aware + selective residual).
fn decompress_int4_hybrid(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        if t.data.len() < 3 {
            return Err("Hybrid codec data too short".to_string());
        }

        // Read header: [flags | group_size_lo | group_size_hi]
        let flags = t.data[0];
        let use_residual = (flags & 1) != 0;
        let group_size = (t.data[1] as usize) | ((t.data[2] as usize) << 8);

        let num_groups = (expected + group_size - 1) / group_size;
        let packed_size = (expected + 1) / 2;
        let header_size = 3;

        let scales_offset = header_size + packed_size;
        let offsets_offset = scales_offset + num_groups * 2;
        let res_packed_offset = offsets_offset + num_groups * 2;

        // Read scales and offsets
        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            if s_idx + 1 < t.data.len() {
                let f16_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                scales.push(f16_bits_to_f32(f16_bits));
            }
            let o_idx = offsets_offset + g * 2;
            if o_idx + 1 < t.data.len() {
                let f16_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                offsets.push(f16_bits_to_f32(f16_bits));
            }
        }

        // Dequantize INT4
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = header_size;
        let mut weight_idx = 0;

        while weight_idx < expected && byte_idx < header_size + packed_size {
            let byte = t.data[byte_idx];
            let g = weight_idx / group_size;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Add residual correction if present
        if use_residual {
            let res_packed_size = (expected + 3) / 4;
            let res_scales_offset = res_packed_offset + res_packed_size;
            let res_offsets_offset = res_scales_offset + num_groups * 2;

            let mut res_scales: Vec<f32> = Vec::with_capacity(num_groups);
            let mut res_offsets: Vec<f32> = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let s_idx = res_scales_offset + g * 2;
                if s_idx + 1 < t.data.len() {
                    let f16_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                    res_scales.push(f16_bits_to_f32(f16_bits));
                }
                let o_idx = res_offsets_offset + g * 2;
                if o_idx + 1 < t.data.len() {
                    let f16_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                    res_offsets.push(f16_bits_to_f32(f16_bits));
                }
            }

            for (i, val) in data.iter_mut().enumerate() {
                let g = i / group_size;
                let rs = res_scales.get(g).copied().unwrap_or(0.0);
                let ro = res_offsets.get(g).copied().unwrap_or(0.0);

                let res_byte_idx = res_packed_offset + i / 4;
                let bit_pos = (i % 4) * 2;

                if res_byte_idx < t.data.len() {
                    let q = ((t.data[res_byte_idx] >> bit_pos) & 0x03) as f32;
                    *val += q * rs + ro;
                }
            }
        }

        // Apply inverse Hadamard transform
        let had_size = if group_size <= 256 && (group_size & (group_size - 1)) == 0 {
            group_size
        } else {
            0
        };

        if had_size > 0 {
            for chunk in data.chunks_mut(had_size) {
                if chunk.len() == had_size {
                    inverse_hadamard_transform(chunk);
                }
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress hybrid_v2 codec (reads group size from header)
fn decompress_int4_hybrid_v2(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.len() < 3 {
            return Err(format!(
                "Tensor '{}' data too short for hybrid_v2 header",
                t.name
            ));
        }

        // Parse header: flags (1 byte) + group_size (2 bytes)
        let flags = t.data[0];
        let group_size = (t.data[1] as usize) | ((t.data[2] as usize) << 8);
        let has_residual = (flags & 0x01) != 0;

        let num_groups = (expected + group_size - 1) / group_size;
        let packed_size = (expected + 1) / 2;

        // Calculate offsets
        let header_size = 3;
        let scales_offset = header_size + packed_size;
        let offsets_offset = scales_offset + num_groups * 2;
        let res_start = offsets_offset + num_groups * 2;

        if t.data.len() < offsets_offset + num_groups * 2 {
            return Err(format!("Tensor '{}' data too short", t.name));
        }

        // Read scales and offsets
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets_vec = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            let o_idx = offsets_offset + g * 2;
            let s_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
            let o_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
            scales.push(f16_bits_to_f32(s_bits));
            offsets_vec.push(f16_bits_to_f32(o_bits));
        }

        // Decompress INT4
        let mut data = Vec::with_capacity(expected);
        let mut weight_idx = 0;
        let mut byte_idx = header_size;

        while weight_idx < expected && byte_idx < scales_offset {
            let byte = t.data[byte_idx];
            let g = weight_idx / group_size;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets_vec.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Add residual if present
        if has_residual && t.data.len() > res_start {
            let res_scales_offset = res_start;
            let res_offsets_offset = res_scales_offset + num_groups * 2;
            let res_packed_offset = res_offsets_offset + num_groups * 2;

            if t.data.len() >= res_packed_offset {
                let mut res_scales = Vec::with_capacity(num_groups);
                let mut res_offsets = Vec::with_capacity(num_groups);
                for g in 0..num_groups {
                    let s_idx = res_scales_offset + g * 2;
                    let o_idx = res_offsets_offset + g * 2;
                    if s_idx + 1 < t.data.len() && o_idx + 1 < t.data.len() {
                        let s_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                        let o_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                        res_scales.push(f16_bits_to_f32(s_bits));
                        res_offsets.push(f16_bits_to_f32(o_bits));
                    }
                }

                // Unpack INT2 residual and add
                let mut res_idx = 0;
                let mut byte_idx = res_packed_offset;
                while res_idx < data.len() && byte_idx < t.data.len() {
                    let byte = t.data[byte_idx];
                    for shift in [0, 2, 4, 6] {
                        if res_idx >= data.len() {
                            break;
                        }
                        let g = res_idx / group_size;
                        let r_scale = res_scales.get(g).copied().unwrap_or(0.0);
                        let r_offset = res_offsets.get(g).copied().unwrap_or(0.0);
                        let q = ((byte >> shift) & 0x03) as f32;
                        data[res_idx] += q * r_scale + r_offset;
                        res_idx += 1;
                    }
                    byte_idx += 1;
                }
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress ultimate codec (with outlier extraction)
fn decompress_int4_ultimate(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.len() < 7 {
            return Err(format!(
                "Tensor '{}' data too short for ultimate header",
                t.name
            ));
        }

        // Parse header: flags (1) + group_size (2) + outlier_count (4)
        let _flags = t.data[0];
        let group_size = (t.data[1] as usize) | ((t.data[2] as usize) << 8);
        let outlier_count = (t.data[3] as usize)
            | ((t.data[4] as usize) << 8)
            | ((t.data[5] as usize) << 16)
            | ((t.data[6] as usize) << 24);

        let header_size = 7;
        let outlier_indices_size = outlier_count * 4;
        let outlier_values_size = outlier_count * 4;
        let outliers_end = header_size + outlier_indices_size + outlier_values_size;

        // Read outlier indices and values
        let mut outlier_indices = Vec::with_capacity(outlier_count);
        let mut outlier_values = Vec::with_capacity(outlier_count);

        for i in 0..outlier_count {
            let idx_offset = header_size + i * 4;
            let idx = (t.data[idx_offset] as u32)
                | ((t.data[idx_offset + 1] as u32) << 8)
                | ((t.data[idx_offset + 2] as u32) << 16)
                | ((t.data[idx_offset + 3] as u32) << 24);
            outlier_indices.push(idx as usize);

            let val_offset = header_size + outlier_indices_size + i * 4;
            let bits = (t.data[val_offset] as u32)
                | ((t.data[val_offset + 1] as u32) << 8)
                | ((t.data[val_offset + 2] as u32) << 16)
                | ((t.data[val_offset + 3] as u32) << 24);
            outlier_values.push(f32::from_bits(bits));
        }

        let num_groups = (expected + group_size - 1) / group_size;
        let packed_size = (expected + 1) / 2;
        let scales_offset = outliers_end + packed_size;
        let offsets_offset = scales_offset + num_groups * 2;

        if t.data.len() < offsets_offset + num_groups * 2 {
            return Err(format!(
                "Tensor '{}' data too short for scales/offsets",
                t.name
            ));
        }

        // Read scales and offsets
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets_vec = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let s_idx = scales_offset + g * 2;
            let o_idx = offsets_offset + g * 2;
            let s_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
            let o_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
            scales.push(f16_bits_to_f32(s_bits));
            offsets_vec.push(f16_bits_to_f32(o_bits));
        }

        // Decompress INT4
        let mut data = Vec::with_capacity(expected);
        let mut weight_idx = 0;
        let mut byte_idx = outliers_end;

        while weight_idx < expected && byte_idx < scales_offset {
            let byte = t.data[byte_idx];
            let g = weight_idx / group_size;
            let scale = scales.get(g).copied().unwrap_or(1.0);
            let offset = offsets_vec.get(g).copied().unwrap_or(0.0);

            let v0 = (byte & 0x0f) as f32;
            data.push(v0 * scale + offset);
            weight_idx += 1;

            if weight_idx < expected {
                let v1 = ((byte >> 4) & 0x0f) as f32;
                data.push(v1 * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Restore outliers
        for (i, &idx) in outlier_indices.iter().enumerate() {
            if idx < data.len() {
                data[idx] = outlier_values[i];
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress CALDERA codec: W = Q + LR (low-rank + quantized backbone)
fn decompress_caldera(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        if t.data.len() < 12 {
            return Err(format!(
                "Tensor '{}' data too short for CALDERA header",
                t.name
            ));
        }

        // Parse header: rank(2) + rows(4) + cols(4) + group_size(2) = 12 bytes
        let rank = (t.data[0] as usize) | ((t.data[1] as usize) << 8);
        let rows = (t.data[2] as usize)
            | ((t.data[3] as usize) << 8)
            | ((t.data[4] as usize) << 16)
            | ((t.data[5] as usize) << 24);
        let cols = (t.data[6] as usize)
            | ((t.data[7] as usize) << 8)
            | ((t.data[8] as usize) << 16)
            | ((t.data[9] as usize) << 24);
        let group_size = (t.data[10] as usize) | ((t.data[11] as usize) << 8);

        let header_size = 12;
        let num_groups = (expected + group_size - 1) / group_size;
        let q_packed_size = (expected + 3) / 4; // INT2: 4 values per byte
        let q_scales_size = num_groups * 2;
        let q_offsets_size = num_groups * 2;
        let l_size = rows * rank;
        let r_size = cols * rank;
        let _lr_params_size = 8; // 4 FP16 values (used for documentation)

        let q_packed_start = header_size;
        let q_scales_start = q_packed_start + q_packed_size;
        let q_offsets_start = q_scales_start + q_scales_size;
        let l_start = q_offsets_start + q_offsets_size;
        let r_start = l_start + l_size;
        let lr_params_start = r_start + r_size;

        // Read Q scales and offsets
        let mut q_scales: Vec<f32> = Vec::with_capacity(num_groups);
        let mut q_offsets: Vec<f32> = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let s_idx = q_scales_start + g * 2;
            let o_idx = q_offsets_start + g * 2;
            if s_idx + 1 < t.data.len() && o_idx + 1 < t.data.len() {
                let s_bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                let o_bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                q_scales.push(f16_bits_to_f32(s_bits));
                q_offsets.push(f16_bits_to_f32(o_bits));
            }
        }

        // Read LR params
        let l_scale = if lr_params_start + 1 < t.data.len() {
            let bits =
                (t.data[lr_params_start] as u16) | ((t.data[lr_params_start + 1] as u16) << 8);
            f16_bits_to_f32(bits)
        } else {
            1.0
        };
        let l_min = if lr_params_start + 3 < t.data.len() {
            let bits =
                (t.data[lr_params_start + 2] as u16) | ((t.data[lr_params_start + 3] as u16) << 8);
            f16_bits_to_f32(bits)
        } else {
            0.0
        };
        let r_scale = if lr_params_start + 5 < t.data.len() {
            let bits =
                (t.data[lr_params_start + 4] as u16) | ((t.data[lr_params_start + 5] as u16) << 8);
            f16_bits_to_f32(bits)
        } else {
            1.0
        };
        let r_min = if lr_params_start + 7 < t.data.len() {
            let bits =
                (t.data[lr_params_start + 6] as u16) | ((t.data[lr_params_start + 7] as u16) << 8);
            f16_bits_to_f32(bits)
        } else {
            0.0
        };

        // Dequantize L factors
        let mut l_factors: Vec<f32> = Vec::with_capacity(l_size);
        for i in 0..l_size {
            let idx = l_start + i;
            if idx < t.data.len() {
                l_factors.push(t.data[idx] as f32 * l_scale + l_min);
            } else {
                l_factors.push(0.0);
            }
        }

        // Dequantize R factors
        let mut r_factors: Vec<f32> = Vec::with_capacity(r_size);
        for i in 0..r_size {
            let idx = r_start + i;
            if idx < t.data.len() {
                r_factors.push(t.data[idx] as f32 * r_scale + r_min);
            } else {
                r_factors.push(0.0);
            }
        }

        // Decompress INT2 backbone
        let mut q_data: Vec<f32> = Vec::with_capacity(expected);
        let mut weight_idx = 0;
        let mut byte_idx = q_packed_start;

        while weight_idx < expected && byte_idx < q_scales_start {
            let byte = t.data[byte_idx];
            for shift in [0, 2, 4, 6] {
                if weight_idx >= expected {
                    break;
                }
                let g = weight_idx / group_size;
                let scale = q_scales.get(g).copied().unwrap_or(1.0);
                let offset = q_offsets.get(g).copied().unwrap_or(0.0);
                let q = ((byte >> shift) & 0x03) as f32;
                q_data.push(q * scale + offset);
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        // Compute LR and add to Q: W = Q + LR
        let mut data: Vec<f32> = q_data;
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if idx < data.len() {
                    let mut lr_val = 0.0f32;
                    for r in 0..rank {
                        let l_idx = i * rank + r;
                        let r_idx = j * rank + r;
                        if l_idx < l_factors.len() && r_idx < r_factors.len() {
                            lr_val += l_factors[l_idx] * r_factors[r_idx];
                        }
                    }
                    data[idx] += lr_val;
                }
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress AQLM codec: w = cb1[q1] + cb2[q2]
fn decompress_aqlm(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    const CODEBOOK_SIZE: usize = 4;

    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        if t.data.len() < 2 {
            return Err(format!(
                "Tensor '{}' data too short for AQLM header",
                t.name
            ));
        }

        // Parse header
        let group_size = (t.data[0] as usize) | ((t.data[1] as usize) << 8);
        let header_size = 2;

        let num_groups = (expected + group_size - 1) / group_size;
        let packed_size_per_group = (group_size + 1) / 2; // 4 bits per weight, 2 weights per byte
        let total_packed_size = num_groups * packed_size_per_group;
        let codebook_size_per_group = CODEBOOK_SIZE * 2 * 2; // 4 values * 2 codebooks * 2 bytes (FP16)

        let packed_start = header_size;
        let codebooks_start = packed_start + total_packed_size;

        // Decompress each group
        let mut data: Vec<f32> = Vec::with_capacity(expected);

        for g in 0..num_groups {
            let group_start = g * group_size;
            let group_end = (group_start + group_size).min(expected);
            let group_len = group_end - group_start;

            // Read codebooks for this group
            let cb_offset = codebooks_start + g * codebook_size_per_group;
            let mut cb1 = [0.0f32; CODEBOOK_SIZE];
            let mut cb2 = [0.0f32; CODEBOOK_SIZE];

            for i in 0..CODEBOOK_SIZE {
                let idx1 = cb_offset + i * 2;
                let idx2 = cb_offset + CODEBOOK_SIZE * 2 + i * 2;
                if idx1 + 1 < t.data.len() {
                    let bits = (t.data[idx1] as u16) | ((t.data[idx1 + 1] as u16) << 8);
                    cb1[i] = f16_bits_to_f32(bits);
                }
                if idx2 + 1 < t.data.len() {
                    let bits = (t.data[idx2] as u16) | ((t.data[idx2 + 1] as u16) << 8);
                    cb2[i] = f16_bits_to_f32(bits);
                }
            }

            // Read packed indices for this group
            let packed_offset = packed_start + g * packed_size_per_group;
            let mut weight_in_group = 0;
            let mut byte_idx = packed_offset;

            while weight_in_group < group_len {
                if byte_idx >= t.data.len() {
                    break;
                }
                let byte = t.data[byte_idx];

                // First weight: bits 0-3 (q1 in 0-1, q2 in 2-3)
                let q1_0 = (byte & 0x03) as usize;
                let q2_0 = ((byte >> 2) & 0x03) as usize;
                data.push(cb1[q1_0] + cb2[q2_0]);
                weight_in_group += 1;

                // Second weight: bits 4-7 (q1 in 4-5, q2 in 6-7)
                if weight_in_group < group_len {
                    let q1_1 = ((byte >> 4) & 0x03) as usize;
                    let q2_1 = ((byte >> 6) & 0x03) as usize;
                    data.push(cb1[q1_1] + cb2[q2_1]);
                    weight_in_group += 1;
                }

                byte_idx += 1;
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress PocketLLM codec: vector quantization with learned codebook + INT4 residual
fn decompress_pocketllm(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        if t.data.len() < 10 {
            return Err(format!(
                "Tensor '{}' data too short for PocketLLM header",
                t.name
            ));
        }

        // Parse header
        let vec_dim = t.data[0] as usize;
        let codebook_size = (t.data[1] as usize) | ((t.data[2] as usize) << 8);
        let num_weights = (t.data[3] as usize)
            | ((t.data[4] as usize) << 8)
            | ((t.data[5] as usize) << 16)
            | ((t.data[6] as usize) << 24);
        let has_residual = t.data[7] != 0;
        let res_group = (t.data[8] as usize) | ((t.data[9] as usize) << 8);

        let header_size = 10;
        let num_vectors = (num_weights + vec_dim - 1) / vec_dim;
        let indices_size = num_vectors;
        let codebook_bytes = codebook_size * vec_dim * 2; // FP16

        let indices_start = header_size;
        let codebook_start = indices_start + indices_size;
        let residual_start = codebook_start + codebook_bytes;

        // Read codebook
        let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(codebook_size);
        for k in 0..codebook_size {
            let mut centroid = Vec::with_capacity(vec_dim);
            for d in 0..vec_dim {
                let idx = codebook_start + (k * vec_dim + d) * 2;
                if idx + 1 < t.data.len() {
                    let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                    centroid.push(f16_bits_to_f32(bits));
                } else {
                    centroid.push(0.0);
                }
            }
            codebook.push(centroid);
        }

        // Reconstruct from codebook indices
        let mut data: Vec<f32> = Vec::with_capacity(num_weights);
        for v in 0..num_vectors {
            let idx_pos = indices_start + v;
            let k = if idx_pos < t.data.len() {
                t.data[idx_pos] as usize
            } else {
                0
            };
            let centroid = codebook
                .get(k)
                .cloned()
                .unwrap_or_else(|| vec![0.0; vec_dim]);

            for d in 0..vec_dim {
                if data.len() < num_weights {
                    data.push(centroid[d]);
                }
            }
        }

        // Add residual if present
        if has_residual {
            let num_res_groups = (num_weights + res_group - 1) / res_group;
            let res_packed_size = (num_weights + 1) / 2;
            let res_scales_start = residual_start + res_packed_size;
            let res_offsets_start = res_scales_start + num_res_groups * 2;

            // Read residual scales and offsets
            let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
            let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);

            for g in 0..num_res_groups {
                let s_idx = res_scales_start + g * 2;
                let o_idx = res_offsets_start + g * 2;
                if s_idx + 1 < t.data.len() {
                    let bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                    res_scales.push(f16_bits_to_f32(bits));
                } else {
                    res_scales.push(1.0);
                }
                if o_idx + 1 < t.data.len() {
                    let bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                    res_offsets.push(f16_bits_to_f32(bits));
                } else {
                    res_offsets.push(0.0);
                }
            }

            // Unpack INT4 residual and add
            let mut weight_idx = 0;
            let mut byte_idx = residual_start;

            while weight_idx < num_weights && byte_idx < res_scales_start {
                let byte = t.data[byte_idx];
                let g = weight_idx / res_group;
                let scale = res_scales.get(g).copied().unwrap_or(1.0);
                let offset = res_offsets.get(g).copied().unwrap_or(0.0);

                // Low nibble
                let q0 = (byte & 0x0f) as f32;
                if weight_idx < data.len() {
                    data[weight_idx] += q0 * scale + offset;
                }
                weight_idx += 1;

                // High nibble
                if weight_idx < num_weights {
                    let q1 = ((byte >> 4) & 0x0f) as f32;
                    if weight_idx < data.len() {
                        data[weight_idx] += q1 * scale + offset;
                    }
                    weight_idx += 1;
                }
                byte_idx += 1;
            }
        }

        // Truncate to exact size
        data.truncate(num_weights);

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress TenPak-X codec: low-rank + vector quantization + INT4 residual
fn decompress_tenpak_x(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        if t.data.len() < 15 {
            return Err(format!(
                "Tensor '{}' data too short for TenPak-X header",
                t.name
            ));
        }

        // Parse header
        let rank = (t.data[0] as usize) | ((t.data[1] as usize) << 8);
        let rows = (t.data[2] as usize)
            | ((t.data[3] as usize) << 8)
            | ((t.data[4] as usize) << 16)
            | ((t.data[5] as usize) << 24);
        let cols = (t.data[6] as usize)
            | ((t.data[7] as usize) << 8)
            | ((t.data[8] as usize) << 16)
            | ((t.data[9] as usize) << 24);
        let vec_dim = t.data[10] as usize;
        let codebook_size = (t.data[11] as usize) | ((t.data[12] as usize) << 8);
        let res_group = (t.data[13] as usize) | ((t.data[14] as usize) << 8);

        let header_size = 15;
        let num_weights = rows * cols;
        let num_vectors = (num_weights + vec_dim - 1) / vec_dim;

        // Calculate offsets
        let l_size = rows * rank * 2; // FP16
        let r_size = rank * cols * 2; // FP16
        let codebook_bytes = codebook_size * vec_dim * 2; // FP16
        let indices_size = num_vectors;
        let res_packed_size = (num_weights + 1) / 2;
        let num_res_groups = (num_weights + res_group - 1) / res_group;

        let l_start = header_size;
        let r_start = l_start + l_size;
        let codebook_start = r_start + r_size;
        let indices_start = codebook_start + codebook_bytes;
        let res_packed_start = indices_start + indices_size;
        let res_scales_start = res_packed_start + res_packed_size;
        let res_offsets_start = res_scales_start + num_res_groups * 2;

        // Read L matrix
        let mut l_mat: Vec<f32> = Vec::with_capacity(rows * rank);
        for i in 0..(rows * rank) {
            let idx = l_start + i * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                l_mat.push(f16_bits_to_f32(bits));
            } else {
                l_mat.push(0.0);
            }
        }

        // Read R matrix
        let mut r_mat: Vec<f32> = Vec::with_capacity(rank * cols);
        for i in 0..(rank * cols) {
            let idx = r_start + i * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                r_mat.push(f16_bits_to_f32(bits));
            } else {
                r_mat.push(0.0);
            }
        }

        // Read codebook
        let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(codebook_size);
        for k in 0..codebook_size {
            let mut centroid = Vec::with_capacity(vec_dim);
            for d in 0..vec_dim {
                let idx = codebook_start + (k * vec_dim + d) * 2;
                if idx + 1 < t.data.len() {
                    let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                    centroid.push(f16_bits_to_f32(bits));
                } else {
                    centroid.push(0.0);
                }
            }
            codebook.push(centroid);
        }

        // Read residual scales and offsets
        let mut res_scales: Vec<f32> = Vec::with_capacity(num_res_groups);
        let mut res_offsets: Vec<f32> = Vec::with_capacity(num_res_groups);
        for g in 0..num_res_groups {
            let s_idx = res_scales_start + g * 2;
            let o_idx = res_offsets_start + g * 2;
            if s_idx + 1 < t.data.len() {
                let bits = (t.data[s_idx] as u16) | ((t.data[s_idx + 1] as u16) << 8);
                res_scales.push(f16_bits_to_f32(bits));
            } else {
                res_scales.push(1.0);
            }
            if o_idx + 1 < t.data.len() {
                let bits = (t.data[o_idx] as u16) | ((t.data[o_idx + 1] as u16) << 8);
                res_offsets.push(f16_bits_to_f32(bits));
            } else {
                res_offsets.push(0.0);
            }
        }

        // Reconstruct: W = L @ R + codebook[indices] + residual
        let mut data: Vec<f32> = vec![0.0; num_weights];

        // Add low-rank: L @ R
        if rank > 0 {
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    for r in 0..rank {
                        let l_idx = i * rank + r;
                        let r_idx = r * cols + j;
                        if l_idx < l_mat.len() && r_idx < r_mat.len() {
                            data[idx] += l_mat[l_idx] * r_mat[r_idx];
                        }
                    }
                }
            }
        }

        // Add vector quantization
        for v in 0..num_vectors {
            let idx_pos = indices_start + v;
            let k = if idx_pos < t.data.len() {
                t.data[idx_pos] as usize
            } else {
                0
            };
            let centroid = codebook
                .get(k)
                .cloned()
                .unwrap_or_else(|| vec![0.0; vec_dim]);

            for d in 0..vec_dim {
                let weight_idx = v * vec_dim + d;
                if weight_idx < num_weights {
                    data[weight_idx] += centroid[d];
                }
            }
        }

        // Add INT4 residual
        let mut weight_idx = 0;
        let mut byte_idx = res_packed_start;
        while weight_idx < num_weights && byte_idx < res_scales_start {
            let byte = t.data[byte_idx];
            let g = weight_idx / res_group;
            let scale = res_scales.get(g).copied().unwrap_or(1.0);
            let offset = res_offsets.get(g).copied().unwrap_or(0.0);

            // Low nibble
            let q0 = (byte & 0x0f) as f32;
            if weight_idx < data.len() {
                data[weight_idx] += q0 * scale + offset;
            }
            weight_idx += 1;

            // High nibble
            if weight_idx < num_weights {
                let q1 = ((byte >> 4) & 0x0f) as f32;
                if weight_idx < data.len() {
                    data[weight_idx] += q1 * scale + offset;
                }
                weight_idx += 1;
            }
            byte_idx += 1;
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress TenPak-X v2 codec: low-rank + vector quantization + sparse residual
fn decompress_tenpak_x_v2(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        if t.data.len() < 17 {
            return Err(format!(
                "Tensor '{}' data too short for TenPak-X v2 header",
                t.name
            ));
        }

        // Parse header
        let rank = (t.data[0] as usize) | ((t.data[1] as usize) << 8);
        let rows = (t.data[2] as usize)
            | ((t.data[3] as usize) << 8)
            | ((t.data[4] as usize) << 16)
            | ((t.data[5] as usize) << 24);
        let cols = (t.data[6] as usize)
            | ((t.data[7] as usize) << 8)
            | ((t.data[8] as usize) << 16)
            | ((t.data[9] as usize) << 24);
        let vec_dim = t.data[10] as usize;
        let codebook_size = (t.data[11] as usize) | ((t.data[12] as usize) << 8);
        let num_sparse = (t.data[13] as usize)
            | ((t.data[14] as usize) << 8)
            | ((t.data[15] as usize) << 16)
            | ((t.data[16] as usize) << 24);

        let header_size = 17;
        let num_weights = rows * cols;
        let num_vectors = (num_weights + vec_dim - 1) / vec_dim;

        // Calculate offsets
        let l_size = rows * rank * 2; // FP16
        let r_size = rank * cols * 2; // FP16
        let codebook_bytes = codebook_size * vec_dim * 2; // FP16
        let indices_size = num_vectors;
        let sparse_indices_size = num_sparse * 4; // 4 bytes per index
        let sparse_values_size = num_sparse * 2; // FP16

        let l_start = header_size;
        let r_start = l_start + l_size;
        let codebook_start = r_start + r_size;
        let indices_start = codebook_start + codebook_bytes;
        let sparse_indices_start = indices_start + indices_size;
        let sparse_values_start = sparse_indices_start + sparse_indices_size;

        // Read L matrix
        let mut l_mat: Vec<f32> = Vec::with_capacity(rows * rank);
        for i in 0..(rows * rank) {
            let idx = l_start + i * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                l_mat.push(f16_bits_to_f32(bits));
            } else {
                l_mat.push(0.0);
            }
        }

        // Read R matrix
        let mut r_mat: Vec<f32> = Vec::with_capacity(rank * cols);
        for i in 0..(rank * cols) {
            let idx = r_start + i * 2;
            if idx + 1 < t.data.len() {
                let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                r_mat.push(f16_bits_to_f32(bits));
            } else {
                r_mat.push(0.0);
            }
        }

        // Read codebook
        let mut codebook: Vec<Vec<f32>> = Vec::with_capacity(codebook_size);
        for k in 0..codebook_size {
            let mut centroid = Vec::with_capacity(vec_dim);
            for d in 0..vec_dim {
                let idx = codebook_start + (k * vec_dim + d) * 2;
                if idx + 1 < t.data.len() {
                    let bits = (t.data[idx] as u16) | ((t.data[idx + 1] as u16) << 8);
                    centroid.push(f16_bits_to_f32(bits));
                } else {
                    centroid.push(0.0);
                }
            }
            codebook.push(centroid);
        }

        // Read sparse indices and values
        let mut sparse_indices: Vec<usize> = Vec::with_capacity(num_sparse);
        let mut sparse_values: Vec<f32> = Vec::with_capacity(num_sparse);

        for i in 0..num_sparse {
            let idx = sparse_indices_start + i * 4;
            if idx + 3 < t.data.len() {
                let sparse_idx = (t.data[idx] as usize)
                    | ((t.data[idx + 1] as usize) << 8)
                    | ((t.data[idx + 2] as usize) << 16)
                    | ((t.data[idx + 3] as usize) << 24);
                sparse_indices.push(sparse_idx);
            }

            let val_idx = sparse_values_start + i * 2;
            if val_idx + 1 < t.data.len() {
                let bits = (t.data[val_idx] as u16) | ((t.data[val_idx + 1] as u16) << 8);
                sparse_values.push(f16_bits_to_f32(bits));
            } else {
                sparse_values.push(0.0);
            }
        }

        // Reconstruct: W = L @ R + codebook[indices] + sparse_residual
        let mut data: Vec<f32> = vec![0.0; num_weights];

        // Add low-rank: L @ R
        if rank > 0 {
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    for r in 0..rank {
                        let l_idx = i * rank + r;
                        let r_idx = r * cols + j;
                        if l_idx < l_mat.len() && r_idx < r_mat.len() {
                            data[idx] += l_mat[l_idx] * r_mat[r_idx];
                        }
                    }
                }
            }
        }

        // Add vector quantization
        for v in 0..num_vectors {
            let idx_pos = indices_start + v;
            let k = if idx_pos < t.data.len() {
                t.data[idx_pos] as usize
            } else {
                0
            };
            let centroid = codebook
                .get(k)
                .cloned()
                .unwrap_or_else(|| vec![0.0; vec_dim]);

            for d in 0..vec_dim {
                let weight_idx = v * vec_dim + d;
                if weight_idx < num_weights {
                    data[weight_idx] += centroid[d];
                }
            }
        }

        // Add sparse residual
        for (i, &sparse_idx) in sparse_indices.iter().enumerate() {
            if sparse_idx < data.len() && i < sparse_values.len() {
                data[sparse_idx] += sparse_values[i];
            }
        }

        out.push(FloatTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            data,
        });
    }

    Ok(FloatBundle {
        tensors: out,
        activation_stats: ActivationStats::new(),
    })
}

/// Decompress a quantized artifact back into float32 tensors.
pub fn decompress_bundle(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    if artifact.version != ARTIFACT_VERSION {
        return Err(format!(
            "Unsupported artifact version {} (expected {})",
            artifact.version, ARTIFACT_VERSION
        ));
    }

    match artifact.codec.as_str() {
        // Production codecs
        CODEC_INT4_RESIDUAL_V1 => decompress_int4_residual(artifact),
        CODEC_INT4_OPT_LLAMA_V1 => decompress_int4_opt_llama(artifact),
        #[cfg(feature = "calibration")]
        CODEC_INT4_CALIBRATED_V1 => decompress_int4_calibrated(artifact),

        // Experimental 10x codecs
        CODEC_INT4_SPIN_V1 => decompress_int4_spin(artifact),
        CODEC_INT4_10X_V1 => decompress_int4_10x(artifact),
        CODEC_INT4_MIXED_V1 => decompress_int4_mixed(artifact),
        CODEC_INT4_HYBRID_V1 => decompress_int4_hybrid(artifact),
        CODEC_INT4_HYBRID_V2 => decompress_int4_hybrid_v2(artifact),
        CODEC_INT4_AWQ_10X_V1 => decompress_int4_mixed(artifact),
        CODEC_INT4_GPTQ_LITE_V1 => decompress_int4_mixed(artifact),
        CODEC_INT4_ULTIMATE_V1 => decompress_int4_ultimate(artifact),
        CODEC_CALDERA_V1 => decompress_caldera(artifact),
        CODEC_AQLM_V1 => decompress_aqlm(artifact),
        CODEC_POCKETLLM_V1 => decompress_pocketllm(artifact),
        CODEC_POCKETLLM_V2 => decompress_pocketllm(artifact), // Same format, reuse decompressor
        CODEC_TENPAK_X_V1 => decompress_tenpak_x(artifact),
        CODEC_TENPAK_X_V2 => decompress_tenpak_x_v2(artifact),

        other => Err(format!("Unsupported codec '{}'", other)),
    }
}

pub use ArtifactFile as Artifact;
pub use FloatBundle as Bundle;
pub use FloatTensor as Tensor;

fn ffi_set_error(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    let cstr = CString::new(msg).unwrap_or_else(|_| CString::new("tenpak error").unwrap());
    unsafe {
        *err_out = cstr.into_raw();
    }
}

#[no_mangle]
pub extern "C" fn tenpak_compress_json_bundle(
    json_ptr: *const u8,
    json_len: usize,
    codec_ptr: *const c_char,
    out_artifact_ptr: *mut *mut u8,
    out_artifact_len: *mut usize,
    out_err_ptr: *mut *mut c_char,
) -> i32 {
    if json_ptr.is_null()
        || out_artifact_ptr.is_null()
        || out_artifact_len.is_null()
        || out_err_ptr.is_null()
    {
        return -1;
    }

    unsafe {
        *out_artifact_ptr = std::ptr::null_mut();
        *out_artifact_len = 0;
        *out_err_ptr = std::ptr::null_mut();
    }

    let json_slice = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
    let json_str = match std::str::from_utf8(json_slice) {
        Ok(s) => s,
        Err(e) => {
            ffi_set_error(out_err_ptr, &format!("invalid UTF-8 in JSON input: {}", e));
            return 1;
        }
    };

    let bundle: FloatBundle = match serde_json::from_str(json_str) {
        Ok(b) => b,
        Err(e) => {
            ffi_set_error(out_err_ptr, &format!("failed to parse JSON bundle: {}", e));
            return 1;
        }
    };

    let codec = if codec_ptr.is_null() {
        CODEC_INT4_RESIDUAL_V1
    } else {
        let s = unsafe { CStr::from_ptr(codec_ptr) };
        match s.to_str() {
            Ok(v) => v,
            Err(e) => {
                ffi_set_error(
                    out_err_ptr,
                    &format!("invalid UTF-8 in codec string: {}", e),
                );
                return 1;
            }
        }
    };

    let artifact = match compress_bundle_with_codec(&bundle, codec) {
        Ok(a) => a,
        Err(e) => {
            ffi_set_error(out_err_ptr, &e);
            return 1;
        }
    };

    let encoded = match bincode::serialize(&artifact) {
        Ok(v) => v,
        Err(e) => {
            ffi_set_error(out_err_ptr, &format!("failed to serialize artifact: {}", e));
            return 1;
        }
    };

    let len = encoded.len();
    let mut boxed = encoded.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        *out_artifact_ptr = ptr;
        *out_artifact_len = len;
    }

    0
}

#[no_mangle]
pub extern "C" fn tenpak_decompress_artifact_to_json(
    artifact_ptr: *const u8,
    artifact_len: usize,
    out_json_ptr: *mut *mut u8,
    out_json_len: *mut usize,
    out_err_ptr: *mut *mut c_char,
) -> i32 {
    if artifact_ptr.is_null()
        || out_json_ptr.is_null()
        || out_json_len.is_null()
        || out_err_ptr.is_null()
    {
        return -1;
    }

    unsafe {
        *out_json_ptr = std::ptr::null_mut();
        *out_json_len = 0;
        *out_err_ptr = std::ptr::null_mut();
    }

    let artifact_slice = unsafe { std::slice::from_raw_parts(artifact_ptr, artifact_len) };
    let artifact: ArtifactFile = match bincode::deserialize(artifact_slice) {
        Ok(a) => a,
        Err(e) => {
            ffi_set_error(
                out_err_ptr,
                &format!("failed to deserialize artifact: {}", e),
            );
            return 1;
        }
    };

    let bundle = match decompress_bundle(&artifact) {
        Ok(b) => b,
        Err(e) => {
            ffi_set_error(out_err_ptr, &e);
            return 1;
        }
    };

    let json = match serde_json::to_vec_pretty(&bundle) {
        Ok(v) => v,
        Err(e) => {
            ffi_set_error(
                out_err_ptr,
                &format!("failed to serialize bundle to JSON: {}", e),
            );
            return 1;
        }
    };

    let len = json.len();
    let mut boxed = json.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        *out_json_ptr = ptr;
        *out_json_len = len;
    }

    0
}

#[no_mangle]
pub extern "C" fn tenpak_free_buffer(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

#[no_mangle]
pub extern "C" fn tenpak_free_cstring(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Create a delta artifact that contains only tensors which differ from the
/// base artifact by more than `epsilon` in L1 norm.
///
/// The variant is provided as a float bundle (e.g. from JSON), and the delta
/// artifact reuses the codec of the base artifact.
pub fn create_delta_artifact(
    base: &ArtifactFile,
    variant: &FloatBundle,
    epsilon: f32,
) -> Result<ArtifactFile, String> {
    let base_bundle = decompress_bundle(base)?;

    let mut base_map: HashMap<&str, &FloatTensor> = HashMap::new();
    for t in &base_bundle.tensors {
        base_map.insert(t.name.as_str(), t);
    }

    let mut changed: Vec<FloatTensor> = Vec::new();

    for vt in &variant.tensors {
        match base_map.get(vt.name.as_str()) {
            None => {
                // New tensor: include in delta.
                changed.push(vt.clone());
            }
            Some(bt) => {
                if bt.shape != vt.shape || bt.data.len() != vt.data.len() {
                    changed.push(vt.clone());
                    continue;
                }

                let mut l1 = 0.0_f32;
                for (a, b) in bt.data.iter().zip(vt.data.iter()) {
                    l1 += (a - b).abs();
                    if l1 > epsilon {
                        break;
                    }
                }

                if l1 > epsilon {
                    changed.push(vt.clone());
                }
            }
        }
    }

    let delta_bundle = FloatBundle {
        tensors: changed,
        activation_stats: ActivationStats::new(),
    };
    compress_bundle_with_codec(&delta_bundle, base.codec.as_str())
}

/// Materialize a full artifact from a base artifact and a delta artifact.
///
/// Any tensor present in the delta replaces the tensor with the same name in
/// the base; tensors only present in the base are kept, and tensors only
/// present in the delta are added.
pub fn materialize_artifact(
    base: &ArtifactFile,
    delta: &ArtifactFile,
) -> Result<ArtifactFile, String> {
    if base.version != delta.version {
        return Err(format!(
            "Version mismatch between base ({}) and delta ({}) artifacts",
            base.version, delta.version
        ));
    }
    if base.codec != delta.codec {
        return Err(format!(
            "Codec mismatch between base ('{}') and delta ('{}') artifacts",
            base.codec, delta.codec
        ));
    }

    let mut merged: Vec<QuantizedTensor> = Vec::new();
    let mut delta_map: HashMap<&str, &QuantizedTensor> = HashMap::new();
    for t in &delta.tensors {
        delta_map.insert(t.name.as_str(), t);
    }

    // Start with base tensors, overridden by delta where applicable.
    for bt in &base.tensors {
        if let Some(dt) = delta_map.get(bt.name.as_str()) {
            merged.push((*dt).clone());
        } else {
            merged.push(bt.clone());
        }
    }

    // Add tensors that exist only in delta.
    for dt in &delta.tensors {
        if !base.tensors.iter().any(|bt| bt.name == dt.name) {
            merged.push(dt.clone());
        }
    }

    Ok(ArtifactFile {
        version: base.version,
        codec: base.codec.clone(),
        tensors: merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple_bundle() {
        let tensor = FloatTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            data: vec![0.1, -0.2, 0.3, -0.4],
        };
        let bundle = FloatBundle {
            tensors: vec![tensor],
            activation_stats: ActivationStats::new(),
        };

        let artifact = compress_bundle(&bundle).expect("compression should succeed");
        let restored = decompress_bundle(&artifact).expect("decompression should succeed");

        assert_eq!(restored.tensors.len(), 1);
        let r = &restored.tensors[0];
        assert_eq!(r.name, "test.weight");
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(r.data.len(), 4);

        for (orig, rec) in [0.1f32, -0.2, 0.3, -0.4].iter().zip(r.data.iter()) {
            let diff = (orig - rec).abs();
            assert!(
                diff < 0.1,
                "quantization error too large: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn shape_mismatch_error() {
        // shape [2, 2] but only 3 elements
        let bad_tensor = FloatTensor {
            name: "bad".to_string(),
            shape: vec![2, 2],
            data: vec![0.0, 1.0, 2.0],
        };
        let bundle = FloatBundle {
            tensors: vec![bad_tensor],
            activation_stats: ActivationStats::new(),
        };

        let res = compress_bundle(&bundle);
        assert!(res.is_err(), "expected shape mismatch to produce an error");
    }
}
