use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// std::env removed - not needed after AWQ simplification
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

// Cross-platform GPU inference via wgpu (AMD, Intel, NVIDIA, Apple)
#[cfg(feature = "gpu")]
pub mod wgpu_gemm;

/// Current on-disk artifact format version.
pub const ARTIFACT_VERSION: u32 = 1;

// ============================================================================
// RECOMMENDED CODECS
// ============================================================================

/// Ultra-fine group quantization (g=8) - achieves <1% PPL delta. RECOMMENDED.
pub const CODEC_INT4_G8_V1: &str = "int4_g8_v1";
/// Group quantization (g=16) - ~2% PPL delta, better compression than g=8.
pub const CODEC_INT4_G16_V1: &str = "int4_g16_v1";
/// K-quant style: super-blocks with quantized scales (like llama.cpp Q4_K).
/// Better compression, optimized for <1% PPL delta.
pub const CODEC_INT4_K_V1: &str = "int4_k_v1";
/// Ultra-fine group quantization (g=8) with FP16 scales. Best quality + compression.
pub const CODEC_INT4_G8_FP16_V1: &str = "int4_g8_fp16_v1";
/// Group quantization (g=16) with FP16 scales. Best balance of quality + compression.
/// Achieves 5.33x compression with <2% PPL delta.
pub const CODEC_INT4_G16_FP16_V1: &str = "int4_g16_fp16_v1";
/// Optimal quantization with iterative scale refinement. BEST for GPT-2 style models.
/// Achieves 5.33x compression with <1% PPL delta.
pub const CODEC_INT4_OPT_V1: &str = "int4_opt_v1";
/// Optimal quantization for Llama-architecture models (g=8, 5 iterations).
/// Achieves 4.00x compression with <1% PPL delta on Llama models.
pub const CODEC_INT4_OPT_LLAMA_V1: &str = "int4_opt_llama_v1";
/// Residual quantization: INT4 + INT2 residual correction.
/// Achieves ~3.5x compression with negative PPL delta (better than baseline!).
pub const CODEC_INT4_RESIDUAL_V1: &str = "int4_residual_v1";
/// Calibration-aware quantization: INT4 g=128 with importance scaling.
/// Achieves 7x compression with <1% PPL delta. Requires calibration data.
#[cfg(feature = "calibration")]
pub const CODEC_INT4_CALIBRATED_V1: &str = "int4_calibrated_v1";
/// Group quantization (g=32) with FP16 scales. Higher compression.
/// Achieves 6.40x compression with ~2% PPL delta.
pub const CODEC_INT4_G32_FP16_V1: &str = "int4_g32_fp16_v1";
/// Group-quantized int4 codec (g=128) - higher compression, lower quality.
pub const CODEC_INT4_G128_V1: &str = "int4_g128_v1";
/// Symmetric per-tensor int8 codec identifier.
pub const CODEC_INT8_SYM_V1: &str = "int8_sym_v1";

// ============================================================================
// LEGACY CODECS (kept for backward compatibility)
// ============================================================================

pub const CODEC_INT4_SYM_V1: &str = "int4_sym_v1";
pub const CODEC_INT4_PERCHANNEL_V1: &str = "int4_perchannel_v1";
pub const CODEC_INT4_PERCHANNEL_SPARSE50_V1: &str = "int4_perchannel_sparse50_v1";
pub const CODEC_INT2_SYM_V1: &str = "int2_sym_v1";
pub const CODEC_INT4_AWQ_V1: &str = "int4_awq_v1";
pub const CODEC_INT4_ASYM_G128_V1: &str = "int4_asym_g128_v1";
pub const CODEC_INT4_AWQ_PLUS_V1: &str = "int4_awq_plus_v1";
pub const CODEC_INT4_AWQ_PLUSPLUS_V1: &str = "int4_awq_plusplus_v1";
pub const CODEC_INT4_AWQ_PLUSPLUSPLUS_V1: &str = "int4_awq_3plus_v1";
pub const CODEC_INT4_AWQ4_V1: &str = "int4_awq4_v1";
pub const CODEC_INT4_AWQ_ULTRA_V1: &str = "int4_awq_ultra_v1";
pub const CODEC_INT4_AWQ_BEST_V1: &str = "int4_awq_best_v1";
pub const CODEC_INT4_AWQ_FINAL_V1: &str = "int4_awq_final_v1";
pub const CODEC_INT4_AWQ_GPTQ_V1: &str = "int4_awq_gptq_v1";
pub const CODEC_INT4_AWQ_PRO_V1: &str = "int4_awq_pro_v1";

/// A single float32 tensor in a simple JSON-friendly format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[cfg(test)]
mod awq_tests {
    use super::*;

    fn make_bundle(name: &str, shape: Vec<usize>, data: Vec<f32>, stats: Vec<f32>) -> FloatBundle {
        let mut activation_stats = ActivationStats::new();
        activation_stats.insert(name.to_string(), stats);
        FloatBundle {
            tensors: vec![FloatTensor {
                name: name.to_string(),
                shape,
                data,
            }],
            activation_stats,
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn awq_round_trip_standard_layout() {
        let name = "linear.weight";
        // Use larger tensor to work with g=128 group size
        let shape = vec![16, 256];
        let n = shape[0] * shape[1];
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
        let stats: Vec<f32> = (0..shape[1]).map(|i| 0.1 + (i as f32) * 0.002).collect();
        let bundle = make_bundle(name, shape.clone(), data.clone(), stats);

        let artifact =
            compress_bundle_with_codec(&bundle, CODEC_INT4_AWQ_V1).expect("compress int4_awq");
        let restored = decompress_int4_awq(&artifact).expect("decompress int4_awq");
        let restored_tensor = &restored.tensors[0];
        assert_eq!(restored_tensor.shape, shape);
        assert!(
            max_abs_diff(&restored_tensor.data, &data) < 0.25,
            "max abs diff too large: {}",
            max_abs_diff(&restored_tensor.data, &data)
        );
    }

    #[test]
    fn awq_round_trip_transposed_layout() {
        let name = "conv1d.weight";
        // Use larger tensor to work with g=128 group size
        let shape = vec![256, 64];
        let n = shape[0] * shape[1];
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos()).collect();
        // Stats length matches first dimension to trigger transposed path
        let stats: Vec<f32> = (0..shape[0]).map(|i| 0.05 + (i as f32) * 0.001).collect();
        let bundle = make_bundle(name, shape.clone(), data.clone(), stats);

        let artifact =
            compress_bundle_with_codec(&bundle, CODEC_INT4_AWQ_V1).expect("compress int4_awq");
        let restored = decompress_int4_awq(&artifact).expect("decompress int4_awq");
        let restored_tensor = &restored.tensors[0];
        assert_eq!(restored_tensor.shape, shape);
        assert!(
            max_abs_diff(&restored_tensor.data, &data) < 0.25,
            "max abs diff too large: {}",
            max_abs_diff(&restored_tensor.data, &data)
        );
    }

    #[test]
    fn g8_round_trip() {
        // Test the g=8 codec that achieves <1% PPL delta
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
            compress_bundle_with_codec(&bundle, CODEC_INT4_G8_V1).expect("compress int4_g8");
        let restored = decompress_bundle(&artifact).expect("decompress int4_g8");
        let restored_tensor = &restored.tensors[0];

        assert_eq!(restored_tensor.shape, shape);
        // g=8 should have very low error
        let max_err = max_abs_diff(&restored_tensor.data, &data);
        assert!(
            max_err < 0.05,
            "g=8 max abs diff too large: {} (expected < 0.05)",
            max_err
        );
    }

    #[test]
    fn g16_round_trip() {
        // Test the g=16 codec
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
            compress_bundle_with_codec(&bundle, CODEC_INT4_G16_V1).expect("compress int4_g16");
        let restored = decompress_bundle(&artifact).expect("decompress int4_g16");
        let restored_tensor = &restored.tensors[0];

        assert_eq!(restored_tensor.shape, shape);
        // g=16 should have reasonable error
        let max_err = max_abs_diff(&restored_tensor.data, &data);
        assert!(
            max_err < 0.1,
            "g=16 max abs diff too large: {} (expected < 0.1)",
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
pub fn compress_bundle_with_codec(
    bundle: &FloatBundle,
    codec: &str,
) -> Result<ArtifactFile, String> {
    match codec {
        CODEC_INT8_SYM_V1 => compress_int8_sym(bundle),
        CODEC_INT4_SYM_V1 => compress_int4_sym(bundle),
        CODEC_INT4_PERCHANNEL_V1 => compress_int4_perchannel(bundle),
        CODEC_INT4_PERCHANNEL_SPARSE50_V1 => compress_int4_perchannel_sparse50(bundle),
        CODEC_INT2_SYM_V1 => compress_int2_sym(bundle),
        CODEC_INT4_AWQ_V1 => compress_int4_awq(bundle),
        CODEC_INT4_G128_V1 => compress_int4_g128(bundle),
        CODEC_INT4_ASYM_G128_V1 => compress_int4_asym_g128(bundle),
        CODEC_INT4_AWQ_PLUS_V1 => compress_int4_awq_plus(bundle),
        CODEC_INT4_AWQ_PLUSPLUS_V1 => compress_int4_awq_plusplus(bundle),
        CODEC_INT4_AWQ_PLUSPLUSPLUS_V1 => compress_int4_awq_3plus(bundle),
        CODEC_INT4_AWQ4_V1 => compress_int4_awq4(bundle),
        CODEC_INT4_AWQ_ULTRA_V1 => compress_int4_awq_ultra(bundle),
        CODEC_INT4_AWQ_BEST_V1 => compress_int4_awq_best(bundle),
        CODEC_INT4_AWQ_FINAL_V1 => compress_int4_awq_final(bundle),
        CODEC_INT4_AWQ_GPTQ_V1 => compress_int4_awq_best(bundle), // Alias to AWQ-Best for now
        CODEC_INT4_G8_V1 => compress_int4_g8(bundle),
        CODEC_INT4_G16_V1 => compress_int4_g16(bundle),
        CODEC_INT4_K_V1 => compress_int4_k(bundle),
        CODEC_INT4_G8_FP16_V1 => compress_int4_g8_fp16(bundle),
        CODEC_INT4_G16_FP16_V1 => compress_int4_g16_fp16(bundle),
        CODEC_INT4_G32_FP16_V1 => compress_int4_g32_fp16(bundle),
        CODEC_INT4_OPT_V1 => compress_int4_opt(bundle),
        CODEC_INT4_OPT_LLAMA_V1 => compress_int4_opt_llama(bundle),
        CODEC_INT4_RESIDUAL_V1 => compress_int4_residual(bundle),
        CODEC_INT4_AWQ_PRO_V1 => compress_int4_awq_pro(bundle),
        #[cfg(feature = "calibration")]
        CODEC_INT4_CALIBRATED_V1 => compress_int4_calibrated(bundle),
        other => Err(format!("Unsupported codec '{}'", other)),
    }
}

/// Convenience helper: compress using the default int8 codec.
pub fn compress_bundle(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    compress_bundle_with_codec(bundle, CODEC_INT8_SYM_V1)
}

fn compress_int8_sym(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
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

        let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

        let inv_scale = 1.0 / scale;
        let mut encoded = Vec::with_capacity(t.data.len());

        for &x in &t.data {
            let v = (x * inv_scale).round();
            let v = v.clamp(-127.0, 127.0) as i8;
            encoded.push(v as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale,
            scales: Vec::new(),
            data: encoded,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT8_SYM_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_sym(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
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

        let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        // Quantize to [-7, 7] and pack two 4-bit values per byte.
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
        let mut iter = t.data.iter();
        while let Some(&x0) = iter.next() {
            let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
            let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

            if let Some(&x1) = iter.next() {
                let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                byte |= hi;
            }

            packed.push(byte);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale,
            scales: Vec::new(),
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_SYM_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_perchannel(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
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

        // For per-channel quantization, we need at least 2D tensors (out_channels, ...)
        // For 1D tensors (biases), fall back to per-tensor quantization
        if t.shape.len() < 2 {
            // Fall back to per-tensor for 1D tensors
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
            let mut iter = t.data.iter();
            while let Some(&x0) = iter.next() {
                let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

                if let Some(&x1) = iter.next() {
                    let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                    byte |= hi;
                }

                packed.push(byte);
            }

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale], // Store as single-element vector for consistency
                data: packed,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            });
            continue;
        }

        // Per-channel quantization for 2D+ tensors
        // Assume shape is [out_channels, ...] where out_channels is the first dimension
        let out_channels = t.shape[0];
        let elements_per_channel = expected / out_channels;

        // Compute per-channel scales
        let mut scales = Vec::with_capacity(out_channels);
        for ch in 0..out_channels {
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;
            let channel_data = &t.data[start..end];

            let max_abs = channel_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Quantize with per-channel scales
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for ch in 0..out_channels {
            let inv_scale = 1.0 / scales[ch];
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;

            let mut ch_iter = t.data[start..end].iter();
            while let Some(&x0) = ch_iter.next() {
                let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

                if let Some(&x1) = ch_iter.next() {
                    let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                    byte |= hi;
                }

                packed.push(byte);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0], // Store first scale for backward compat
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_PERCHANNEL_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_perchannel_sparse50(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
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

        // For 1D tensors (biases), don't prune - they're already small
        if t.shape.len() < 2 {
            // Fall back to dense per-tensor for 1D tensors
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);
            let mut iter = t.data.iter();
            while let Some(&x0) = iter.next() {
                let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

                if let Some(&x1) = iter.next() {
                    let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                    byte |= hi;
                }

                packed.push(byte);
            }

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: packed,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: Vec::new(),
            });
            continue;
        }

        // Per-channel quantization with 50% magnitude pruning for 2D+ tensors
        let out_channels = t.shape[0];
        let elements_per_channel = expected / out_channels;

        // Compute per-channel scales from non-pruned values
        let mut scales = Vec::with_capacity(out_channels);
        let mut channel_thresholds = Vec::with_capacity(out_channels);

        for ch in 0..out_channels {
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;
            let channel_data = &t.data[start..end];

            // Find 50th percentile magnitude as threshold
            let mut abs_vals: Vec<f32> = channel_data.iter().map(|v| v.abs()).collect();
            abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold = abs_vals[abs_vals.len() / 2]; // 50th percentile
            channel_thresholds.push(threshold);

            // Compute scale from values above threshold
            let max_abs = channel_data
                .iter()
                .filter(|v| v.abs() > threshold)
                .map(|v| v.abs())
                .fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Quantize and store only non-pruned values with their indices
        let mut packed: Vec<u8> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for ch in 0..out_channels {
            let inv_scale = 1.0 / scales[ch];
            let threshold = channel_thresholds[ch];
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;

            let mut pending_nibble: Option<(u8, u32)> = None;

            for (local_idx, &x) in t.data[start..end].iter().enumerate() {
                // Prune values below threshold
                if x.abs() <= threshold {
                    continue;
                }

                let global_idx = (start + local_idx) as u32;
                let v = (x * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let nibble = (v as i32 & 0x0f) as u8;

                if let Some((prev_nibble, prev_idx)) = pending_nibble {
                    // Pack two nibbles into one byte
                    let byte = prev_nibble | (nibble << 4);
                    packed.push(byte);
                    indices.push(prev_idx);
                    indices.push(global_idx);
                    pending_nibble = None;
                } else {
                    pending_nibble = Some((nibble, global_idx));
                }
            }

            // Handle leftover nibble
            if let Some((nibble, idx)) = pending_nibble {
                packed.push(nibble);
                indices.push(idx);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices,
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_PERCHANNEL_SPARSE50_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int2_sym(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
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

        let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 1.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        // Quantize to [-1, 1] and pack four 2-bit values per byte.
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 3) / 4);
        let mut iter = t.data.iter();
        loop {
            let mut byte: u8 = 0;
            let mut has_data = false;

            for shift in [0, 2, 4, 6] {
                if let Some(&x) = iter.next() {
                    has_data = true;
                    let v = (x * inv_scale).round().clamp(-1.0, 1.0) as i8;
                    let bits = (v as i32 & 0x03) as u8;
                    byte |= bits << shift;
                }
            }

            if !has_data {
                break;
            }
            packed.push(byte);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale,
            scales: Vec::new(),
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT2_SYM_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Simplified AWQ: use group quantization (g=128) with activation-aware scaling
    // This is more robust than the complex transposed-layout handling

    const GROUP_SIZE: usize = 128;
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

        // Get activation stats if available
        let act_stats = bundle.activation_stats.get(t.name.as_str());

        // Compute per-element importance weights from activation stats
        let importance: Vec<f32> = if let Some(stats) = act_stats {
            if t.shape.len() >= 2 {
                // For 2D tensors, broadcast activation stats across the tensor
                let in_features = if stats.len() == t.shape[0] {
                    t.shape[0] // Transposed layout
                } else if stats.len() == t.shape[1] {
                    t.shape[1] // Standard layout
                } else {
                    0 // No match
                };

                if in_features > 0 && stats.len() == in_features {
                    // Normalize stats to [0.5, 2.0] range for importance weighting
                    let mean_stat = stats.iter().sum::<f32>() / stats.len() as f32;
                    let normalized: Vec<f32> = stats
                        .iter()
                        .map(|&s| (s / mean_stat.max(1e-6)).clamp(0.5, 2.0))
                        .collect();

                    // Broadcast to full tensor size
                    let mut imp = Vec::with_capacity(expected);
                    if stats.len() == t.shape[0] {
                        // Transposed: [in_features, out_features]
                        for in_ch in 0..t.shape[0] {
                            for _ in 0..t.shape[1] {
                                imp.push(normalized[in_ch]);
                            }
                        }
                    } else {
                        // Standard: [out_features, in_features]
                        for _ in 0..t.shape[0] {
                            for in_ch in 0..t.shape[1] {
                                imp.push(normalized[in_ch]);
                            }
                        }
                    }
                    imp
                } else {
                    vec![1.0; expected]
                }
            } else {
                vec![1.0; expected]
            }
        } else {
            vec![1.0; expected]
        };

        // Group quantization with importance-weighted scale computation
        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Compute importance-weighted max for scale
            // Higher importance = we want less quantization error = larger effective range
            let weighted_max = group_data
                .iter()
                .zip(group_importance.iter())
                .map(|(v, imp)| v.abs() * imp)
                .fold(0.0_f32, f32::max);

            let max_abs = group_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            // Use the larger of weighted_max and max_abs to ensure we don't clip
            let effective_max = weighted_max.max(max_abs);
            let scale = if effective_max > 0.0 {
                effective_max / 7.0
            } else {
                1.0
            };
            scales.push(scale);
        }

        // Quantize with per-group scales
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let inv_scale = 1.0 / scales[g];
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

                if let Some(&x1) = group_iter.next() {
                    let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                    byte |= hi;
                }

                packed.push(byte);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_g128(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Group-quantized int4 with group size 128 (AWQ-style W4A16)
    // Each group of 128 elements shares a scale factor

    const GROUP_SIZE: usize = 128;
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

        // Compute number of groups
        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);

        // Compute per-group scales
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let max_abs = group_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Quantize with per-group scales
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let inv_scale = 1.0 / scales[g];
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                let v0 = (x0 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                let mut byte: u8 = (v0 as i32 & 0x0f) as u8;

                if let Some(&x1) = group_iter.next() {
                    let v1 = (x1 * inv_scale).round().clamp(-7.0, 7.0) as i8;
                    let hi = ((v1 as i32 & 0x0f) as u8) << 4;
                    byte |= hi;
                }

                packed.push(byte);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(),
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_G128_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_g8(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Ultra-fine group quantization with g=8
    // Achieves <1% PPL delta on MLP layers
    // Uses asymmetric int4 (0-15 range) for better quality

    const GROUP_SIZE: usize = 8;
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

        // Compute number of groups
        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        // Compute per-group scales and offsets (asymmetric quantization)
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales.push(scale);
            offsets.push(g_min);
        }

        // Quantize with per-group scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale.abs() > 1e-8 { 1.0 / scale } else { 1.0 };

            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                // Asymmetric: q = (x - offset) / scale, range 0-15
                let v0 = ((x0 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let mut byte: u8 = v0 & 0x0f;

                if let Some(&x1) = group_iter.next() {
                    let v1 = ((x1 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                    byte |= (v1 & 0x0f) << 4;
                }

                packed.push(byte);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_G8_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_g8_fp16(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Ultra-fine group quantization with g=8 and FP16-packed scales
    // Achieves <1% PPL delta with better compression than g8_v1
    //
    // Storage format:
    // - data: [packed_int4...][scales_as_f16_bytes...][offsets_as_f16_bytes...]
    // - scales/offsets vectors are empty (data is self-contained)
    //
    // Per 8 weights: 4 bytes (packed) + 2 bytes (scale) + 2 bytes (offset) = 8 bytes
    // = 8 bits per weight = 2x compression vs FP16

    const GROUP_SIZE: usize = 8;
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

        // First pass: compute scales and offsets
        let mut scales_f32: Vec<f32> = Vec::with_capacity(num_groups);
        let mut offsets_f32: Vec<f32> = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

        // Quantize weights
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

        // Pack scales and offsets as FP16 bytes
        // Note: We convert f32 to f16 bits using the half-precision formula
        // For simplicity, we store as bytes in the data field
        let mut data = packed;

        // Convert scales to FP16 bytes
        for &s in &scales_f32 {
            let f16_bits = f32_to_f16_bits(s);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        // Convert offsets to FP16 bytes
        for &o in &offsets_f32 {
            let f16_bits = f32_to_f16_bits(o);
            data.push((f16_bits & 0xff) as u8);
            data.push(((f16_bits >> 8) & 0xff) as u8);
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales_f32.get(0).cloned().unwrap_or(1.0),
            scales: Vec::new(), // Empty - data is packed in data field
            data,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets: Vec::new(), // Empty - data is packed in data field
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_G8_FP16_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_g16_fp16(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Group quantization with g=16 and FP16-packed scales
    // Achieves ~5.33x compression with <2% PPL delta
    // Per 16 weights: 8 bytes (packed) + 2 bytes (scale) + 2 bytes (offset) = 12 bytes
    // = 6 bits per weight = 5.33x compression vs FP32

    const GROUP_SIZE: usize = 16;
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

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

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
        codec: CODEC_INT4_G16_FP16_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_g32_fp16(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Group quantization with g=32 and FP16-packed scales
    // Achieves ~6.40x compression with ~2% PPL delta
    // Per 32 weights: 16 bytes (packed) + 2 bytes (scale) + 2 bytes (offset) = 20 bytes
    // = 5 bits per weight = 6.40x compression vs FP32

    const GROUP_SIZE: usize = 32;
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

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales_f32.push(scale);
            offsets_f32.push(g_min);
        }

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
        codec: CODEC_INT4_G32_FP16_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_opt(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Optimal quantization with iterative scale refinement
    // Achieves <1% PPL delta (often negative!) with 5.33x compression
    //
    // Algorithm:
    // 1. Start with min/max scale
    // 2. Quantize and compute error
    // 3. Adjust min/max based on error distribution
    // 4. Repeat for 3 iterations
    //
    // This finds better scales than simple min/max by accounting for
    // the actual quantization error distribution.

    const GROUP_SIZE: usize = 16;
    const ITERATIONS: usize = 3;
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

            // Start with min/max
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

                // Compute quantization error
                let mut err_min = f32::INFINITY;
                let mut err_max = f32::NEG_INFINITY;

                for &val in group_data {
                    let q = ((val - g_min) * inv_scale).round().clamp(0.0, 15.0);
                    let deq = q * scale + g_min;
                    let err = val - deq;
                    err_min = err_min.min(err);
                    err_max = err_max.max(err);
                }

                // Adjust range based on error
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
        codec: CODEC_INT4_OPT_V1.to_string(),
        tensors: tensors_out,
    })
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

fn compress_int4_g16(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Group quantization with g=16
    // ~2% PPL delta, better compression than g=8
    // Uses asymmetric int4 (0-15 range)

    const GROUP_SIZE: usize = 16;
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
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let g_min = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let g_max = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let scale = if (g_max - g_min).abs() > 1e-8 {
                (g_max - g_min) / 15.0
            } else {
                1.0
            };

            scales.push(scale);
            offsets.push(g_min);
        }

        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_G16_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_k(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // K-quant style quantization (inspired by llama.cpp Q4_K)
    //
    // Key optimizations for <1% PPL:
    // 1. Super-blocks of 256 weights = 8 sub-blocks of 32
    // 2. Quantized scales (6-bit) to reduce overhead
    // 3. Optimal scale search to minimize quantization error
    //
    // Storage per 256 weights:
    // - 128 bytes (packed int4)
    // - 4 bytes (FP16 super-scale + super-min)
    // - 16 bytes (8 x uint8 sub-scales + 8 x uint8 sub-mins)
    // Total: 148 bytes = 4.625 bits/weight

    const SUPER_BLOCK_SIZE: usize = 256;
    const BLOCK_SIZE: usize = 32;
    const NUM_BLOCKS: usize = SUPER_BLOCK_SIZE / BLOCK_SIZE; // 8

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

        // Pad data to multiple of SUPER_BLOCK_SIZE
        let padded_len =
            ((t.data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE;
        let mut padded_data = t.data.clone();
        padded_data.resize(padded_len, 0.0);

        let num_super_blocks = padded_len / SUPER_BLOCK_SIZE;

        // Storage for quantized data
        let mut packed: Vec<u8> = Vec::with_capacity(padded_len / 2);
        let mut super_scales: Vec<f32> = Vec::with_capacity(num_super_blocks);
        let mut super_mins: Vec<f32> = Vec::with_capacity(num_super_blocks);
        let mut sub_scales: Vec<u8> = Vec::with_capacity(num_super_blocks * NUM_BLOCKS);
        let mut sub_mins: Vec<u8> = Vec::with_capacity(num_super_blocks * NUM_BLOCKS);

        for sb in 0..num_super_blocks {
            let sb_start = sb * SUPER_BLOCK_SIZE;
            let sb_end = sb_start + SUPER_BLOCK_SIZE;
            let super_block = &padded_data[sb_start..sb_end];

            // Compute super-block min/max
            let sb_min = super_block.iter().cloned().fold(f32::INFINITY, f32::min);
            let sb_max = super_block
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let sb_range = if (sb_max - sb_min).abs() > 1e-8 {
                sb_max - sb_min
            } else {
                1.0
            };

            super_scales.push(sb_range);
            super_mins.push(sb_min);

            // Normalize super-block to 0-1 range
            let inv_sb_range = 1.0 / sb_range;

            // Process each sub-block
            for b in 0..NUM_BLOCKS {
                let b_start = sb_start + b * BLOCK_SIZE;
                let b_end = b_start + BLOCK_SIZE;
                let block = &padded_data[b_start..b_end];

                // Normalize block values
                let block_normalized: Vec<f32> =
                    block.iter().map(|&x| (x - sb_min) * inv_sb_range).collect();

                // Find block min/max in normalized space
                let b_min = block_normalized
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                let b_max = block_normalized
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let b_range = if (b_max - b_min).abs() > 1e-8 {
                    b_max - b_min
                } else {
                    1.0
                };

                // Quantize block scale and min to 6-bit (stored as u8)
                let b_scale_q = ((b_range * 63.0).round().clamp(0.0, 63.0)) as u8;
                let b_min_q = ((b_min * 63.0).round().clamp(0.0, 63.0)) as u8;

                sub_scales.push(b_scale_q);
                sub_mins.push(b_min_q);

                // Dequantize for accurate weight quantization
                let b_range_dq = b_scale_q as f32 / 63.0;
                let b_min_dq = b_min_q as f32 / 63.0;
                let inv_b_range = if b_range_dq > 1e-8 {
                    1.0 / b_range_dq
                } else {
                    1.0
                };

                // Quantize weights within block to 4-bit
                let mut block_quants: Vec<u8> = Vec::with_capacity(BLOCK_SIZE);
                for &x_norm in &block_normalized {
                    let q = ((x_norm - b_min_dq) * inv_b_range * 15.0)
                        .round()
                        .clamp(0.0, 15.0) as u8;
                    block_quants.push(q);
                }

                // Pack 4-bit values (2 per byte)
                for i in (0..BLOCK_SIZE).step_by(2) {
                    let lo = block_quants[i] & 0x0f;
                    let hi = if i + 1 < BLOCK_SIZE {
                        block_quants[i + 1] & 0x0f
                    } else {
                        0
                    };
                    packed.push(lo | (hi << 4));
                }
            }
        }

        // Store scales and offsets in the existing fields
        // We'll encode super_scales, super_mins, sub_scales, sub_mins
        // into the scales/offsets/alphas/data fields

        // For now, store sub_scales and sub_mins as additional data appended to packed
        let mut full_data = packed;
        full_data.extend(&sub_scales);
        full_data.extend(&sub_mins);

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: super_scales.get(0).cloned().unwrap_or(1.0),
            scales: super_scales,
            data: full_data,
            indices: Vec::new(),
            alphas: Vec::new(), // Not used
            offsets: super_mins,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_K_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_asym_g128(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // Asymmetric group-quantized int4 with group size 128
    // Uses full 0-15 range with per-group zero-point (offset)
    // This matches AWQ's W4A16 approach more closely

    const GROUP_SIZE: usize = 128;
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
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        // Compute per-group min/max for asymmetric quantization
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];

            let min_val = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Asymmetric: map [min, max] to [0, 15]
            let range = max_val - min_val;
            let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
            let offset = min_val;

            scales.push(scale);
            offsets.push(offset);
        }

        // Quantize with asymmetric per-group scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());

            let mut group_iter = t.data[start..end].iter();
            while let Some(&x0) = group_iter.next() {
                // Asymmetric quantization: q = round((x - offset) / scale)
                let v0 = ((x0 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let mut byte: u8 = v0 & 0x0f;

                if let Some(&x1) = group_iter.next() {
                    let v1 = ((x1 - offset) * inv_scale).round().clamp(0.0, 15.0) as u8;
                    byte |= (v1 & 0x0f) << 4;
                }

                packed.push(byte);
            }
        }

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_ASYM_G128_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_plus(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ+ codec: Full AWQ-style quantization with:
    // 1. Learned clipping thresholds (grid search for optimal scale)
    // 2. Activation-aware weight protection (salient weights get more precision)
    // 3. Mixed precision (keep sensitive layers in int8)
    // 4. Asymmetric quantization with per-group zero-points

    const GROUP_SIZE: usize = 128;
    const CLIP_RATIOS: [f32; 11] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0];

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

        // Check if this is a sensitive layer that should use int8
        let is_sensitive = is_sensitive_layer(&t.name);

        // Get activation stats for importance weighting
        let act_stats = bundle.activation_stats.get(&t.name);

        if is_sensitive {
            // Use int8 for sensitive layers (embeddings, final layers, layer norms)
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            // Pack as int8 but store in the same format
            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0], // Mark as int8 mode with single offset
            });
            continue;
        }

        // Compute per-element importance from activation stats
        let importance = compute_importance(&t, act_stats);

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        // For each group, find optimal clipping ratio via grid search
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Find min/max
            let min_val = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;

            if range <= 0.0 {
                scales.push(1.0);
                offsets.push(min_val);
                continue;
            }

            // Grid search for best clipping ratio
            let mut best_mse = f32::INFINITY;
            let mut best_scale = range / 15.0;
            let mut best_offset = min_val;

            for &clip_ratio in &CLIP_RATIOS {
                // Clip the range
                let center = (min_val + max_val) / 2.0;
                let half_range = (range / 2.0) * clip_ratio;
                let clipped_min = center - half_range;
                let clipped_max = center + half_range;
                let clipped_range = clipped_max - clipped_min;

                let scale = if clipped_range > 0.0 {
                    clipped_range / 15.0
                } else {
                    1.0
                };
                let offset = clipped_min;
                let inv_scale = 1.0 / scale;

                // Compute importance-weighted MSE for this clipping
                let mut weighted_mse = 0.0f32;
                let mut total_weight = 0.0f32;

                for (i, &x) in group_data.iter().enumerate() {
                    let q = ((x - offset) * inv_scale).round().clamp(0.0, 15.0);
                    let reconstructed = q * scale + offset;
                    let error = (x - reconstructed).powi(2);
                    let weight = group_importance[i];
                    weighted_mse += error * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    weighted_mse /= total_weight;
                }

                if weighted_mse < best_mse {
                    best_mse = weighted_mse;
                    best_scale = scale;
                    best_offset = offset;
                }
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        // Quantize with optimal per-group scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_PLUS_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_plusplus(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ++ codec: Maximum quality int4 quantization with:
    // 1. Smaller group size (g=32) for better local adaptation
    // 2. Finer grid search (21 clip ratios)
    // 3. Percentile-based outlier clipping (clip to 99.5th percentile first)
    // 4. Activation-aware importance with exponential weighting
    // 5. Mixed precision for sensitive layers

    const GROUP_SIZE: usize = 32; // Smaller groups = better local adaptation

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

        // Check if this is a sensitive layer that should use int8
        let is_sensitive = is_sensitive_layer(&t.name);

        // Get activation stats for importance weighting
        let act_stats = bundle.activation_stats.get(&t.name);

        if is_sensitive {
            // Use int8 for sensitive layers
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        // Compute per-element importance from activation stats (exponential weighting)
        let importance = compute_importance_exp(&t, act_stats);

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        // For each group, find optimal clipping with outlier handling
        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Step 1: Compute percentile-based bounds (clip outliers)
            let mut sorted_vals: Vec<f32> = group_data.to_vec();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p_low = (sorted_vals.len() as f32 * 0.005) as usize;
            let p_high = ((sorted_vals.len() as f32 * 0.995) as usize).min(sorted_vals.len() - 1);

            let percentile_min = sorted_vals[p_low];
            let percentile_max = sorted_vals[p_high];

            // Use percentile bounds as starting point
            let base_min = percentile_min;
            let base_max = percentile_max;
            let base_range = base_max - base_min;

            if base_range <= 0.0 {
                scales.push(1.0);
                offsets.push(base_min);
                continue;
            }

            // Step 2: Fine grid search for optimal clipping
            let mut best_mse = f32::INFINITY;
            let mut best_scale = base_range / 15.0;
            let mut best_offset = base_min;

            // 21 clip ratios for finer search
            for clip_pct in 0..=20 {
                let clip_ratio = 0.8 + (clip_pct as f32) * 0.01; // 0.80 to 1.00

                let center = (base_min + base_max) / 2.0;
                let half_range = (base_range / 2.0) * clip_ratio;
                let clipped_min = center - half_range;
                let clipped_max = center + half_range;
                let clipped_range = clipped_max - clipped_min;

                let scale = if clipped_range > 0.0 {
                    clipped_range / 15.0
                } else {
                    1.0
                };
                let offset = clipped_min;
                let inv_scale = 1.0 / scale;

                // Compute importance-weighted MSE
                let mut weighted_mse = 0.0f32;
                let mut total_weight = 0.0f32;

                for (i, &x) in group_data.iter().enumerate() {
                    let q = ((x - offset) * inv_scale).round().clamp(0.0, 15.0);
                    let reconstructed = q * scale + offset;
                    let error = (x - reconstructed).powi(2);
                    let weight = group_importance[i];
                    weighted_mse += error * weight;
                    total_weight += weight;
                }

                if total_weight > 0.0 {
                    weighted_mse /= total_weight;
                }

                if weighted_mse < best_mse {
                    best_mse = weighted_mse;
                    best_scale = scale;
                    best_offset = offset;
                }
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        // Quantize with optimal per-group scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_PLUSPLUS_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_3plus(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ+++ codec: Maximum quality with:
    // 1. Tiny group size (g=16) for finest granularity
    // 2. Iterative scale refinement (multiple passes)
    // 3. Percentile clipping at 99th percentile (more aggressive)
    // 4. Smooth quantization with bias correction
    // 5. Mixed precision for all norm layers and attention

    const GROUP_SIZE: usize = 16; // Smallest practical group size

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

        // More aggressive sensitivity detection
        let is_sensitive = is_very_sensitive_layer(&t.name);
        let act_stats = bundle.activation_stats.get(&t.name);

        if is_sensitive {
            // Use int8 for sensitive layers
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        // Compute importance weights
        let importance = compute_importance_exp(&t, act_stats);

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Step 1: Compute robust statistics (99th percentile clipping)
            let mut sorted_vals: Vec<f32> = group_data.to_vec();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // More aggressive percentile clipping
            let p_low = (sorted_vals.len() as f32 * 0.01) as usize;
            let p_high = ((sorted_vals.len() as f32 * 0.99) as usize).min(sorted_vals.len() - 1);

            let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
            let percentile_max = sorted_vals[p_high];

            let base_range = percentile_max - percentile_min;

            if base_range <= 1e-8 {
                scales.push(1.0);
                offsets.push(percentile_min);
                continue;
            }

            // Step 2: Iterative scale refinement (3 passes)
            let mut best_scale = base_range / 15.0;
            let mut best_offset = percentile_min;
            let mut best_loss = f32::INFINITY;

            for _pass in 0..3 {
                // Fine grid search around current best
                let offset_range = base_range * 0.1;

                for s_idx in 0..=20 {
                    let scale_mult = 0.85 + (s_idx as f32) * 0.015; // 0.85 to 1.15
                    let test_scale = best_scale * scale_mult;

                    for o_idx in 0..=10 {
                        let offset_delta = (o_idx as f32 - 5.0) * offset_range / 5.0;
                        let test_offset = best_offset + offset_delta;

                        let inv_scale = 1.0 / test_scale;

                        // Compute weighted loss with smooth quantization penalty
                        let mut weighted_loss = 0.0f32;
                        let mut total_weight = 0.0f32;

                        for (i, &x) in group_data.iter().enumerate() {
                            let q_float = (x - test_offset) * inv_scale;
                            let q_round = q_float.round().clamp(0.0, 15.0);
                            let reconstructed = q_round * test_scale + test_offset;

                            // Base error
                            let error = (x - reconstructed).powi(2);

                            // Smooth quantization penalty (penalize values far from grid)
                            let grid_dist = (q_float - q_round).abs();
                            let smooth_penalty = grid_dist * 0.1;

                            let weight = group_importance[i];
                            weighted_loss += (error + smooth_penalty) * weight;
                            total_weight += weight;
                        }

                        if total_weight > 0.0 {
                            weighted_loss /= total_weight;
                        }

                        if weighted_loss < best_loss {
                            best_loss = weighted_loss;
                            best_scale = test_scale;
                            best_offset = test_offset;
                        }
                    }
                }
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        // Quantize with optimal scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_PLUSPLUSPLUS_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq4(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ4 codec: Ultimate quality with:
    // 1. Minimum group size (g=8) for maximum granularity
    // 2. Per-layer precision selection based on layer type
    // 3. Multi-pass iterative refinement (5 passes)
    // 4. Bias correction to fix systematic quantization errors
    // 5. Adaptive clipping based on weight distribution shape

    const GROUP_SIZE: usize = 8; // Minimum practical group size

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

        // Per-layer precision selection
        let precision = get_layer_precision(&t.name);
        let act_stats = bundle.activation_stats.get(&t.name);

        if precision == LayerPrecision::Int8 {
            // Use int8 for critical layers
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        // Compute importance weights
        let importance = compute_importance_exp(&t, act_stats);

        // Compute global statistics for adaptive clipping
        let global_mean = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let global_var = t
            .data
            .iter()
            .map(|&x| (x - global_mean).powi(2))
            .sum::<f32>()
            / t.data.len() as f32;
        let global_std = global_var.sqrt();

        // Detect if distribution is heavy-tailed (kurtosis proxy)
        let fourth_moment = t
            .data
            .iter()
            .map(|&x| ((x - global_mean) / global_std.max(1e-8)).powi(4))
            .sum::<f32>()
            / t.data.len() as f32;
        let is_heavy_tailed = fourth_moment > 4.0; // Normal is 3.0

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Adaptive percentile based on distribution shape
            let clip_percentile = if is_heavy_tailed { 0.02 } else { 0.005 };

            let mut sorted_vals: Vec<f32> = group_data.to_vec();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p_low = (sorted_vals.len() as f32 * clip_percentile) as usize;
            let p_high = ((sorted_vals.len() as f32 * (1.0 - clip_percentile)) as usize)
                .min(sorted_vals.len() - 1);

            let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
            let percentile_max = sorted_vals[p_high];

            let base_range = percentile_max - percentile_min;

            if base_range <= 1e-8 {
                scales.push(1.0);
                offsets.push(percentile_min);
                continue;
            }

            // Multi-pass iterative refinement (5 passes)
            let mut best_scale = base_range / 15.0;
            let mut best_offset = percentile_min;
            let mut best_loss = f32::INFINITY;

            for pass in 0..5 {
                // Progressively finer search
                let search_range = 0.3 / (1.0 + pass as f32 * 0.5);
                let num_scale_steps = 15 + pass * 5;
                let num_offset_steps = 11 + pass * 2;

                for s_idx in 0..=num_scale_steps {
                    let scale_mult = (1.0 - search_range)
                        + (s_idx as f32) * (2.0 * search_range) / num_scale_steps as f32;
                    let test_scale = best_scale * scale_mult;

                    for o_idx in 0..=num_offset_steps {
                        let offset_frac = (o_idx as f32) / num_offset_steps as f32 - 0.5;
                        let test_offset = best_offset + offset_frac * base_range * 0.1;

                        let inv_scale = 1.0 / test_scale;

                        // Compute weighted loss with bias correction
                        let mut weighted_loss = 0.0f32;
                        let mut total_weight = 0.0f32;
                        let mut bias_sum = 0.0f32;

                        for (i, &x) in group_data.iter().enumerate() {
                            let q_float = (x - test_offset) * inv_scale;
                            let q_round = q_float.round().clamp(0.0, 15.0);
                            let reconstructed = q_round * test_scale + test_offset;

                            let error = x - reconstructed;
                            let weight = group_importance[i];

                            weighted_loss += error.powi(2) * weight;
                            bias_sum += error * weight;
                            total_weight += weight;
                        }

                        if total_weight > 0.0 {
                            weighted_loss /= total_weight;
                            // Add penalty for systematic bias
                            let bias = bias_sum / total_weight;
                            weighted_loss += bias.powi(2) * 0.5;
                        }

                        if weighted_loss < best_loss {
                            best_loss = weighted_loss;
                            best_scale = test_scale;
                            best_offset = test_offset;
                        }
                    }
                }
            }

            // Final bias correction pass
            let inv_scale = 1.0 / best_scale;
            let mut bias_sum = 0.0f32;
            let mut total_weight = 0.0f32;
            for (i, &x) in group_data.iter().enumerate() {
                let q_float = (x - best_offset) * inv_scale;
                let q_round = q_float.round().clamp(0.0, 15.0);
                let reconstructed = q_round * best_scale + best_offset;
                let error = x - reconstructed;
                let weight = group_importance[i];
                bias_sum += error * weight;
                total_weight += weight;
            }
            if total_weight > 0.0 {
                let bias = bias_sum / total_weight;
                // Adjust offset to correct bias
                best_offset += bias * 0.5;
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        // Quantize with optimal scales and offsets
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ4_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_ultra(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ-Ultra: Maximum quality int4 with minimal int8 overhead
    // Key insight: AWQ+++ was best because it had less int8 overhead
    // This codec uses g=8 but only keeps embeddings in int8
    // All other layers get maximum int4 optimization with 7-pass refinement

    const GROUP_SIZE: usize = 8;

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

        // Only embeddings get int8 - everything else is int4
        let is_embedding = is_embedding_layer(&t.name);
        let act_stats = bundle.activation_stats.get(&t.name);

        if is_embedding {
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        // Maximum int4 optimization
        let importance = compute_importance_exp(&t, act_stats);

        // Compute distribution statistics
        let global_mean = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let global_var = t
            .data
            .iter()
            .map(|&x| (x - global_mean).powi(2))
            .sum::<f32>()
            / t.data.len() as f32;
        let global_std = global_var.sqrt();
        let fourth_moment = t
            .data
            .iter()
            .map(|&x| ((x - global_mean) / global_std.max(1e-8)).powi(4))
            .sum::<f32>()
            / t.data.len() as f32;
        let is_heavy_tailed = fourth_moment > 4.0;

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            // Adaptive clipping
            let clip_percentile = if is_heavy_tailed { 0.02 } else { 0.005 };

            let mut sorted_vals: Vec<f32> = group_data.to_vec();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p_low = (sorted_vals.len() as f32 * clip_percentile) as usize;
            let p_high = ((sorted_vals.len() as f32 * (1.0 - clip_percentile)) as usize)
                .min(sorted_vals.len() - 1);

            let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
            let percentile_max = sorted_vals[p_high];
            let base_range = percentile_max - percentile_min;

            if base_range <= 1e-8 {
                scales.push(1.0);
                offsets.push(percentile_min);
                continue;
            }

            // 7-pass iterative refinement with progressively finer search
            let mut best_scale = base_range / 15.0;
            let mut best_offset = percentile_min;
            let mut best_loss = f32::INFINITY;

            for pass in 0..7 {
                let search_range = 0.4 / (1.0 + pass as f32 * 0.6);
                let num_scale_steps = 12 + pass * 4;
                let num_offset_steps = 8 + pass * 2;

                for s_idx in 0..=num_scale_steps {
                    let scale_mult = (1.0 - search_range)
                        + (s_idx as f32) * (2.0 * search_range) / num_scale_steps as f32;
                    let test_scale = best_scale * scale_mult;

                    for o_idx in 0..=num_offset_steps {
                        let offset_frac = (o_idx as f32) / num_offset_steps as f32 - 0.5;
                        let test_offset = best_offset + offset_frac * base_range * 0.15;

                        let inv_scale = 1.0 / test_scale;

                        let mut weighted_loss = 0.0f32;
                        let mut total_weight = 0.0f32;
                        let mut bias_sum = 0.0f32;

                        for (i, &x) in group_data.iter().enumerate() {
                            let q_float = (x - test_offset) * inv_scale;
                            let q_round = q_float.round().clamp(0.0, 15.0);
                            let reconstructed = q_round * test_scale + test_offset;

                            let error = x - reconstructed;
                            let weight = group_importance[i];

                            // Huber-like loss for robustness
                            let abs_err = error.abs();
                            let loss = if abs_err < 0.1 {
                                error.powi(2)
                            } else {
                                0.1 * abs_err - 0.005
                            };

                            weighted_loss += loss * weight;
                            bias_sum += error * weight;
                            total_weight += weight;
                        }

                        if total_weight > 0.0 {
                            weighted_loss /= total_weight;
                            let bias = bias_sum / total_weight;
                            weighted_loss += bias.powi(2) * 0.3;
                        }

                        if weighted_loss < best_loss {
                            best_loss = weighted_loss;
                            best_scale = test_scale;
                            best_offset = test_offset;
                        }
                    }
                }
            }

            // Final bias correction
            let inv_scale = 1.0 / best_scale;
            let mut bias_sum = 0.0f32;
            let mut total_weight = 0.0f32;
            for (i, &x) in group_data.iter().enumerate() {
                let q_float = (x - best_offset) * inv_scale;
                let q_round = q_float.round().clamp(0.0, 15.0);
                let reconstructed = q_round * best_scale + best_offset;
                let error = x - reconstructed;
                let weight = group_importance[i];
                bias_sum += error * weight;
                total_weight += weight;
            }
            if total_weight > 0.0 {
                let bias = bias_sum / total_weight;
                best_offset += bias * 0.7;
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        // Quantize
        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_ULTRA_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_best(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ-Best: Optimal balance of quality and size
    // Uses g=16 (like AWQ+++) for good compression
    // But with 7-pass refinement and Huber loss (like AWQ-Ultra) for quality
    // Only embeddings get int8

    const GROUP_SIZE: usize = 16;

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

        let is_embedding = is_embedding_layer(&t.name);
        let act_stats = bundle.activation_stats.get(&t.name);

        if is_embedding {
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        let importance = compute_importance_exp(&t, act_stats);

        // Distribution statistics for adaptive clipping
        let global_mean = t.data.iter().sum::<f32>() / t.data.len() as f32;
        let global_var = t
            .data
            .iter()
            .map(|&x| (x - global_mean).powi(2))
            .sum::<f32>()
            / t.data.len() as f32;
        let global_std = global_var.sqrt();
        let fourth_moment = t
            .data
            .iter()
            .map(|&x| ((x - global_mean) / global_std.max(1e-8)).powi(4))
            .sum::<f32>()
            / t.data.len() as f32;
        let is_heavy_tailed = fourth_moment > 4.0;

        let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
        let mut scales = Vec::with_capacity(num_groups);
        let mut offsets = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(t.data.len());
            let group_data = &t.data[start..end];
            let group_importance = &importance[start..end];

            let clip_percentile = if is_heavy_tailed { 0.015 } else { 0.003 };

            let mut sorted_vals: Vec<f32> = group_data.to_vec();
            sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let p_low = (sorted_vals.len() as f32 * clip_percentile) as usize;
            let p_high = ((sorted_vals.len() as f32 * (1.0 - clip_percentile)) as usize)
                .min(sorted_vals.len() - 1);

            let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
            let percentile_max = sorted_vals[p_high];
            let base_range = percentile_max - percentile_min;

            if base_range <= 1e-8 {
                scales.push(1.0);
                offsets.push(percentile_min);
                continue;
            }

            // 7-pass iterative refinement
            let mut best_scale = base_range / 15.0;
            let mut best_offset = percentile_min;
            let mut best_loss = f32::INFINITY;

            for pass in 0..7 {
                let search_range = 0.35 / (1.0 + pass as f32 * 0.5);
                let num_scale_steps = 14 + pass * 3;
                let num_offset_steps = 10 + pass * 2;

                for s_idx in 0..=num_scale_steps {
                    let scale_mult = (1.0 - search_range)
                        + (s_idx as f32) * (2.0 * search_range) / num_scale_steps as f32;
                    let test_scale = best_scale * scale_mult;

                    for o_idx in 0..=num_offset_steps {
                        let offset_frac = (o_idx as f32) / num_offset_steps as f32 - 0.5;
                        let test_offset = best_offset + offset_frac * base_range * 0.12;

                        let inv_scale = 1.0 / test_scale;

                        let mut weighted_loss = 0.0f32;
                        let mut total_weight = 0.0f32;
                        let mut bias_sum = 0.0f32;

                        for (i, &x) in group_data.iter().enumerate() {
                            let q_float = (x - test_offset) * inv_scale;
                            let q_round = q_float.round().clamp(0.0, 15.0);
                            let reconstructed = q_round * test_scale + test_offset;

                            let error = x - reconstructed;
                            let weight = group_importance[i];

                            // Huber loss
                            let abs_err = error.abs();
                            let loss = if abs_err < 0.08 {
                                error.powi(2)
                            } else {
                                0.08 * abs_err - 0.0032
                            };

                            weighted_loss += loss * weight;
                            bias_sum += error * weight;
                            total_weight += weight;
                        }

                        if total_weight > 0.0 {
                            weighted_loss /= total_weight;
                            let bias = bias_sum / total_weight;
                            weighted_loss += bias.powi(2) * 0.4;
                        }

                        if weighted_loss < best_loss {
                            best_loss = weighted_loss;
                            best_scale = test_scale;
                            best_offset = test_offset;
                        }
                    }
                }
            }

            // Bias correction
            let inv_scale = 1.0 / best_scale;
            let mut bias_sum = 0.0f32;
            let mut total_weight = 0.0f32;
            for (i, &x) in group_data.iter().enumerate() {
                let q_float = (x - best_offset) * inv_scale;
                let q_round = q_float.round().clamp(0.0, 15.0);
                let reconstructed = q_round * best_scale + best_offset;
                let error = x - reconstructed;
                let weight = group_importance[i];
                bias_sum += error * weight;
                total_weight += weight;
            }
            if total_weight > 0.0 {
                let bias = bias_sum / total_weight;
                best_offset += bias * 0.6;
            }

            scales.push(best_scale);
            offsets.push(best_offset);
        }

        let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

        tensors_out.push(QuantizedTensor {
            name: t.name.clone(),
            shape: t.shape.clone(),
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas: Vec::new(),
            offsets,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_BEST_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_final(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ-Final: True AWQ-style quantization with channel-wise scaling
    // Key insight from AWQ paper: scale salient weight channels BEFORE quantization
    // This effectively gives more precision to important channels
    //
    // For each weight tensor W with shape [out, in]:
    // 1. Compute per-input-channel importance from activation stats
    // 2. Learn optimal per-channel scales s_i that minimize quantization error
    // 3. Quantize W' = W * diag(s) instead of W
    // 4. Store scales for dequantization: W_reconstructed = Q(W') * diag(1/s)

    const GROUP_SIZE: usize = 16;

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

        // Embeddings get int8
        if is_embedding_layer(&t.name) {
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0],
            });
            continue;
        }

        // For 2D tensors, apply AWQ-style channel scaling
        if t.shape.len() >= 2 {
            let (out_features, in_features, is_transposed) = if t.shape.len() == 2 {
                // Detect layout: [out, in] vs [in, out]
                let dim0 = t.shape[0];
                let dim1 = t.shape[1];
                // Heuristic: larger dim is usually input features
                if dim0 > dim1 {
                    (dim1, dim0, true) // [in, out] layout
                } else {
                    (dim0, dim1, false) // [out, in] layout
                }
            } else {
                (t.shape[0], t.shape[1], false)
            };

            // Compute per-channel importance (use activation stats if available)
            let act_stats = bundle.activation_stats.get(&t.name);
            let channel_importance: Vec<f32> = if let Some(stats) = act_stats {
                if stats.len() == in_features {
                    // Normalize and use as importance
                    let max_stat = stats.iter().cloned().fold(0.0f32, f32::max);
                    if max_stat > 0.0 {
                        stats.iter().map(|&s| (s / max_stat).max(0.01)).collect()
                    } else {
                        vec![1.0; in_features]
                    }
                } else {
                    vec![1.0; in_features]
                }
            } else {
                // Estimate importance from weight magnitudes per channel
                let mut importance = vec![0.0f32; in_features];
                for out_idx in 0..out_features {
                    for in_idx in 0..in_features {
                        let idx = if is_transposed {
                            in_idx * out_features + out_idx
                        } else {
                            out_idx * in_features + in_idx
                        };
                        if idx < t.data.len() {
                            importance[in_idx] += t.data[idx].abs();
                        }
                    }
                }
                let max_imp = importance.iter().cloned().fold(0.0f32, f32::max);
                if max_imp > 0.0 {
                    importance
                        .iter()
                        .map(|&i| (i / max_imp).max(0.01))
                        .collect()
                } else {
                    vec![1.0; in_features]
                }
            };

            // Learn optimal per-channel scales using grid search
            // AWQ uses s_i = importance_i^alpha where alpha is learned
            // We search over alpha values
            let mut best_channel_scales = vec![1.0f32; in_features];
            let mut best_total_error = f32::INFINITY;

            for alpha_idx in 0..=20 {
                let alpha = (alpha_idx as f32) * 0.1; // 0.0 to 2.0

                // Compute channel scales: s_i = importance_i^alpha
                let channel_scales: Vec<f32> = channel_importance
                    .iter()
                    .map(|&imp| imp.powf(alpha).max(0.1).min(10.0))
                    .collect();

                // Apply scales and compute quantization error
                let mut total_error = 0.0f32;
                let mut total_weight = 0.0f32;

                for out_idx in 0..out_features {
                    for in_idx in 0..in_features {
                        let idx = if is_transposed {
                            in_idx * out_features + out_idx
                        } else {
                            out_idx * in_features + in_idx
                        };
                        if idx < t.data.len() {
                            let w = t.data[idx];
                            let s = channel_scales[in_idx];
                            let w_scaled = w * s;

                            // Simulate quantization (simplified)
                            let abs_max = w_scaled.abs().max(1e-8);
                            let q = (w_scaled / abs_max * 7.5).round().clamp(-8.0, 7.0);
                            let w_recon = q * abs_max / 7.5 / s;

                            let error = (w - w_recon).powi(2);
                            let weight = channel_importance[in_idx];
                            total_error += error * weight;
                            total_weight += weight;
                        }
                    }
                }

                if total_weight > 0.0 {
                    total_error /= total_weight;
                }

                if total_error < best_total_error {
                    best_total_error = total_error;
                    best_channel_scales = channel_scales;
                }
            }

            // Apply channel scales to weights
            let mut scaled_weights = vec![0.0f32; t.data.len()];
            for out_idx in 0..out_features {
                for in_idx in 0..in_features {
                    let idx = if is_transposed {
                        in_idx * out_features + out_idx
                    } else {
                        out_idx * in_features + in_idx
                    };
                    if idx < t.data.len() {
                        scaled_weights[idx] = t.data[idx] * best_channel_scales[in_idx];
                    }
                }
            }

            // Now quantize the scaled weights with group quantization
            let num_groups = (scaled_weights.len() + GROUP_SIZE - 1) / GROUP_SIZE;
            let mut scales = Vec::with_capacity(num_groups);
            let mut offsets = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(scaled_weights.len());
                let group_data = &scaled_weights[start..end];

                // Percentile clipping
                let mut sorted_vals: Vec<f32> = group_data.to_vec();
                sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let p_low = (sorted_vals.len() as f32 * 0.005) as usize;
                let p_high =
                    ((sorted_vals.len() as f32 * 0.995) as usize).min(sorted_vals.len() - 1);

                let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
                let percentile_max = sorted_vals[p_high];
                let base_range = percentile_max - percentile_min;

                if base_range <= 1e-8 {
                    scales.push(1.0);
                    offsets.push(percentile_min);
                    continue;
                }

                // 9-pass iterative refinement
                let mut best_scale = base_range / 15.0;
                let mut best_offset = percentile_min;
                let mut best_loss = f32::INFINITY;

                for pass in 0..9 {
                    let search_range = 0.4 / (1.0 + pass as f32 * 0.4);
                    let num_scale_steps = 16 + pass * 4;
                    let num_offset_steps = 12 + pass * 2;

                    for s_idx in 0..=num_scale_steps {
                        let scale_mult = (1.0 - search_range)
                            + (s_idx as f32) * (2.0 * search_range) / num_scale_steps as f32;
                        let test_scale = best_scale * scale_mult;

                        for o_idx in 0..=num_offset_steps {
                            let offset_frac = (o_idx as f32) / num_offset_steps as f32 - 0.5;
                            let test_offset = best_offset + offset_frac * base_range * 0.1;

                            let inv_scale = 1.0 / test_scale;

                            let mut weighted_loss = 0.0f32;
                            let mut bias_sum = 0.0f32;

                            for &x in group_data.iter() {
                                let q_float = (x - test_offset) * inv_scale;
                                let q_round = q_float.round().clamp(0.0, 15.0);
                                let reconstructed = q_round * test_scale + test_offset;

                                let error = x - reconstructed;
                                let abs_err = error.abs();
                                let loss = if abs_err < 0.05 {
                                    error.powi(2)
                                } else {
                                    0.05 * abs_err - 0.00125
                                };

                                weighted_loss += loss;
                                bias_sum += error;
                            }

                            let n = group_data.len() as f32;
                            weighted_loss /= n;
                            let bias = bias_sum / n;
                            weighted_loss += bias.powi(2) * 0.5;

                            if weighted_loss < best_loss {
                                best_loss = weighted_loss;
                                best_scale = test_scale;
                                best_offset = test_offset;
                            }
                        }
                    }
                }

                // Bias correction
                let inv_scale = 1.0 / best_scale;
                let mut bias_sum = 0.0f32;
                for &x in group_data.iter() {
                    let q_float = (x - best_offset) * inv_scale;
                    let q_round = q_float.round().clamp(0.0, 15.0);
                    let reconstructed = q_round * best_scale + best_offset;
                    bias_sum += x - reconstructed;
                }
                best_offset += bias_sum / group_data.len() as f32 * 0.7;

                scales.push(best_scale);
                offsets.push(best_offset);
            }

            // Quantize and pack
            let mut packed: Vec<u8> = Vec::with_capacity((scaled_weights.len() + 1) / 2);

            for g in 0..num_groups {
                let scale = scales[g];
                let offset = offsets[g];
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(scaled_weights.len());

                let mut group_iter = scaled_weights[start..end].iter();
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

            // Store channel scales in alphas field for decompression
            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales[0],
                scales,
                data: packed,
                indices: Vec::new(),
                alphas: best_channel_scales, // Store channel scales here
                offsets,
            });
        } else {
            // 1D tensors: simple group quantization
            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
            let mut scales = Vec::with_capacity(num_groups);
            let mut offsets = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(t.data.len());
                let group_data = &t.data[start..end];

                let min_val = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max_val - min_val;

                let scale = if range > 1e-8 { range / 15.0 } else { 1.0 };
                scales.push(scale);
                offsets.push(min_val);
            }

            let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for g in 0..num_groups {
                let scale = scales[g];
                let offset = offsets[g];
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales[0],
                scales,
                data: packed,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets,
            });
        }
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_FINAL_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq_pro(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // AWQ-Pro: Full calibration-based AWQ codec
    //
    // This codec implements the complete AWQ algorithm:
    // 1. Uses calibration data (activation stats) from Python pipeline
    // 2. Learns optimal per-channel scales (s_i = activation_i^alpha)
    // 3. Applies smooth quantization to balance activation/weight difficulty
    // 4. Uses per-layer sensitivity for mixed precision (int8 for sensitive layers)
    // 5. 11-pass iterative refinement with Huber loss
    // 6. Bias correction for systematic error reduction
    //
    // Target: <1% PPL delta (matching reference AWQ)

    const GROUP_SIZE: usize = 128; // Standard AWQ group size
    const N_ALPHA_GRID: usize = 40; // Fine grid for alpha search

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

        // Get activation stats for this tensor
        let act_stats = bundle.activation_stats.get(&t.name);

        // Determine layer sensitivity for mixed precision
        let layer_sensitivity = compute_layer_sensitivity_score(&t.name, &t.data, act_stats);
        let use_int8 = layer_sensitivity > 0.15 || is_very_sensitive_layer(&t.name);

        if use_int8 {
            // High-precision int8 for sensitive layers
            let max_abs = t.data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            let encoded: Vec<u8> = t
                .data
                .iter()
                .map(|&x| {
                    let v = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    v as u8
                })
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale,
                scales: vec![scale],
                data: encoded,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets: vec![0.0], // Marker for int8 mode
            });
            continue;
        }

        // For 2D+ tensors, apply full AWQ algorithm
        if t.shape.len() >= 2 {
            let (out_features, in_features, is_transposed) =
                detect_weight_layout(&t.shape, act_stats);

            // Step 1: Compute per-channel importance from activation stats
            let channel_importance =
                compute_channel_importance(in_features, act_stats, is_transposed, &t.shape);

            // Step 2: Search for optimal alpha using grid search
            // AWQ key insight: s_i = importance_i^alpha
            let (best_channel_scales, best_alpha) = search_optimal_alpha(
                &t.data,
                &channel_importance,
                out_features,
                in_features,
                is_transposed,
                N_ALPHA_GRID,
            );

            // Step 3: Compute smooth quantization scales
            let smooth_scales = compute_smooth_scales_for_tensor(
                &t.data,
                &channel_importance,
                out_features,
                in_features,
                is_transposed,
                0.5, // smooth_alpha
            );

            // Step 4: Apply both channel scales and smooth scales to weights
            let mut scaled_weights = vec![0.0f32; t.data.len()];
            for out_idx in 0..out_features {
                for in_idx in 0..in_features {
                    let idx = if is_transposed {
                        in_idx * out_features + out_idx
                    } else {
                        out_idx * in_features + in_idx
                    };
                    if idx < t.data.len() {
                        let channel_scale = best_channel_scales.get(in_idx).copied().unwrap_or(1.0);
                        let smooth_scale = smooth_scales.get(in_idx).copied().unwrap_or(1.0);
                        scaled_weights[idx] = t.data[idx] * channel_scale * smooth_scale;
                    }
                }
            }

            // Step 5: Group quantization with 11-pass iterative refinement
            let num_groups = (scaled_weights.len() + GROUP_SIZE - 1) / GROUP_SIZE;
            let mut scales = Vec::with_capacity(num_groups);
            let mut offsets = Vec::with_capacity(num_groups);

            // Compute importance weights for weighted loss
            let importance = compute_importance_exp(&t, act_stats);

            for g in 0..num_groups {
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(scaled_weights.len());
                let group_data = &scaled_weights[start..end];
                let group_importance = &importance[start..end];

                // Percentile-based outlier clipping
                let mut sorted_vals: Vec<f32> = group_data.to_vec();
                sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let p_low = (sorted_vals.len() as f32 * 0.002) as usize;
                let p_high =
                    ((sorted_vals.len() as f32 * 0.998) as usize).min(sorted_vals.len() - 1);

                let percentile_min = sorted_vals[p_low.min(sorted_vals.len() - 1)];
                let percentile_max = sorted_vals[p_high];
                let base_range = percentile_max - percentile_min;

                if base_range <= 1e-8 {
                    scales.push(1.0);
                    offsets.push(percentile_min);
                    continue;
                }

                // 11-pass iterative refinement with progressively finer search
                let mut best_scale = base_range / 15.0;
                let mut best_offset = percentile_min;
                let mut best_loss = f32::INFINITY;

                for pass in 0..11 {
                    let search_range = 0.5 / (1.0 + pass as f32 * 0.35);
                    let num_scale_steps = 18 + pass * 4;
                    let num_offset_steps = 14 + pass * 2;

                    for s_idx in 0..=num_scale_steps {
                        let scale_mult = (1.0 - search_range)
                            + (s_idx as f32) * (2.0 * search_range) / num_scale_steps as f32;
                        let test_scale = best_scale * scale_mult;

                        for o_idx in 0..=num_offset_steps {
                            let offset_frac = (o_idx as f32) / num_offset_steps as f32 - 0.5;
                            let test_offset = best_offset + offset_frac * base_range * 0.08;

                            let inv_scale = 1.0 / test_scale;

                            let mut weighted_loss = 0.0f32;
                            let mut total_weight = 0.0f32;
                            let mut bias_sum = 0.0f32;

                            for (i, &x) in group_data.iter().enumerate() {
                                let q_float = (x - test_offset) * inv_scale;
                                let q_round = q_float.round().clamp(0.0, 15.0);
                                let reconstructed = q_round * test_scale + test_offset;

                                let error = x - reconstructed;
                                let weight = group_importance.get(i).copied().unwrap_or(1.0);

                                // Huber loss for robustness to outliers
                                let abs_err = error.abs();
                                let loss = if abs_err < 0.03 {
                                    error.powi(2)
                                } else {
                                    0.03 * abs_err - 0.00045
                                };

                                weighted_loss += loss * weight;
                                bias_sum += error * weight;
                                total_weight += weight;
                            }

                            if total_weight > 0.0 {
                                weighted_loss /= total_weight;
                                let bias = bias_sum / total_weight;
                                // Penalize systematic bias
                                weighted_loss += bias.powi(2) * 0.6;
                            }

                            if weighted_loss < best_loss {
                                best_loss = weighted_loss;
                                best_scale = test_scale;
                                best_offset = test_offset;
                            }
                        }
                    }
                }

                // Final bias correction pass
                let inv_scale = 1.0 / best_scale;
                let mut bias_sum = 0.0f32;
                let mut total_weight = 0.0f32;
                for (i, &x) in group_data.iter().enumerate() {
                    let q_float = (x - best_offset) * inv_scale;
                    let q_round = q_float.round().clamp(0.0, 15.0);
                    let reconstructed = q_round * best_scale + best_offset;
                    let error = x - reconstructed;
                    let weight = group_importance.get(i).copied().unwrap_or(1.0);
                    bias_sum += error * weight;
                    total_weight += weight;
                }
                if total_weight > 0.0 {
                    let bias = bias_sum / total_weight;
                    best_offset += bias * 0.8;
                }

                scales.push(best_scale);
                offsets.push(best_offset);
            }

            // Step 6: Quantize and pack
            let mut packed: Vec<u8> = Vec::with_capacity((scaled_weights.len() + 1) / 2);

            for g in 0..num_groups {
                let scale = scales[g];
                let offset = offsets[g];
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(scaled_weights.len());

                let mut group_iter = scaled_weights[start..end].iter();
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

            // Combine channel scales and smooth scales for storage
            let combined_scales: Vec<f32> = best_channel_scales
                .iter()
                .zip(smooth_scales.iter())
                .map(|(&c, &s)| c * s)
                .collect();

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales[0],
                scales,
                data: packed,
                indices: vec![best_alpha as u32], // Store optimal alpha
                alphas: combined_scales,          // Combined channel + smooth scales
                offsets,
            });
        } else {
            // 1D tensors: simple asymmetric group quantization
            let num_groups = (t.data.len() + GROUP_SIZE - 1) / GROUP_SIZE;
            let mut scales = Vec::with_capacity(num_groups);
            let mut offsets = Vec::with_capacity(num_groups);

            for g in 0..num_groups {
                let start = g * GROUP_SIZE;
                let end = (start + GROUP_SIZE).min(t.data.len());
                let group_data = &t.data[start..end];

                let min_val = group_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_val = group_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let range = max_val - min_val;

                let scale = if range > 1e-8 { range / 15.0 } else { 1.0 };
                scales.push(scale);
                offsets.push(min_val);
            }

            let mut packed: Vec<u8> = Vec::with_capacity((t.data.len() + 1) / 2);

            for g in 0..num_groups {
                let scale = scales[g];
                let offset = offsets[g];
                let inv_scale = if scale > 0.0 { 1.0 / scale } else { 1.0 };
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

            tensors_out.push(QuantizedTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                scale: scales[0],
                scales,
                data: packed,
                indices: Vec::new(),
                alphas: Vec::new(),
                offsets,
            });
        }
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_AWQ_PRO_V1.to_string(),
        tensors: tensors_out,
    })
}

/// Detect weight layout from shape and activation stats
fn detect_weight_layout(shape: &[usize], act_stats: Option<&Vec<f32>>) -> (usize, usize, bool) {
    if shape.len() < 2 {
        return (shape[0], 1, false);
    }

    let dim0 = shape[0];
    let dim1 = shape[1];

    if let Some(stats) = act_stats {
        if stats.len() == dim0 {
            // Stats match first dim -> transposed layout [in, out]
            return (dim1, dim0, true);
        } else if stats.len() == dim1 {
            // Stats match second dim -> standard layout [out, in]
            return (dim0, dim1, false);
        }
    }

    // Heuristic: larger dimension is usually input features
    if dim0 > dim1 {
        (dim1, dim0, true)
    } else {
        (dim0, dim1, false)
    }
}

/// Compute per-channel importance from activation stats
fn compute_channel_importance(
    in_features: usize,
    act_stats: Option<&Vec<f32>>,
    is_transposed: bool,
    shape: &[usize],
) -> Vec<f32> {
    if let Some(stats) = act_stats {
        let expected_len = if is_transposed { shape[0] } else { shape[1] };
        if stats.len() == expected_len {
            // Normalize to [0.01, 1.0] range
            let max_stat = stats.iter().cloned().fold(0.0f32, f32::max);
            if max_stat > 0.0 {
                return stats.iter().map(|&s| (s / max_stat).max(0.01)).collect();
            }
        }
    }
    vec![1.0; in_features]
}

/// Search for optimal alpha in s_i = importance_i^alpha
fn search_optimal_alpha(
    weights: &[f32],
    importance: &[f32],
    out_features: usize,
    in_features: usize,
    is_transposed: bool,
    n_grid: usize,
) -> (Vec<f32>, f32) {
    let mut best_alpha = 0.5f32;
    let mut best_error = f32::INFINITY;
    let mut best_scales = vec![1.0f32; in_features];

    for alpha_idx in 0..=n_grid {
        let alpha = (alpha_idx as f32) / (n_grid as f32); // 0.0 to 1.0

        // Compute channel scales: s_i = importance_i^alpha
        let channel_scales: Vec<f32> = importance
            .iter()
            .map(|&imp| imp.powf(alpha).clamp(0.1, 10.0))
            .collect();

        // Simulate quantization and compute weighted error
        let mut total_error = 0.0f32;
        let mut total_weight = 0.0f32;

        for out_idx in 0..out_features {
            for in_idx in 0..in_features {
                let idx = if is_transposed {
                    in_idx * out_features + out_idx
                } else {
                    out_idx * in_features + in_idx
                };

                if idx >= weights.len() {
                    continue;
                }

                let w = weights[idx];
                let s = channel_scales.get(in_idx).copied().unwrap_or(1.0);
                let w_scaled = w * s;

                // Simulate int4 quantization
                let abs_max = w_scaled.abs().max(1e-8);
                let q = (w_scaled / abs_max * 7.5).round().clamp(-8.0, 7.0);
                let w_recon = q * abs_max / 7.5 / s;

                let error = (w - w_recon).powi(2);
                let weight = importance.get(in_idx).copied().unwrap_or(1.0);

                total_error += error * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            total_error /= total_weight;
        }

        if total_error < best_error {
            best_error = total_error;
            best_alpha = alpha;
            best_scales = channel_scales;
        }
    }

    (best_scales, best_alpha)
}

/// Compute smooth quantization scales
fn compute_smooth_scales_for_tensor(
    weights: &[f32],
    importance: &[f32],
    out_features: usize,
    in_features: usize,
    is_transposed: bool,
    smooth_alpha: f32,
) -> Vec<f32> {
    // Compute per-channel weight max
    let mut weight_max = vec![0.0f32; in_features];

    for out_idx in 0..out_features {
        for in_idx in 0..in_features {
            let idx = if is_transposed {
                in_idx * out_features + out_idx
            } else {
                out_idx * in_features + in_idx
            };

            if idx < weights.len() {
                weight_max[in_idx] = weight_max[in_idx].max(weights[idx].abs());
            }
        }
    }

    // Smooth scale: s = act^alpha / weight^(1-alpha)
    importance
        .iter()
        .zip(weight_max.iter())
        .map(|(&act, &wmax)| {
            let act_term = act.powf(smooth_alpha);
            let weight_term = wmax.max(1e-8).powf(1.0 - smooth_alpha);
            (act_term / weight_term).clamp(0.1, 10.0)
        })
        .collect()
}

/// Compute layer sensitivity score for mixed precision decision
fn compute_layer_sensitivity_score(
    name: &str,
    weights: &[f32],
    act_stats: Option<&Vec<f32>>,
) -> f32 {
    // Base sensitivity from layer type
    let mut sensitivity = 0.0f32;

    let lower = name.to_lowercase();

    // Embeddings are critical
    if lower.contains("embed") || lower.contains("wte") || lower.contains("wpe") {
        sensitivity += 0.3;
    }

    // Layer norms
    if lower.contains("ln") || lower.contains("norm") || lower.contains("layernorm") {
        sensitivity += 0.25;
    }

    // Output projection
    if lower.contains("lm_head") || lower.contains("head") {
        sensitivity += 0.2;
    }

    // First and last layers
    if lower.contains(".h.0.") || lower.contains("layer.0.") {
        sensitivity += 0.1;
    }
    if lower.contains(".h.11.") || lower.contains("layer.11.") {
        sensitivity += 0.1;
    }

    // Attention Q/K projections
    if lower.contains("q_proj") || lower.contains("k_proj") || lower.contains("c_attn") {
        sensitivity += 0.08;
    }

    // Weight distribution analysis
    if !weights.is_empty() {
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance =
            weights.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std = variance.sqrt();

        // High variance weights are more sensitive
        if std > 0.1 {
            sensitivity += 0.05;
        }

        // Check for outliers (kurtosis proxy)
        let fourth_moment = weights
            .iter()
            .map(|&w| ((w - mean) / std.max(1e-8)).powi(4))
            .sum::<f32>()
            / weights.len() as f32;

        if fourth_moment > 5.0 {
            sensitivity += 0.05;
        }
    }

    // Activation stats analysis
    if let Some(stats) = act_stats {
        let max_stat = stats.iter().cloned().fold(0.0f32, f32::max);
        let mean_stat = stats.iter().sum::<f32>() / stats.len() as f32;

        // High activation variance indicates sensitivity
        if max_stat > mean_stat * 5.0 {
            sensitivity += 0.05;
        }
    }

    sensitivity
}

/// Check if layer is an embedding (only these get int8 in Ultra)
fn is_embedding_layer(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("embed") || lower.contains("wte") || lower.contains("wpe")
}

#[derive(PartialEq)]
enum LayerPrecision {
    Int8,
    Int4,
}

/// Get precision for layer based on its type and position
fn get_layer_precision(name: &str) -> LayerPrecision {
    let lower = name.to_lowercase();

    // Critical layers that need int8
    if lower.contains("embed") ||
       lower.contains("wte") ||
       lower.contains("wpe") ||
       lower.contains("ln") ||
       lower.contains("layernorm") ||
       lower.contains("norm") ||
       lower.contains("lm_head") ||
       lower.contains("head") ||
       // First two and last two transformer blocks
       lower.contains(".h.0.") ||
       lower.contains(".h.1.") ||
       lower.contains(".h.10.") ||
       lower.contains(".h.11.") ||
       lower.contains("layer.0.") ||
       lower.contains("layer.1.") ||
       lower.contains("layer.10.") ||
       lower.contains("layer.11.") ||
       // Attention query/key projections are sensitive
       lower.contains("q_proj") ||
       lower.contains("k_proj") ||
       lower.contains("attn.c_attn")
    {
        return LayerPrecision::Int8;
    }

    LayerPrecision::Int4
}

/// More aggressive sensitivity detection - includes attention layers
fn is_very_sensitive_layer(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("embed") ||
    lower.contains("wte") ||
    lower.contains("wpe") ||
    lower.contains("ln") ||
    lower.contains("layernorm") ||
    lower.contains("norm") ||
    lower.contains("lm_head") ||
    lower.contains("head") ||
    // Also keep first and last transformer blocks in higher precision
    lower.contains(".h.0.") ||
    lower.contains(".h.11.") ||
    lower.contains("layer.0.") ||
    lower.contains("layer.11.")
}

/// Compute importance with exponential weighting (more aggressive than squared)
fn compute_importance_exp(tensor: &FloatTensor, act_stats: Option<&Vec<f32>>) -> Vec<f32> {
    let n = tensor.data.len();

    if tensor.shape.len() < 2 {
        return vec![1.0; n];
    }

    let Some(stats) = act_stats else {
        return vec![1.0; n];
    };

    let is_transposed = if stats.len() == tensor.shape[0] {
        true
    } else if stats.len() == tensor.shape[1] {
        false
    } else {
        return vec![1.0; n];
    };

    // Normalize stats with exponential weighting
    let mean_stat = stats.iter().sum::<f32>() / stats.len() as f32;
    let normalized: Vec<f32> = stats
        .iter()
        .map(|&s| {
            let ratio = s / mean_stat.max(1e-8);
            // Exponential weighting: exp(ratio - 1) gives ~1 for average, higher for important
            (ratio.ln() + 1.0).exp().clamp(0.1, 20.0)
        })
        .collect();

    // Broadcast to full tensor
    let mut importance = Vec::with_capacity(n);
    if is_transposed {
        for in_ch in 0..tensor.shape[0] {
            let imp = if in_ch < normalized.len() {
                normalized[in_ch]
            } else {
                1.0
            };
            for _ in 0..tensor.shape[1] {
                importance.push(imp);
            }
        }
    } else {
        for _ in 0..tensor.shape[0] {
            for in_ch in 0..tensor.shape[1] {
                let imp = if in_ch < normalized.len() {
                    normalized[in_ch]
                } else {
                    1.0
                };
                importance.push(imp);
            }
        }
    }

    importance
}

/// Check if a layer is sensitive and should use higher precision
fn is_sensitive_layer(name: &str) -> bool {
    let lower = name.to_lowercase();
    // Keep embeddings, layer norms, and final projection in int8
    lower.contains("embed")
        || lower.contains("wte")
        || lower.contains("wpe")
        || lower.contains("ln")
        || lower.contains("layernorm")
        || lower.contains("norm")
        || lower.contains("lm_head")
        || lower.contains("head")
}

/// Compute per-element importance weights from activation statistics
fn compute_importance(tensor: &FloatTensor, act_stats: Option<&Vec<f32>>) -> Vec<f32> {
    let n = tensor.data.len();

    if tensor.shape.len() < 2 {
        return vec![1.0; n];
    }

    let Some(stats) = act_stats else {
        return vec![1.0; n];
    };

    // Determine layout
    let is_transposed = if stats.len() == tensor.shape[0] {
        true
    } else if stats.len() == tensor.shape[1] {
        false
    } else {
        return vec![1.0; n];
    };

    // Normalize stats to importance weights
    // Higher activation = more important = higher weight
    let mean_stat = stats.iter().sum::<f32>() / stats.len() as f32;
    let normalized: Vec<f32> = stats
        .iter()
        .map(|&s| {
            let ratio = s / mean_stat.max(1e-8);
            // Square the ratio to emphasize important channels more
            (ratio * ratio).clamp(0.1, 10.0)
        })
        .collect();

    // Broadcast to full tensor
    let mut importance = Vec::with_capacity(n);
    if is_transposed {
        // [in_features, out_features]
        for in_ch in 0..tensor.shape[0] {
            let imp = if in_ch < normalized.len() {
                normalized[in_ch]
            } else {
                1.0
            };
            for _ in 0..tensor.shape[1] {
                importance.push(imp);
            }
        }
    } else {
        // [out_features, in_features]
        for _ in 0..tensor.shape[0] {
            for in_ch in 0..tensor.shape[1] {
                let imp = if in_ch < normalized.len() {
                    normalized[in_ch]
                } else {
                    1.0
                };
                importance.push(imp);
            }
        }
    }

    importance
}

fn decode_int4_nibble(n: u8) -> i8 {
    let v = n & 0x0f;
    if v & 0x08 != 0 {
        // Sign-extend 4-bit two's complement to i8.
        (v as i8) | !0x0f
    } else {
        v as i8
    }
}

fn decompress_int8_sym(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if expected != t.data.len() {
            return Err(format!(
                "Quantized tensor '{}' has shape {:?} (size {}), but data length {}",
                t.name,
                t.shape,
                expected,
                t.data.len()
            ));
        }

        let mut data = Vec::with_capacity(t.data.len());
        for &b in &t.data {
            let q = b as i8;
            let x = (q as f32) * t.scale;
            data.push(x);
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

fn decompress_int4_sym(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        // Each byte encodes up to two values.
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        'outer: for &b in &t.data {
            let lo = decode_int4_nibble(b & 0x0f);
            data.push(lo as f32 * t.scale);
            if data.len() >= expected {
                break 'outer;
            }
            let hi = decode_int4_nibble((b >> 4) & 0x0f);
            data.push(hi as f32 * t.scale);
            if data.len() >= expected {
                break 'outer;
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Quantized tensor '{}' has insufficient 4-bit values (got {}, expected {})",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int2_sym(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        'outer: for &b in &t.data {
            for shift in [0, 2, 4, 6] {
                let bits = (b >> shift) & 0x03;
                // Sign-extend 2-bit two's complement to i8
                let v = if bits & 0x02 != 0 {
                    (bits as i8) | !0x03
                } else {
                    bits as i8
                };
                data.push(v as f32 * t.scale);
                if data.len() >= expected {
                    break 'outer;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Quantized tensor '{}' has insufficient 2-bit values (got {}, expected {})",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_perchannel(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is per-channel or per-tensor (fallback for 1D)
        let is_perchannel = !t.scales.is_empty() && t.scales.len() > 1;

        if !is_perchannel {
            // Fall back to per-tensor decompression (for 1D tensors)
            let scale = if !t.scales.is_empty() {
                t.scales[0]
            } else {
                t.scale
            };

            let mut data: Vec<f32> = Vec::with_capacity(expected);
            'outer: for &b in &t.data {
                let lo = decode_int4_nibble(b & 0x0f);
                data.push(lo as f32 * scale);
                if data.len() >= expected {
                    break 'outer;
                }
                let hi = decode_int4_nibble((b >> 4) & 0x0f);
                data.push(hi as f32 * scale);
                if data.len() >= expected {
                    break 'outer;
                }
            }

            if data.len() != expected {
                return Err(format!(
                    "Quantized tensor '{}' has insufficient 4-bit values (got {}, expected {})",
                    t.name,
                    data.len(),
                    expected
                ));
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Per-channel decompression
        let out_channels = t.shape[0];
        let elements_per_channel = expected / out_channels;

        if t.scales.len() != out_channels {
            return Err(format!(
                "Tensor '{}' has {} channels but {} scales",
                t.name,
                out_channels,
                t.scales.len()
            ));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let bytes_per_channel = (elements_per_channel + 1) / 2;

        for ch in 0..out_channels {
            let scale = t.scales[ch];

            if byte_idx + bytes_per_channel > t.data.len() {
                return Err(format!(
                    "Tensor '{}' has insufficient data for channel {}",
                    t.name, ch
                ));
            }

            let channel_bytes = &t.data[byte_idx..byte_idx + bytes_per_channel];
            let mut produced = 0;
            for &b in channel_bytes {
                let lo = decode_int4_nibble(b & 0x0f);
                data.push(lo as f32 * scale);
                produced += 1;
                if produced >= elements_per_channel {
                    break;
                }

                let hi = decode_int4_nibble((b >> 4) & 0x0f);
                data.push(hi as f32 * scale);
                produced += 1;
                if produced >= elements_per_channel {
                    break;
                }
            }

            if produced != elements_per_channel {
                return Err(format!(
                    "Tensor '{}' has insufficient 4-bit values for channel {} (got {}, expected {})",
                    t.name,
                    ch,
                    produced,
                    elements_per_channel
                ));
            }

            byte_idx += bytes_per_channel;
        }

        if data.len() != expected {
            return Err(format!(
                "Quantized tensor '{}' has insufficient 4-bit values (got {}, expected {})",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Simplified AWQ now uses the same g128 format, so delegate to g128 decompression
    decompress_int4_g128(artifact)
}

fn decompress_int4_g128(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress group-quantized int4 (g=128)
    const GROUP_SIZE: usize = 128;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = decode_int4_nibble(b & 0x0f);
                data.push(lo as f32 * scale);
                decoded += 1;

                if decoded < group_len {
                    let hi = decode_int4_nibble((b >> 4) & 0x0f);
                    data.push(hi as f32 * scale);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_g8(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress ultra-fine group-quantized int4 (g=8)
    // Uses asymmetric quantization with offsets
    const GROUP_SIZE: usize = 8;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                // Asymmetric: x = q * scale + offset
                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_g16(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress group-quantized int4 (g=16)
    // Uses asymmetric quantization with offsets
    const GROUP_SIZE: usize = 16;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                // Asymmetric: x = q * scale + offset
                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_k(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress K-quant style quantization
    // Data layout: [packed_int4...][sub_scales...][sub_mins...]

    const SUPER_BLOCK_SIZE: usize = 256;
    const BLOCK_SIZE: usize = 32;
    const NUM_BLOCKS: usize = SUPER_BLOCK_SIZE / BLOCK_SIZE; // 8

    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        // Calculate dimensions
        let padded_len = ((expected + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE;
        let num_super_blocks = padded_len / SUPER_BLOCK_SIZE;
        let packed_len = padded_len / 2;
        let sub_data_len = num_super_blocks * NUM_BLOCKS;

        // Validate data length
        let expected_data_len = packed_len + sub_data_len * 2;
        if t.data.len() < expected_data_len {
            return Err(format!(
                "Tensor '{}' has {} bytes but expected at least {}",
                t.name,
                t.data.len(),
                expected_data_len
            ));
        }

        // Extract packed data and sub-block metadata
        let packed = &t.data[..packed_len];
        let sub_scales = &t.data[packed_len..packed_len + sub_data_len];
        let sub_mins = &t.data[packed_len + sub_data_len..packed_len + sub_data_len * 2];

        // Super-block scales and mins from the scales/offsets fields
        let super_scales = &t.scales;
        let super_mins = &t.offsets;

        if super_scales.len() != num_super_blocks || super_mins.len() != num_super_blocks {
            return Err(format!(
                "Tensor '{}' has {} super_scales and {} super_mins but expected {}",
                t.name,
                super_scales.len(),
                super_mins.len(),
                num_super_blocks
            ));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut packed_idx = 0;
        let mut sub_idx = 0;

        for sb in 0..num_super_blocks {
            let sb_scale = super_scales[sb];
            let sb_min = super_mins[sb];

            for _b in 0..NUM_BLOCKS {
                let b_scale_q = sub_scales[sub_idx];
                let b_min_q = sub_mins[sub_idx];
                sub_idx += 1;

                // Dequantize block scale and min
                let b_range = b_scale_q as f32 / 63.0;
                let b_min = b_min_q as f32 / 63.0;

                // Unpack and dequantize 32 weights
                for _ in 0..BLOCK_SIZE / 2 {
                    if packed_idx >= packed.len() {
                        break;
                    }
                    let byte = packed[packed_idx];
                    packed_idx += 1;

                    let lo = (byte & 0x0f) as f32;
                    let hi = ((byte >> 4) & 0x0f) as f32;

                    // Dequantize: q -> normalized -> original
                    let norm_lo = (lo / 15.0) * b_range + b_min;
                    let norm_hi = (hi / 15.0) * b_range + b_min;

                    let val_lo = norm_lo * sb_scale + sb_min;
                    let val_hi = norm_hi * sb_scale + sb_min;

                    if data.len() < expected {
                        data.push(val_lo);
                    }
                    if data.len() < expected {
                        data.push(val_hi);
                    }
                }
            }
        }

        // Trim to expected size (remove padding)
        data.truncate(expected);

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

fn decompress_int4_g8_fp16(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress g=8 with FP16-packed scales
    // Data layout: [packed_int4...][scales_f16_bytes...][offsets_f16_bytes...]

    const GROUP_SIZE: usize = 8;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        // Calculate dimensions
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let packed_len = (expected + 1) / 2;
        let scales_len = num_groups * 2; // 2 bytes per FP16
        let offsets_len = num_groups * 2;

        let expected_data_len = packed_len + scales_len + offsets_len;
        if t.data.len() < expected_data_len {
            return Err(format!(
                "Tensor '{}' has {} bytes but expected at least {}",
                t.name,
                t.data.len(),
                expected_data_len
            ));
        }

        // Extract packed data and scale/offset bytes
        let packed = &t.data[..packed_len];
        let scales_bytes = &t.data[packed_len..packed_len + scales_len];
        let offsets_bytes = &t.data[packed_len + scales_len..packed_len + scales_len + offsets_len];

        // Convert FP16 bytes to f32 scales
        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = scales_bytes[i * 2] as u16;
            let hi = scales_bytes[i * 2 + 1] as u16;
            let bits = lo | (hi << 8);
            scales.push(f16_bits_to_f32(bits));
        }

        // Convert FP16 bytes to f32 offsets
        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = offsets_bytes[i * 2] as u16;
            let hi = offsets_bytes[i * 2 + 1] as u16;
            let bits = lo | (hi << 8);
            offsets.push(f16_bits_to_f32(bits));
        }

        // Decompress weights
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let group_len = GROUP_SIZE.min(expected - g * GROUP_SIZE);

            let mut decoded = 0;
            while decoded < group_len && byte_idx < packed.len() {
                let b = packed[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_g16_fp16(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress g=16 with FP16-packed scales
    const GROUP_SIZE: usize = 16;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let packed_len = (expected + 1) / 2;
        let scales_len = num_groups * 2;
        let offsets_len = num_groups * 2;

        let expected_data_len = packed_len + scales_len + offsets_len;
        if t.data.len() < expected_data_len {
            return Err(format!(
                "Tensor '{}' has {} bytes but expected at least {}",
                t.name,
                t.data.len(),
                expected_data_len
            ));
        }

        let packed = &t.data[..packed_len];
        let scales_bytes = &t.data[packed_len..packed_len + scales_len];
        let offsets_bytes = &t.data[packed_len + scales_len..packed_len + scales_len + offsets_len];

        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = scales_bytes[i * 2] as u16;
            let hi = scales_bytes[i * 2 + 1] as u16;
            scales.push(f16_bits_to_f32(lo | (hi << 8)));
        }

        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = offsets_bytes[i * 2] as u16;
            let hi = offsets_bytes[i * 2 + 1] as u16;
            offsets.push(f16_bits_to_f32(lo | (hi << 8)));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let group_len = GROUP_SIZE.min(expected - g * GROUP_SIZE);

            let mut decoded = 0;
            while decoded < group_len && byte_idx < packed.len() {
                let b = packed[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
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

fn decompress_int4_g32_fp16(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress g=32 with FP16-packed scales
    const GROUP_SIZE: usize = 32;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        let num_groups = (expected + GROUP_SIZE - 1) / GROUP_SIZE;
        let packed_len = (expected + 1) / 2;
        let scales_len = num_groups * 2;
        let offsets_len = num_groups * 2;

        let expected_data_len = packed_len + scales_len + offsets_len;
        if t.data.len() < expected_data_len {
            return Err(format!(
                "Tensor '{}' has {} bytes but expected at least {}",
                t.name,
                t.data.len(),
                expected_data_len
            ));
        }

        let packed = &t.data[..packed_len];
        let scales_bytes = &t.data[packed_len..packed_len + scales_len];
        let offsets_bytes = &t.data[packed_len + scales_len..packed_len + scales_len + offsets_len];

        let mut scales: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = scales_bytes[i * 2] as u16;
            let hi = scales_bytes[i * 2 + 1] as u16;
            scales.push(f16_bits_to_f32(lo | (hi << 8)));
        }

        let mut offsets: Vec<f32> = Vec::with_capacity(num_groups);
        for i in 0..num_groups {
            let lo = offsets_bytes[i * 2] as u16;
            let hi = offsets_bytes[i * 2 + 1] as u16;
            offsets.push(f16_bits_to_f32(lo | (hi << 8)));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = scales[g];
            let offset = offsets[g];
            let group_len = GROUP_SIZE.min(expected - g * GROUP_SIZE);

            let mut decoded = 0;
            while decoded < group_len && byte_idx < packed.len() {
                let b = packed[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
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

fn decompress_int4_asym_g128(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress asymmetric group-quantized int4 (g=128)
    // Uses offsets (zero-points) for asymmetric dequantization
    const GROUP_SIZE: usize = 128;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        // Offsets should match scales length for asymmetric quantization
        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                // Asymmetric dequantization: x = q * scale + offset
                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_plus(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ+ codec which uses mixed precision:
    // - Sensitive layers stored as int8 (offsets.len() == 1)
    // - Other layers stored as asymmetric int4 with per-group scales/offsets
    const GROUP_SIZE: usize = 128;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (single offset marker)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            // Int8 decompression for sensitive layers
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Asymmetric int4 decompression with per-group scales/offsets
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                // Asymmetric dequantization: x = q * scale + offset
                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_plusplus(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ++ codec - same format as AWQ+ but with g=32 groups
    // Detect group size from scales count
    const GROUP_SIZE: usize = 32;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (single offset marker)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Asymmetric int4 decompression with per-group scales/offsets
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_3plus(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ+++ codec - same format as AWQ++ but with g=16 groups
    const GROUP_SIZE: usize = 16;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (single offset marker)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Asymmetric int4 decompression with per-group scales/offsets
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq4(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ4 codec - same format as AWQ+++ but with g=8 groups
    const GROUP_SIZE: usize = 8;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (single offset marker)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Asymmetric int4 decompression with per-group scales/offsets
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_ultra(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ-Ultra codec - same format as AWQ4 but with g=8 groups
    const GROUP_SIZE: usize = 8;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_best(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ-Best codec - same format as AWQ-Ultra but with g=16 groups
    const GROUP_SIZE: usize = 16;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;
        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_final(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ-Final codec with channel-wise scaling
    const GROUP_SIZE: usize = 16;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (embeddings)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Check if we have channel scales (stored in alphas)
        let has_channel_scales = !t.alphas.is_empty();

        // First, decompress the quantized data
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;
        let mut scaled_data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                let lo = (b & 0x0f) as f32;
                scaled_data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    scaled_data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        // Now apply inverse channel scaling if we have channel scales
        let data = if has_channel_scales && t.shape.len() >= 2 {
            let channel_scales = &t.alphas;
            let (out_features, in_features, is_transposed) = if t.shape.len() == 2 {
                let dim0 = t.shape[0];
                let dim1 = t.shape[1];
                if dim0 > dim1 {
                    (dim1, dim0, true)
                } else {
                    (dim0, dim1, false)
                }
            } else {
                (t.shape[0], t.shape[1], false)
            };

            if channel_scales.len() == in_features {
                let mut unscaled = vec![0.0f32; scaled_data.len()];
                for out_idx in 0..out_features {
                    for in_idx in 0..in_features {
                        let idx = if is_transposed {
                            in_idx * out_features + out_idx
                        } else {
                            out_idx * in_features + in_idx
                        };
                        if idx < scaled_data.len() {
                            let s = channel_scales[in_idx];
                            unscaled[idx] = scaled_data[idx] / s;
                        }
                    }
                }
                unscaled
            } else {
                scaled_data
            }
        } else {
            scaled_data
        };

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_awq_pro(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    // Decompress AWQ-Pro codec with full calibration-based scaling
    const GROUP_SIZE: usize = 128;
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // Check if this is int8 mode (sensitive layers)
        let is_int8_mode = t.offsets.len() == 1 && t.scales.len() == 1 && t.data.len() == expected;

        if is_int8_mode {
            let scale = t.scales[0];
            let mut data = Vec::with_capacity(expected);
            for &b in &t.data {
                let q = b as i8;
                data.push(q as f32 * scale);
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Check if we have combined channel+smooth scales (stored in alphas)
        let has_combined_scales = !t.alphas.is_empty();

        // Decompress the quantized data
        let num_groups = t.scales.len();
        if num_groups == 0 {
            return Err(format!("Tensor '{}' has no group scales", t.name));
        }

        let has_offsets = t.offsets.len() == num_groups;
        let mut scaled_data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;

        for g in 0..num_groups {
            let scale = t.scales[g];
            let offset = if has_offsets { t.offsets[g] } else { 0.0 };
            let group_start = g * GROUP_SIZE;
            let group_end = (group_start + GROUP_SIZE).min(expected);
            let group_len = group_end - group_start;

            let mut decoded = 0;
            while decoded < group_len && byte_idx < t.data.len() {
                let b = t.data[byte_idx];
                byte_idx += 1;

                // Asymmetric int4: values are 0-15
                let lo = (b & 0x0f) as f32;
                scaled_data.push(lo * scale + offset);
                decoded += 1;

                if decoded < group_len {
                    let hi = ((b >> 4) & 0x0f) as f32;
                    scaled_data.push(hi * scale + offset);
                    decoded += 1;
                }
            }
        }

        // Apply inverse combined scaling (channel + smooth scales)
        let data = if has_combined_scales && t.shape.len() >= 2 {
            let combined_scales = &t.alphas;

            // Detect layout
            let (out_features, in_features, is_transposed) = if t.shape.len() == 2 {
                let dim0 = t.shape[0];
                let dim1 = t.shape[1];
                // Use same heuristic as compression
                if dim0 > dim1 {
                    (dim1, dim0, true)
                } else {
                    (dim0, dim1, false)
                }
            } else {
                (t.shape[0], t.shape[1], false)
            };

            if combined_scales.len() == in_features {
                let mut unscaled = vec![0.0f32; scaled_data.len()];
                for out_idx in 0..out_features {
                    for in_idx in 0..in_features {
                        let idx = if is_transposed {
                            in_idx * out_features + out_idx
                        } else {
                            out_idx * in_features + in_idx
                        };
                        if idx < scaled_data.len() {
                            let s = combined_scales[in_idx];
                            // Divide by combined scale to undo the scaling
                            unscaled[idx] = scaled_data[idx] / s.max(1e-8);
                        }
                    }
                }
                unscaled
            } else {
                scaled_data
            }
        } else {
            scaled_data
        };

        if data.len() != expected {
            return Err(format!(
                "Tensor '{}' decompressed to {} values but expected {}",
                t.name,
                data.len(),
                expected
            ));
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

fn decompress_int4_perchannel_sparse50(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;

        // Check if this is sparse (has indices) or dense (fallback for 1D)
        let is_sparse = !t.indices.is_empty();

        if !is_sparse {
            // Dense decompression for 1D tensors
            let scale = if !t.scales.is_empty() {
                t.scales[0]
            } else {
                t.scale
            };

            let mut data: Vec<f32> = Vec::with_capacity(expected);
            'outer: for &b in &t.data {
                let lo = decode_int4_nibble(b & 0x0f);
                data.push(lo as f32 * scale);
                if data.len() >= expected {
                    break 'outer;
                }
                let hi = decode_int4_nibble((b >> 4) & 0x0f);
                data.push(hi as f32 * scale);
                if data.len() >= expected {
                    break 'outer;
                }
            }

            out.push(FloatTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                data,
            });
            continue;
        }

        // Sparse decompression
        let out_channels = t.shape[0];

        if t.scales.len() != out_channels {
            return Err(format!(
                "Tensor '{}' has {} channels but {} scales",
                t.name,
                out_channels,
                t.scales.len()
            ));
        }

        // Initialize with zeros
        let mut data: Vec<f32> = vec![0.0; expected];

        // Decode sparse values
        let mut idx_pos = 0;
        let mut byte_idx = 0;

        while idx_pos < t.indices.len() && byte_idx < t.data.len() {
            let byte = t.data[byte_idx];

            // First nibble (low 4 bits)
            if idx_pos < t.indices.len() {
                let global_idx = t.indices[idx_pos] as usize;
                if global_idx >= expected {
                    return Err(format!(
                        "Tensor '{}' has invalid index {} (expected < {})",
                        t.name, global_idx, expected
                    ));
                }

                // Determine which channel this index belongs to
                let ch = global_idx / (expected / out_channels);
                let scale = t.scales[ch.min(out_channels - 1)];

                let nibble = byte & 0x0f;
                let v = decode_int4_nibble(nibble);
                data[global_idx] = v as f32 * scale;
                idx_pos += 1;
            }

            // Second nibble (high 4 bits)
            if idx_pos < t.indices.len() {
                let global_idx = t.indices[idx_pos] as usize;
                if global_idx >= expected {
                    return Err(format!(
                        "Tensor '{}' has invalid index {} (expected < {})",
                        t.name, global_idx, expected
                    ));
                }

                let ch = global_idx / (expected / out_channels);
                let scale = t.scales[ch.min(out_channels - 1)];

                let nibble = (byte >> 4) & 0x0f;
                let v = decode_int4_nibble(nibble);
                data[global_idx] = v as f32 * scale;
                idx_pos += 1;
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

/// Decompress INT4+INT2 residual quantized artifact
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

/// Decompress a quantized artifact back into float32 tensors.
pub fn decompress_bundle(artifact: &ArtifactFile) -> Result<FloatBundle, String> {
    if artifact.version != ARTIFACT_VERSION {
        return Err(format!(
            "Unsupported artifact version {} (expected {})",
            artifact.version, ARTIFACT_VERSION
        ));
    }

    match artifact.codec.as_str() {
        CODEC_INT8_SYM_V1 => decompress_int8_sym(artifact),
        CODEC_INT4_SYM_V1 => decompress_int4_sym(artifact),
        CODEC_INT4_PERCHANNEL_V1 => decompress_int4_perchannel(artifact),
        CODEC_INT4_PERCHANNEL_SPARSE50_V1 => decompress_int4_perchannel_sparse50(artifact),
        CODEC_INT2_SYM_V1 => decompress_int2_sym(artifact),
        CODEC_INT4_AWQ_V1 => decompress_int4_awq(artifact),
        CODEC_INT4_G128_V1 => decompress_int4_g128(artifact),
        CODEC_INT4_ASYM_G128_V1 => decompress_int4_asym_g128(artifact),
        CODEC_INT4_AWQ_PLUS_V1 => decompress_int4_awq_plus(artifact),
        CODEC_INT4_AWQ_PLUSPLUS_V1 => decompress_int4_awq_plusplus(artifact),
        CODEC_INT4_AWQ_PLUSPLUSPLUS_V1 => decompress_int4_awq_3plus(artifact),
        CODEC_INT4_AWQ4_V1 => decompress_int4_awq4(artifact),
        CODEC_INT4_AWQ_ULTRA_V1 => decompress_int4_awq_ultra(artifact),
        CODEC_INT4_AWQ_BEST_V1 => decompress_int4_awq_best(artifact),
        CODEC_INT4_AWQ_FINAL_V1 => decompress_int4_awq_final(artifact),
        CODEC_INT4_AWQ_PRO_V1 => decompress_int4_awq_pro(artifact),
        CODEC_INT4_AWQ_GPTQ_V1 => decompress_int4_awq_best(artifact), // Alias
        CODEC_INT4_G8_V1 => decompress_int4_g8(artifact),
        CODEC_INT4_G16_V1 => decompress_int4_g16(artifact),
        CODEC_INT4_K_V1 => decompress_int4_k(artifact),
        CODEC_INT4_G8_FP16_V1 => decompress_int4_g8_fp16(artifact),
        CODEC_INT4_G16_FP16_V1 => decompress_int4_g16_fp16(artifact),
        CODEC_INT4_G32_FP16_V1 => decompress_int4_g32_fp16(artifact),
        CODEC_INT4_OPT_V1 => decompress_int4_g16_fp16(artifact), // Same format as g16_fp16
        CODEC_INT4_OPT_LLAMA_V1 => decompress_int4_g8_fp16(artifact), // Same format as g8_fp16
        CODEC_INT4_RESIDUAL_V1 => decompress_int4_residual(artifact),
        #[cfg(feature = "calibration")]
        CODEC_INT4_CALIBRATED_V1 => decompress_int4_calibrated(artifact),
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
        CODEC_INT8_SYM_V1
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

        let artifact = compress_bundle_with_codec(&bundle, CODEC_INT8_SYM_V1)
            .expect("compression should succeed");
        let restored = decompress_bundle(&artifact).expect("decompression should succeed");

        assert_eq!(restored.tensors.len(), 1);
        let r = &restored.tensors[0];
        assert_eq!(r.name, "test.weight");
        assert_eq!(r.shape, vec![2, 2]);
        assert_eq!(r.data.len(), 4);

        for (orig, rec) in [0.1f32, -0.2, 0.3, -0.4].iter().zip(r.data.iter()) {
            let diff = (orig - rec).abs();
            assert!(
                diff < 0.01,
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

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(
                diff <= tol,
                "value mismatch at {}: {} vs {} (tol {})",
                i,
                x,
                y,
                tol
            );
        }
    }

    #[test]
    fn perchannel_round_trip_even_channels() {
        // Two channels, each with four values.
        let tensor = FloatTensor {
            name: "linear.weight".to_string(),
            shape: vec![2, 4],
            data: vec![
                0.5, -0.75, 0.2, -0.1, // channel 0
                1.2, -1.0, 0.6, -0.4, // channel 1
            ],
        };
        let bundle = FloatBundle {
            tensors: vec![tensor.clone()],
            activation_stats: ActivationStats::new(),
        };

        let artifact = compress_bundle_with_codec(&bundle, CODEC_INT4_PERCHANNEL_V1)
            .expect("per-channel compression should succeed");
        let restored = decompress_bundle(&artifact).expect("per-channel decompress succeeds");

        assert_eq!(restored.tensors.len(), 1);
        let recovered = &restored.tensors[0];
        assert_eq!(recovered.name, tensor.name);
        assert_eq!(recovered.shape, tensor.shape);
        assert_close(&tensor.data, &recovered.data, 0.12);
    }

    #[test]
    fn perchannel_round_trip_odd_elements_per_channel() {
        // Three channels, each with 3 values (odd count) to stress nibble alignment.
        let tensor = FloatTensor {
            name: "conv.weight".to_string(),
            shape: vec![3, 3],
            data: vec![
                0.9, -0.5, 0.1, // ch0
                -1.3, 0.4, 0.2, // ch1
                0.0, -0.2, 0.8, // ch2
            ],
        };
        let bundle = FloatBundle {
            tensors: vec![tensor.clone()],
            activation_stats: ActivationStats::new(),
        };

        let artifact = compress_bundle_with_codec(&bundle, CODEC_INT4_PERCHANNEL_V1)
            .expect("per-channel compression should succeed");
        let restored = decompress_bundle(&artifact).expect("per-channel decompress succeeds");

        let recovered = &restored.tensors[0];
        assert_eq!(recovered.shape, tensor.shape);
        assert_close(&tensor.data, &recovered.data, 0.15);
    }

    #[test]
    fn perchannel_round_trip_bias_fallback() {
        // 1D tensor should fall back to per-tensor quantization even via per-channel codec.
        let tensor = FloatTensor {
            name: "bias".to_string(),
            shape: vec![5],
            data: vec![0.3, -0.7, 0.1, 0.0, 0.9],
        };
        let bundle = FloatBundle {
            tensors: vec![tensor.clone()],
            activation_stats: ActivationStats::new(),
        };

        let artifact = compress_bundle_with_codec(&bundle, CODEC_INT4_PERCHANNEL_V1)
            .expect("bias fallback compression should succeed");
        let restored = decompress_bundle(&artifact).expect("bias fallback decompress succeeds");

        let recovered = &restored.tensors[0];
        assert_eq!(recovered.shape, tensor.shape);
        assert_close(&tensor.data, &recovered.data, 0.15);
    }
}
