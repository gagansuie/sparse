use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Current on-disk artifact format version.
pub const ARTIFACT_VERSION: u32 = 1;
/// Symmetric per-tensor int8 codec identifier.
pub const CODEC_INT8_SYM_V1: &str = "int8_sym_v1";
/// Symmetric per-tensor int4 codec identifier (two 4-bit values per byte).
pub const CODEC_INT4_SYM_V1: &str = "int4_sym_v1";
/// Symmetric per-channel int4 codec identifier (per-output-channel quantization).
pub const CODEC_INT4_PERCHANNEL_V1: &str = "int4_perchannel_v1";
/// Per-channel int4 with 50% magnitude pruning (sparse).
pub const CODEC_INT4_PERCHANNEL_SPARSE50_V1: &str = "int4_perchannel_sparse50_v1";
/// Symmetric per-tensor int2 codec identifier (four 2-bit values per byte).
pub const CODEC_INT2_SYM_V1: &str = "int2_sym_v1";
/// Activation-aware int4 codec (AWQ-style with salient weight preservation).
pub const CODEC_INT4_AWQ_V1: &str = "int4_awq_v1";
/// Group-quantized int4 codec (g=128, similar to AWQ W4A16 group size).
pub const CODEC_INT4_G128_V1: &str = "int4_g128_v1";

/// A single float32 tensor in a simple JSON-friendly format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
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
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT2_SYM_V1.to_string(),
        tensors: tensors_out,
    })
}

fn compress_int4_awq(bundle: &FloatBundle) -> Result<ArtifactFile, String> {
    // True AWQ-style activation-aware quantization:
    // 1. Use activation stats to compute per-input-channel alphas
    // 2. Scale weights by alphas to equalize importance
    // 3. Quantize scaled weights with per-output-channel scales
    // 4. Store alphas for reconstruction

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

        // For 1D tensors (biases), fall back to per-tensor
        if t.shape.len() < 2 {
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
            });
            continue;
        }

        // AWQ per-channel quantization for 2D+ tensors
        let out_channels = t.shape[0];
        let in_features = t.shape[1];
        let elements_per_channel = expected / out_channels;

        // Step 1: Compute per-input-channel alphas from activation stats
        let act_stats = bundle.activation_stats.get(t.name.as_str());
        let alphas: Vec<f32> = if let Some(stats) = act_stats {
            if stats.len() == in_features {
                // Use activation magnitudes to compute alphas
                // alpha[i] = max(act_mean[i], epsilon) to avoid division by zero
                stats.iter().map(|&a| a.max(1e-5)).collect()
            } else {
                vec![1.0; in_features]
            }
        } else {
            vec![1.0; in_features]
        };

        // Step 2: Apply per-input-channel scaling (W' = W * diag(alpha))
        let mut scaled_data = t.data.clone();
        for ch in 0..out_channels {
            let start = ch * elements_per_channel;
            for i in 0..in_features {
                let idx = start + i;
                if idx < scaled_data.len() {
                    scaled_data[idx] *= alphas[i];
                }
            }
        }

        // Step 3: Compute per-output-channel scales on scaled weights
        let mut scales = Vec::with_capacity(out_channels);
        for ch in 0..out_channels {
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;
            let channel_data = &scaled_data[start..end];

            let max_abs = channel_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Step 4: Quantize scaled weights
        let mut packed: Vec<u8> = Vec::with_capacity((scaled_data.len() + 1) / 2);
        for ch in 0..out_channels {
            let inv_scale = 1.0 / scales[ch];
            let start = ch * elements_per_channel;
            let end = start + elements_per_channel;

            let mut ch_iter = scaled_data[start..end].iter();
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
            scale: scales[0],
            scales,
            data: packed,
            indices: Vec::new(),
            alphas,
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
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_G128_V1.to_string(),
        tensors: tensors_out,
    })
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
    // Decompress AWQ-quantized int4 with per-input-channel alpha unscaling
    let mut out = Vec::with_capacity(artifact.tensors.len());

    for t in &artifact.tensors {
        let expected = expected_len(&t.shape)?;
        if t.data.is_empty() && expected > 0 {
            return Err(format!(
                "Quantized tensor '{}' has no data but expected {} values",
                t.name, expected
            ));
        }

        // For 1D tensors, fall back to simple per-tensor decompression
        if t.shape.len() < 2 {
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

        // AWQ decompression for 2D+ tensors
        let out_channels = t.shape[0];
        let in_features = t.shape[1];
        let elements_per_channel = expected / out_channels;

        // Validate alphas
        if t.alphas.is_empty() {
            // No alphas stored, fall back to per-channel decompression
            return decompress_int4_perchannel(artifact);
        }

        if t.alphas.len() != in_features {
            return Err(format!(
                "Tensor '{}' has {} input features but {} alphas",
                t.name,
                in_features,
                t.alphas.len()
            ));
        }

        let mut data: Vec<f32> = Vec::with_capacity(expected);
        let mut byte_idx = 0;
        let bytes_per_channel = (elements_per_channel + 1) / 2;

        for ch in 0..out_channels {
            let scale = if !t.scales.is_empty() {
                t.scales[ch]
            } else {
                t.scale
            };

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
                let idx_in_channel = produced;
                let alpha = if idx_in_channel < t.alphas.len() {
                    t.alphas[idx_in_channel]
                } else {
                    1.0
                };
                // Undo alpha scaling: W = (Q * scale) / alpha
                data.push((lo as f32 * scale) / alpha);
                produced += 1;
                if produced >= elements_per_channel {
                    break;
                }

                let hi = decode_int4_nibble((b >> 4) & 0x0f);
                let idx_in_channel = produced;
                let alpha = if idx_in_channel < t.alphas.len() {
                    t.alphas[idx_in_channel]
                } else {
                    1.0
                };
                data.push((hi as f32 * scale) / alpha);
                produced += 1;
                if produced >= elements_per_channel {
                    break;
                }
            }

            byte_idx += bytes_per_channel;
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
