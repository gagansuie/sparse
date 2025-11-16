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
}

/// A quantized tensor: encoded values plus a per-tensor scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub scale: f32,
    /// Raw encoded bytes. Interpretation depends on the codec.
    pub data: Vec<u8>,
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
            data: encoded,
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
            data: packed,
        });
    }

    Ok(ArtifactFile {
        version: ARTIFACT_VERSION,
        codec: CODEC_INT4_SYM_V1.to_string(),
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

    Ok(FloatBundle { tensors: out })
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

    Ok(FloatBundle { tensors: out })
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

    let delta_bundle = FloatBundle { tensors: changed };
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
        };

        let res = compress_bundle(&bundle);
        assert!(res.is_err(), "expected shape mismatch to produce an error");
    }
}
