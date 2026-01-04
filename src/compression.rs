//! Zstd compression module for delta files
//!
//! Provides high-performance compression/decompression using zstd.
//! Typically achieves 2x additional compression on INT8 deltas.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::{Read, Write};

/// Compression level presets
#[derive(Clone, Copy)]
pub enum CompressionLevel {
    Fast = 1,
    Default = 3,
    Best = 19,
}

/// Internal: Compress data using zstd (returns Vec)
fn compress_zstd_internal(data: &[u8], level: i32) -> Result<Vec<u8>, String> {
    let mut encoder =
        zstd::Encoder::new(Vec::new(), level).map_err(|e| format!("Zstd init failed: {}", e))?;

    encoder
        .write_all(data)
        .map_err(|e| format!("Zstd write failed: {}", e))?;

    encoder
        .finish()
        .map_err(|e| format!("Zstd finish failed: {}", e))
}

/// Internal: Decompress zstd data (returns Vec)
fn decompress_zstd_internal(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoder =
        zstd::Decoder::new(data).map_err(|e| format!("Zstd decode init failed: {}", e))?;

    let mut decompressed = Vec::new();
    decoder
        .read_to_end(&mut decompressed)
        .map_err(|e| format!("Zstd decompress failed: {}", e))?;

    Ok(decompressed)
}

/// Compress data using zstd
#[pyfunction]
#[pyo3(signature = (data, level=3))]
pub fn compress_zstd<'py>(py: Python<'py>, data: &[u8], level: i32) -> PyResult<&'py PyBytes> {
    let compressed = compress_zstd_internal(data, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    Ok(PyBytes::new(py, &compressed))
}

/// Decompress zstd data
#[pyfunction]
pub fn decompress_zstd<'py>(py: Python<'py>, data: &[u8]) -> PyResult<&'py PyBytes> {
    let decompressed = decompress_zstd_internal(data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    Ok(PyBytes::new(py, &decompressed))
}

/// Compress INT8 delta tensor with zstd
#[pyfunction]
#[pyo3(signature = (quantized, scale, level=3))]
pub fn compress_int8_delta_zstd<'py>(
    py: Python<'py>,
    quantized: Vec<i8>,
    scale: f32,
    level: i32,
) -> PyResult<(&'py PyBytes, f32)> {
    let bytes: Vec<u8> = quantized.iter().map(|&x| x as u8).collect();
    let compressed = compress_zstd_internal(&bytes, level)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    Ok((PyBytes::new(py, &compressed), scale))
}

/// Decompress INT8 delta tensor from zstd
#[pyfunction]
pub fn decompress_int8_delta_zstd(compressed: &[u8], scale: f32) -> PyResult<(Vec<i8>, f32)> {
    let decompressed = decompress_zstd_internal(compressed)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
    let quantized: Vec<i8> = decompressed.iter().map(|&x| x as i8).collect();
    Ok((quantized, scale))
}

/// Get compression stats
#[pyfunction]
pub fn get_compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
    if compressed_size == 0 {
        return f64::INFINITY;
    }
    original_size as f64 / compressed_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let compressed = compress_zstd_internal(&data, 3).unwrap();
        let decompressed = decompress_zstd_internal(&compressed).unwrap();
        assert_eq!(data, decompressed);
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_int8_delta_roundtrip() {
        let quantized: Vec<i8> = (0..1000)
            .map(|i| ((i % 256) as u8).wrapping_sub(128) as i8)
            .collect();
        let bytes: Vec<u8> = quantized.iter().map(|&x| x as u8).collect();
        let compressed = compress_zstd_internal(&bytes, 3).unwrap();
        let decompressed = decompress_zstd_internal(&compressed).unwrap();
        let recovered: Vec<i8> = decompressed.iter().map(|&x| x as i8).collect();
        assert_eq!(quantized, recovered);
    }
}
