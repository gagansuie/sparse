use pyo3::prelude::*;

mod compress;
mod compression;
mod decompress;
mod delta_compute;
mod gpu;
mod quantize;
mod reconstruct;
mod streaming;
mod utils;

pub use compress::*;
pub use compression::*;
pub use decompress::*;
pub use delta_compute::*;
pub use gpu::*;
pub use quantize::*;
pub use reconstruct::*;
pub use streaming::*;

/// Sparse Core - High-performance delta compression and quantization
#[pymodule]
fn sparse_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core compression classes
    m.add_class::<SparseDeltaCompressor>()?;
    m.add_class::<Int8Quantizer>()?;
    m.add_class::<ReconstructionResult>()?;

    // INT8 quantization
    m.add_function(wrap_pyfunction!(compress_sparse_delta, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_sparse_delta, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8, m)?)?;

    // Fast reconstruction functions (for <10s delta application)
    m.add_function(wrap_pyfunction!(apply_sparse_delta_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(apply_int8_delta_inplace, m)?)?;
    m.add_function(wrap_pyfunction!(apply_deltas_batch, m)?)?;

    // Full Rust reconstruction (safetensors-based)
    m.add_function(wrap_pyfunction!(reconstruct_model_fast, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_int8_apply, m)?)?;

    // Zstd compression
    m.add_function(wrap_pyfunction!(compress_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(compress_int8_delta_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_int8_delta_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(get_compression_ratio, m)?)?;

    // Streaming reconstruction
    m.add_class::<StreamingReconstructor>()?;
    m.add_class::<StreamingStats>()?;
    m.add_class::<BenchmarkResult>()?;

    // GPU acceleration
    m.add_class::<GpuOptimizedOps>()?;
    m.add_class::<GpuBenchmark>()?;
    m.add_function(wrap_pyfunction!(benchmark_gpu_ops, m)?)?;

    // Fast delta computation
    m.add_class::<DeltaStats>()?;
    m.add_function(wrap_pyfunction!(compute_delta_simd, m)?)?;

    Ok(())
}
