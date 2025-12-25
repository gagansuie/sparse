use pyo3::prelude::*;

mod compress;
mod decompress;
mod quantize;
mod utils;

pub use compress::*;
pub use decompress::*;
pub use quantize::*;

/// Sparse Core - High-performance delta compression and quantization
#[pymodule]
fn sparse_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SparseDeltaCompressor>()?;
    m.add_class::<Int8Quantizer>()?;
    m.add_function(wrap_pyfunction!(compress_sparse_delta, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_sparse_delta, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8, m)?)?;
    Ok(())
}
