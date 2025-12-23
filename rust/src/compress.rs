use crate::utils::{find_nonzero, find_nonzero_parallel};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// High-performance sparse delta compressor
#[pyclass]
pub struct SparseDeltaCompressor {
    threshold: f32,
    parallel: bool,
}

#[pymethods]
impl SparseDeltaCompressor {
    #[new]
    #[pyo3(signature = (threshold=1e-6, parallel=true))]
    fn new(threshold: f32, parallel: bool) -> Self {
        SparseDeltaCompressor {
            threshold,
            parallel,
        }
    }

    /// Compress a 1D tensor delta to sparse representation
    fn compress_1d<'py>(
        &self,
        py: Python<'py>,
        delta: PyReadonlyArray1<f32>,
    ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<f32>>)> {
        let delta_view = delta.as_slice()?;

        let (indices, values) = if self.parallel && delta_view.len() > 10000 {
            find_nonzero_parallel(delta_view, self.threshold)
        } else {
            find_nonzero(delta_view, self.threshold)
        };

        let indices_py = PyArray1::from_vec(py, indices).to_owned();
        let values_py = PyArray1::from_vec(py, values).to_owned();

        Ok((indices_py, values_py))
    }

    /// Compress a 2D tensor delta to sparse representation
    fn compress_2d<'py>(
        &self,
        py: Python<'py>,
        delta: PyReadonlyArray2<f32>,
    ) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<f32>>)> {
        let delta_array = delta.as_array();
        let flattened = delta_array.as_slice().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Array must be contiguous")
        })?;

        let (indices, values) = if self.parallel && flattened.len() > 10000 {
            find_nonzero_parallel(flattened, self.threshold)
        } else {
            find_nonzero(flattened, self.threshold)
        };

        let indices_py = PyArray1::from_vec(py, indices).to_owned();
        let values_py = PyArray1::from_vec(py, values).to_owned();

        Ok((indices_py, values_py))
    }

    /// Get compression statistics
    fn get_stats(
        &self,
        original_size: usize,
        compressed_size: usize,
    ) -> PyResult<CompressionStats> {
        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            f64::INFINITY
        };

        let sparsity = 1.0 - (compressed_size as f64 / original_size as f64);

        Ok(CompressionStats {
            original_size,
            compressed_size,
            compression_ratio,
            sparsity,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CompressionStats {
    #[pyo3(get)]
    original_size: usize,
    #[pyo3(get)]
    compressed_size: usize,
    #[pyo3(get)]
    compression_ratio: f64,
    #[pyo3(get)]
    sparsity: f64,
}

/// Standalone function for quick compression
#[pyfunction]
#[pyo3(signature = (delta, threshold=1e-6, parallel=true))]
pub fn compress_sparse_delta<'py>(
    py: Python<'py>,
    delta: PyReadonlyArray1<f32>,
    threshold: f32,
    parallel: bool,
) -> PyResult<(Py<PyArray1<u32>>, Py<PyArray1<f32>>)> {
    let compressor = SparseDeltaCompressor::new(threshold, parallel);
    compressor.compress_1d(py, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression() {
        let compressor = SparseDeltaCompressor::new(1e-6, false);
        assert_eq!(compressor.threshold, 1e-6);
    }
}
