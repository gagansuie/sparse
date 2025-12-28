use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// High-performance INT8 quantizer
#[pyclass]
pub struct Int8Quantizer {
    parallel_threshold: usize,
}

#[pymethods]
impl Int8Quantizer {
    #[new]
    #[pyo3(signature = (parallel_threshold=10000))]
    fn new(parallel_threshold: usize) -> Self {
        Int8Quantizer { parallel_threshold }
    }

    /// Quantize FP32 tensor to INT8
    fn quantize<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
    ) -> PyResult<(Py<PyArray1<i8>>, f32)> {
        let data_slice = data.as_slice()?;

        // Find max absolute value for scaling
        let max_abs = if data_slice.len() > self.parallel_threshold {
            data_slice
                .par_iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        } else {
            data_slice
                .iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0)
        };

        let scale = if max_abs < 1e-10 {
            1.0
        } else {
            max_abs / 127.0
        };

        // Quantize
        let quantized: Vec<i8> = if data_slice.len() > self.parallel_threshold {
            data_slice
                .par_iter()
                .map(|&x| {
                    let scaled = x / scale;
                    scaled.clamp(-127.0, 127.0).round() as i8
                })
                .collect()
        } else {
            data_slice
                .iter()
                .map(|&x| {
                    let scaled = x / scale;
                    scaled.clamp(-127.0, 127.0).round() as i8
                })
                .collect()
        };

        Ok((PyArray1::from_vec(py, quantized).to_owned(), scale))
    }

    /// Dequantize INT8 back to FP32
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        quantized: PyReadonlyArray1<i8>,
        scale: f32,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let quantized_slice = quantized.as_slice()?;

        let dequantized: Vec<f32> = if quantized_slice.len() > self.parallel_threshold {
            quantized_slice
                .par_iter()
                .map(|&q| (q as f32) * scale)
                .collect()
        } else {
            quantized_slice
                .iter()
                .map(|&q| (q as f32) * scale)
                .collect()
        };

        Ok(PyArray1::from_vec(py, dequantized).to_owned())
    }
}

/// Standalone quantization function
#[pyfunction]
pub fn quantize_int8<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f32>,
) -> PyResult<(Py<PyArray1<i8>>, f32)> {
    let quantizer = Int8Quantizer::new(10000);
    quantizer.quantize(py, data)
}

/// Standalone dequantization function
#[pyfunction]
pub fn dequantize_int8<'py>(
    py: Python<'py>,
    quantized: PyReadonlyArray1<i8>,
    scale: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let quantizer = Int8Quantizer::new(10000);
    quantizer.dequantize(py, quantized, scale)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_quantize_logic() {
        let data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5];

        // Find max abs
        let max_abs = data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;

        // Quantize
        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| {
                let scaled = x / scale;
                scaled.clamp(-127.0f32, 127.0f32).round() as i8
            })
            .collect();

        // Dequantize
        let dequantized: Vec<f32> = quantized.iter().map(|&q| (q as f32) * scale).collect();

        // Check approximate equality
        for (original, deq) in data.iter().zip(dequantized.iter()) {
            assert!(
                (original - deq).abs() < 0.01,
                "Quantization error too large"
            );
        }
    }
}
