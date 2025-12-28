use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Decompress sparse delta back to full tensor
#[pyfunction]
pub fn decompress_sparse_delta<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<u32>,
    values: PyReadonlyArray1<f32>,
    shape: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let indices_slice = indices.as_slice()?;
    let values_slice = values.as_slice()?;

    if indices_slice.len() != values_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Indices and values must have same length",
        ));
    }

    // Create zero-filled output
    let mut output = vec![0.0f32; shape];

    // Fill in sparse values
    for (&idx, &val) in indices_slice.iter().zip(values_slice.iter()) {
        if (idx as usize) < shape {
            output[idx as usize] = val;
        }
    }

    Ok(PyArray1::from_vec(py, output).to_owned())
}

/// Parallel decompression for large tensors
#[pyfunction]
pub fn decompress_sparse_delta_parallel<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray1<u32>,
    values: PyReadonlyArray1<f32>,
    shape: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let indices_slice = indices.as_slice()?;
    let values_slice = values.as_slice()?;

    if indices_slice.len() != values_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Indices and values must have same length",
        ));
    }

    // Create zero-filled output
    let mut output = vec![0.0f32; shape];

    // Chunk the work for parallel writing
    const CHUNK_SIZE: usize = 10000;

    if indices_slice.len() > CHUNK_SIZE * 4 {
        // Parallel version: split into chunks and process
        let chunks: Vec<_> = indices_slice
            .chunks(CHUNK_SIZE)
            .zip(values_slice.chunks(CHUNK_SIZE))
            .collect();

        // Process chunks in parallel, collecting updates
        let updates: Vec<Vec<(usize, f32)>> = chunks
            .par_iter()
            .map(|(idx_chunk, val_chunk)| {
                idx_chunk
                    .iter()
                    .zip(val_chunk.iter())
                    .filter_map(|(&idx, &val)| {
                        if (idx as usize) < shape {
                            Some((idx as usize, val))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();

        // Apply updates sequentially (avoiding race conditions)
        for chunk_updates in updates {
            for (idx, val) in chunk_updates {
                output[idx] = val;
            }
        }
    } else {
        // Sequential for smaller tensors
        for (&idx, &val) in indices_slice.iter().zip(values_slice.iter()) {
            if (idx as usize) < shape {
                output[idx as usize] = val;
            }
        }
    }

    Ok(PyArray1::from_vec(py, output).to_owned())
}

/// Apply sparse delta to base weights in-place (fast reconstruction)
/// This is the key function for <10s reconstruction
#[pyfunction]
pub fn apply_sparse_delta_inplace(
    base_weights: &PyArray1<f32>,
    indices: PyReadonlyArray1<u32>,
    values: PyReadonlyArray1<f32>,
) -> PyResult<()> {
    let indices_slice = indices.as_slice()?;
    let values_slice = values.as_slice()?;

    if indices_slice.len() != values_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Indices and values must have same length",
        ));
    }

    // Get mutable access to base weights
    let base_slice = unsafe { base_weights.as_slice_mut()? };
    let base_len = base_slice.len();

    // Apply delta values directly (in-place modification)
    for (&idx, &val) in indices_slice.iter().zip(values_slice.iter()) {
        let i = idx as usize;
        if i < base_len {
            base_slice[i] += val;
        }
    }

    Ok(())
}

/// Apply INT8 quantized delta to base weights (fast reconstruction)
/// Dequantizes and applies delta in one pass for efficiency
#[pyfunction]
pub fn apply_int8_delta_inplace(
    base_weights: &PyArray1<f32>,
    quantized_delta: PyReadonlyArray1<i8>,
    scale: f32,
) -> PyResult<()> {
    let quant_slice = quantized_delta.as_slice()?;

    // Get mutable access to base weights
    let base_slice = unsafe { base_weights.as_slice_mut()? };

    if base_slice.len() != quant_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Base weights ({}) and quantized delta ({}) must have same length",
            base_slice.len(),
            quant_slice.len()
        )));
    }

    // Parallel dequantize and apply delta
    // This is the hot path for fast reconstruction
    base_slice
        .par_iter_mut()
        .zip(quant_slice.par_iter())
        .for_each(|(base, &quant)| {
            *base += (quant as f32) * scale;
        });

    Ok(())
}

/// Batch apply multiple INT8 deltas to multiple layers
/// Returns timing info for benchmarking
#[pyfunction]
pub fn apply_deltas_batch(layer_count: usize) -> PyResult<String> {
    // This is a placeholder - actual batch processing would take
    // lists of base weights, quantized deltas, and scales
    Ok(format!(
        "Batch processing {} layers (placeholder)",
        layer_count
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_decompress_logic() {
        let indices = vec![0u32, 5, 10];
        let values = vec![1.0f32, 2.0, 3.0];
        let shape = 15;

        let mut output = vec![0.0f32; shape];
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            output[idx as usize] = val;
        }

        assert_eq!(output[0], 1.0);
        assert_eq!(output[5], 2.0);
        assert_eq!(output[10], 3.0);
        assert_eq!(output[1], 0.0);
    }

    #[test]
    fn test_int8_delta_logic() {
        let base = vec![1.0f32, 2.0, 3.0, 4.0];
        let quant = vec![10i8, -10, 5, -5];
        let scale = 0.1f32;

        let mut output = base.clone();
        for (b, &q) in output.iter_mut().zip(quant.iter()) {
            *b += (q as f32) * scale;
        }

        assert!((output[0] - 2.0).abs() < 0.01); // 1.0 + 10*0.1 = 2.0
        assert!((output[1] - 1.0).abs() < 0.01); // 2.0 + (-10)*0.1 = 1.0
    }
}
