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
}
