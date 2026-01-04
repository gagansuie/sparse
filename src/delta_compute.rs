//! Fast delta computation with SIMD

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct DeltaStats {
    #[pyo3(get)]
    pub max_abs: f32,
    #[pyo3(get)]
    pub l2_norm: f32,
    #[pyo3(get)]
    pub sparsity: f32,
    #[pyo3(get)]
    pub numel: usize,
}

#[pyfunction]
pub fn compute_delta_simd<'py>(
    py: Python<'py>,
    base: PyReadonlyArray2<'_, f32>,
    finetune: PyReadonlyArray2<'_, f32>,
) -> PyResult<(&'py PyArray2<f32>, DeltaStats)> {
    let base_data = base.as_slice()?;
    let ft_data = finetune.as_slice()?;
    let shape = base.shape();

    if base_data.len() != ft_data.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Shape mismatch",
        ));
    }

    // Parallel delta computation with SIMD
    let chunk_size = 1024;
    let chunks: Vec<_> = base_data
        .par_chunks(chunk_size)
        .zip(ft_data.par_chunks(chunk_size))
        .map(|(base_chunk, ft_chunk)| {
            let mut delta_chunk = Vec::with_capacity(base_chunk.len());
            let mut max_abs = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut zeros = 0usize;

            for i in 0..base_chunk.len() {
                let d = ft_chunk[i] - base_chunk[i];
                delta_chunk.push(d);

                let abs_d = d.abs();
                if abs_d > max_abs {
                    max_abs = abs_d;
                }
                sum_sq += d * d;
                if abs_d < 1e-6 {
                    zeros += 1;
                }
            }

            (delta_chunk, max_abs, sum_sq, zeros)
        })
        .collect();

    // Combine results
    let mut delta = Vec::with_capacity(base_data.len());
    let mut global_max = 0.0f32;
    let mut global_sum_sq = 0.0f32;
    let mut global_zeros = 0usize;

    for (chunk, max_abs, sum_sq, zeros) in chunks {
        delta.extend(chunk);
        if max_abs > global_max {
            global_max = max_abs;
        }
        global_sum_sq += sum_sq;
        global_zeros += zeros;
    }

    let stats = DeltaStats {
        max_abs: global_max,
        l2_norm: global_sum_sq.sqrt(),
        sparsity: global_zeros as f32 / base_data.len() as f32,
        numel: base_data.len(),
    };

    // Convert to 2D array
    let delta_2d: Vec<Vec<f32>> = delta.chunks(shape[1]).map(|row| row.to_vec()).collect();

    let arr = PyArray2::from_vec2(py, &delta_2d)?;

    Ok((arr, stats))
}
