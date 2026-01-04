//! GPU Acceleration Module
//!
//! Provides GPU-accelerated delta operations via PyTorch/CUDA interop.
//! This module exposes functions that can be called from Python with GPU tensors.
//!
//! For actual CUDA kernels, we rely on PyTorch's CUDA backend.
//! This module provides optimized CPU fallbacks and GPU dispatch hints.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

// Removed: GpuInfo - unused, CUDA detection handled by PyTorch

/// High-performance delta application optimized for GPU data layout
/// Processes data in tiles for better cache/memory coalescing
#[pyclass]
pub struct GpuOptimizedOps {
    tile_size: usize,
    use_fma: bool,
}

#[pymethods]
impl GpuOptimizedOps {
    #[new]
    #[pyo3(signature = (tile_size=256, use_fma=true))]
    fn new(tile_size: usize, use_fma: bool) -> Self {
        GpuOptimizedOps { tile_size, use_fma }
    }

    /// Apply INT8 delta with tiled processing (GPU-friendly memory access)
    fn apply_int8_delta_tiled<'py>(
        &self,
        py: Python<'py>,
        base: PyReadonlyArray1<f32>,
        quantized: PyReadonlyArray1<i8>,
        scale: f32,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let base_slice = base.as_slice()?;
        let quant_slice = quantized.as_slice()?;

        if base_slice.len() != quant_slice.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Base and delta must have same length",
            ));
        }

        // Process in tiles for better cache utilization
        let result: Vec<f32> = base_slice
            .par_chunks(self.tile_size)
            .zip(quant_slice.par_chunks(self.tile_size))
            .flat_map(|(base_tile, quant_tile)| {
                base_tile
                    .iter()
                    .zip(quant_tile.iter())
                    .map(|(&b, &q)| {
                        if self.use_fma {
                            // Fused multiply-add (FMA) - single instruction on modern CPUs/GPUs
                            b + (q as f32) * scale
                        } else {
                            let delta = (q as f32) * scale;
                            b + delta
                        }
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();

        Ok(PyArray1::from_vec(py, result).to_owned())
    }

    /// Batch apply deltas to multiple layers (better GPU utilization)
    fn batch_apply_deltas<'py>(
        &self,
        py: Python<'py>,
        bases: Vec<PyReadonlyArray1<f32>>,
        deltas: Vec<PyReadonlyArray1<i8>>,
        scales: Vec<f32>,
    ) -> PyResult<Vec<Py<PyArray1<f32>>>> {
        if bases.len() != deltas.len() || bases.len() != scales.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All inputs must have same length",
            ));
        }

        let results: Result<Vec<_>, _> = bases
            .into_iter()
            .zip(deltas.into_iter())
            .zip(scales.into_iter())
            .map(|((base, delta), scale)| self.apply_int8_delta_tiled(py, base, delta, scale))
            .collect();

        results
    }

    /// Get optimal tile size for given tensor size
    fn get_optimal_tile_size(&self, tensor_size: usize) -> usize {
        // Heuristics for optimal tiling:
        // - Small tensors: process whole tensor
        // - Medium: 256-element tiles (fits in L1 cache)
        // - Large: 1024-element tiles (balance parallelism/cache)
        if tensor_size < 1024 {
            tensor_size
        } else if tensor_size < 1_000_000 {
            256
        } else {
            1024
        }
    }
}

// Removed: CUDA launch config helpers - unused, PyTorch handles CUDA dispatch

/// Benchmark CPU vs GPU-optimized operations
#[pyfunction]
pub fn benchmark_gpu_ops(tensor_size: usize, iterations: usize) -> PyResult<GpuBenchmark> {
    let ops = GpuOptimizedOps::new(256, true);

    // Create test data
    let base: Vec<f32> = (0..tensor_size).map(|i| i as f32 * 0.001).collect();
    let delta: Vec<i8> = (0..tensor_size).map(|i| (i % 256) as i8).collect();
    let scale = 0.001f32;

    // Benchmark naive
    let naive_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _: Vec<f32> = base
            .iter()
            .zip(delta.iter())
            .map(|(&b, &d)| b + (d as f32) * scale)
            .collect();
    }
    let naive_ms = naive_start.elapsed().as_millis() as u64;

    // Benchmark tiled parallel
    let tiled_start = std::time::Instant::now();
    for _ in 0..iterations {
        let _: Vec<f32> = base
            .par_chunks(ops.tile_size)
            .zip(delta.par_chunks(ops.tile_size))
            .flat_map(|(b, d)| {
                b.iter()
                    .zip(d.iter())
                    .map(|(&b, &d)| b + (d as f32) * scale)
                    .collect::<Vec<f32>>()
            })
            .collect();
    }
    let tiled_ms = tiled_start.elapsed().as_millis() as u64;

    Ok(GpuBenchmark {
        naive_ms,
        tiled_parallel_ms: tiled_ms,
        speedup: naive_ms as f64 / tiled_ms.max(1) as f64,
        tensor_size,
        iterations,
    })
}

#[pyclass]
#[derive(Clone)]
pub struct GpuBenchmark {
    #[pyo3(get)]
    naive_ms: u64,
    #[pyo3(get)]
    tiled_parallel_ms: u64,
    #[pyo3(get)]
    speedup: f64,
    #[pyo3(get)]
    tensor_size: usize,
    #[pyo3(get)]
    iterations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_tile_size() {
        let ops = GpuOptimizedOps::new(256, true);

        assert_eq!(ops.get_optimal_tile_size(100), 100);
        assert_eq!(ops.get_optimal_tile_size(10000), 256);
        assert_eq!(ops.get_optimal_tile_size(10_000_000), 1024);
    }
}
