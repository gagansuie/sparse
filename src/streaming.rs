//! Streaming Reconstruction Module
//!
//! Pipelines I/O and compute for large model reconstruction.
//! Uses async channels to overlap disk reads with delta application.

use half::f16;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::Instant;

/// A chunk of work for the streaming pipeline
#[derive(Clone)]
pub struct LayerChunk {
    pub name: String,
    pub base_data: Vec<f16>,
    pub delta_data: Option<Vec<i8>>,
    pub scale: f32,
}

/// Result of processing a layer
pub struct ProcessedLayer {
    pub name: String,
    pub data: Vec<f16>,
}

/// Streaming reconstruction pipeline
#[pyclass]
pub struct StreamingReconstructor {
    num_workers: usize,
    chunk_size: usize,
    #[allow(dead_code)]
    prefetch_count: usize,
}

#[pymethods]
impl StreamingReconstructor {
    #[new]
    #[pyo3(signature = (num_workers=4, chunk_size=1048576, prefetch_count=2))]
    fn new(num_workers: usize, chunk_size: usize, prefetch_count: usize) -> Self {
        StreamingReconstructor {
            num_workers,
            chunk_size,
            prefetch_count,
        }
    }

    /// Process layers in a streaming pipeline
    fn process_layers_streaming(&self, layer_count: usize) -> PyResult<StreamingStats> {
        let start = Instant::now();

        // Create channels for pipeline stages
        let (load_tx, load_rx): (Sender<LayerChunk>, Receiver<LayerChunk>) = channel();
        let (process_tx, process_rx): (Sender<ProcessedLayer>, Receiver<ProcessedLayer>) =
            channel();

        // Stage 1: Simulate loading (would be actual I/O in production)
        let chunk_size = self.chunk_size;
        let loader_handle = thread::spawn(move || {
            for i in 0..layer_count {
                let chunk = LayerChunk {
                    name: format!("layer_{}", i),
                    base_data: vec![f16::from_f32(0.0); chunk_size],
                    delta_data: Some(vec![1i8; chunk_size]),
                    scale: 0.001,
                };
                if load_tx.send(chunk).is_err() {
                    break;
                }
            }
        });

        // Stage 2: Process deltas in parallel
        let num_workers = self.num_workers;
        let processor_handle = thread::spawn(move || {
            // Use rayon thread pool for parallel processing
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_workers)
                .build()
                .unwrap()
                .install(|| {
                    for chunk in load_rx {
                        let mut data = chunk.base_data;

                        if let Some(delta) = chunk.delta_data {
                            // Apply delta in parallel
                            data.par_iter_mut()
                                .zip(delta.par_iter())
                                .for_each(|(b, &d)| {
                                    *b = f16::from_f32(b.to_f32() + (d as f32) * chunk.scale);
                                });
                        }

                        let processed = ProcessedLayer {
                            name: chunk.name,
                            data,
                        };

                        if process_tx.send(processed).is_err() {
                            break;
                        }
                    }
                });
        });

        // Stage 3: Collect results (would be saving in production)
        let mut processed_count = 0;
        let mut total_elements = 0;

        for result in process_rx {
            processed_count += 1;
            total_elements += result.data.len();
        }

        loader_handle.join().ok();
        processor_handle.join().ok();

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(StreamingStats {
            layers_processed: processed_count,
            total_elements,
            elapsed_ms,
            throughput_mbs: (total_elements * 2) as f64 / (elapsed_ms as f64 / 1000.0) / 1e6,
        })
    }

    /// Benchmark streaming vs sequential processing
    fn benchmark(
        &self,
        layer_count: usize,
        elements_per_layer: usize,
    ) -> PyResult<BenchmarkResult> {
        // Sequential baseline
        let seq_start = Instant::now();
        let mut seq_data: Vec<f16> = vec![f16::from_f32(0.0); elements_per_layer];
        let delta: Vec<i8> = vec![1i8; elements_per_layer];

        for _ in 0..layer_count {
            for (b, &d) in seq_data.iter_mut().zip(delta.iter()) {
                *b = f16::from_f32(b.to_f32() + (d as f32) * 0.001);
            }
        }
        let seq_ms = seq_start.elapsed().as_millis() as u64;

        // Parallel baseline
        let par_start = Instant::now();
        let mut par_data: Vec<f16> = vec![f16::from_f32(0.0); elements_per_layer];

        for _ in 0..layer_count {
            par_data
                .par_iter_mut()
                .zip(delta.par_iter())
                .for_each(|(b, &d)| {
                    *b = f16::from_f32(b.to_f32() + (d as f32) * 0.001);
                });
        }
        let par_ms = par_start.elapsed().as_millis() as u64;

        Ok(BenchmarkResult {
            sequential_ms: seq_ms,
            parallel_ms: par_ms,
            speedup: seq_ms as f64 / par_ms.max(1) as f64,
            elements_processed: layer_count * elements_per_layer,
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct StreamingStats {
    #[pyo3(get)]
    layers_processed: usize,
    #[pyo3(get)]
    total_elements: usize,
    #[pyo3(get)]
    elapsed_ms: u64,
    #[pyo3(get)]
    throughput_mbs: f64,
}

#[pyclass]
#[derive(Clone)]
pub struct BenchmarkResult {
    #[pyo3(get)]
    sequential_ms: u64,
    #[pyo3(get)]
    parallel_ms: u64,
    #[pyo3(get)]
    speedup: f64,
    #[pyo3(get)]
    elements_processed: usize,
}

/// Prefetch buffer for streaming I/O
pub struct PrefetchBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> PrefetchBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        PrefetchBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) -> bool {
        if self.buffer.len() < self.capacity {
            self.buffer.push_back(item);
            true
        } else {
            false
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_front()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_buffer() {
        let mut buffer: PrefetchBuffer<i32> = PrefetchBuffer::new(3);
        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));
        assert!(!buffer.push(4)); // Full

        assert_eq!(buffer.pop(), Some(1));
        assert!(buffer.push(4)); // Now has space
    }

    #[test]
    fn test_layer_chunk() {
        let chunk = LayerChunk {
            name: "test".to_string(),
            base_data: vec![f16::from_f32(1.0); 100],
            delta_data: Some(vec![10i8; 100]),
            scale: 0.1,
        };

        assert_eq!(chunk.base_data.len(), 100);
        assert_eq!(chunk.delta_data.as_ref().unwrap().len(), 100);
    }
}
