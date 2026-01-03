/// Utility functions for sparse operations

/// Count non-zero elements above threshold using SIMD-friendly loops
#[inline]
pub fn count_nonzero(data: &[f32], threshold: f32) -> usize {
    data.iter()
        .filter(|&&x| x.abs() >= threshold)
        .count()
}

/// Find indices and values above threshold
pub fn find_nonzero(data: &[f32], threshold: f32) -> (Vec<u32>, Vec<f32>) {
    let capacity = count_nonzero(data, threshold);
    let mut indices = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    
    for (idx, &val) in data.iter().enumerate() {
        if val.abs() >= threshold {
            indices.push(idx as u32);
            values.push(val);
        }
    }
    
    (indices, values)
}

/// Parallel version using rayon
pub fn find_nonzero_parallel(data: &[f32], threshold: f32) -> (Vec<u32>, Vec<f32>) {
    use rayon::prelude::*;
    
    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 8192;
    
    let results: Vec<(Vec<u32>, Vec<f32>)> = data
        .par_chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let offset = chunk_idx * CHUNK_SIZE;
            let mut indices = Vec::new();
            let mut values = Vec::new();
            
            for (idx, &val) in chunk.iter().enumerate() {
                if val.abs() >= threshold {
                    indices.push((offset + idx) as u32);
                    values.push(val);
                }
            }
            
            (indices, values)
        })
        .collect();
    
    // Merge results
    let total_elements: usize = results.iter().map(|(i, _)| i.len()).sum();
    let mut indices = Vec::with_capacity(total_elements);
    let mut values = Vec::with_capacity(total_elements);
    
    for (chunk_indices, chunk_values) in results {
        indices.extend(chunk_indices);
        values.extend(chunk_values);
    }
    
    (indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nonzero() {
        let data = vec![0.0, 1.5, 0.0, 2.3, 0.0, -1.2];
        let (indices, values) = find_nonzero(&data, 1e-6);
        
        assert_eq!(indices, vec![1, 3, 5]);
        assert_eq!(values, vec![1.5, 2.3, -1.2]);
    }

    #[test]
    fn test_find_nonzero_parallel() {
        let data: Vec<f32> = (0..100000)
            .map(|i| if i % 10 == 0 { i as f32 } else { 0.0 })
            .collect();
        
        let (indices, values) = find_nonzero_parallel(&data, 1e-6);
        
        assert_eq!(indices.len(), 10000);
        assert_eq!(values.len(), 10000);
    }
}
