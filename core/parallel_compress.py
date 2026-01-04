"""
Parallel layer processing for faster compression.
"""

import torch
from typing import Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def compute_layer_delta_worker(args: Tuple) -> Tuple[str, torch.Tensor, Dict]:
    """Worker function for parallel delta computation."""
    name, base_weight_bytes, finetune_weight_bytes, shape, dtype = args
    
    # Reconstruct tensors from bytes (make writable copy)
    base_weight = torch.frombuffer(base_weight_bytes, dtype=dtype).reshape(shape).clone()
    finetune_weight = torch.frombuffer(finetune_weight_bytes, dtype=dtype).reshape(shape).clone()
    
    # Compute delta
    delta = finetune_weight - base_weight
    
    # Compute stats
    stats = {
        "max_abs": delta.abs().max().item(),
        "l2_norm": delta.norm().item(),
        "sparsity": (delta.abs() < 1e-6).float().mean().item(),
        "numel": delta.numel(),
    }
    
    return name, delta, stats


def compute_deltas_parallel(
    base_params: Dict[str, torch.Tensor],
    finetune_params: Dict[str, torch.Tensor],
    max_workers: int = None,
) -> Dict[str, Tuple[torch.Tensor, Dict]]:
    """
    Compute deltas for all layers in parallel.
    
    Args:
        base_params: Base model parameters
        finetune_params: Fine-tuned model parameters
        max_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        Dict mapping layer names to (delta, stats) tuples
    """
    if max_workers is None:
        max_workers = mp.cpu_count()
    
    # Prepare work items
    work_items = []
    for name in base_params.keys():
        if name not in finetune_params:
            continue
        
        base_weight = base_params[name].data.cpu()
        finetune_weight = finetune_params[name].data.cpu()
        
        if base_weight.shape != finetune_weight.shape:
            continue
        
        # Convert to bytes for pickling
        work_items.append((
            name,
            base_weight.numpy().tobytes(),
            finetune_weight.numpy().tobytes(),
            base_weight.shape,
            base_weight.dtype
        ))
    
    # Process in parallel
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_layer_delta_worker, item): item[0]
            for item in work_items
        }
        
        for future in as_completed(futures):
            name, delta, stats = future.result()
            results[name] = (delta, stats)
    
    return results


def batch_layer_processing(
    param_names: List[str],
    base_params: Dict,
    finetune_params: Dict,
    batch_size: int = 10,
) -> List[Tuple[str, torch.Tensor, Dict]]:
    """
    Process layers in batches for memory efficiency.
    
    Args:
        param_names: List of parameter names
        base_params: Base model parameters
        finetune_params: Fine-tuned model parameters
        batch_size: Number of layers to process at once
    
    Yields:
        Tuples of (name, delta, stats)
    """
    for i in range(0, len(param_names), batch_size):
        batch = param_names[i:i + batch_size]
        batch_base = {n: base_params[n] for n in batch if n in base_params}
        batch_ft = {n: finetune_params[n] for n in batch if n in finetune_params}
        
        # Process batch in parallel
        results = compute_deltas_parallel(batch_base, batch_ft, max_workers=batch_size)
        
        for name in batch:
            if name in results:
                delta, stats = results[name]
                yield name, delta, stats
