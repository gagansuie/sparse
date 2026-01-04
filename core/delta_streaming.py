"""
Streaming delta compression for large models (30B+).
Automatically used by compress_delta() when model size is detected.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import json
from datetime import datetime

from core.lazy_loading import LazyModelLoader


def _compress_delta_streaming(
    base_model_id: str,
    finetune_model_id: str,
    base_loader: LazyModelLoader,
    ft_loader: LazyModelLoader,
    output_path: str,
    use_parallel: bool,
    log: Callable,
    progress_callback: Optional[Callable] = None,
):
    """
    Streaming compression for large models.
    
    Uses lazy loading to process layers one-by-one without loading entire model.
    Optionally uses parallel processing for layer computation.
    """
    from core.lazy_loading import compute_deltas_streaming
    from core.parallel_compress import compute_deltas_parallel
    from core.delta import (
        DeltaManifest,
        compress_delta_sparse,
        compress_delta_int8,
        compute_model_hash,
    )
    from core.smart_heuristics import choose_optimal_compression
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    deltas_path = output_path / "deltas"
    deltas_path.mkdir(exist_ok=True)
    
    log("Computing deltas with streaming...", 0.30)
    
    layer_deltas = []
    total_original_size = 0
    total_compressed_size = 0
    changed_params = 0
    total_params = 0
    
    base_layers = base_loader.get_layer_names()
    total_layers = len(base_layers)
    
    processed = 0
    for name, delta, stats in compute_deltas_streaming(base_loader, ft_loader):
        processed += 1
        progress = 0.30 + (processed / total_layers) * 0.60
        
        if processed % 50 == 0:
            log(f"  Processed {processed}/{total_layers} layers", progress)
        
        # Choose compression method with smart heuristics
        from core.smart_heuristics import choose_optimal_compression
        try:
            method = choose_optimal_compression(delta, stats, name)
        except Exception:
            # Fallback
            from core.delta import choose_delta_method
            method = choose_delta_method(delta, stats)
        
        # Track sizes
        original_size = delta.numel() * 2
        total_original_size += original_size
        total_params += delta.numel()
        
        layer_info = {
            "name": name,
            "shape": list(delta.shape),
            "dtype": str(delta.dtype),
            "method": method,
            "stats": stats,
        }
        
        if method == "zero":
            layer_info["compressed_size"] = 0
            compressed_size = 0
            
        elif method == "sparse":
            indices, values, comp_ratio = compress_delta_sparse(delta)
            
            torch.save({
                "indices": indices,
                "values": values,
            }, deltas_path / f"{name.replace('.', '_')}.pt")
            
            compressed_size = indices.numel() * 4 + values.numel() * 2
            layer_info["compressed_size"] = compressed_size
            layer_info["num_nonzero"] = values.numel()
            changed_params += values.numel()
            
        elif method == "int8":
            quantized_bytes, scale, comp_ratio = compress_delta_int8(delta)
            
            with open(deltas_path / f"{name.replace('.', '_')}.bin", "wb") as f:
                f.write(quantized_bytes)
            
            compressed_size = len(quantized_bytes)
            layer_info["compressed_size"] = compressed_size
            layer_info["scale"] = scale
            changed_params += delta.numel()
            
        elif method == "hybrid":
            # Hybrid: quantize non-sparse part
            sparse_mask = delta.abs() > stats.get("sparsity_threshold", 1e-6)
            sparse_indices = sparse_mask.nonzero(as_tuple=True)
            sparse_values = delta[sparse_indices]
            
            torch.save({
                "indices": torch.stack(sparse_indices, dim=1),
                "values": sparse_values,
            }, deltas_path / f"{name.replace('.', '_')}_sparse.pt")
            
            # Dense part compressed as INT8
            dense_part = delta.clone()
            dense_part[sparse_indices] = 0
            
            if dense_part.abs().max() > 0:
                quantized_bytes, scale, _ = compress_delta_int8(dense_part)
                with open(deltas_path / f"{name.replace('.', '_')}_dense.bin", "wb") as f:
                    f.write(quantized_bytes)
                dense_size = len(quantized_bytes)
            else:
                scale = 0.0
                dense_size = 0
            
            compressed_size = sparse_values.numel() * 6 + dense_size
            layer_info["compressed_size"] = compressed_size
            layer_info["num_nonzero"] = sparse_values.numel()
            layer_info["scale"] = scale
            changed_params += delta.numel()
        
        total_compressed_size += compressed_size
        layer_deltas.append(layer_info)
    
    # Create manifest
    log("Saving manifest...", 0.95)
    
    manifest = DeltaManifest(
        base_model_id=base_model_id,
        finetune_model_id=finetune_model_id,
        layer_deltas=layer_deltas,
        num_layers=len(layer_deltas),
        total_params=total_params,
        changed_params=changed_params,
        compression_ratio=total_original_size / max(total_compressed_size, 1),
        delta_type="model_delta",
        creation_time=datetime.now().isoformat(),
        base_model_hash=None,  # Skip hash for streaming
        finetune_model_hash=None,
    )
    
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    log(f"Delta compression complete!", 1.0)
    log(f"  Compression ratio: {manifest.compression_ratio:.2f}x")
    log(f"  Changed params: {changed_params:,} / {total_params:,} ({100*changed_params/total_params:.1f}%)")
    
    return manifest
