"""
Smart compression heuristics based on layer analysis.
"""

import torch
from typing import Dict


def analyze_layer_type(layer_name: str) -> str:
    """Determine layer type from name."""
    name_lower = layer_name.lower()
    
    if any(x in name_lower for x in ['attn', 'attention', 'self_attn', 'q_proj', 'k_proj', 'v_proj']):
        return 'attention'
    elif any(x in name_lower for x in ['mlp', 'ffn', 'fc', 'dense']):
        return 'feedforward'
    elif 'embed' in name_lower or 'wte' in name_lower or 'wpe' in name_lower:
        return 'embedding'
    elif 'ln' in name_lower or 'norm' in name_lower or 'layernorm' in name_lower:
        return 'normalization'
    elif 'lm_head' in name_lower or 'output' in name_lower:
        return 'output'
    else:
        return 'unknown'


def analyze_delta_distribution(delta: torch.Tensor) -> Dict:
    """Analyze delta tensor distribution."""
    delta_abs = delta.abs().flatten()
    
    # Sample if tensor is too large (>10M elements)
    if delta_abs.numel() > 10_000_000:
        indices = torch.randperm(delta_abs.numel())[:1_000_000]
        delta_abs = delta_abs[indices]
    
    return {
        "mean": delta.mean().item(),
        "std": delta.std().item(),
        "max": delta_abs.max().item(),
        "min": delta_abs.min().item(),
        "median": delta_abs.median().item(),
        "q25": delta_abs.quantile(0.25).item(),
        "q75": delta_abs.quantile(0.75).item(),
        "q95": delta_abs.quantile(0.95).item(),
        "q99": delta_abs.quantile(0.99).item(),
    }


def choose_optimal_compression(
    delta: torch.Tensor,
    stats: Dict[str, float],
    layer_name: str,
) -> str:
    """
    Choose optimal compression method based on layer analysis.
    
    Returns:
        One of "zero", "sparse", "int8", "hybrid"
    """
    layer_type = analyze_layer_type(layer_name)
    distribution = analyze_delta_distribution(delta)
    
    # Zero delta (no change)
    if stats["max_abs"] < 1e-8:
        return "zero"
    
    # Attention layers: often sparse due to selective updates
    if layer_type == 'attention':
        if stats["sparsity"] > 0.85:
            return "sparse"
        elif stats["max_abs"] < 0.1:
            return "int8"
        else:
            return "hybrid"  # Mix of sparse and INT8
    
    # Feed-forward layers: typically dense updates
    elif layer_type == 'feedforward':
        if stats["sparsity"] > 0.95:
            return "sparse"
        else:
            return "int8"
    
    # Embedding layers: very sparse updates (only used tokens change)
    elif layer_type == 'embedding':
        if stats["sparsity"] > 0.80:
            return "sparse"
        else:
            return "int8"
    
    # Normalization layers: small deltas, good for INT8
    elif layer_type == 'normalization':
        return "int8"
    
    # Output/LM head: similar to feed-forward
    elif layer_type == 'output':
        if stats["sparsity"] > 0.90:
            return "sparse"
        else:
            return "int8"
    
    # Default heuristic
    if stats["sparsity"] > 0.90:
        return "sparse"
    elif stats["max_abs"] < 0.15:
        return "int8"
    else:
        return "hybrid"


def get_compression_params(method: str, delta: torch.Tensor, stats: Dict) -> Dict:
    """
    Get optimal compression parameters for chosen method.
    
    Returns:
        Dict with compression-specific parameters
    """
    if method == "sparse":
        # Adaptive threshold based on distribution
        distribution = analyze_delta_distribution(delta)
        threshold = max(1e-6, distribution["q25"])  # Use 25th percentile as threshold
        
        return {
            "threshold": threshold,
            "store_indices": True,
        }
    
    elif method == "int8":
        # Determine if we need per-channel or global quantization
        if delta.dim() >= 2 and delta.shape[0] > 1:
            # Per-channel for matrices
            return {
                "per_channel": True,
                "symmetric": True,
            }
        else:
            return {
                "per_channel": False,
                "symmetric": True,
            }
    
    elif method == "hybrid":
        # Combine sparse and INT8
        distribution = analyze_delta_distribution(delta)
        
        return {
            "sparse_threshold": distribution["q50"],  # Sparsify bottom 50%
            "quantize_remaining": True,
            "int8_params": {
                "per_channel": True,
                "symmetric": True,
            }
        }
    
    return {}


def estimate_compression_ratio(
    delta: torch.Tensor,
    method: str,
    params: Dict,
) -> float:
    """
    Estimate compression ratio for given method.
    
    Returns:
        Estimated compression ratio
    """
    numel = delta.numel()
    original_bytes = numel * 2  # FP16
    
    if method == "zero":
        return float('inf')
    
    elif method == "sparse":
        threshold = params.get("threshold", 1e-6)
        nnz = (delta.abs() > threshold).sum().item()
        # indices (4 bytes) + values (1 byte INT8) per non-zero
        compressed_bytes = nnz * 5
        return original_bytes / max(compressed_bytes, 1)
    
    elif method == "int8":
        # INT8 + scale
        compressed_bytes = numel + 4  # 1 byte per value + 4 bytes for scale
        return original_bytes / compressed_bytes
    
    elif method == "hybrid":
        threshold = params.get("sparse_threshold", 1e-4)
        nnz = (delta.abs() > threshold).sum().item()
        # Sparse part: indices + quantized values
        sparse_bytes = nnz * 5
        # Dense part: quantized remaining values
        dense_count = numel - nnz
        dense_bytes = dense_count + 4
        compressed_bytes = sparse_bytes + dense_bytes
        return original_bytes / max(compressed_bytes, 1)
    
    return 2.0  # Default estimate
