"""
Sparse Bit Allocation - Assign compression strategies per layer

Uses Fisher importance scores to allocate different quantization
configurations to different layers. More important layers get
better quality (smaller groups), less important get more compression.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LayerAllocation:
    """Compression configuration for a single layer."""
    name: str
    method: str  # 'int4', 'int4_residual', 'calibrated_vq', etc.
    importance: float
    rank: int = 0  # For low-rank methods
    group_size: int = 256
    codebook_id: str = 'medium'
    vq_vec_dim: int = 4
    vq_codebook_size: int = 256
    bits_per_weight: float = 4.0
    sparsity: float = 0.0


def allocate_bits(
    model: nn.Module,
    fisher_scores: Dict[str, float],
    target: str = "quality"  # "quality", "balanced", "size"
) -> Dict[str, LayerAllocation]:
    """Allocate compression settings per layer based on importance.
    
    Production config (v10/v68):
    - Attention layers: INT4 g=256 (more sensitive)
    - MLP layers: INT4 g=2048 (more robust)
    
    Args:
        model: The model to allocate for
        fisher_scores: Fisher importance scores per layer
        target: Optimization target
            - "quality": Conservative groups, best PPL
            - "balanced": v10 config (7.42x @ +1.47% PPL)
            - "size": Aggressive groups, max compression
            
    Returns:
        Dict mapping layer names to LayerAllocation
    """
    print("[ALLOCATE] Assigning compression strategies...", flush=True)
    
    allocations = {}
    linear_layers = []
    
    # Collect linear layers with their importance
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D":
            # Skip embeddings and lm_head
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue
            fisher = fisher_scores.get(f"{name}.weight", 0.1)
            linear_layers.append((name, module, fisher))
    
    # Sort by importance (highest first)
    linear_layers.sort(key=lambda x: x[2], reverse=True)
    
    # Configuration presets
    configs = {
        "quality": {
            "attn_group": 128,
            "mlp_group": 512,
            "method": "int4_awq",
        },
        "balanced": {
            "attn_group": 256,
            "mlp_group": 2048,
            "method": "int4_awq",
        },
        "size": {
            "attn_group": 512,
            "mlp_group": 4096,
            "method": "int4_awq",
        },
        "v11": {
            "awq_top_pct": 15,
            "attn_group": 256,
            "mlp_group": 2048,
            "outlier_pct": 0.5,
            "vq_vec_dim": 4,
            "vq_codebook_size": 256,
            "method": "hybrid",
        },
    }
    
    cfg = configs.get(target, configs["balanced"])
    
    num_layers = len(linear_layers)
    awq_top_k = 0
    if target == "v11":
        awq_top_k = max(1, int(num_layers * cfg.get("awq_top_pct", 0) / 100.0))

    for i, (name, module, importance) in enumerate(linear_layers):
        name_lower = name.lower()
        
        # Determine if attention or MLP
        is_attention = any(x in name_lower for x in [
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 
            'attn', 'attention', 'query', 'key', 'value'
        ])
        
        is_attention_in_proj = any(x in name_lower for x in [
            'q_proj', 'k_proj', 'v_proj', 'c_attn'
        ])
        
        layer_method = cfg.get("method", "int4_awq")
        outlier_pct = float(cfg.get("outlier_pct", 0.5))
        vq_vec_dim = int(cfg.get("vq_vec_dim", 4))
        vq_codebook_size = int(cfg.get("vq_codebook_size", 256))

        # Compute AWQ cost (bits/weight)
        if is_attention:
            awq_group_size = cfg["attn_group"]
        else:
            awq_group_size = cfg["mlp_group"]

        awq_bits = 4.0 + 16.0 / awq_group_size
        if outlier_pct > 0:
            awq_bits += 16.0 * outlier_pct / 100.0

        if target == "v11":
            # Top layers: always AWQ
            if i < awq_top_k or is_attention:
                layer_method = "int4_awq"
                group_size = awq_group_size
                bits = awq_bits
            else:
                # Remaining layers: use VQ only if it actually improves bits/weight
                m, n = module.weight.shape
                pad_n = (vq_vec_dim - n % vq_vec_dim) % vq_vec_dim
                n_padded = n + pad_n
                num_vectors = (m * n_padded) // vq_vec_dim
                effective_codebook_size = min(vq_codebook_size, num_vectors) if num_vectors > 0 else 1
                compressed_bits = (num_vectors * 8.0) + (effective_codebook_size * vq_vec_dim * 16.0)
                vq_bits = compressed_bits / float(m * n)

                if vq_bits < awq_bits:
                    layer_method = "calibrated_vq"
                    group_size = vq_vec_dim
                    bits = vq_bits
                    vq_codebook_size = effective_codebook_size
                else:
                    layer_method = "int4_awq"
                    group_size = awq_group_size
                    bits = awq_bits
        else:
            if layer_method == "calibrated_vq":
                m, n = module.weight.shape
                pad_n = (vq_vec_dim - n % vq_vec_dim) % vq_vec_dim
                n_padded = n + pad_n
                num_vectors = (m * n_padded) // vq_vec_dim
                effective_codebook_size = min(vq_codebook_size, num_vectors) if num_vectors > 0 else 1
                compressed_bits = (num_vectors * 8.0) + (effective_codebook_size * vq_vec_dim * 16.0)
                bits = compressed_bits / float(m * n)
                group_size = vq_vec_dim
                vq_codebook_size = effective_codebook_size
            else:
                group_size = awq_group_size
                bits = awq_bits

        alloc = LayerAllocation(
            name=name,
            method=layer_method,
            importance=importance,
            group_size=group_size,
            vq_vec_dim=vq_vec_dim,
            vq_codebook_size=vq_codebook_size,
            bits_per_weight=bits,
        )
        allocations[name] = alloc
    
    # Log summary
    total_params = sum(m.weight.numel() for _, m, _ in linear_layers)
    total_bits = sum(
        allocations[name].bits_per_weight * module.weight.numel()
        for name, module, _ in linear_layers
    )
    avg_bits = total_bits / total_params if total_params > 0 else 4.0
    expected_compression = 16.0 / avg_bits
    
    print(f"[ALLOCATE] {len(allocations)} layers configured", flush=True)
    print(f"[ALLOCATE] Expected: {avg_bits:.2f} bits/weight, {expected_compression:.2f}x compression", flush=True)
    
    return allocations


def allocate_bits_adaptive(
    model: nn.Module,
    fisher_scores: Dict[str, float],
    target_bits: float = 3.5
) -> Dict[str, LayerAllocation]:
    """Adaptive bit allocation to hit a target bits-per-weight.
    
    Uses importance-based tiering:
    - Top 10%: g=64 (protected)
    - 10-30%: g=128 (high)
    - 30-60%: g=256 (medium)
    - Bottom 40%: g=512 (aggressive)
    
    Args:
        model: The model to allocate for
        fisher_scores: Fisher importance scores
        target_bits: Target average bits per weight
        
    Returns:
        Dict mapping layer names to LayerAllocation
    """
    print(f"[ALLOCATE] Adaptive allocation targeting {target_bits} bits/weight...", flush=True)
    
    allocations = {}
    linear_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed' in name.lower() or 'lm_head' in name.lower():
                continue
            fisher = fisher_scores.get(f"{name}.weight", 0.1)
            linear_layers.append((name, module, fisher))
    
    linear_layers.sort(key=lambda x: x[2], reverse=True)
    num_layers = len(linear_layers)
    
    for i, (name, module, importance) in enumerate(linear_layers):
        percentile = i / num_layers if num_layers > 0 else 0
        
        if percentile < 0.10:
            # Top 10%: Protected
            group_size = 64
        elif percentile < 0.30:
            # 10-30%: High importance
            group_size = 128
        elif percentile < 0.60:
            # 30-60%: Medium
            group_size = 256
        else:
            # Bottom 40%: Aggressive
            group_size = 512
        
        bits = 4.0 + 16.0 / group_size
        
        alloc = LayerAllocation(
            name=name,
            method='int4',
            importance=importance,
            group_size=group_size,
            bits_per_weight=bits
        )
        allocations[name] = alloc
    
    # Log results
    total_params = sum(m.weight.numel() for _, m, _ in linear_layers)
    total_bits = sum(
        allocations[name].bits_per_weight * module.weight.numel()
        for name, module, _ in linear_layers
    )
    avg_bits = total_bits / total_params if total_params > 0 else 4.0
    
    print(f"[ALLOCATE] Result: {avg_bits:.2f} bits/weight (target: {target_bits})", flush=True)
    
    return allocations
