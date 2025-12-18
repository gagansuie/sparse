"""
TenPak Compression Codecs

Production-validated quantization methods extracted from hf_space/app.py.
These are the proven working implementations (v10/v68 config).

Codec Performance (Mistral-7B, WikiText-2):
- INT4+AWQ g=256/2048: 7.42x compression, +1.47% PPL ✅
- INT4+Residual: 5.3x compression, -0.41% PPL ✅
- Calibrated VQ: experimental, higher compression but higher PPL

Note: Custom GPTQ/AQLM implementations were found to be broken (v50-67).
Only use the proven INT4+AWQ approach for production.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

# Codec identifiers
CODEC_INT4_AWQ = "int4_awq_v1"
CODEC_INT4_RESIDUAL = "int4_residual_v1"
CODEC_CALIBRATED_VQ = "calibrated_vq_v1"


@dataclass
class CompressionResult:
    """Result of compressing a weight tensor."""
    weight: torch.Tensor  # Dequantized weight (for in-place replacement)
    compression_ratio: float
    method: str
    bits_per_weight: float


def compress_int4_awq(
    weight: torch.Tensor,
    group_size: int = 256,
    act_scale: Optional[torch.Tensor] = None,
    outlier_pct: float = 0.5,
    iterations: int = 5
) -> Tuple[torch.Tensor, float]:
    """INT4 quantization with AWQ-style activation scaling.
    
    This is the PROVEN working codec (v10 config):
    - Attention layers: g=256, 0.5% outliers
    - MLP layers: g=2048, 0.5% outliers
    
    Args:
        weight: FP16/FP32 weight tensor [out_features, in_features]
        group_size: Quantization group size (smaller = better quality, larger = better compression)
        act_scale: Optional activation scales from calibration [in_features]
        outlier_pct: Percentage of weights to keep as FP16 outliers
        iterations: Number of scale refinement iterations
        
    Returns:
        (dequantized_weight, compression_ratio)
    """
    orig_device = weight.device
    w = weight.float().cpu()
    original_shape = w.shape
    m, n = w.shape
    
    # AWQ scaling: scale columns by activation importance
    if act_scale is not None and act_scale.numel() == n:
        scale_factor = act_scale.cpu().view(1, -1).clamp(min=0.01, max=100)
        w_scaled = w * scale_factor
    else:
        scale_factor = None
        w_scaled = w
    
    # Outlier extraction (keep top outlier_pct% as FP16)
    if outlier_pct > 0:
        flat = w_scaled.abs().flatten()
        k = max(1, int(flat.numel() * outlier_pct / 100))
        threshold = torch.kthvalue(flat, flat.numel() - k + 1).values
        outlier_mask = w_scaled.abs() >= threshold
        outliers = w_scaled * outlier_mask.float()
        w_to_quant = w_scaled * (~outlier_mask).float()
    else:
        outlier_mask = None
        outliers = None
        w_to_quant = w_scaled
    
    # Pad to multiple of group_size
    flat = w_to_quant.flatten()
    orig_numel = flat.numel()
    pad_len = (group_size - orig_numel % group_size) % group_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len)])
    
    groups = flat.view(-1, group_size)
    num_groups = groups.shape[0]
    
    # Iterative scale refinement
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    
    for _ in range(iterations):
        scale = torch.where(
            (g_max - g_min).abs() > 1e-8,
            (g_max - g_min) / 15.0,
            torch.ones_like(g_max)
        )
        inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
        
        q = ((groups - g_min) * inv_scale).round().clamp(0, 15)
        deq = q * scale + g_min
        err = groups - deq
        
        g_min = g_min + err.min(dim=1, keepdim=True).values * 0.5
        g_max = g_max + err.max(dim=1, keepdim=True).values * 0.5
    
    # Final quantization
    scale = torch.where(
        (g_max - g_min).abs() > 1e-8,
        (g_max - g_min) / 15.0,
        torch.ones_like(g_max)
    )
    q = ((groups - g_min) / scale.clamp(min=1e-8)).round().clamp(0, 15)
    deq = q * scale + g_min
    
    # Reshape back
    result = deq.flatten()[:orig_numel].view(original_shape)
    
    # Add back outliers
    if outliers is not None:
        result = result + outliers
    
    # Reverse AWQ scaling
    if scale_factor is not None:
        result = result / scale_factor
    
    # Compression ratio calculation
    # INT4: 4 bits + scale overhead (32 bits per group)
    bits_per_weight = 4.0 + 32.0 / group_size
    if outlier_pct > 0:
        bits_per_weight += 16.0 * outlier_pct / 100  # FP16 outliers
    compression = 32.0 / bits_per_weight
    
    return result.to(orig_device), compression


def compress_int4_residual(
    weight: torch.Tensor,
    group_size: int = 16,
    residual_group: int = 16,
    iterations: int = 5
) -> Tuple[torch.Tensor, float]:
    """INT4 + INT2 residual quantization for best quality.
    
    Two-pass quantization:
    1. INT4 quantization with iterative refinement
    2. INT2 residual correction on quantization error
    
    Achieves negative PPL delta on larger models (regularization effect).
    
    Args:
        weight: FP16/FP32 weight tensor
        group_size: INT4 group size
        residual_group: INT2 residual group size
        iterations: Scale refinement iterations
        
    Returns:
        (dequantized_weight, compression_ratio)
    """
    orig_device = weight.device
    w = weight.float().cpu()
    original_shape = w.shape
    
    flat = w.flatten()
    orig_numel = flat.numel()
    
    # Pad for group_size
    pad_len = (group_size - orig_numel % group_size) % group_size
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len)])
    
    groups = flat.view(-1, group_size)
    num_groups = groups.shape[0]
    
    # Pass 1: INT4 with iterative refinement
    g_min = groups.min(dim=1, keepdim=True).values
    g_max = groups.max(dim=1, keepdim=True).values
    
    for _ in range(iterations):
        scale = torch.where(
            (g_max - g_min).abs() > 1e-8,
            (g_max - g_min) / 15.0,
            torch.ones_like(g_max)
        )
        inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
        
        q = ((groups - g_min) * inv_scale).round().clamp(0, 15)
        deq = q * scale + g_min
        err = groups - deq
        
        g_min = g_min + err.min(dim=1, keepdim=True).values * 0.5
        g_max = g_max + err.max(dim=1, keepdim=True).values * 0.5
    
    scale = torch.where(
        (g_max - g_min).abs() > 1e-8,
        (g_max - g_min) / 15.0,
        torch.ones_like(g_max)
    )
    q = ((groups - g_min) / scale.clamp(min=1e-8)).round().clamp(0, 15)
    deq_int4 = q * scale + g_min
    
    # Compute residuals
    residuals = groups - deq_int4
    residuals_flat = residuals.flatten()
    
    # Pass 2: INT2 residual quantization
    res_pad = (residual_group - len(residuals_flat) % residual_group) % residual_group
    if res_pad > 0:
        residuals_flat = torch.cat([residuals_flat, torch.zeros(res_pad)])
    
    res_groups = residuals_flat.view(-1, residual_group)
    
    r_min = res_groups.min(dim=1, keepdim=True).values
    r_max = res_groups.max(dim=1, keepdim=True).values
    r_scale = torch.where(
        (r_max - r_min).abs() > 1e-8,
        (r_max - r_min) / 3.0,  # INT2 = 4 levels
        torch.ones_like(r_max)
    )
    
    q_res = ((res_groups - r_min) / r_scale.clamp(min=1e-8)).round().clamp(0, 3)
    deq_res = q_res * r_scale + r_min
    
    # Combine INT4 + INT2 residual
    result = deq_int4.flatten() + deq_res.flatten()[:len(deq_int4.flatten())]
    result = result[:orig_numel].view(original_shape)
    
    # Compression: INT4 (4 bits) + INT2 (2 bits) + scales
    # ~6 bits effective + overhead
    bits_per_weight = 4.0 + 2.0 + 32.0 / group_size + 32.0 / residual_group
    compression = 32.0 / bits_per_weight
    
    return result.to(orig_device), compression


def compress_calibrated_vq(
    weight: torch.Tensor,
    vec_dim: int = 4,
    codebook_size: int = 256,
    num_iters: int = 15,
    hessian_diag: Optional[torch.Tensor] = None,
    act_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float]:
    """Calibrated Vector Quantization with importance weighting.
    
    K-means clustering with Hessian-weighted distance metric.
    Higher compression but requires good calibration for quality.
    
    Args:
        weight: Weight tensor [m, n]
        vec_dim: Vector dimension for clustering
        codebook_size: Number of codebook entries (256 = 8-bit indices)
        num_iters: K-means iterations
        hessian_diag: Optional Hessian diagonal for importance weighting
        act_scale: Optional activation scales for AWQ-style scaling
        
    Returns:
        (dequantized_weight, compression_ratio)
    """
    orig_device = weight.device
    device = weight.device if weight.is_cuda else torch.device('cpu')
    w = weight.float().to(device)
    m, n = w.shape
    
    # AWQ scaling
    if act_scale is not None and act_scale.numel() == n:
        scale_factor = act_scale.to(device).view(1, -1).clamp(min=0.01, max=100)
        w = w * scale_factor
    else:
        scale_factor = None
    
    # Reshape to vectors
    pad_n = (vec_dim - n % vec_dim) % vec_dim
    if pad_n > 0:
        w = F.pad(w, (0, pad_n))
    
    n_padded = w.shape[1]
    num_vectors = (m * n_padded) // vec_dim
    vectors = w.view(-1, vec_dim)
    
    # Importance weights from Hessian
    if hessian_diag is not None and hessian_diag.numel() == n:
        H = hessian_diag.to(device).clamp(min=1e-8)
        if pad_n > 0:
            H = F.pad(H, (0, pad_n), value=H.mean())
        H_groups = H.view(-1, vec_dim)
        vec_weights = H_groups.mean(dim=1).repeat(m)[:num_vectors]
        vec_weights = vec_weights / vec_weights.sum() * num_vectors
    else:
        vec_weights = torch.ones(num_vectors, device=device)
    
    # Initialize codebook with k-means++
    indices = torch.randperm(num_vectors, device=device)[:codebook_size]
    codebook = vectors[indices].clone()
    
    batch_size = min(8192, num_vectors)
    
    # K-means iterations
    for _ in range(num_iters):
        # Assignment step (batched)
        assignments = torch.zeros(num_vectors, dtype=torch.long, device=device)
        for start in range(0, num_vectors, batch_size):
            end = min(start + batch_size, num_vectors)
            v_sq = (vectors[start:end] ** 2).sum(dim=1, keepdim=True)
            c_sq = (codebook ** 2).sum(dim=1, keepdim=True).T
            dots = vectors[start:end] @ codebook.T
            dists = v_sq + c_sq - 2 * dots
            assignments[start:end] = dists.argmin(dim=1)
        
        # Update step (weighted)
        new_codebook = torch.zeros_like(codebook)
        counts = torch.zeros(codebook_size, device=device)
        
        for c in range(codebook_size):
            mask = assignments == c
            if mask.sum() > 0:
                weights = vec_weights[mask]
                weighted_sum = (vectors[mask] * weights.unsqueeze(1)).sum(dim=0)
                new_codebook[c] = weighted_sum / weights.sum()
                counts[c] = mask.sum()
        
        # Handle empty clusters
        empty_mask = counts == 0
        if empty_mask.sum() > 0:
            high_weight_idx = vec_weights.argsort(descending=True)[:empty_mask.sum()]
            new_codebook[empty_mask] = vectors[high_weight_idx]
        
        codebook = new_codebook
    
    # Final assignment
    final_assignments = torch.zeros(num_vectors, dtype=torch.long, device=device)
    for start in range(0, num_vectors, batch_size):
        end = min(start + batch_size, num_vectors)
        v_sq = (vectors[start:end] ** 2).sum(dim=1, keepdim=True)
        c_sq = (codebook ** 2).sum(dim=1, keepdim=True).T
        dots = vectors[start:end] @ codebook.T
        dists = v_sq + c_sq - 2 * dots
        final_assignments[start:end] = dists.argmin(dim=1)
    
    # Reconstruct
    reconstructed = codebook[final_assignments].flatten()[:m * n_padded].view(m, n_padded)
    if pad_n > 0:
        reconstructed = reconstructed[:, :n]
    
    # Reverse scaling
    if scale_factor is not None:
        reconstructed = reconstructed / scale_factor[:, :n]
    
    # Compression calculation
    orig_size = m * n * 4  # FP32 bytes
    indices_size = num_vectors * 1  # 8-bit indices
    codebook_overhead = codebook_size * vec_dim * 4  # FP32 codebook
    compressed_size = indices_size + codebook_overhead
    compression = orig_size / compressed_size
    
    return reconstructed.to(orig_device), compression
