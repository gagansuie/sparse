#!/usr/bin/env python3
"""
Novel compression techniques combining AWQ/GPTQ insights without calibration.

Key Insights:
- AWQ: Only ~1% of weights are critical (can we detect from statistics?)
- GPTQ: Compensate quantization error by adjusting remaining weights
- Sparsity: Works if we know which weights to prune (importance)

Novel Approaches:
1. Outlier-Preserving: Keep high-magnitude weights at full precision
2. Correlation-Based Compensation: GPTQ-style without Hessian
3. Smart PQ: Percentile initialization for codebook
4. Multi-Stage Residual: INT4 + INT2 + INT1
5. Layer Sensitivity: First/last layers preserved
6. Fisher-Free Importance: Weight statistics as importance proxy
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_ppl(model, tokenizer, texts, max_samples=15, max_length=256):
    model.eval()
    device = next(model.parameters()).device
    nll, ntokens = 0.0, 0
    with torch.no_grad():
        for text in texts[:max_samples]:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            ids = enc["input_ids"].to(device)
            out = model(ids, labels=ids)
            nll += out.loss.item() * ids.numel()
            ntokens += ids.numel()
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


# =============================================================================
# APPROACH 1: Outlier-Preserving Quantization (AWQ-inspired)
# =============================================================================
def outlier_preserving_quant(weight, outlier_pct=0.01, group_size=16):
    """
    Keep top outlier_pct weights at full precision, quantize the rest.
    AWQ insight: Only ~1% of weights are critical.
    
    Storage: (1-outlier_pct) * 4 bits + outlier_pct * 16 bits + mask
    For 1%: 0.99*4 + 0.01*16 + 0.125 = 4.23 bits ≈ 7.5x compression
    """
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    # Find outliers (top percentile by magnitude)
    num_outliers = max(1, int(n * outlier_pct))
    _, outlier_indices = torch.topk(flat.abs(), num_outliers)
    
    # Create mask for outliers
    outlier_mask = torch.zeros(n, dtype=torch.bool, device=weight.device)
    outlier_mask[outlier_indices] = True
    
    # Store outliers at full precision
    outlier_values = flat[outlier_mask].clone()
    
    # Quantize non-outliers with INT4
    non_outlier_mask = ~outlier_mask
    non_outlier_vals = flat.clone()
    non_outlier_vals[outlier_mask] = 0  # Zero out outliers for quantization
    
    # Simple INT4 quantization on non-outliers
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        non_outlier_vals = torch.cat([non_outlier_vals, torch.zeros(pad, device=weight.device)])
    
    groups = non_outlier_vals.view(-1, group_size)
    g_min, g_max = groups.min(1).values, groups.max(1).values
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    result = deq.flatten()[:n]
    
    # Restore outliers at full precision
    result[outlier_mask] = outlier_values
    
    return result.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 2: Correlation-Based Error Compensation (GPTQ-inspired)
# =============================================================================
def correlation_compensated_quant(weight, group_size=8, compensation_strength=0.5):
    """
    After quantizing, compensate error using weight correlations.
    GPTQ uses Hessian; we use weight covariance as approximation.
    
    For each quantized weight, distribute error to correlated neighbors.
    """
    original_shape = weight.shape
    if weight.dim() == 1:
        weight = weight.unsqueeze(0)
    
    # Reshape to [out_features, in_features] or similar
    if weight.dim() > 2:
        weight = weight.view(weight.shape[0], -1)
    
    out_features, in_features = weight.shape
    result = weight.clone().float()
    
    # Process column by column (GPTQ style)
    for col in range(in_features):
        col_data = result[:, col]
        
        # Quantize this column
        g_min, g_max = col_data.min(), col_data.max()
        scale = (g_max - g_min) / 15.0 if (g_max - g_min).abs() > 1e-8 else 1.0
        
        q = ((col_data - g_min) / scale).round().clamp(0, 15)
        deq = q * scale + g_min
        
        # Compute quantization error
        error = col_data - deq
        
        # Distribute error to remaining columns based on correlation
        if col < in_features - 1:
            remaining_cols = result[:, col+1:]
            
            # Compute correlation between this column and remaining
            # Use simplified correlation: just propagate fraction of error
            error_per_remaining = error.unsqueeze(1) * compensation_strength / (in_features - col - 1)
            remaining_cols += error_per_remaining
        
        result[:, col] = deq
    
    return result.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 3: Smart Product Quantization with Percentile Init
# =============================================================================
def smart_pq(weight, codebook_size=256, vector_size=8, use_residual=True):
    """
    Product quantization with smart initialization.
    Instead of random k-means, use percentile-based codebook.
    """
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (vector_size - n % vector_size) % vector_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    vectors = flat.view(-1, vector_size)
    num_vectors = vectors.shape[0]
    
    # Smart initialization: use percentiles of vector norms
    norms = vectors.norm(dim=1)
    percentiles = torch.linspace(0, 1, codebook_size)
    
    # Select representative vectors at each percentile
    sorted_indices = norms.argsort()
    codebook_indices = (percentiles * (num_vectors - 1)).long()
    codebook = vectors[sorted_indices[codebook_indices]].clone()
    
    # One iteration of k-means refinement
    dists = torch.cdist(vectors, codebook)
    assignments = dists.argmin(dim=1)
    
    for i in range(codebook_size):
        mask = assignments == i
        if mask.sum() > 0:
            codebook[i] = vectors[mask].mean(dim=0)
    
    # Final assignment
    dists = torch.cdist(vectors, codebook)
    assignments = dists.argmin(dim=1)
    reconstructed = codebook[assignments]
    
    # Add residual correction
    if use_residual:
        residual = vectors - reconstructed
        # Quantize residual with INT2 (4 levels per element)
        r_min = residual.min(dim=1, keepdim=True).values
        r_max = residual.max(dim=1, keepdim=True).values
        r_scale = (r_max - r_min) / 3.0
        r_scale = torch.where(r_scale.abs() < 1e-8, torch.ones_like(r_scale), r_scale)
        r_q = ((residual - r_min) / r_scale).round().clamp(0, 3)
        r_deq = r_q * r_scale + r_min
        reconstructed = reconstructed + r_deq
    
    result = reconstructed.flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 4: Multi-Stage Residual (INT4 + INT2 + INT1)
# =============================================================================
def multi_stage_residual(weight, group_size=16, iterations=5):
    """
    Three-stage quantization:
    Stage 1: INT4 (4 bits)
    Stage 2: INT2 residual (2 bits)
    Stage 3: INT1 residual (1 bit - just sign)
    
    Total: ~7 bits with triple correction = ~4.5x compression
    But potentially better quality than INT4 alone.
    """
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
    
    # Stage 1: INT4 with iterative refinement
    g_min, g_max = groups.min(1).values, groups.max(1).values
    for _ in range(iterations):
        scale = (g_max - g_min) / 15.0
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(1).values * 0.5
        g_max = g_max + err.max(1).values * 0.5
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q1 = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    stage1 = q1 * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Stage 2: INT2 on residual
    residual1 = groups - stage1
    r1_min, r1_max = residual1.min(1).values, residual1.max(1).values
    r1_scale = (r1_max - r1_min) / 3.0
    r1_scale = torch.where(r1_scale.abs() < 1e-8, torch.ones_like(r1_scale), r1_scale)
    q2 = ((residual1 - r1_min.unsqueeze(1)) / r1_scale.unsqueeze(1)).round().clamp(0, 3)
    stage2 = q2 * r1_scale.unsqueeze(1) + r1_min.unsqueeze(1)
    
    # Stage 3: INT1 on remaining residual (just sign correction)
    residual2 = groups - stage1 - stage2
    r2_scale = residual2.abs().mean(dim=1, keepdim=True)
    q3 = (residual2 > 0).float() * 2 - 1  # -1 or +1
    stage3 = q3 * r2_scale * 0.5
    
    result = (stage1 + stage2 + stage3).flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 5: Layer-Position Aware Quantization
# =============================================================================
def layer_position_quant(weight, layer_idx, total_layers, group_size=8):
    """
    First and last layers get gentle quantization (g=8).
    Middle layers get aggressive quantization (g=32 or INT3).
    """
    # Determine aggressiveness based on layer position
    position_ratio = layer_idx / total_layers
    
    if position_ratio < 0.15 or position_ratio > 0.85:
        # First/last 15% of layers: conservative
        effective_group = 8
        bits = 4
    elif position_ratio < 0.3 or position_ratio > 0.7:
        # Next 15%: moderate
        effective_group = 16
        bits = 4
    else:
        # Middle 40%: aggressive
        effective_group = 32
        bits = 4  # Could try 3 but it's too aggressive
    
    levels = 2**bits - 1
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (effective_group - n % effective_group) % effective_group
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, effective_group)
    g_min, g_max = groups.min(1).values, groups.max(1).values
    
    # Iterative refinement
    for _ in range(5):
        scale = (g_max - g_min) / levels
        scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
        q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, levels)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = groups - deq
        g_min = g_min + err.min(1).values * 0.5
        g_max = g_max + err.max(1).values * 0.5
    
    scale = (g_max - g_min) / levels
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, levels)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    result = deq.flatten()[:n]
    return result.view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 6: Fisher-Free Importance (Weight Statistics as Importance)
# =============================================================================
def fisher_free_importance_quant(weight, group_size=8, protect_pct=0.05):
    """
    Estimate weight importance from statistics:
    - High magnitude = important
    - High variance within group = important
    - Outliers = important
    
    Protect important weights with lower quantization error.
    """
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    num_groups = flat.numel() // group_size
    groups = flat.view(num_groups, group_size)
    
    # Compute importance score per group
    group_magnitude = groups.abs().mean(dim=1)
    group_variance = groups.var(dim=1)
    group_max = groups.abs().max(dim=1).values
    
    # Combined importance: magnitude * variance * max
    importance = group_magnitude * (1 + group_variance) * group_max
    importance = importance / importance.max()  # Normalize
    
    # Determine which groups to protect
    num_protected = int(num_groups * protect_pct)
    _, protected_indices = torch.topk(importance, num_protected)
    protected_mask = torch.zeros(num_groups, dtype=torch.bool, device=weight.device)
    protected_mask[protected_indices] = True
    
    result = torch.zeros_like(groups)
    
    # Protected groups: Use g=4 (more precision)
    for g in range(num_groups):
        group = groups[g]
        if protected_mask[g]:
            # Higher precision for important groups
            sub_group_size = 4
            sub_groups = group.view(-1, sub_group_size)
            g_min, g_max = sub_groups.min(1).values, sub_groups.max(1).values
            scale = (g_max - g_min) / 15.0
            scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
            q = ((sub_groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
            deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
            result[g] = deq.flatten()
        else:
            # Normal quantization
            g_min, g_max = group.min(), group.max()
            scale = (g_max - g_min) / 15.0 if (g_max - g_min).abs() > 1e-8 else 1.0
            q = ((group - g_min) / scale).round().clamp(0, 15)
            result[g] = q * scale + g_min
    
    return result.flatten()[:n].view(original_shape).to(weight.dtype)


# =============================================================================
# APPROACH 7: Hybrid Sparsity + Quantization with Magnitude Pruning
# =============================================================================
def magnitude_aware_sparse_quant(weight, sparsity=0.3, group_size=8):
    """
    Instead of uniform sparsity, prune based on group importance.
    Prune entire groups (structured) rather than individual weights.
    
    Groups with lowest L2 norm are pruned (set to mean of remaining).
    """
    original_shape = weight.shape
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    num_groups = flat.numel() // group_size
    groups = flat.view(num_groups, group_size)
    
    # Compute group importance (L2 norm)
    group_norms = groups.norm(dim=1)
    
    # Find groups to prune (lowest norms)
    num_prune = int(num_groups * sparsity)
    _, prune_indices = torch.topk(group_norms, num_prune, largest=False)
    prune_mask = torch.zeros(num_groups, dtype=torch.bool, device=weight.device)
    prune_mask[prune_indices] = True
    
    # Compute mean value to use for pruned groups
    kept_groups = groups[~prune_mask]
    mean_value = kept_groups.mean()
    
    result = groups.clone()
    
    # Prune groups: set to mean (or zero)
    result[prune_mask] = mean_value
    
    # Quantize remaining groups
    for g in range(num_groups):
        if not prune_mask[g]:
            group = result[g]
            g_min, g_max = group.min(), group.max()
            scale = (g_max - g_min) / 15.0 if (g_max - g_min).abs() > 1e-8 else 1.0
            q = ((group - g_min) / scale).round().clamp(0, 15)
            result[g] = q * scale + g_min
    
    return result.flatten()[:n].view(original_shape).to(weight.dtype)


# =============================================================================
# TEST HARNESS
# =============================================================================
def test(model, orig, tok, texts, name, fn, baseline, comp):
    print(f"  {name}...", end=" ", flush=True)
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                getattr(layer.mlp, p).weight.data = fn(orig[i][p].clone(), i, len(model.model.layers))
    ppl = compute_ppl(model, tok, texts)
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) {comp} {status}")
    return ppl, delta


def main():
    print("=" * 70)
    print("Novel Compression Techniques (AWQ/GPTQ-inspired, No Calibration)")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:30]
    
    orig = {}
    total_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        orig[i] = {}
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, p):
                orig[i][p] = getattr(layer.mlp, p).weight.data.clone()
    
    def restore():
        for i, layer in enumerate(model.model.layers):
            for p in ["gate_proj", "up_proj", "down_proj"]:
                if p in orig[i]:
                    getattr(layer.mlp, p).weight.data = orig[i][p].clone()
    
    baseline = compute_ppl(model, tok, texts)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print("\n" + "-" * 70)
    
    configs = [
        ("Outlier 1% preserve", lambda w, i, t: outlier_preserving_quant(w, 0.01, 16), "~7.5x"),
        ("Outlier 5% preserve", lambda w, i, t: outlier_preserving_quant(w, 0.05, 16), "~6x"),
        ("Correlation compensate", lambda w, i, t: correlation_compensated_quant(w, 8, 0.5), "~4x"),
        ("Smart PQ + residual", lambda w, i, t: smart_pq(w, 256, 8, True), "~8x"),
        ("Smart PQ no residual", lambda w, i, t: smart_pq(w, 256, 8, False), "~16x"),
        ("Multi-stage residual", lambda w, i, t: multi_stage_residual(w, 16, 5), "~4.5x"),
        ("Layer-position aware", lambda w, i, t: layer_position_quant(w, i, t, 8), "~5.5x"),
        ("Fisher-free importance", lambda w, i, t: fisher_free_importance_quant(w, 8, 0.1), "~4x"),
        ("Magnitude sparse 20%", lambda w, i, t: magnitude_aware_sparse_quant(w, 0.2, 8), "~5x"),
        ("Magnitude sparse 30%", lambda w, i, t: magnitude_aware_sparse_quant(w, 0.3, 8), "~5.7x"),
    ]
    
    results = []
    for name, fn, comp in configs:
        restore()
        ppl, delta = test(model, orig, tok, texts, name, fn, baseline, comp)
        results.append((name, comp, ppl, delta))
        gc.collect()
    
    print("\n" + "=" * 70)
    print("SUMMARY - Novel Techniques")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print()
    print(f"{'Technique':<25} {'Comp':>6} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 55)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<25} {comp:>6} {ppl:>8.2f} {delta:>+6.1f}% {status}")


if __name__ == "__main__":
    main()
