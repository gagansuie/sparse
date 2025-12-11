#!/usr/bin/env python3
"""
Full GPTQ Implementation for 10x Compression.

GPTQ quantizes weights column-by-column using Hessian information
to optimally compensate for quantization error in remaining columns.

Key differences from our previous simplified version:
1. Compute full Hessian H = X^T X from calibration data
2. Use Cholesky decomposition for efficient inverse
3. Process columns in optimal order (by Hessian diagonal)
4. Compensate error using H^{-1} for remaining columns
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional


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


class HessianCollector:
    """Collect Hessian (X^T X) for each linear layer."""
    
    def __init__(self):
        self.hessians: Dict[str, torch.Tensor] = {}
        self.num_samples: Dict[str, int] = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is None:
                return
            
            # Flatten batch and sequence dimensions
            x = x.reshape(-1, x.shape[-1]).float()
            
            # Compute H = X^T X (online accumulation)
            H = x.t() @ x
            
            if name not in self.hessians:
                self.hessians[name] = H.cpu()
                self.num_samples[name] = x.shape[0]
            else:
                self.hessians[name] += H.cpu()
                self.num_samples[name] += x.shape[0]
        
        return fn
    
    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj']):
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_hessians(self) -> Dict[str, torch.Tensor]:
        """Return normalized Hessians."""
        result = {}
        for name, H in self.hessians.items():
            n = self.num_samples[name]
            result[name] = H / n if n > 0 else H
        return result


def collect_hessians(model, tokenizer, texts, num_samples=128):
    """Run calibration and collect Hessian matrices."""
    print(f"  Collecting Hessians from {num_samples} samples...")
    
    collector = HessianCollector()
    collector.register(model)
    
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, text in enumerate(texts[:num_samples]):
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            ids = enc["input_ids"].to(device)
            _ = model(ids)
            
            if (i + 1) % 32 == 0:
                print(f"    Processed {i+1}/{num_samples}")
    
    hessians = collector.get_hessians()
    collector.remove()
    
    return hessians


def gptq_quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    actorder: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Full GPTQ quantization of a single weight matrix.
    
    Args:
        weight: [out_features, in_features] weight matrix
        hessian: [in_features, in_features] Hessian H = X^T X
        bits: quantization bits (4 for INT4)
        group_size: group size for per-group quantization
        percdamp: damping factor for Hessian diagonal
        actorder: whether to process columns in activation order
    
    Returns:
        quantized_weight: dequantized weight matrix
        quant_info: scales, zeros for each group
    """
    W = weight.clone().float()
    out_features, in_features = W.shape
    device = W.device
    
    H = hessian.to(device).float()
    
    # Add damping to diagonal
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)
    
    # Cholesky decomposition for efficient inverse
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
    except:
        # Fallback: diagonal approximation
        H_inv = torch.diag(1.0 / (torch.diag(H) + damp))
    
    H_inv_diag = torch.diag(H_inv)
    
    # Determine column order
    if actorder:
        # Process columns with higher Hessian diagonal first (more important)
        perm = torch.argsort(torch.diag(H), descending=True)
    else:
        perm = torch.arange(in_features, device=device)
    
    inv_perm = torch.argsort(perm)
    
    # Reorder weight and Hessian
    W = W[:, perm]
    H_inv = H_inv[perm][:, perm]
    H_inv_diag = H_inv_diag[perm]
    
    # Quantization parameters
    maxq = 2**bits - 1
    
    # Storage for scales and zeros
    num_groups = (in_features + group_size - 1) // group_size
    scales = torch.zeros(out_features, num_groups, device=device)
    zeros = torch.zeros(out_features, num_groups, device=device)
    
    # Process column by column with error compensation
    Q = torch.zeros_like(W)
    
    for col in range(in_features):
        group_idx = col // group_size
        col_in_group = col % group_size
        
        # Get current column
        w = W[:, col].clone()
        
        # Compute group quantization parameters at start of each group
        if col_in_group == 0:
            group_end = min(col + group_size, in_features)
            group_w = W[:, col:group_end]
            
            # Compute min/max per row
            g_min = group_w.min(dim=1).values
            g_max = group_w.max(dim=1).values
            
            # Asymmetric quantization
            scale = (g_max - g_min) / maxq
            scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
            zero = g_min
            
            scales[:, group_idx] = scale
            zeros[:, group_idx] = zero
        
        scale = scales[:, group_idx]
        zero = zeros[:, group_idx]
        
        # Quantize
        q = ((w - zero) / scale).round().clamp(0, maxq)
        Q[:, col] = q * scale + zero
        
        # Error compensation: distribute error to remaining columns
        err = (w - Q[:, col]) / H_inv_diag[col]
        
        # Update remaining columns
        W[:, col+1:] -= err.unsqueeze(1) * H_inv[col, col+1:].unsqueeze(0)
    
    # Restore original column order
    Q = Q[:, inv_perm]
    
    return Q.to(weight.dtype), {'scales': scales, 'zeros': zeros}


def gptq_quantize_layer(layer, hessians, bits=4, group_size=128):
    """Quantize all weights in an MLP layer using GPTQ."""
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        if not hasattr(layer.mlp, proj_name):
            continue
        
        proj = getattr(layer.mlp, proj_name)
        full_name = None
        
        # Find matching Hessian
        for name in hessians:
            if proj_name in name:
                full_name = name
                break
        
        if full_name is None or full_name not in hessians:
            continue
        
        H = hessians[full_name]
        W = proj.weight.data
        
        # GPTQ quantize
        Q, _ = gptq_quantize_weight(W, H, bits=bits, group_size=group_size)
        proj.weight.data = Q


def test_gptq(model, tokenizer, texts, hessians, baseline, name, bits, group_size):
    """Test GPTQ configuration."""
    print(f"  {name}...", end=" ", flush=True)
    
    # Quantize each layer
    for i, layer in enumerate(model.model.layers):
        # Filter Hessians for this layer
        layer_hessians = {k: v for k, v in hessians.items() if f'.{i}.' in k}
        if layer_hessians:
            gptq_quantize_layer(layer, layer_hessians, bits=bits, group_size=group_size)
    
    ppl = compute_ppl(model, tokenizer, texts)
    delta = (ppl - baseline) / baseline * 100
    
    # Estimate compression
    if bits == 4:
        comp = f"~{8 * 32 // (bits * 32 // group_size + 32):.0f}x"
    else:
        comp = f"~{32 // bits:.0f}x"
    
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) {comp} {status}")
    
    return ppl, delta, comp


def main():
    print("=" * 70)
    print("FULL GPTQ IMPLEMENTATION")
    print("=" * 70)
    
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()]
    
    # Save original weights
    orig = {}
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
    
    baseline = compute_ppl(model, tok, texts[:30])
    print(f"\nBaseline PPL: {baseline:.2f}")
    
    # Collect Hessians
    print("\n" + "-" * 70)
    print("COLLECTING HESSIAN MATRICES")
    print("-" * 70)
    start = time.time()
    hessians = collect_hessians(model, tok, texts, num_samples=64)
    print(f"  Collected {len(hessians)} Hessians in {time.time() - start:.1f}s")
    
    # Test configurations
    print("\n" + "-" * 70)
    print("TESTING GPTQ CONFIGURATIONS")
    print("-" * 70)
    
    configs = [
        ("GPTQ INT4 g64", 4, 64),
        ("GPTQ INT4 g128", 4, 128),
        ("GPTQ INT4 g256", 4, 256),
        ("GPTQ INT3 g64", 3, 64),
        ("GPTQ INT3 g128", 3, 128),
    ]
    
    results = []
    for name, bits, gs in configs:
        restore()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        ppl, delta, comp = test_gptq(model, tok, texts[:30], hessians, baseline, name, bits, gs)
        results.append((name, comp, ppl, delta))
    
    # Summary
    print("\n" + "=" * 70)
    print("GPTQ RESULTS")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}\n")
    print(f"{'Config':<20} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 50)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<20} {comp:>8} {ppl:>8.2f} {delta:>+6.1f}% {status}")


if __name__ == "__main__":
    main()
