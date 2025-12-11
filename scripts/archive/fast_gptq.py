#!/usr/bin/env python3
"""
Fast GPTQ Implementation - Block-wise processing for speed.

Instead of column-by-column, process blocks of columns together.
Trades some optimality for 10-100x speedup.
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
from typing import Dict, Tuple


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


class FastHessianCollector:
    """Collect diagonal Hessian approximation (much faster)."""
    
    def __init__(self):
        self.hessians: Dict[str, torch.Tensor] = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is None:
                return
            
            # Diagonal Hessian approximation: sum of squared activations
            x = x.reshape(-1, x.shape[-1]).float()
            h_diag = (x ** 2).sum(dim=0)
            
            if name not in self.hessians:
                self.hessians[name] = h_diag.cpu()
            else:
                self.hessians[name] += h_diag.cpu()
        
        return fn
    
    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_hessians(self) -> Dict[str, torch.Tensor]:
        return self.hessians


def collect_hessians_fast(model, tokenizer, texts, num_samples=64):
    """Fast Hessian collection (diagonal only)."""
    print(f"  Collecting diagonal Hessians from {num_samples} samples...")
    
    collector = FastHessianCollector()
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
    
    hessians = collector.get_hessians()
    collector.remove()
    
    return hessians


def fast_gptq_quantize(
    weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """
    Fast GPTQ: Block-wise quantization with diagonal Hessian.
    
    Process blocks of columns together instead of column-by-column.
    Uses diagonal Hessian approximation for speed.
    """
    W = weight.clone().float()
    out_features, in_features = W.shape
    device = W.device
    
    H_diag = hessian_diag.to(device).float()
    
    # Ensure H_diag matches weight dimensions
    if len(H_diag) < in_features:
        H_diag = torch.cat([H_diag, torch.ones(in_features - len(H_diag), device=device)])
    H_diag = H_diag[:in_features]
    
    # Add damping
    damp = percdamp * H_diag.mean()
    H_diag = H_diag + damp
    H_inv_diag = 1.0 / H_diag
    
    maxq = 2**bits - 1
    Q = torch.zeros_like(W)
    
    # Process in blocks
    for block_start in range(0, in_features, block_size):
        block_end = min(block_start + block_size, in_features)
        block_W = W[:, block_start:block_end].clone()
        
        # Compute group quantization parameters
        for col in range(block_W.shape[1]):
            global_col = block_start + col
            group_idx = global_col // group_size
            col_in_group = global_col % group_size
            
            # Compute scale at start of each group
            if col_in_group == 0:
                group_end = min(global_col + group_size, in_features)
                group_W = W[:, global_col:group_end]
                g_min = group_W.min(dim=1).values
                g_max = group_W.max(dim=1).values
                scale = (g_max - g_min) / maxq
                scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
                zero = g_min
            
            # Quantize column
            w = block_W[:, col]
            q = ((w - zero) / scale).round().clamp(0, maxq)
            Q[:, global_col] = q * scale + zero
            
            # Error compensation to remaining columns in block
            if col + 1 < block_W.shape[1]:
                err = (w - Q[:, global_col]) * H_inv_diag[global_col]
                # Distribute error based on Hessian importance
                for j in range(col + 1, block_W.shape[1]):
                    block_W[:, j] -= err * (H_diag[block_start + j] / H_diag.sum()) * 0.1
    
    return Q.to(weight.dtype)


def quantize_with_gptq(model, hessians, bits=4, group_size=128):
    """Apply fast GPTQ to all MLP layers."""
    for i, layer in enumerate(model.model.layers):
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            if not hasattr(layer.mlp, proj):
                continue
            
            proj_module = getattr(layer.mlp, proj)
            W = proj_module.weight.data
            
            # Find matching Hessian
            H_diag = None
            for name, h in hessians.items():
                if f'.{i}.' in name and proj in name:
                    H_diag = h
                    break
            
            if H_diag is None:
                H_diag = torch.ones(W.shape[1])
            
            Q = fast_gptq_quantize(W, H_diag, bits=bits, group_size=group_size)
            proj_module.weight.data = Q


def test_config(model, orig, tok, texts, hessians, baseline, name, bits, gs):
    """Test a GPTQ configuration."""
    print(f"  {name}...", end=" ", flush=True)
    
    # Restore original weights
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                getattr(layer.mlp, p).weight.data = orig[i][p].clone()
    
    # Apply GPTQ
    start = time.time()
    quantize_with_gptq(model, hessians, bits=bits, group_size=gs)
    quant_time = time.time() - start
    
    # Evaluate
    ppl = compute_ppl(model, tok, texts)
    delta = (ppl - baseline) / baseline * 100
    
    # Estimate compression
    overhead = 32 / gs * 2  # scale + zero per group
    bpw = bits + overhead / (32 / bits)
    comp = 32 / bpw
    
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) ~{comp:.1f}x [{quant_time:.1f}s] {status}")
    
    return ppl, delta, comp


def main():
    print("=" * 70)
    print("FAST GPTQ IMPLEMENTATION")
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
    
    baseline = compute_ppl(model, tok, texts[:30])
    print(f"\nBaseline PPL: {baseline:.2f}")
    
    # Collect Hessians
    print("\n" + "-" * 70)
    start = time.time()
    hessians = collect_hessians_fast(model, tok, texts, num_samples=64)
    print(f"  Collected {len(hessians)} Hessians in {time.time() - start:.1f}s")
    
    # Test configurations
    print("\n" + "-" * 70)
    print("TESTING FAST GPTQ")
    print("-" * 70)
    
    configs = [
        ("INT4 g64", 4, 64),
        ("INT4 g128", 4, 128),
        ("INT4 g256", 4, 256),
        ("INT4 g512", 4, 512),
        ("INT3 g64", 3, 64),
        ("INT3 g128", 3, 128),
        ("INT3 g256", 3, 256),
        ("INT2 g64", 2, 64),
    ]
    
    results = []
    for name, bits, gs in configs:
        gc.collect()
        ppl, delta, comp = test_config(model, orig, tok, texts[:30], hessians, baseline, name, bits, gs)
        results.append((name, comp, ppl, delta))
    
    # Summary
    print("\n" + "=" * 70)
    print("FAST GPTQ RESULTS")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}\n")
    print(f"{'Config':<15} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 45)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<15} {comp:>7.1f}x {ppl:>8.2f} {delta:>+6.1f}% {status}")
    
    # Best configs
    print("\n" + "-" * 45)
    best_ok = [r for r in results if r[3] < 1]
    if best_ok:
        print("✅ Best <1% PPL:")
        for r in sorted(best_ok, key=lambda x: -x[1]):
            print(f"   {r[0]}: {r[1]:.1f}x compression, {r[3]:+.2f}% PPL")
    
    best_almost = [r for r in results if 1 <= r[3] < 2]
    if best_almost:
        print("\n⚠️ Best <2% PPL:")
        for r in sorted(best_almost, key=lambda x: -x[1]):
            print(f"   {r[0]}: {r[1]:.1f}x compression, {r[3]:+.2f}% PPL")


if __name__ == "__main__":
    main()
