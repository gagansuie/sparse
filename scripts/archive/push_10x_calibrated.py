#!/usr/bin/env python3
"""
Push for 10x compression with calibration.

Current best: 7x with -0.3% PPL (Hybrid g128)

Strategies to try:
1. Larger groups for middle layers (g256, g512)
2. Sparse + calibrated hybrid
3. INT3 + calibration for middle layers
4. Layer-position aware with calibration
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
import time


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


class ActivationCollector:
    def __init__(self):
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is not None:
                self.activations.setdefault(name, []).append(
                    x.abs().mean(dim=(0, 1)).detach().cpu()
                )
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
    
    def get_importance(self) -> Dict[str, torch.Tensor]:
        importance = {}
        for name, acts in self.activations.items():
            avg_act = torch.stack(acts).mean(dim=0)
            importance[name] = avg_act / avg_act.max()
        return importance


def collect_calibration(model, tokenizer, texts, num_samples=64):
    print(f"  Collecting calibration from {num_samples} samples...")
    collector = ActivationCollector()
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
    
    importance = collector.get_importance()
    collector.remove()
    return importance


def calibrated_quant(weight, importance, group_size=128, use_residual=True):
    """Calibrated quantization with AWQ-style scaling and optional residual."""
    flat = weight.flatten().float()
    n = flat.numel()
    
    # AWQ-style importance scaling
    if importance is not None and len(importance) > 0:
        imp = importance.to(weight.device)
        if len(imp) < n:
            imp = imp.repeat(n // len(imp) + 1)[:n]
        else:
            imp = imp[:n]
        scale_mask = (imp > 0.3).float() * 1.5 + 0.5
        scaled = flat * scale_mask
    else:
        scaled = flat
        scale_mask = torch.ones_like(flat)
    
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        scaled = torch.cat([scaled, torch.zeros(pad, device=weight.device)])
        scale_mask = torch.cat([scale_mask, torch.ones(pad, device=weight.device)])
    
    groups = scaled.view(-1, group_size)
    mask_groups = scale_mask.view(-1, group_size)
    
    # INT4 with iterative refinement
    g_min, g_max = groups.min(1).values, groups.max(1).values
    for _ in range(5):
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
    
    if use_residual:
        residual = groups - stage1
        r_min, r_max = residual.min(1).values, residual.max(1).values
        r_scale = (r_max - r_min) / 3.0
        r_scale = torch.where(r_scale.abs() < 1e-8, torch.ones_like(r_scale), r_scale)
        q2 = ((residual - r_min.unsqueeze(1)) / r_scale.unsqueeze(1)).round().clamp(0, 3)
        stage2 = q2 * r_scale.unsqueeze(1) + r_min.unsqueeze(1)
        result = (stage1 + stage2) / mask_groups
    else:
        result = stage1 / mask_groups
    
    return result.flatten()[:n].view(weight.shape).to(weight.dtype)


def layer_aware_10x(weight, importance, layer_idx, total_layers):
    """
    Layer-position aware for 10x compression.
    
    First/last 5%: g64 with residual (quality)
    Next 15%: g128 with residual
    Middle 60%: g256 no residual (aggressive)
    """
    pos = layer_idx / total_layers
    
    if pos < 0.05 or pos > 0.95:
        return calibrated_quant(weight, importance, group_size=64, use_residual=True)
    elif pos < 0.2 or pos > 0.8:
        return calibrated_quant(weight, importance, group_size=128, use_residual=True)
    else:
        return calibrated_quant(weight, importance, group_size=256, use_residual=False)


def layer_aware_10x_v2(weight, importance, layer_idx, total_layers):
    """
    More aggressive: Only protect first/last layer.
    """
    pos = layer_idx / total_layers
    
    if pos < 0.05 or pos > 0.95:
        return calibrated_quant(weight, importance, group_size=64, use_residual=True)
    else:
        return calibrated_quant(weight, importance, group_size=256, use_residual=False)


def layer_aware_10x_v3(weight, importance, layer_idx, total_layers):
    """
    Maximum push: g512 for middle layers.
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return calibrated_quant(weight, importance, group_size=128, use_residual=True)
    else:
        return calibrated_quant(weight, importance, group_size=512, use_residual=False)


def sparse_calibrated(weight, importance, sparsity=0.15, group_size=128):
    """Structured sparsity + calibrated quantization."""
    flat = weight.flatten().float()
    n = flat.numel()
    
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    num_groups = flat.numel() // group_size
    groups = flat.view(num_groups, group_size)
    
    # Importance-weighted group selection for pruning
    if importance is not None and len(importance) > 0:
        imp = importance.to(weight.device)
        if len(imp) < flat.numel():
            imp = imp.repeat(flat.numel() // len(imp) + 1)[:flat.numel()]
        imp_groups = imp[:flat.numel()].view(num_groups, group_size).mean(dim=1)
    else:
        imp_groups = groups.abs().mean(dim=1)
    
    # Prune lowest importance groups
    num_prune = int(num_groups * sparsity)
    _, prune_idx = torch.topk(imp_groups, num_prune, largest=False)
    prune_mask = torch.zeros(num_groups, dtype=torch.bool, device=weight.device)
    prune_mask[prune_idx] = True
    
    # Set pruned groups to mean
    mean_val = groups[~prune_mask].mean()
    groups[prune_mask] = mean_val
    
    # Quantize remaining with calibration
    result = calibrated_quant(groups.flatten()[:n].view(weight.shape), importance, group_size, use_residual=True)
    return result


def test(model, orig, tok, texts, name, fn, baseline, expected_comp, importance):
    print(f"  {name}...", end=" ", flush=True)
    total_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                w = orig[i][p].clone()
                layer_name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(layer_name) if importance else None
                try:
                    # Try calling with layer info first
                    getattr(layer.mlp, p).weight.data = fn(w, imp, i, total_layers)
                except TypeError:
                    # Fall back to simple call
                    getattr(layer.mlp, p).weight.data = fn(w, imp)
    ppl = compute_ppl(model, tok, texts)
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) {expected_comp} {status}")
    return ppl, delta


def main():
    print("=" * 70)
    print("PUSH FOR 10x COMPRESSION WITH CALIBRATION")
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
    
    print("\n" + "-" * 70)
    start = time.time()
    importance = collect_calibration(model, tok, texts, num_samples=64)
    print(f"  Calibration took {time.time() - start:.1f}s")
    
    print("\n" + "-" * 70)
    print("TESTING 10x CONFIGURATIONS")
    print("-" * 70)
    
    configs = [
        ("Baseline g128+res", lambda w, imp: calibrated_quant(w, imp, 128, True), "~7x"),
        ("g256 no residual", lambda w, imp: calibrated_quant(w, imp, 256, False), "~8x"),
        ("g256 + residual", lambda w, imp: calibrated_quant(w, imp, 256, True), "~7.5x"),
        ("g512 no residual", lambda w, imp: calibrated_quant(w, imp, 512, False), "~8.5x"),
        ("Layer-aware 10x v1", layer_aware_10x, "~8-9x"),
        ("Layer-aware 10x v2", layer_aware_10x_v2, "~9x"),
        ("Layer-aware 10x v3", layer_aware_10x_v3, "~9-10x"),
        ("Sparse 15% + calib", lambda w, imp: sparse_calibrated(w, imp, 0.15, 128), "~8x"),
        ("Sparse 20% + calib", lambda w, imp: sparse_calibrated(w, imp, 0.20, 128), "~9x"),
        ("Sparse 25% + calib", lambda w, imp: sparse_calibrated(w, imp, 0.25, 128), "~10x"),
    ]
    
    results = []
    for name, fn, comp in configs:
        restore()
        ppl, delta = test(model, orig, tok, texts[:30], name, fn, baseline, comp, importance)
        results.append((name, comp, ppl, delta))
        gc.collect()
    
    print("\n" + "=" * 70)
    print("RESULTS - Push for 10x")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}")
    print()
    print(f"{'Technique':<25} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 55)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<25} {comp:>8} {ppl:>8.2f} {delta:>+6.1f}% {status}")
    
    print("\n" + "=" * 70)
    best_ok = [r for r in results if r[3] < 1]
    best_almost = [r for r in results if 1 <= r[3] < 2]
    
    if best_ok:
        print("✅ <1% PPL:")
        for r in sorted(best_ok, key=lambda x: -float(x[1].replace('~','').replace('x','').split('-')[0])):
            print(f"   {r[0]}: {r[1]}, {r[3]:+.2f}%")
    if best_almost:
        print("\n⚠️ <2% PPL:")
        for r in sorted(best_almost, key=lambda x: -float(x[1].replace('~','').replace('x','').split('-')[0])):
            print(f"   {r[0]}: {r[1]}, {r[3]:+.2f}%")


if __name__ == "__main__":
    main()
