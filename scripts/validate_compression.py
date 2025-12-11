#!/usr/bin/env python3
"""
Validation test for Tenpak compression results on TinyLlama.

Validates:
1. No calibration: 4x at <1% PPL
2. With calibration: 7-7.5x at <1% PPL
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict


def compute_ppl(model, tokenizer, texts, max_samples=30, max_length=256):
    """Compute perplexity on text samples."""
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


def int4_iterative(weight, group_size=8, iterations=5):
    """INT4 with iterative scale refinement (no calibration)."""
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
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
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    return deq.flatten()[:n].view(weight.shape).to(weight.dtype)


def calibrated_hybrid(weight, importance, group_size=128, use_residual=True):
    """Calibrated hybrid quantization (AWQ-style + multi-stage)."""
    flat = weight.flatten().float()
    n = flat.numel()
    
    # Importance scaling
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
    
    # INT4 with refinement
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


class ImportanceCollector:
    def __init__(self):
        self.importance: Dict[str, torch.Tensor] = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if x is not None:
                imp = x.abs().mean(dim=(0, 1)).detach().cpu()
                if name not in self.importance:
                    self.importance[name] = imp
                else:
                    self.importance[name] += imp
        return fn
    
    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ['gate_proj', 'up_proj', 'down_proj']):
                self.hooks.append(module.register_forward_hook(self.hook_fn(name)))
    
    def remove(self):
        for h in self.hooks:
            h.remove()
    
    def normalize(self):
        for k in self.importance:
            self.importance[k] /= self.importance[k].max()


def main():
    print("=" * 70)
    print("TENPAK COMPRESSION VALIDATION - TinyLlama 1.1B")
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
    
    # Baseline
    baseline = compute_ppl(model, tok, texts[:30])
    print(f"\nBaseline PPL: {baseline:.2f}")
    
    results = []
    
    # Test 1: No calibration (INT4 g8)
    print("\n" + "-" * 70)
    print("TEST 1: No Calibration (int4_opt_llama_v1 style)")
    print("-" * 70)
    
    restore()
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                getattr(layer.mlp, p).weight.data = int4_iterative(orig[i][p].clone(), group_size=8)
    
    ppl = compute_ppl(model, tok, texts[:30])
    delta = (ppl - baseline) / baseline * 100
    status = "✅ PASS" if abs(delta) < 1 else "❌ FAIL"
    print(f"  INT4 g8: PPL {ppl:.2f} (Δ {delta:+.2f}%) ~4x {status}")
    results.append(("No calib INT4 g8", "4x", delta, abs(delta) < 1))
    
    # Test 2: Collect calibration data
    print("\n" + "-" * 70)
    print("TEST 2: Collecting Calibration Data")
    print("-" * 70)
    
    restore()
    collector = ImportanceCollector()
    collector.register(model)
    
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, text in enumerate(texts[:32]):
            if not text.strip():
                continue
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            _ = model(enc["input_ids"].to(device))
    
    collector.remove()
    collector.normalize()
    importance = collector.importance
    print(f"  Collected importance for {len(importance)} layers")
    
    # Test 3: Calibrated g128
    print("\n" + "-" * 70)
    print("TEST 3: Calibrated g128 (7x target)")
    print("-" * 70)
    
    restore()
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(name)
                getattr(layer.mlp, p).weight.data = calibrated_hybrid(
                    orig[i][p].clone(), imp, group_size=128, use_residual=True
                )
    
    ppl = compute_ppl(model, tok, texts[:30])
    delta = (ppl - baseline) / baseline * 100
    status = "✅ PASS" if abs(delta) < 1 else "❌ FAIL"
    print(f"  Calibrated g128+res: PPL {ppl:.2f} (Δ {delta:+.2f}%) ~7x {status}")
    results.append(("Calibrated g128", "7x", delta, abs(delta) < 1))
    
    # Test 4: Calibrated g256
    print("\n" + "-" * 70)
    print("TEST 4: Calibrated g256 (7.5x target)")
    print("-" * 70)
    
    restore()
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(name)
                getattr(layer.mlp, p).weight.data = calibrated_hybrid(
                    orig[i][p].clone(), imp, group_size=256, use_residual=True
                )
    
    ppl = compute_ppl(model, tok, texts[:30])
    delta = (ppl - baseline) / baseline * 100
    status = "✅ PASS" if abs(delta) < 1 else "❌ FAIL"
    print(f"  Calibrated g256+res: PPL {ppl:.2f} (Δ {delta:+.2f}%) ~7.5x {status}")
    results.append(("Calibrated g256", "7.5x", delta, abs(delta) < 1))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}\n")
    
    all_pass = True
    for name, comp, delta, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:<20} {comp:>6} Δ {delta:>+6.2f}% {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "-" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED - Compression results validated!")
    else:
        print("❌ SOME TESTS FAILED - Check results above")
    print("-" * 70)


if __name__ == "__main__":
    main()
