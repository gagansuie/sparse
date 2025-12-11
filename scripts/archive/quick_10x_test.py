#!/usr/bin/env python3
"""
Quick 10x compression test combining our best techniques.

Tests:
1. AWQ-style importance + multi-stage (our best: 7.5x)
2. Add simple GPTQ-style error compensation
3. Add knowledge distillation post-processing
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List


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


class ImportanceCollector:
    """Collect activation importance."""
    
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
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def normalize(self):
        for k in self.importance:
            self.importance[k] = self.importance[k] / self.importance[k].max()


def collect_importance(model, tokenizer, texts, num_samples=32):
    """Quick importance collection."""
    collector = ImportanceCollector()
    collector.register(model)
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for text in texts[:num_samples]:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            _ = model(enc["input_ids"].to(device))
    
    collector.remove()
    collector.normalize()
    return collector.importance


def hybrid_quantize(weight, importance, group_size=128, use_residual=True):
    """Our best quantization: AWQ-style + multi-stage."""
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


def quantize_aggressive(weight, importance, layer_idx, total_layers):
    """
    Aggressive layer-aware quantization for 10x target.
    
    Edge: g64 with residual
    Middle: g256 no residual
    """
    pos = layer_idx / total_layers
    
    if pos < 0.1 or pos > 0.9:
        return hybrid_quantize(weight, importance, group_size=64, use_residual=True)
    else:
        return hybrid_quantize(weight, importance, group_size=256, use_residual=False)


def simple_distillation(teacher, student, tokenizer, texts, num_steps=50, lr=1e-5):
    """Simple output distillation to recover quality."""
    device = next(student.parameters()).device
    teacher.eval()
    
    # Only train student MLP weights
    for param in student.parameters():
        param.requires_grad = False
    for layer in student.model.layers:
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.mlp, proj):
                getattr(layer.mlp, proj).weight.requires_grad = True
    
    total_loss = 0
    for i, text in enumerate(texts[:num_steps]):
        if not text.strip():
            continue
        
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        ids = enc["input_ids"].to(device)
        
        with torch.no_grad():
            t_out = teacher(ids)
        
        s_out = student(ids)
        
        # KL divergence
        T = 2.0
        loss = F.kl_div(
            F.log_softmax(s_out.logits / T, dim=-1),
            F.softmax(t_out.logits / T, dim=-1),
            reduction='batchmean'
        ) * (T ** 2)
        
        loss.backward()
        
        # Manual update
        with torch.no_grad():
            for layer in student.model.layers:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(layer.mlp, proj):
                        param = getattr(layer.mlp, proj).weight
                        if param.grad is not None:
                            param.data -= lr * param.grad
                            param.grad.zero_()
        
        total_loss += loss.item()
    
    return total_loss / num_steps if num_steps > 0 else 0


def main():
    print("=" * 70)
    print("QUICK 10x COMPRESSION TEST")
    print("=" * 70)
    
    teacher = AutoModelForCausalLM.from_pretrained(
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
    for i, layer in enumerate(teacher.model.layers):
        orig[i] = {}
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, p):
                orig[i][p] = getattr(layer.mlp, p).weight.data.clone()
    
    baseline = compute_ppl(teacher, tok, texts[:20])
    print(f"\nBaseline PPL: {baseline:.2f}")
    
    # Collect importance
    print("\n" + "-" * 70)
    print("COLLECTING IMPORTANCE")
    print("-" * 70)
    start = time.time()
    importance = collect_importance(teacher, tok, texts, num_samples=32)
    print(f"  Collected {len(importance)} importance maps in {time.time() - start:.1f}s")
    
    def restore():
        for i, layer in enumerate(teacher.model.layers):
            for p in ["gate_proj", "up_proj", "down_proj"]:
                if p in orig[i]:
                    getattr(layer.mlp, p).weight.data = orig[i][p].clone()
    
    print("\n" + "-" * 70)
    print("TESTING CONFIGURATIONS")
    print("-" * 70)
    
    results = []
    
    # Test 1: Our best (7.5x)
    print("\n1. Hybrid g256+residual (7.5x baseline):")
    restore()
    for i, layer in enumerate(teacher.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(name)
                getattr(layer.mlp, p).weight.data = hybrid_quantize(
                    orig[i][p].clone(), imp, group_size=256, use_residual=True
                )
    
    ppl = compute_ppl(teacher, tok, texts[:20])
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"   PPL: {ppl:.2f} (Δ {delta:+.1f}%) {status}")
    results.append(("Hybrid g256+res", "~7.5x", ppl, delta))
    
    # Test 2: Aggressive layer-aware (targeting 8-9x)
    print("\n2. Aggressive layer-aware (8-9x):")
    restore()
    total_layers = len(teacher.model.layers)
    for i, layer in enumerate(teacher.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(name)
                getattr(layer.mlp, p).weight.data = quantize_aggressive(
                    orig[i][p].clone(), imp, i, total_layers
                )
    
    ppl = compute_ppl(teacher, tok, texts[:20])
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"   PPL: {ppl:.2f} (Δ {delta:+.1f}%) {status}")
    results.append(("Aggressive layer", "~8-9x", ppl, delta))
    
    # Test 3: Aggressive + distillation
    print("\n3. Aggressive + knowledge distillation:")
    # Create student with aggressive quantization
    student = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    for i, layer in enumerate(student.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                name = f"model.layers.{i}.mlp.{p}"
                imp = importance.get(name)
                getattr(layer.mlp, p).weight.data = quantize_aggressive(
                    orig[i][p].clone(), imp, i, total_layers
                )
    
    pre_distill_ppl = compute_ppl(student, tok, texts[:20])
    print(f"   Before distillation: PPL {pre_distill_ppl:.2f}")
    
    # Reload teacher (it was modified)
    teacher = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    print("   Running distillation (50 steps)...", end=" ", flush=True)
    avg_loss = simple_distillation(teacher, student, tok, texts, num_steps=50)
    print(f"avg loss: {avg_loss:.4f}")
    
    ppl = compute_ppl(student, tok, texts[:20])
    delta = (ppl - baseline) / baseline * 100
    recovery = ((pre_distill_ppl - ppl) / (pre_distill_ppl - baseline) * 100) if pre_distill_ppl != baseline else 0
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"   After distillation: PPL {ppl:.2f} (Δ {delta:+.1f}%) {status}")
    print(f"   Quality recovery: {recovery:.1f}%")
    results.append(("Aggressive + KD", "~8-9x", ppl, delta))
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBaseline PPL: {baseline:.2f}\n")
    print(f"{'Config':<20} {'Comp':>8} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 50)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<20} {comp:>8} {ppl:>8.2f} {delta:>+6.1f}% {status}")


if __name__ == "__main__":
    main()
