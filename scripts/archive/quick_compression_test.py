#!/usr/bin/env python3
"""
Quick test of compression limits without calibration.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import gc
import math
import torch
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


def quantize_iterative(w, g=8, iters=5, bits=4):
    """Iterative scale refinement with configurable bits."""
    levels = 2**bits - 1
    flat = w.flatten().float()
    n = flat.numel()
    pad = (g - n % g) % g
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=w.device)])
    
    groups = flat.view(-1, g)
    g_min, g_max = groups.min(1).values, groups.max(1).values
    
    for _ in range(iters):
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
    
    return deq.flatten()[:n].view(w.shape).to(w.dtype)


def test(model, orig, tok, texts, name, fn, baseline, comp):
    print(f"  {name}...", end=" ", flush=True)
    for i, layer in enumerate(model.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig[i]:
                getattr(layer.mlp, p).weight.data = fn(orig[i][p].clone())
    ppl = compute_ppl(model, tok, texts)
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f"PPL: {ppl:.2f} (Δ {delta:+.1f}%) {comp} {status}")
    return ppl, delta


def main():
    print("=" * 60)
    print("Quick Compression Limit Test")
    print("=" * 60)
    
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
    print("\n" + "-" * 60)
    
    configs = [
        ("INT4 g8 i5", lambda w: quantize_iterative(w, 8, 5, 4), "4.0x"),
        ("INT4 g16 i5", lambda w: quantize_iterative(w, 16, 5, 4), "5.3x"),
        ("INT4 g32 i5", lambda w: quantize_iterative(w, 32, 5, 4), "6.4x"),
        ("INT4 g64 i5", lambda w: quantize_iterative(w, 64, 5, 4), "7.1x"),
        ("INT4 g128 i5", lambda w: quantize_iterative(w, 128, 5, 4), "7.5x"),
        ("INT3 g8 i5", lambda w: quantize_iterative(w, 8, 5, 3), "5.3x"),
        ("INT3 g16 i5", lambda w: quantize_iterative(w, 16, 5, 3), "6.4x"),
        ("INT3 g32 i5", lambda w: quantize_iterative(w, 32, 5, 3), "8.0x"),
        ("INT2 g8 i5", lambda w: quantize_iterative(w, 8, 5, 2), "8.0x"),
        ("INT2 g16 i5", lambda w: quantize_iterative(w, 16, 5, 2), "10.7x"),
    ]
    
    results = []
    for name, fn, comp in configs:
        restore()
        ppl, delta = test(model, orig, tok, texts, name, fn, baseline, comp)
        results.append((name, comp, ppl, delta))
        gc.collect()
    
    print("\n" + "=" * 60)
    print("SUMMARY - Maximum Compression Without Calibration")
    print("=" * 60)
    print(f"\n{'Config':<15} {'Comp':>6} {'PPL':>8} {'Δ':>7} Status")
    print("-" * 45)
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<15} {comp:>6} {ppl:>8.2f} {delta:>+6.1f}% {status}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    best_ok = [r for r in results if r[3] < 1]
    best_almost = [r for r in results if 1 <= r[3] < 2]
    
    if best_ok:
        b = max(best_ok, key=lambda x: float(x[1].replace('x', '')))
        print(f"✅ Best <1% PPL: {b[0]} at {b[1]} compression")
    if best_almost:
        b = max(best_almost, key=lambda x: float(x[1].replace('x', '')))
        print(f"⚠️ Best <2% PPL: {b[0]} at {b[1]} compression")
    
    print("\n10x compression with <1% PPL requires calibration (AWQ/GPTQ style)")


if __name__ == "__main__":
    main()
