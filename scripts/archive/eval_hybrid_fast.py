#!/usr/bin/env python3
"""
FAST HYBRID VALIDATION - Quick sanity check in ~2 minutes

Tests the core hypothesis:
1. Does low-rank SVD preserve model quality at all?
2. If yes, at what rank_ratio does it break?

Skip codebook quantization entirely for now - just validate SVD works.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time

def evaluate_ppl_fast(model, encodings, max_steps=5):
    """Super fast PPL check - just 5 steps."""
    print(f"  PPL eval ({max_steps} steps)...", end="", flush=True)
    model.eval()
    seq_len = encodings.input_ids.size(1)
    nlls = []
    stride = 256
    max_length = 512
    
    for step, begin_loc in enumerate(range(0, seq_len - 1, stride)):
        if step >= max_steps:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - (0 if step == 0 else begin_loc)
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss.cpu())
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    print(f" {ppl:.2f}", flush=True)
    return ppl


class SVDLinear(nn.Module):
    """Linear layer with SVD low-rank approximation (no codebook)."""
    
    def __init__(self, U, S, Vh, bias=None):
        super().__init__()
        # Store as (U @ diag(S)) and Vh for efficient forward
        self.register_buffer('US', U @ torch.diag(S))  # (out, rank)
        self.register_buffer('Vh', Vh)  # (rank, in)
        self.register_buffer('bias', bias)
    
    def forward(self, x):
        # x @ W.T = x @ (U @ S @ Vh).T = x @ Vh.T @ S @ U.T
        out = F.linear(x, self.Vh)  # (batch, rank)
        out = F.linear(out, self.US)  # (batch, out)
        if self.bias is not None:
            out = out + self.bias
        return out


def svd_compress_layer(weight, rank_ratio):
    """Compress weight matrix using truncated SVD."""
    # weight is (out_features, in_features)
    out_features, in_features = weight.shape
    rank = max(1, int(min(out_features, in_features) * rank_ratio))
    
    # Truncated SVD
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    U = U[:, :rank].half()
    S = S[:rank].half()
    Vh = Vh[:rank, :].half()
    
    return U, S, Vh, rank


def test_single_layer(model, layer_idx, proj_name, rank_ratio, encodings, baseline_ppl):
    """Test SVD on a single layer and measure PPL impact."""
    # Get the layer
    if hasattr(model, 'transformer'):  # GPT-2
        layer = model.transformer.h[layer_idx]
        mlp = layer.mlp
        is_gpt2 = True
    else:  # LLaMA
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        is_gpt2 = False
    
    original = getattr(mlp, proj_name)
    
    # Get weight (handle GPT-2 Conv1D)
    if is_gpt2:
        weight = original.weight.data.t().contiguous()  # (out, in)
        bias = original.bias.data.clone() if original.bias is not None else None
    else:
        weight = original.weight.data
        bias = original.bias.data.clone() if original.bias is not None else None
    
    # SVD compress
    U, S, Vh, rank = svd_compress_layer(weight, rank_ratio)
    
    # Calculate compression
    original_params = weight.numel()
    compressed_params = U.numel() + S.numel() + Vh.numel()
    compression = original_params / compressed_params
    
    # Replace layer
    new_layer = SVDLinear(U, S, Vh, bias)
    setattr(mlp, proj_name, new_layer)
    
    # Measure PPL
    ppl = evaluate_ppl_fast(model, encodings)
    ppl_delta = (ppl - baseline_ppl) / baseline_ppl * 100
    
    return {
        'rank': rank,
        'compression': compression,
        'ppl': ppl,
        'ppl_delta': ppl_delta,
    }


def main():
    print("=" * 60)
    print("FAST HYBRID VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Use GPT-2
    model_name = "gpt2"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset (smaller)
    print("Loading WikiText-2 (10k tokens)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=10000)
    
    # Baseline
    print("\n--- BASELINE ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()
    baseline_ppl = evaluate_ppl_fast(model, encodings, max_steps=10)
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    del model
    
    # Test different rank ratios on layer 0, c_fc (the big one)
    print("\n--- SVD RANK RATIO SWEEP (Layer 0, c_fc) ---")
    print(f"{'Rank%':<8} {'Rank':<6} {'Compress':<10} {'PPL':<8} {'Δ%':<10} {'Status'}")
    print("-" * 55)
    
    rank_ratios = [0.8, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    for rank_ratio in rank_ratios:
        # Fresh model each time
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.eval()
        
        result = test_single_layer(model, 0, 'c_fc', rank_ratio, encodings, baseline_ppl)
        
        status = "✓ OK" if abs(result['ppl_delta']) < 1.0 else "✗ FAIL" if result['ppl_delta'] > 10 else "~ MARGINAL"
        print(f"{rank_ratio*100:<8.0f} {result['rank']:<6} {result['compression']:<10.2f}x {result['ppl']:<8.2f} {result['ppl_delta']:<+10.2f} {status}")
        
        del model
        
        # Early stop if catastrophic
        if result['ppl_delta'] > 100:
            print(f"\n⚠️  Catastrophic failure at {rank_ratio*100}% rank - stopping sweep")
            break
    
    # If any rank ratio worked, test full model
    print("\n--- FULL MODEL TEST (best working rank) ---")
    
    # Find best working rank_ratio (PPL delta < 5%)
    best_ratio = None
    for rank_ratio in [0.5, 0.3, 0.2]:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.eval()
        
        print(f"\nTesting rank_ratio={rank_ratio} on ALL layers...")
        
        total_original = 0
        total_compressed = 0
        
        for layer_idx in range(len(model.transformer.h)):
            layer = model.transformer.h[layer_idx]
            mlp = layer.mlp
            
            for proj_name in ['c_fc', 'c_proj']:
                original = getattr(mlp, proj_name)
                weight = original.weight.data.t().contiguous()
                bias = original.bias.data.clone() if original.bias is not None else None
                
                U, S, Vh, rank = svd_compress_layer(weight, rank_ratio)
                
                total_original += weight.numel() * 2  # FP16
                total_compressed += (U.numel() + S.numel() + Vh.numel()) * 2
                
                new_layer = SVDLinear(U, S, Vh, bias)
                setattr(mlp, proj_name, new_layer)
        
        compression = total_original / total_compressed
        ppl = evaluate_ppl_fast(model, encodings, max_steps=10)
        ppl_delta = (ppl - baseline_ppl) / baseline_ppl * 100
        
        print(f"  Compression: {compression:.2f}x")
        print(f"  PPL: {ppl:.2f} (Δ {ppl_delta:+.2f}%)")
        
        if ppl_delta < 5:
            best_ratio = rank_ratio
            print(f"  ✓ VIABLE - rank_ratio={rank_ratio} works!")
        else:
            print(f"  ✗ Too much degradation")
        
        del model
        
        if best_ratio:
            break
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Completed in {elapsed:.1f}s")
    print("=" * 60)
    
    if best_ratio:
        print(f"\n✓ SVD low-rank works at rank_ratio={best_ratio}")
        print("  Next step: Add codebook quantization to U, S, Vh factors")
    else:
        print("\n✗ SVD low-rank alone doesn't preserve quality")
        print("  Hybrid approach unlikely to work - need different strategy")


if __name__ == "__main__":
    main()
