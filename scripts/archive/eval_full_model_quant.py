#!/usr/bin/env python3
"""
FULL MODEL QUANTIZATION (like llama.cpp)

llama.cpp quantizes EVERYTHING:
- MLP layers (c_fc, c_proj)
- Attention layers (c_attn, c_proj in attention)
- Embeddings (wte, wpe)
- Output layer (lm_head)

Let's do the same and see the real compression!
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    device = next(model.parameters()).device
    nll = 0.0
    ntokens = 0
    
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            nll += outputs.loss.item() * input_ids.numel()
            ntokens += input_ids.numel()
    
    return math.exp(nll / ntokens) if ntokens > 0 else float("nan")


# =============================================================================
# Q4_K QUANTIZATION (from previous script)
# =============================================================================

def q4_k_quantize(weight: torch.Tensor):
    weight = weight.float()
    flat = weight.flatten()
    numel = flat.numel()
    
    SUPER_BLOCK_SIZE = 256
    BLOCK_SIZE = 32
    NUM_BLOCKS = 8
    
    padded = ((numel + SUPER_BLOCK_SIZE - 1) // SUPER_BLOCK_SIZE) * SUPER_BLOCK_SIZE
    if padded > numel:
        flat = F.pad(flat, (0, padded - numel))
    
    super_blocks = flat.view(-1, SUPER_BLOCK_SIZE)
    num_super = super_blocks.shape[0]
    
    sb_min = super_blocks.min(dim=1).values
    sb_max = super_blocks.max(dim=1).values
    sb_range = torch.clamp(sb_max - sb_min, min=1e-8)
    
    normalized = (super_blocks - sb_min.unsqueeze(1)) / sb_range.unsqueeze(1)
    blocks = normalized.view(num_super, NUM_BLOCKS, BLOCK_SIZE)
    
    block_min = blocks.min(dim=2).values
    block_max = blocks.max(dim=2).values
    block_range = torch.clamp(block_max - block_min, min=1e-8)
    
    block_scales_q = (block_range * 63).round().clamp(0, 63).to(torch.uint8)
    block_mins_q = (block_min * 63).round().clamp(0, 63).to(torch.uint8)
    
    block_range_dq = block_scales_q.float() / 63
    block_min_dq = block_mins_q.float() / 63
    
    block_normalized = (blocks - block_min_dq.unsqueeze(2)) / block_range_dq.unsqueeze(2).clamp(min=1e-8)
    q = (block_normalized * 15).round().clamp(0, 15).to(torch.uint8)
    
    q_flat = q.flatten()
    packed = (q_flat[0::2] | (q_flat[1::2] << 4)).to(torch.uint8)
    
    return {
        'packed': packed,
        'd': sb_range.half(),
        'dmin': sb_min.half(),
        'scales': block_scales_q,
        'mins': block_mins_q,
        'shape': weight.shape,
        'numel': numel,
    }


def q4_k_dequantize(data: dict) -> torch.Tensor:
    packed = data['packed']
    d = data['d'].float()
    dmin = data['dmin'].float()
    scales = data['scales'].float() / 63
    mins = data['mins'].float() / 63
    shape = data['shape']
    numel = data['numel']
    
    num_super = d.numel()
    
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).flatten().float()
    
    blocks = q.view(num_super, 8, 32)
    
    block_range = scales.view(num_super, 8, 1)
    block_min = mins.view(num_super, 8, 1)
    normalized = (blocks / 15) * block_range + block_min
    
    super_range = d.view(num_super, 1, 1)
    super_min = dmin.view(num_super, 1, 1)
    weight = normalized * super_range + super_min
    
    weight = weight.flatten()[:numel]
    return weight.view(shape)


def q4_k_storage_bytes(data: dict) -> int:
    num_super = data['d'].numel()
    return data['packed'].numel() + num_super * 4 + num_super * 16


# =============================================================================
# QUANTIZED LAYERS
# =============================================================================

class QuantLinear(nn.Module):
    def __init__(self, data, bias=None, transpose=False):
        super().__init__()
        self.data = data
        self.transpose = transpose
        self.register_buffer('bias', bias)
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = q4_k_dequantize(self.data)
            if self.transpose:
                self._w = self._w.t()
        return F.linear(x, self._w.to(x.dtype).to(x.device), self.bias)


class QuantEmbedding(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._w = None
    
    def forward(self, x):
        if self._w is None:
            self._w = q4_k_dequantize(self.data)
        return F.embedding(x, self._w.to(torch.float32))


# =============================================================================
# FULL MODEL QUANTIZATION
# =============================================================================

def quantize_full_model(model):
    """Quantize entire GPT-2 model like llama.cpp does."""
    total_orig_fp16 = 0
    total_orig_fp32 = 0
    total_quant = 0
    
    # 1. Embeddings (wte, wpe)
    print("  Quantizing embeddings...")
    for name in ['wte', 'wpe']:
        emb = getattr(model.transformer, name)
        weight = emb.weight.data
        total_orig_fp16 += weight.numel() * 2
        total_orig_fp32 += weight.numel() * 4
        
        data = q4_k_quantize(weight)
        total_quant += q4_k_storage_bytes(data)
        
        setattr(model.transformer, name, QuantEmbedding(data))
    
    # 2. Transformer blocks
    print("  Quantizing transformer blocks...")
    for layer_idx, block in enumerate(model.transformer.h):
        # Attention: c_attn (QKV projection), c_proj (output projection)
        attn = block.attn
        for name in ['c_attn', 'c_proj']:
            layer = getattr(attn, name)
            weight = layer.weight.data.t().contiguous()  # Conv1D transpose
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig_fp16 += weight.numel() * 2
            total_orig_fp32 += weight.numel() * 4
            
            data = q4_k_quantize(weight)
            total_quant += q4_k_storage_bytes(data)
            
            setattr(attn, name, QuantLinear(data, bias))
        
        # MLP: c_fc, c_proj
        mlp = block.mlp
        for name in ['c_fc', 'c_proj']:
            layer = getattr(mlp, name)
            weight = layer.weight.data.t().contiguous()
            bias = layer.bias.data.clone() if layer.bias is not None else None
            
            total_orig_fp16 += weight.numel() * 2
            total_orig_fp32 += weight.numel() * 4
            
            data = q4_k_quantize(weight)
            total_quant += q4_k_storage_bytes(data)
            
            setattr(mlp, name, QuantLinear(data, bias))
        
        if layer_idx % 4 == 0:
            print(f"    Layer {layer_idx}/{len(model.transformer.h)}")
    
    # 3. Output layer (lm_head) - usually tied to wte, but let's handle it
    # In GPT-2, lm_head shares weights with wte, so we skip it
    
    return total_orig_fp16, total_orig_fp32, total_quant


def main():
    print("=" * 70)
    print("FULL MODEL QUANTIZATION (like llama.cpp)")
    print("=" * 70)
    
    model_name = "gpt2"
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()][:128]
    
    # Baseline
    print("\nComputing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    baseline_ppl = compute_perplexity(model, tokenizer, texts)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Count total model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1e6:.2f} MB")
    print(f"Model size (FP16): {total_params * 2 / 1e6:.2f} MB")
    del model
    
    # Quantize
    print("\n" + "=" * 60)
    print("QUANTIZING FULL MODEL")
    print("=" * 60)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    orig_fp16, orig_fp32, quant_bytes = quantize_full_model(model)
    
    compress_fp16 = orig_fp16 / quant_bytes
    compress_fp32 = orig_fp32 / quant_bytes
    bits_per_weight = (quant_bytes * 8) / (orig_fp16 / 2)
    
    print(f"\nQuantized weights: {orig_fp16/1e6:.2f} MB (FP16) → {quant_bytes/1e6:.2f} MB")
    print(f"Compression vs FP16: {compress_fp16:.2f}x")
    print(f"Compression vs FP32: {compress_fp32:.2f}x")
    print(f"Bits per weight: {bits_per_weight:.2f}")
    
    # Evaluate
    print("\nComputing quantized PPL...")
    ppl = compute_perplexity(model, tokenizer, texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    print(f"Quantized PPL: {ppl:.4f} (Δ {delta:+.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Baseline PPL:        {baseline_ppl:.4f}")
    print(f"Quantized PPL:       {ppl:.4f}")
    print(f"PPL Delta:           {delta:+.2f}%")
    print()
    print(f"Compression vs FP16: {compress_fp16:.2f}x")
    print(f"Compression vs FP32: {compress_fp32:.2f}x  ← (llama.cpp reports this)")
    print(f"Bits per weight:     {bits_per_weight:.2f}")
    print("=" * 70)
    
    # Comparison
    print("\n📊 COMPARISON WITH llama.cpp:")
    print("-" * 50)
    print(f"{'Method':<20} {'vs FP32':<12} {'PPL Δ':<10}")
    print("-" * 50)
    print(f"{'llama.cpp Q4_K_M':<20} {'~4x':<12} {'<1%':<10}")
    print(f"{'Tenpak Q4_K (full)':<20} {f'{compress_fp32:.2f}x':<12} {f'{delta:+.2f}%':<10}")
    print("-" * 50)
    
    if compress_fp32 >= 4.0 and abs(delta) < 5.0:
        print("\n🎯 We're in the same ballpark as llama.cpp!")
    
    if abs(delta) > 5.0:
        print("\n⚠️  High PPL delta - attention layers may be too sensitive")
        print("   Try: Keep attention in higher precision (Q6_K or Q8)")


if __name__ == "__main__":
    main()
