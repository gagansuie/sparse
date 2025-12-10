#!/usr/bin/env python3
"""
QAT-Lite: Quick Quantization-Aware Training

Instead of full QAT (days of training), we do lightweight fine-tuning:
1. Quantize weights with straight-through estimator (STE)
2. Fine-tune for just 100-500 steps on calibration data
3. Learn better scales through gradient descent

This should recover most of the quality lost from aggressive quantization.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_perplexity(model, tokenizer, texts, max_length=512):
    """Compute perplexity on text samples."""
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


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator for quantization."""
    
    @staticmethod
    def forward(ctx, x, scale, zero, bits):
        max_val = 2 ** bits - 1
        x_q = ((x - zero) / scale).round().clamp(0, max_val)
        x_deq = x_q * scale + zero
        return x_deq
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output, None, None, None


def ste_quantize(x, scale, zero, bits):
    return STEQuantize.apply(x, scale, zero, bits)


class QuantizedLinear(nn.Module):
    """Linear layer with learnable quantization parameters."""
    
    def __init__(self, linear, bits=3, group_size=32):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Copy original weight
        weight = linear.weight.data.float()
        
        # Initialize quantization parameters per group
        # Weight shape: [out, in]
        ncol = weight.shape[1]
        num_groups = (ncol + group_size - 1) // group_size
        
        # Learnable scales and zeros
        scales = []
        zeros = []
        
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, ncol)
            w_group = weight[:, start:end]
            
            w_min = w_group.min().item()
            w_max = w_group.max().item()
            
            max_val = 2 ** bits - 1
            scale = (w_max - w_min) / max_val if abs(w_max - w_min) > 1e-8 else 1.0
            
            scales.append(scale)
            zeros.append(w_min)
        
        self.register_buffer('weight_fp', weight)
        self.scales = nn.Parameter(torch.tensor(scales))
        self.zeros = nn.Parameter(torch.tensor(zeros))
        
        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Quantize weight using STE
        weight = self.weight_fp.clone()
        ncol = weight.shape[1]
        
        for g in range(len(self.scales)):
            start = g * self.group_size
            end = min(start + self.group_size, ncol)
            
            scale = self.scales[g].abs().clamp(min=1e-8)
            zero = self.zeros[g]
            
            weight[:, start:end] = ste_quantize(
                weight[:, start:end], scale, zero, self.bits
            )
        
        return F.linear(x, weight, self.bias)


def replace_linear_with_quantized(model, bits=3, group_size=32, target_modules=None):
    """Replace Linear layers with QuantizedLinear."""
    from transformers.pytorch_utils import Conv1D
    
    for name, module in model.named_modules():
        if target_modules and not any(t in name for t in target_modules):
            continue
        
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            quant_linear = QuantizedLinear(module, bits=bits, group_size=group_size)
            setattr(parent, child_name, quant_linear)
            print(f"  Quantized: {name}")
        
        elif isinstance(module, Conv1D):
            # Conv1D has weight [in, out], convert to Linear style
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            # Create equivalent Linear
            linear = nn.Linear(module.weight.shape[0], module.weight.shape[1], bias=True)
            linear.weight.data = module.weight.data.T.contiguous()
            linear.bias.data = module.bias.data.clone()
            
            quant_linear = QuantizedLinear(linear, bits=bits, group_size=group_size)
            # Store original shape for proper forward
            quant_linear.is_conv1d = True
            setattr(parent, child_name, quant_linear)
            print(f"  Quantized (Conv1D): {name}")


def qat_fine_tune(model, tokenizer, texts, num_steps=100, lr=1e-4):
    """Quick fine-tuning with quantized weights."""
    model.train()
    device = next(model.parameters()).device
    
    # Only optimize quantization parameters (scales, zeros)
    quant_params = []
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            quant_params.extend([module.scales, module.zeros])
    
    if not quant_params:
        print("No quantization parameters found!")
        return
    
    optimizer = AdamW(quant_params, lr=lr)
    
    print(f"\nQAT Fine-tuning ({num_steps} steps, {len(quant_params)} param groups)...")
    
    # Prepare data
    all_text = " ".join(texts[:32])
    enc = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")
    
    print("  Fine-tuning complete!")


def main():
    print("=" * 70)
    print("QAT-Lite: Quick Quantization-Aware Training")
    print("Target: 10x compression with <1% PPL via lightweight fine-tuning")
    print("=" * 70)
    
    model_name = "gpt2"
    bits = 3
    group_size = 32
    
    print(f"\nConfig: {bits}-bit, group_size={group_size}")
    
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Loading WikiText-2...")
    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts_train = [x["text"] for x in ds_train if x["text"].strip()][:128]
    
    ds_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts_test = [x["text"] for x in ds_test if x["text"].strip()][:80]
    
    # Baseline
    print("\nComputing baseline PPL...")
    baseline_ppl = compute_perplexity(model, tokenizer, texts_test)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Quantize only MLP layers (attention is too sensitive)
    print("\nQuantizing MLP layers...")
    replace_linear_with_quantized(
        model, 
        bits=bits, 
        group_size=group_size,
        target_modules=['mlp.c_fc', 'mlp.c_proj']
    )
    
    # Evaluate before fine-tuning
    print("\nPPL before QAT...")
    pre_qat_ppl = compute_perplexity(model, tokenizer, texts_test)
    pre_qat_delta = (pre_qat_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"Pre-QAT PPL: {pre_qat_ppl:.4f} (Δ {pre_qat_delta:+.2f}%)")
    
    # QAT Fine-tuning
    qat_fine_tune(model, tokenizer, texts_train, num_steps=200, lr=5e-4)
    
    # Evaluate after fine-tuning
    print("\nPPL after QAT...")
    post_qat_ppl = compute_perplexity(model, tokenizer, texts_test)
    post_qat_delta = (post_qat_ppl - baseline_ppl) / baseline_ppl * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline PPL:    {baseline_ppl:.4f}")
    print(f"Pre-QAT PPL:     {pre_qat_ppl:.4f} (Δ {pre_qat_delta:+.2f}%)")
    print(f"Post-QAT PPL:    {post_qat_ppl:.4f} (Δ {post_qat_delta:+.2f}%)")
    print(f"Improvement:     {pre_qat_delta - post_qat_delta:+.2f} percentage points")
    
    status = "🎯 SUCCESS!" if post_qat_delta < 1.0 else "✓ Good" if post_qat_delta < 5.0 else "~ OK" if post_qat_delta < 10.0 else "✗ Needs work"
    print(f"\nStatus: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
