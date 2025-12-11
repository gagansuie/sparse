#!/usr/bin/env python3
"""
Knowledge Distillation for Post-Quantization Recovery.

After aggressive quantization, use the original (teacher) model to guide
the quantized (student) model to recover lost quality.

Techniques:
1. Output distillation: Match logits between teacher and student
2. Layer-wise distillation: Match intermediate hidden states
3. Attention transfer: Match attention patterns
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
from typing import Dict, List, Optional
import copy


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


def quantize_weight(weight, group_size=128):
    """Simple INT4 quantization for aggressive compression."""
    flat = weight.flatten().float()
    n = flat.numel()
    pad = (group_size - n % group_size) % group_size
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=weight.device)])
    
    groups = flat.view(-1, group_size)
    g_min = groups.min(1).values
    g_max = groups.max(1).values
    
    scale = (g_max - g_min) / 15.0
    scale = torch.where(scale.abs() < 1e-8, torch.ones_like(scale), scale)
    
    q = ((groups - g_min.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    return deq.flatten()[:n].view(weight.shape).to(weight.dtype)


def quantize_model(model, group_size=128):
    """Quantize all MLP weights."""
    for layer in model.model.layers:
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            if hasattr(layer.mlp, proj):
                w = getattr(layer.mlp, proj).weight.data
                getattr(layer.mlp, proj).weight.data = quantize_weight(w, group_size)


class DistillationTrainer:
    """Train quantized student to match teacher outputs."""
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        tokenizer,
        temperature: float = 2.0,
        alpha: float = 0.5,  # Weight for distillation loss vs task loss
    ):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels=None):
        """
        Compute distillation loss.
        
        KL divergence between softened student and teacher distributions,
        plus optional cross-entropy loss with true labels.
        """
        T = self.temperature
        
        # Soft targets from teacher
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        
        # KL divergence (scaled by T^2 as per Hinton et al.)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        if labels is not None:
            # Hard target cross-entropy
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return kl_loss
    
    def train_step(self, input_ids, learning_rate=1e-5):
        """Single training step."""
        device = next(self.student.parameters()).device
        input_ids = input_ids.to(device)
        
        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits
        
        # Get student outputs
        student_outputs = self.student(input_ids)
        student_logits = student_outputs.logits
        
        # Compute loss
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        # Backward pass (only update MLP weights)
        loss.backward()
        
        # Manual SGD update for MLP weights only
        with torch.no_grad():
            for layer in self.student.model.layers:
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(layer.mlp, proj):
                        param = getattr(layer.mlp, proj).weight
                        if param.grad is not None:
                            param.data -= learning_rate * param.grad
                            param.grad.zero_()
        
        return loss.item()
    
    def train(self, texts, num_epochs=1, batch_size=1, learning_rate=1e-5, max_length=256):
        """Train student with distillation."""
        device = next(self.student.parameters()).device
        
        total_loss = 0
        num_steps = 0
        
        for epoch in range(num_epochs):
            for i, text in enumerate(texts):
                if not text.strip():
                    continue
                
                enc = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                )
                input_ids = enc["input_ids"].to(device)
                
                if input_ids.numel() < 2:
                    continue
                
                loss = self.train_step(input_ids, learning_rate)
                total_loss += loss
                num_steps += 1
                
                if (i + 1) % 50 == 0:
                    avg_loss = total_loss / num_steps if num_steps > 0 else 0
                    print(f"    Step {i+1}, Avg Loss: {avg_loss:.4f}")
        
        return total_loss / num_steps if num_steps > 0 else 0


class LayerWiseDistillation:
    """Match intermediate hidden states between teacher and student."""
    
    def __init__(self, teacher, student, tokenizer):
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        
        self.teacher_hiddens = {}
        self.student_hiddens = {}
        self.hooks = []
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def _make_hook(self, storage, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[name] = output[0].detach()
            else:
                storage[name] = output.detach()
        return hook
    
    def register_hooks(self):
        """Register hooks to capture hidden states."""
        for i, layer in enumerate(self.teacher.model.layers):
            hook = layer.register_forward_hook(
                self._make_hook(self.teacher_hiddens, f"layer_{i}")
            )
            self.hooks.append(hook)
        
        for i, layer in enumerate(self.student.model.layers):
            hook = layer.register_forward_hook(
                self._make_hook(self.student_hiddens, f"layer_{i}")
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_layer_loss(self):
        """Compute MSE loss between teacher and student hidden states."""
        total_loss = 0
        for name in self.teacher_hiddens:
            if name in self.student_hiddens:
                t_hidden = self.teacher_hiddens[name]
                s_hidden = self.student_hiddens[name]
                total_loss += F.mse_loss(s_hidden, t_hidden)
        return total_loss
    
    def train_step(self, input_ids, learning_rate=1e-5):
        """Single layer-wise distillation step."""
        device = next(self.student.parameters()).device
        input_ids = input_ids.to(device)
        
        # Clear storage
        self.teacher_hiddens.clear()
        self.student_hiddens.clear()
        
        # Forward passes
        with torch.no_grad():
            _ = self.teacher(input_ids)
        
        _ = self.student(input_ids)
        
        # Compute loss
        loss = self.compute_layer_loss()
        
        if loss.requires_grad:
            loss.backward()
            
            # Update MLP weights
            with torch.no_grad():
                for layer in self.student.model.layers:
                    for proj in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(layer.mlp, proj):
                            param = getattr(layer.mlp, proj).weight
                            if param.grad is not None:
                                param.data -= learning_rate * param.grad
                                param.grad.zero_()
        
        return loss.item() if torch.is_tensor(loss) else loss


def test_distillation(teacher, student, tokenizer, texts, baseline, name, 
                      trainer_class, num_samples=100, **kwargs):
    """Test a distillation configuration."""
    print(f"  {name}...", end=" ", flush=True)
    
    # Train
    if trainer_class == DistillationTrainer:
        trainer = trainer_class(teacher, student, tokenizer, **kwargs)
        trainer.train(texts[:num_samples], num_epochs=1, learning_rate=1e-5)
    elif trainer_class == LayerWiseDistillation:
        trainer = trainer_class(teacher, student, tokenizer)
        trainer.register_hooks()
        
        device = next(student.parameters()).device
        for i, text in enumerate(texts[:num_samples]):
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            trainer.train_step(enc["input_ids"].to(device))
            if (i + 1) % 50 == 0:
                print(f".", end="", flush=True)
        
        trainer.remove_hooks()
    
    # Evaluate
    ppl = compute_ppl(student, tokenizer, texts[:30])
    delta = (ppl - baseline) / baseline * 100
    status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
    print(f" PPL: {ppl:.2f} (Δ {delta:+.1f}%) {status}")
    
    return ppl, delta


def main():
    print("=" * 70)
    print("KNOWLEDGE DISTILLATION FOR POST-QUANTIZATION RECOVERY")
    print("=" * 70)
    
    # Load teacher (original model)
    print("\nLoading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x["text"].strip()]
    
    baseline = compute_ppl(teacher, tok, texts[:30])
    print(f"\nTeacher (FP16) PPL: {baseline:.2f}")
    
    # Create quantized student
    print("\nCreating quantized student (INT4 g128 = ~8x compression)...")
    student = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
    )
    
    # Save original weights for comparison
    orig_weights = {}
    for i, layer in enumerate(student.model.layers):
        orig_weights[i] = {}
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if hasattr(layer.mlp, p):
                orig_weights[i][p] = getattr(layer.mlp, p).weight.data.clone()
    
    # Aggressively quantize student
    quantize_model(student, group_size=128)
    
    quant_ppl = compute_ppl(student, tok, texts[:30])
    quant_delta = (quant_ppl - baseline) / baseline * 100
    print(f"Quantized student PPL: {quant_ppl:.2f} (Δ {quant_delta:+.1f}%)")
    
    # Test distillation configurations
    print("\n" + "-" * 70)
    print("TESTING DISTILLATION METHODS")
    print("-" * 70)
    
    results = []
    
    # Test 1: Output distillation with different temperatures
    print("\n1. Output Distillation:")
    
    for temp in [1.0, 2.0, 4.0]:
        # Reset student
        for i, layer in enumerate(student.model.layers):
            for p in ["gate_proj", "up_proj", "down_proj"]:
                if p in orig_weights[i]:
                    getattr(layer.mlp, p).weight.data = orig_weights[i][p].clone()
        quantize_model(student, group_size=128)
        
        ppl, delta = test_distillation(
            teacher, student, tok, texts, baseline,
            f"T={temp}", DistillationTrainer,
            temperature=temp, alpha=0.7
        )
        results.append((f"Output T={temp}", "~8x", ppl, delta))
        gc.collect()
    
    # Test 2: Layer-wise distillation
    print("\n2. Layer-wise Distillation:")
    
    # Reset student
    for i, layer in enumerate(student.model.layers):
        for p in ["gate_proj", "up_proj", "down_proj"]:
            if p in orig_weights[i]:
                getattr(layer.mlp, p).weight.data = orig_weights[i][p].clone()
    quantize_model(student, group_size=128)
    
    ppl, delta = test_distillation(
        teacher, student, tok, texts, baseline,
        "Layer-wise MSE", LayerWiseDistillation,
        num_samples=100
    )
    results.append(("Layer-wise MSE", "~8x", ppl, delta))
    
    # Summary
    print("\n" + "=" * 70)
    print("DISTILLATION RESULTS")
    print("=" * 70)
    print(f"\nTeacher PPL: {baseline:.2f}")
    print(f"Quantized (no distill) PPL: {quant_ppl:.2f} (Δ {quant_delta:+.1f}%)\n")
    
    print(f"{'Method':<20} {'Comp':>6} {'PPL':>8} {'Δ':>7} {'Recovery':>10}")
    print("-" * 55)
    
    for name, comp, ppl, delta in sorted(results, key=lambda x: x[3]):
        recovery = ((quant_delta - delta) / quant_delta * 100) if quant_delta != 0 else 0
        status = "✅" if delta < 1 else "⚠️" if delta < 2 else "❌"
        print(f"{name:<20} {comp:>6} {ppl:>8.2f} {delta:>+6.1f}% {recovery:>+8.1f}% {status}")


if __name__ == "__main__":
    main()
