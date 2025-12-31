"""
Sparse Calibration - Collect statistics for compression

Collects Fisher information, activation scales, and Hessian diagonal
for importance-aware quantization (AWQ/GPTQ style).
"""

import itertools

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm import tqdm


def collect_calibration_stats(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    num_samples: int = 64,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Collect calibration data for compression.
    
    Collects:
    - Fisher information (gradient-based importance)
    - Activation scales (AWQ-style input magnitudes)
    - Hessian diagonal (for GPTQ-style error compensation)
    - Input samples (for on-demand full Hessian computation)
    
    Args:
        model: The model to calibrate
        tokenizer: Tokenizer for the model
        texts: Calibration texts
        num_samples: Number of samples to use
        device: Device to run on
        
    Returns:
        (fisher_scores, activation_scales, hessian_diags, input_samples)
    """
    print(f"[CALIBRATE] Collecting stats from {num_samples} samples...", flush=True)
    
    model.eval()
    fisher_accum = {}
    activation_accum = {}
    hessian_accum = {}
    input_samples = {}
    nsamples = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                x = inp[0].detach()
                if x.dim() >= 2:
                    x_flat = x.view(-1, x.shape[-1])
                    
                    # AWQ-style activation scale
                    act_scale = x_flat.abs().mean(dim=0)
                    if name not in activation_accum:
                        activation_accum[name] = act_scale.cpu()
                    else:
                        activation_accum[name] = activation_accum[name] + act_scale.cpu()
                    
                    # Hessian diagonal
                    h_diag = (x_flat ** 2).sum(dim=0)
                    if name not in hessian_accum:
                        hessian_accum[name] = h_diag.cpu()
                        nsamples[name] = x_flat.shape[0]
                    else:
                        hessian_accum[name] = hessian_accum[name] + h_diag.cpu()
                        nsamples[name] += x_flat.shape[0]
                    
                    # Store subset of input samples for on-demand Hessian
                    if name not in input_samples:
                        input_samples[name] = [x_flat[:32].cpu()]
                    elif len(input_samples[name]) < 8:
                        input_samples[name].append(x_flat[:32].cpu())
        return hook
    
    # Register hooks on Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D":
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
    
    # Collect Fisher scores (gradient-based)
    model.train()
    for i, text in enumerate(tqdm(texts[:num_samples], desc="Calibrating")):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        if tokens.input_ids.shape[1] < 2:
            continue
        
        try:
            outputs = model(**tokens, labels=tokens.input_ids)
            loss = outputs.loss
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_sq = param.grad.pow(2).mean().item()
                    fisher_accum[name] = fisher_accum.get(name, 0) + grad_sq
            
            model.zero_grad()
        except Exception as e:
            print(f"[CALIBRATE] Error at sample {i}: {e}")
            continue
        
        if i % 20 == 0:
            torch.cuda.empty_cache() if device == 'cuda' else None
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    model.eval()
    
    # Normalize Fisher scores
    if fisher_accum:
        max_score = max(fisher_accum.values()) + 1e-10
        fisher_accum = {k: v / max_score for k, v in fisher_accum.items()}
    
    # Normalize activation scales
    for name in activation_accum:
        activation_accum[name] = (activation_accum[name] / num_samples).sqrt().clamp(min=1e-5)
    
    # Normalize Hessian diagonal
    for name in hessian_accum:
        if name in nsamples and nsamples[name] > 0:
            hessian_accum[name] = (hessian_accum[name] / nsamples[name]).clamp(min=1e-8)
    
    # Concatenate input samples
    for name in input_samples:
        input_samples[name] = torch.cat(input_samples[name], dim=0)
    
    print(f"[CALIBRATE] Collected stats for {len(fisher_accum)} layers", flush=True)
    return fisher_accum, activation_accum, hessian_accum, input_samples


def compute_ppl(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_samples: int = 100,
    streaming: bool = False,
    max_length: int = 512,
    stride: int = 512,
) -> float:
    """Compute perplexity on evaluation texts.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        texts: Evaluation texts
        device: Device to run on
        max_samples: Maximum samples to evaluate
        
    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        if streaming:
            max_model_len = getattr(model.config, "max_position_embeddings", None)
            if max_model_len is not None:
                max_length = min(max_length, int(max_model_len))
            if stride < 1:
                stride = max_length
            if stride > max_length:
                stride = max_length

            eval_texts = [
                t for t in itertools.islice(texts, max_samples)
                if isinstance(t, str) and t.strip()
            ]
            if not eval_texts:
                return float('inf')

            text = "\n\n".join(eval_texts)
            encodings = tokenizer(text, return_tensors="pt")
            input_ids_all = encodings["input_ids"][0]
            seq_len = input_ids_all.size(0)
            if seq_len < 2:
                return float('inf')

            prev_end = 0
            for begin in tqdm(range(0, seq_len - 1, stride), desc="Evaluating PPL (streaming)"):
                end = min(begin + max_length, seq_len)
                trg_len = end - prev_end
                if trg_len <= 0:
                    break

                input_ids = input_ids_all[begin:end].unsqueeze(0).to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                try:
                    outputs = model(input_ids, labels=target_ids)
                    total_loss += outputs.loss.item() * trg_len
                    total_tokens += trg_len
                except Exception:
                    pass

                prev_end = end
                if end == seq_len:
                    break
        else:
            for text in tqdm(itertools.islice(texts, max_samples), total=max_samples, desc="Evaluating PPL"):
                tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                if tokens.input_ids.shape[1] < 2:
                    continue
                
                try:
                    outputs = model(**tokens, labels=tokens.input_ids)
                    total_loss += outputs.loss.item() * tokens.input_ids.shape[1]
                    total_tokens += tokens.input_ids.shape[1]
                except Exception:
                    continue
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    
    return ppl
