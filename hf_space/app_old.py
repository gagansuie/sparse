#!/usr/bin/env python3
"""
TenPak 7B Model Compression Evaluation - HuggingFace Space

Evaluates calibrated compression on 7B models with GPU.
"""

import gradio as gr
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
import json
import os
import gc
import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTQConfig
from datasets import load_dataset


def get_compression_config(sensitivity: float) -> Dict:
    """
    INT2/INT3/INT4 mixed precision based on sensitivity.
    Target: 10x overall compression with <1% PPL.
    
    Math: 10x = 3.2 bits/weight average
    - Top 20%: 8 bits (INT4 g=8 + residual) 
    - Next 30%: 4 bits (INT4 g=32)
    - Next 30%: 3 bits (INT3 g=16)
    - Bottom 20%: 2 bits (INT2 g=8)
    Weighted avg: 0.2*8 + 0.3*4 + 0.3*3 + 0.2*2 = 4.1 bits = ~7.8x
    
    Push harder for 10x:
    - Top 10%: 6 bits
    - Next 20%: 4 bits  
    - Next 40%: 2.5 bits
    - Bottom 30%: 2 bits
    Weighted avg: 0.1*6 + 0.2*4 + 0.4*2.5 + 0.3*2 = 2.8 bits = ~11.4x
    """
    if sensitivity > 0.9:
        # Critical 10%: INT4 g=8 + INT2 residual (~6 bits, ~5.3x)
        return {"group_size": 8, "use_residual": True, "residual_bits": 2, "use_additive": False, "quant_bits": 4}
    elif sensitivity > 0.7:
        # Important 20%: INT4 g=16 (~4.5 bits, ~7x)
        return {"group_size": 16, "use_residual": False, "residual_bits": 0, "use_additive": False, "quant_bits": 4}
    elif sensitivity > 0.3:
        # Medium 40%: INT3 g=8 (~3.5 bits, ~9x)
        return {"group_size": 8, "use_residual": False, "residual_bits": 0, "use_additive": False, "quant_bits": 3}
    else:
        # Low 30%: INT2 g=8 (~2.5 bits, ~13x)
        return {"group_size": 8, "use_residual": False, "residual_bits": 0, "use_additive": False, "quant_bits": 2}


def compute_calibrated_sensitivity(model, tokenizer, device, num_samples=8) -> Dict[str, float]:
    """Compute sensitivity using actual activation scales (AWQ-style)."""
    print("Computing calibrated sensitivity with activation scales...")
    
    # Get calibration data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [item["text"] for item in dataset if len(item["text"]) > 200][:num_samples]
    
    # Collect activation scales per layer
    activation_scales = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                x = input[0]
                if x is not None and hasattr(x, 'abs'):
                    scale = x.abs().mean().item()
                    if name not in activation_scales:
                        activation_scales[name] = []
                    activation_scales[name].append(scale)
        return hook
    
    # Register hooks on linear layers
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if len(module.weight.shape) == 2 and 'embed' not in name.lower():
                hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            try:
                model(input_ids)
            except:
                pass
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Compute sensitivity from activation scales
    # Higher activation = more important = higher sensitivity
    sensitivity = {}
    for name, scales in activation_scales.items():
        sensitivity[name] = np.mean(scales) if scales else 0.0
    
    # Also factor in weight magnitude (AWQ insight)
    for name, module in model.named_modules():
        if name in sensitivity and hasattr(module, 'weight'):
            weight_scale = module.weight.data.abs().mean().item()
            # Combine: sensitivity = activation_scale * weight_scale^0.5
            sensitivity[name] = sensitivity[name] * (weight_scale ** 0.5)
    
    # Normalize to 0-1
    if sensitivity:
        max_s, min_s = max(sensitivity.values()), min(sensitivity.values())
        range_s = max_s - min_s + 1e-8
        for name in sensitivity:
            sensitivity[name] = (sensitivity[name] - min_s) / range_s
    
    print(f"Calibrated {len(sensitivity)} layers")
    return sensitivity


def quantize_weight_additive(weight: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, float]:
    """
    Additive codebook quantization (AQLM-lite).
    Uses 2 small codebooks that sum together for better representation.
    Target: 2-3 bits per weight for 10x+ compression.
    """
    original_size = weight.numel() * 4  # FP32 bytes
    vec_dim = config.get("vec_dim", 8)  # Vector dimension
    num_codebooks = config.get("num_codebooks", 2)  # Additive codebooks
    codebook_bits = config.get("codebook_bits", 4)  # 16 entries per codebook
    codebook_size = 1 << codebook_bits
    
    # Reshape to vectors
    flat = weight.flatten().float()
    n = len(flat)
    pad_size = (vec_dim - (n % vec_dim)) % vec_dim
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
    
    vectors = flat.view(-1, vec_dim)  # [num_vectors, vec_dim]
    num_vectors = vectors.shape[0]
    
    # Initialize codebooks with k-means++ style
    codebooks = []
    residual = vectors.clone()
    
    for cb_idx in range(num_codebooks):
        # Fast k-means (3 iterations for speed)
        # Initialize with random samples
        indices = torch.randperm(num_vectors)[:codebook_size]
        centroids = residual[indices].clone()
        
        for _ in range(3):  # Fast k-means iterations
            # Assign vectors to nearest centroid
            dists = torch.cdist(residual, centroids)  # [num_vectors, codebook_size]
            assignments = dists.argmin(dim=1)
            
            # Update centroids
            for k in range(codebook_size):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = residual[mask].mean(dim=0)
        
        # Final assignment
        dists = torch.cdist(residual, centroids)
        assignments = dists.argmin(dim=1)
        
        # Store codebook and update residual
        codebooks.append((centroids, assignments))
        reconstructed = centroids[assignments]
        residual = residual - reconstructed
    
    # Reconstruct from all codebooks
    final_reconstructed = torch.zeros_like(vectors)
    for centroids, assignments in codebooks:
        final_reconstructed += centroids[assignments]
    
    # Compressed size calculation:
    # - Each vector needs num_codebooks * codebook_bits bits for indices
    # - Plus codebook storage: num_codebooks * codebook_size * vec_dim * 2 bytes (FP16)
    indices_size = num_vectors * num_codebooks * codebook_bits / 8
    codebook_storage = num_codebooks * codebook_size * vec_dim * 2
    compressed_size = indices_size + codebook_storage
    
    final = final_reconstructed.flatten()[:n].view(weight.shape)
    compression = original_size / compressed_size
    
    return final, compression


def quantize_weight(weight: torch.Tensor, config: Dict) -> Tuple[torch.Tensor, float]:
    """Quantize a weight tensor with variable bit width (INT2/INT3/INT4)."""
    if config.get("use_additive", False):
        return quantize_weight_additive(weight, config)
    
    original_size = weight.numel() * 4  # FP32 bytes
    group_size = config["group_size"]
    use_residual = config.get("use_residual", False)
    residual_bits = config.get("residual_bits", 0)
    quant_bits = config.get("quant_bits", 4)  # Support INT2, INT3, INT4
    max_val = (1 << quant_bits) - 1  # 3 for INT2, 7 for INT3, 15 for INT4
    
    flat = weight.flatten().float()
    n = len(flat)
    pad_size = (group_size - (n % group_size)) % group_size
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, device=flat.device)])
    
    groups = flat.view(-1, group_size)
    mins = groups.min(dim=1, keepdim=True).values
    maxs = groups.max(dim=1, keepdim=True).values
    scales = (maxs - mins) / max_val
    scales = scales.clamp(min=1e-8)
    
    quantized = ((groups - mins) / scales).round().clamp(0, max_val)
    dequantized = quantized * scales + mins
    
    # Compressed size: quant_bits per weight + scales/mins per group
    compressed_size = (n * quant_bits / 8) + (n / group_size) * 4
    
    if use_residual and residual_bits > 0:
        residual = groups - dequantized
        r_mins = residual.min(dim=1, keepdim=True).values
        r_maxs = residual.max(dim=1, keepdim=True).values
        r_max_val = (1 << residual_bits) - 1
        r_scales = (r_maxs - r_mins) / r_max_val
        r_scales = r_scales.clamp(min=1e-8)
        
        r_quantized = ((residual - r_mins) / r_scales).round().clamp(0, r_max_val)
        r_dequantized = r_quantized * r_scales + r_mins
        dequantized = dequantized + r_dequantized
        compressed_size += (n * residual_bits / 8) + (n / group_size) * 4
    
    final = dequantized.flatten()[:n].view(weight.shape)
    return final, original_size / compressed_size


def compute_ppl(model, tokenizer, texts: List[str], device: str) -> float:
    """Compute perplexity with memory-efficient batching."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    num_texts = len(texts)
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 5 == 0:
                print(f"[DEBUG] PPL progress: {i}/{num_texts}", flush=True)
            
            try:
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                input_ids = tokens["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    continue
                
                outputs = model(input_ids, labels=input_ids)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item() * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]
                
                del outputs, input_ids, tokens
                
            except Exception as e:
                print(f"[DEBUG] Error at sample {i}: {e}", flush=True)
                continue
            
            # Aggressive memory cleanup every 5 samples
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
    
    print(f"[DEBUG] PPL computation done: {total_tokens} tokens processed", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.exp(total_loss / max(total_tokens, 1))


def compute_fp32_param_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total


def compute_model_bytes(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def load_wikitext_texts(split: str, min_chars: int, max_samples: int) -> List[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [item["text"] for item in dataset if len(item["text"]) > min_chars][:max_samples]


def evaluate_model_gptq(
    model_name: str,
    eval_samples: int,
    gptq_bits: int,
    gptq_group_size: int,
    gptq_calib_samples: int,
    progress=gr.Progress(),
):
    gptq_bits = int(gptq_bits)
    gptq_group_size = int(gptq_group_size)
    gptq_calib_samples = int(gptq_calib_samples)
    eval_samples = int(eval_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    progress(0.05, desc="Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    progress(0.10, desc="Loading evaluation data...")
    eval_texts = load_wikitext_texts("test", min_chars=100, max_samples=eval_samples)

    progress(0.20, desc="Loading baseline model (FP16) for baseline PPL...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    baseline_model.eval()

    original_size = compute_fp32_param_bytes(baseline_model)

    progress(0.30, desc="Computing baseline PPL...")
    baseline_ppl = compute_ppl(baseline_model, tokenizer, eval_texts, device)

    del baseline_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    progress(0.40, desc="Preparing calibration data...")
    calib_texts = load_wikitext_texts("train", min_chars=200, max_samples=gptq_calib_samples)
    
    # Pre-tokenize calibration data to avoid RoPE dimension mismatch
    # The optimum quantizer needs properly formatted inputs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    calib_dataset = []
    max_length = min(2048, getattr(tokenizer, 'model_max_length', 2048))
    for text in calib_texts:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Keep 2D shape (1, seq_len) - optimum expects batch dimension
        calib_dataset.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        })

    quant_dir = os.path.join(
        "/tmp",
        "gptq_cache",
        model_name.replace("/", "__") + f"__gptq_{gptq_bits}bit_{gptq_group_size}g",
    )

    if os.path.isdir(quant_dir) and os.path.exists(os.path.join(quant_dir, "config.json")):
        progress(0.50, desc="Loading cached GPTQ model...")
        model = AutoModelForCausalLM.from_pretrained(
            quant_dir,
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    else:
        progress(0.50, desc=f"Quantizing with GPTQ ({gptq_bits}-bit, g={gptq_group_size})...")
        gptq_config = GPTQConfig(
            bits=gptq_bits,
            group_size=gptq_group_size,
            dataset=calib_dataset,
            tokenizer=tokenizer,
            use_cuda_fp16=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=gptq_config,
            low_cpu_mem_usage=True,
        )
        try:
            os.makedirs(quant_dir, exist_ok=True)
            model.save_pretrained(quant_dir)
            tokenizer.save_pretrained(quant_dir)
        except Exception as e:
            print(f"Warning: failed to cache quantized model: {e}")

    print("[DEBUG] GPTQ quantization complete, setting model to eval mode...")
    model.eval()
    print("[DEBUG] Computing compressed size...")
    compressed_size = compute_model_bytes(model)
    overall_compression = original_size / max(compressed_size, 1)
    print(f"[DEBUG] Compression: {overall_compression:.2f}x")

    progress(0.85, desc="Computing quantized PPL...")
    print(f"[DEBUG] Starting PPL computation on {len(eval_texts)} samples...")
    quantized_ppl = compute_ppl(model, tokenizer, eval_texts, device)
    print(f"[DEBUG] Quantized PPL: {quantized_ppl:.4f}")

    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    status = "✅ PASS" if abs(ppl_delta) < 1.0 else "❌ FAIL"

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    result = f"""
## Results for {model_name} (GPTQ)

| Metric | Value |
|--------|-------|
| **Compression** | **{overall_compression:.2f}x** |
| Baseline PPL | {baseline_ppl:.4f} |
| Quantized PPL | {quantized_ppl:.4f} |
| **PPL Delta** | **{ppl_delta:+.2f}%** |
| **Status** | **{status}** |

### Details
- Original size (FP32 params): {original_size / 1e9:.2f} GB
- Compressed size (quantized params+buffers): {compressed_size / 1e9:.2f} GB
- Bits: {gptq_bits}
- Group size: {gptq_group_size}
- Calibration samples: {gptq_calib_samples}
- Device: {device}
"""
    progress(1.0, desc="Done!")
    return result


def _evaluate_model_inner(model_name: str, max_samples: int, method: str, gptq_bits: int, gptq_group_size: int, gptq_calib_samples: int, progress=gr.Progress()):
    """Inner evaluation function."""
    print(f"[DEBUG] Starting evaluation: {model_name}, method={method}", flush=True)
    
    if method == "GPTQ (gptqmodel)":
        return evaluate_model_gptq(
            model_name=model_name,
            eval_samples=max_samples,
            gptq_bits=gptq_bits,
            gptq_group_size=gptq_group_size,
            gptq_calib_samples=gptq_calib_samples,
            progress=progress,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    progress(0.1, desc="Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Force GPU-only loading - fail fast if not enough memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Model loaded successfully on {device}")
    
    progress(0.2, desc="Computing calibrated sensitivity (AWQ-style)...")
    sensitivity = compute_calibrated_sensitivity(model, tokenizer, device, num_samples=8)
    
    progress(0.3, desc="Loading evaluation data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [item["text"] for item in dataset if len(item["text"]) > 100][:max_samples]
    
    progress(0.4, desc="Computing baseline PPL...")
    baseline_ppl = compute_ppl(model, tokenizer, texts, device)
    
    # Get layers
    layers = [(n, m) for n, m in model.named_modules() 
              if hasattr(m, 'weight') and m.weight is not None 
              and len(m.weight.shape) == 2 and 'embed' not in n.lower()]
    
    progress(0.5, desc=f"Quantizing {len(layers)} layers...")
    total_original, total_compressed = 0, 0
    
    for i, (name, module) in enumerate(layers):
        weight = module.weight.data.float()
        original_size = weight.numel() * 4
        total_original += original_size
        
        config = get_compression_config(sensitivity.get(name, 0.5))
        quantized, compression = quantize_weight(weight, config)
        module.weight.data = quantized.half().to(module.weight.device)
        
        total_compressed += original_size / compression
        progress(0.5 + 0.3 * (i / len(layers)), desc=f"Quantizing layer {i+1}/{len(layers)}")
        
        del weight, quantized
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    overall_compression = total_original / total_compressed
    
    progress(0.85, desc="Computing quantized PPL...")
    quantized_ppl = compute_ppl(model, tokenizer, texts, device)
    
    ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
    status = "✅ PASS" if abs(ppl_delta) < 1.0 else "❌ FAIL"
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    result = f"""
## Results for {model_name}

| Metric | Value |
|--------|-------|
| **Compression** | **{overall_compression:.2f}x** |
| Baseline PPL | {baseline_ppl:.4f} |
| Quantized PPL | {quantized_ppl:.4f} |
| **PPL Delta** | **{ppl_delta:+.2f}%** |
| **Status** | **{status}** |

### Details
- Original size: {total_original / 1e9:.2f} GB
- Compressed size: {total_compressed / 1e9:.2f} GB
- Layers quantized: {len(layers)}
- Device: {device}
"""
    
    progress(1.0, desc="Done!")
    return result


def evaluate_model(model_name: str, max_samples: int, method: str, gptq_bits: int, gptq_group_size: int, gptq_calib_samples: int, progress=gr.Progress()):
    """Main evaluation function with error handling."""
    try:
        return _evaluate_model_inner(model_name, max_samples, method, gptq_bits, gptq_group_size, gptq_calib_samples, progress)
    except Exception as e:
        import traceback
        error_msg = f"## Error\n\n```\n{traceback.format_exc()}\n```"
        print(f"[ERROR] {e}", flush=True)
        return error_msg

# Note: @spaces.GPU decorator removed - causing silent failures
# ZeroGPU will auto-allocate GPU when CUDA is used


# Gradio interface
with gr.Blocks(title="TenPak 7B Compression Evaluation") as demo:
    gr.Markdown("""
    # TenPak 7B Model Compression Evaluation
    
    Evaluate calibrated compression on 7B models. Target: **8-10x compression with <1% PPL delta**.
    
    Based on scaling analysis:
    - GPT-2 (124M): 4.82x, +0.68% PPL
    - TinyLlama (1.1B): 4.58x, +0.15% PPL  
    - GPT-2 XL (1.5B): 6.03x, **-0.21% PPL**
    
    **Hypothesis**: 7B models should achieve 8-10x compression with <1% PPL.
    """)
    
    with gr.Row():
        model_input = gr.Dropdown(
            choices=[
                "meta-llama/Llama-2-7b-hf",  # 7B - needs A10G, requires license
                "mistralai/Mistral-7B-v0.1",  # 7B - open, needs A10G
                "Qwen/Qwen2-7B",  # 7B - open, needs A10G
                "microsoft/phi-2",  # 2.7B - fits on T4
                "Qwen/Qwen2-1.5B",  # 1.5B - fits on T4
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B
            ],
            value="mistralai/Mistral-7B-v0.1",
            label="Model (7B models need A10G GPU - $1/hour)"
        )
        samples_input = gr.Slider(10, 50, value=30, step=5, label="Evaluation Samples")

    with gr.Row():
        method_input = gr.Dropdown(
            choices=["GPTQ (gptqmodel)", "TenPak (naive)"],
            value="GPTQ (gptqmodel)",
            label="Compression Method",
        )
        gptq_bits_input = gr.Dropdown(
            choices=[3, 4],
            value=3,
            label="GPTQ Bits",
        )
        gptq_group_input = gr.Dropdown(
            choices=[64, 128, 256],
            value=128,
            label="GPTQ Group Size",
        )
        gptq_calib_input = gr.Slider(16, 256, value=64, step=16, label="GPTQ Calibration Samples")
    
    run_btn = gr.Button("Run Evaluation", variant="primary")
    output = gr.Markdown()
    
    run_btn.click(evaluate_model, inputs=[model_input, samples_input, method_input, gptq_bits_input, gptq_group_input, gptq_calib_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
