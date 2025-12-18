#!/usr/bin/env python3
"""
TenPak 7B Model Compression Evaluation - HuggingFace Space

Pure Python implementation of TenPak codecs for fast evaluation.
No JSON serialization - operates directly on PyTorch tensors.
"""

import gradio as gr
import gc
import torch
import numpy as np
from typing import Dict, List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Available codecs (Python implementations)
CODECS = {
    "int4_opt_llama": "INT4 Optimized (g=8, 5 iter) - ~4x compression",
    "int4_residual": "INT4 + INT2 Residual (g=16) - ~3.2x, best quality",
    "int4_g128": "INT4 Large Groups (g=128) - ~7.5x compression",
    "int4_g256": "INT4 Max Compression (g=256) - ~7.8x compression",
}


def quantize_int4_group(weight: torch.Tensor, group_size: int = 8, iterations: int = 5) -> Tuple[torch.Tensor, float]:
    """
    INT4 quantization with iterative scale refinement.
    
    Port of Rust compress_int4_opt_llama algorithm.
    Returns (dequantized_weight, compression_ratio).
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad to multiple of group_size
    pad_len = (group_size - (n % group_size)) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    # Compute scales and offsets with iterative refinement
    g_min = weight_groups.min(dim=1).values
    g_max = weight_groups.max(dim=1).values
    
    for _ in range(iterations):
        scale = torch.where(
            (g_max - g_min).abs() > 1e-8,
            (g_max - g_min) / 15.0,
            torch.ones_like(g_max)
        )
        inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
        
        # Quantize and compute error
        q = ((weight_groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 15)
        deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
        err = weight_groups - deq
        
        err_min = err.min(dim=1).values
        err_max = err.max(dim=1).values
        
        g_min = g_min + err_min * 0.5
        g_max = g_max + err_max * 0.5
    
    # Final quantization
    scale = torch.where(
        (g_max - g_min).abs() > 1e-8,
        (g_max - g_min) / 15.0,
        torch.ones_like(g_max)
    )
    inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
    
    q = ((weight_groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Remove padding and reshape
    deq_flat = deq.flatten()[:n]
    result = deq_flat.view(original_shape)
    
    # Compression ratio: 4 bits per weight + scales/offsets overhead
    # Packed: 0.5 bytes/weight, scales: 2 bytes per group, offsets: 2 bytes per group
    packed_size = n / 2  # 4 bits packed
    meta_size = num_groups * 4  # 2 bytes scale + 2 bytes offset per group
    compressed_size = packed_size + meta_size
    original_size = n * 4  # FP32 = 4 bytes
    compression = original_size / compressed_size
    
    return result, compression


def quantize_int4_residual(weight: torch.Tensor, group_size: int = 16) -> Tuple[torch.Tensor, float]:
    """
    INT4 + INT2 residual quantization for best quality.
    
    Port of Rust compress_int4_residual_v1 algorithm.
    """
    original_shape = weight.shape
    weight_flat = weight.flatten().float()
    n = weight_flat.numel()
    
    # Pad to multiple of group_size
    pad_len = (group_size - (n % group_size)) % group_size
    if pad_len > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_len, device=weight.device)])
    
    num_groups = weight_flat.numel() // group_size
    weight_groups = weight_flat.view(num_groups, group_size)
    
    # Primary INT4 quantization
    g_min = weight_groups.min(dim=1).values
    g_max = weight_groups.max(dim=1).values
    
    scale = torch.where(
        (g_max - g_min).abs() > 1e-8,
        (g_max - g_min) / 15.0,
        torch.ones_like(g_max)
    )
    inv_scale = torch.where(scale.abs() > 1e-8, 1.0 / scale, torch.ones_like(scale))
    
    q = ((weight_groups - g_min.unsqueeze(1)) * inv_scale.unsqueeze(1)).round().clamp(0, 15)
    deq = q * scale.unsqueeze(1) + g_min.unsqueeze(1)
    
    # Compute residual
    residual = weight_groups - deq
    
    # INT2 quantization of residual
    r_min = residual.min(dim=1).values
    r_max = residual.max(dim=1).values
    
    r_scale = torch.where(
        (r_max - r_min).abs() > 1e-8,
        (r_max - r_min) / 3.0,  # INT2 = 4 levels (0-3)
        torch.ones_like(r_max)
    )
    r_inv_scale = torch.where(r_scale.abs() > 1e-8, 1.0 / r_scale, torch.ones_like(r_scale))
    
    r_q = ((residual - r_min.unsqueeze(1)) * r_inv_scale.unsqueeze(1)).round().clamp(0, 3)
    r_deq = r_q * r_scale.unsqueeze(1) + r_min.unsqueeze(1)
    
    # Final reconstruction
    result_groups = deq + r_deq
    result_flat = result_groups.flatten()[:n]
    result = result_flat.view(original_shape)
    
    # Compression ratio: INT4 (0.5 bytes) + INT2 (0.25 bytes) + metadata
    int4_size = n / 2
    int2_size = n / 4
    meta_size = num_groups * 8  # 4 bytes for INT4 meta + 4 bytes for INT2 meta
    compressed_size = int4_size + int2_size + meta_size
    original_size = n * 4
    compression = original_size / compressed_size
    
    return result, compression


def quantize_weight(weight: torch.Tensor, codec: str) -> Tuple[torch.Tensor, float]:
    """Quantize a weight tensor using the specified codec."""
    if codec == "int4_opt_llama":
        return quantize_int4_group(weight, group_size=8, iterations=5)
    elif codec == "int4_residual":
        return quantize_int4_residual(weight, group_size=16)
    elif codec == "int4_g128":
        return quantize_int4_group(weight, group_size=128, iterations=3)
    elif codec == "int4_g256":
        return quantize_int4_group(weight, group_size=256, iterations=3)
    else:
        return quantize_int4_group(weight, group_size=8, iterations=5)


def compute_ppl(model, tokenizer, texts: List[str], device: str) -> float:
    """Compute perplexity on text samples."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"[PPL] {i}/{len(texts)}", flush=True)
            
            try:
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = tokens["input_ids"].to(device)
                if input_ids.shape[1] < 2:
                    continue
                
                outputs = model(input_ids, labels=input_ids)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item() * input_ids.shape[1]
                    total_tokens += input_ids.shape[1]
                
                del outputs, input_ids, tokens
            except Exception as e:
                print(f"[PPL] Error at {i}: {e}", flush=True)
                continue
            
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return np.exp(total_loss / max(total_tokens, 1))


def evaluate_model(model_name: str, codec: str, max_samples: int, progress=gr.Progress()):
    """Main evaluation function using pure Python quantization."""
    try:
        print(f"[START] Model: {model_name}, Codec: {codec}", flush=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {device}", flush=True)
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        
        progress(0.05, desc="Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        progress(0.1, desc="Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"[INFO] Model loaded", flush=True)
        
        progress(0.15, desc="Loading evaluation data...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100][:max_samples]
        
        progress(0.2, desc="Computing baseline PPL...")
        baseline_ppl = compute_ppl(model, tokenizer, texts, device)
        print(f"[INFO] Baseline PPL: {baseline_ppl:.4f}", flush=True)
        
        # Get layers to quantize
        layers = [(n, m) for n, m in model.named_modules() 
                  if hasattr(m, 'weight') and m.weight is not None 
                  and len(m.weight.shape) == 2 and 'embed' not in n.lower()]
        
        progress(0.3, desc=f"Quantizing {len(layers)} layers...")
        total_original = 0
        total_compressed = 0
        
        for i, (name, module) in enumerate(layers):
            weight = module.weight.data
            original_size = weight.numel() * 4  # FP32 bytes
            total_original += original_size
            
            # Quantize and get dequantized weight
            deq_weight, compression = quantize_weight(weight, codec)
            
            # Replace weight with dequantized version
            module.weight.data = deq_weight.to(weight.dtype).to(weight.device)
            
            total_compressed += original_size / compression
            
            if i % 20 == 0:
                print(f"[QUANT] {i}/{len(layers)} layers", flush=True)
                progress(0.3 + 0.4 * (i / len(layers)), desc=f"Quantizing layer {i+1}/{len(layers)}")
            
            # Memory cleanup
            del weight, deq_weight
            if i % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        overall_compression = total_original / total_compressed
        print(f"[INFO] Compression: {overall_compression:.2f}x", flush=True)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress(0.8, desc="Computing quantized PPL...")
        quantized_ppl = compute_ppl(model, tokenizer, texts, device)
        print(f"[INFO] Quantized PPL: {quantized_ppl:.4f}", flush=True)
        
        ppl_delta = (quantized_ppl - baseline_ppl) / baseline_ppl * 100
        status = "✅ PASS" if abs(ppl_delta) < 1.0 else ("⚠️ MARGINAL" if abs(ppl_delta) < 5.0 else "❌ FAIL")
        
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        result = f"""
## Results for {model_name}

| Metric | Value |
|--------|-------|
| **Codec** | **{codec}** |
| **Compression** | **{overall_compression:.2f}x** |
| Baseline PPL | {baseline_ppl:.4f} |
| Quantized PPL | {quantized_ppl:.4f} |
| **PPL Delta** | **{ppl_delta:+.2f}%** |
| **Status** | **{status}** |

### Details
- Original size (FP32): {total_original / 1e9:.2f} GB
- Compressed size: {total_compressed / 1e6:.1f} MB
- Layers quantized: {len(layers)}
- Device: {device}

### Codec Description
{CODECS.get(codec, codec)}
"""
        progress(1.0, desc="Done!")
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"## Error\n\n```\n{traceback.format_exc()}\n```"
        print(f"[ERROR] {e}", flush=True)
        return error_msg


# Gradio interface
with gr.Blocks(title="TenPak Compression Evaluation") as demo:
    gr.Markdown("""
    # TenPak Model Compression Evaluation
    
    **Pure Python implementation** - No JSON serialization, operates directly on PyTorch tensors.
    
    ### Available Codecs
    - **int4_opt_llama**: Best balance (g=8, 5 iter) - ~4x compression, <1% PPL
    - **int4_residual**: Best quality (INT4+INT2) - ~3.2x compression, negative PPL delta
    - **int4_g128**: High compression (g=128) - ~7.5x compression
    - **int4_g256**: Max compression (g=256) - ~7.8x compression
    
    Target: **<1% PPL delta** at various compression ratios.
    """)
    
    with gr.Row():
        model_input = gr.Dropdown(
            choices=[
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B - fast testing
                "Qwen/Qwen2-1.5B",  # 1.5B
                "microsoft/phi-2",  # 2.7B
                "mistralai/Mistral-7B-v0.1",  # 7B
                "Qwen/Qwen2-7B",  # 7B
            ],
            value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            label="Model"
        )
        codec_input = gr.Dropdown(
            choices=list(CODECS.keys()),
            value="int4_opt_llama",
            label="Codec"
        )
        samples_input = gr.Slider(10, 50, value=20, step=5, label="Evaluation Samples")
    
    run_btn = gr.Button("Run Evaluation", variant="primary")
    output = gr.Markdown()
    
    run_btn.click(evaluate_model, inputs=[model_input, codec_input, samples_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
