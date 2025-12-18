#!/usr/bin/env python3
"""
TenPak 7B Model Compression Evaluation - HuggingFace Space

Uses the actual Rust tenpak binary for compression evaluation.
"""

import gradio as gr
import json
import os
import gc
import subprocess
import tempfile
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Available codecs from tenpak Rust implementation
CODECS = {
    "int4_residual_v1": "INT4 + INT2 Residual (Best Quality, ~3.2x)",
    "int4_opt_llama_v1": "INT4 Optimized Llama (Good Balance, ~4x)",
    "int4_spin_v1": "SpinQuant INT4 (Experimental, ~6-8x)",
    "int4_10x_v1": "INT4 Max Compression (Experimental, ~7-8x)",
    "int4_mixed_v1": "Mixed Precision by Layer (Experimental, ~6-8x)",
    "int4_hybrid_v1": "Hybrid SpinQuant+Mixed (Experimental, ~6-7x)",
    "int4_hybrid_v2": "Hybrid v2 Conservative (Experimental, ~5-6x)",
    "caldera_v1": "CALDERA Low-Rank+Quant (Experimental, ~8-10x)",
    "aqlm_v1": "AQLM Additive Quant (Experimental, ~8-10x)",
    "pocketllm_v1": "PocketLLM Vector Quant (Experimental, ~10x+)",
    "pocketllm_v2": "PocketLLM v2 Quality (Experimental, ~6-8x)",
    "tenpak_x_v1": "TenPak-X Hybrid (Novel, ~8-10x)",
    "tenpak_x_v2": "TenPak-X v2 Max (Novel, ~8-10x)",
}

def get_tenpak_binary():
    """Find the tenpak binary."""
    # In HF Space, binary is in same directory as app.py
    script_dir = Path(__file__).parent
    binary = script_dir / "tenpak"
    if binary.exists():
        # Make sure it's executable
        binary.chmod(0o755)
        return str(binary)
    raise FileNotFoundError("tenpak binary not found")


def extract_weights_to_file(model, output_path: str) -> Tuple[int, int]:
    """Extract model weights directly to JSON file to minimize memory usage.
    
    Returns (num_tensors, original_size_bytes).
    """
    original_size = 0
    num_tensors = 0
    
    with open(output_path, 'w') as f:
        f.write('{"tensors": [')
        first = True
        
        for name, module in model.named_modules():
            if not hasattr(module, 'weight') or module.weight is None:
                continue
            if len(module.weight.shape) != 2:
                continue
            if 'embed' in name.lower():
                continue
            
            # Get weight and convert to list
            weight = module.weight.data.float().cpu()
            shape = list(weight.shape)
            data = weight.flatten().tolist()
            original_size += len(data) * 4  # FP32 = 4 bytes
            
            # Write tensor JSON
            if not first:
                f.write(',')
            first = False
            
            tensor_json = json.dumps({"name": name, "shape": shape, "data": data})
            f.write(tensor_json)
            num_tensors += 1
            
            # Free memory immediately
            del weight, data
            gc.collect()
            
            if num_tensors % 10 == 0:
                print(f"[EXTRACT] {num_tensors} tensors, {original_size / 1e9:.2f} GB", flush=True)
        
        f.write('], "activation_stats": {}}')
    
    return num_tensors, original_size


def load_weights_from_bundle(model, bundle: Dict):
    """Load weights from tenpak FloatBundle back into model."""
    tensor_map = {t["name"]: t for t in bundle["tensors"]}
    
    for name, module in model.named_modules():
        if name in tensor_map:
            t = tensor_map[name]
            weight_data = torch.tensor(t["data"], dtype=torch.float32)
            weight_data = weight_data.view(t["shape"])
            module.weight.data = weight_data.to(module.weight.dtype).to(module.weight.device)


def compress_with_tenpak(input_json: str, codec: str, tmpdir: str) -> Tuple[int, str]:
    """Compress bundle using tenpak CLI.
    
    Args:
        input_json: Path to input JSON bundle file
        codec: Codec to use
        tmpdir: Temp directory for output files
        
    Returns:
        (artifact_size, output_json_path)
    """
    binary = get_tenpak_binary()
    
    artifact_bin = os.path.join(tmpdir, "artifact.bin")
    output_json = os.path.join(tmpdir, "output.json")
    
    # Compress
    print(f"[COMPRESS] Running tenpak compress with {codec}...", flush=True)
    result = subprocess.run(
        [binary, "compress", "-i", input_json, "-o", artifact_bin, "--codec", codec],
        capture_output=True,
        text=True,
        timeout=600  # 10 minute timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compression failed: {result.stderr}")
    
    # Read artifact size
    artifact_size = os.path.getsize(artifact_bin)
    print(f"[COMPRESS] Artifact size: {artifact_size / 1e6:.1f} MB", flush=True)
    
    # Decompress
    print(f"[DECOMPRESS] Running tenpak decompress...", flush=True)
    result = subprocess.run(
        [binary, "decompress", "-i", artifact_bin, "-o", output_json],
        capture_output=True,
        text=True,
        timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Decompression failed: {result.stderr}")
    
    return artifact_size, output_json


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
    """Main evaluation function using tenpak Rust binary."""
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
        
        progress(0.3, desc="Extracting weights...")
        with tempfile.TemporaryDirectory() as tmpdir:
            input_json = os.path.join(tmpdir, "input.json")
            
            # Stream weights directly to file (memory efficient)
            num_tensors, original_size = extract_weights_to_file(model, input_json)
            print(f"[INFO] Extracted {num_tensors} tensors, {original_size / 1e9:.2f} GB", flush=True)
            
            # Force garbage collection after extraction
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            progress(0.4, desc=f"Compressing with {codec}...")
            artifact_size, output_json = compress_with_tenpak(input_json, codec, tmpdir)
            
            compression = original_size / artifact_size
            print(f"[INFO] Compression: {compression:.2f}x ({artifact_size / 1e6:.1f} MB)", flush=True)
            
            progress(0.7, desc="Loading quantized weights...")
            # Load decompressed bundle from file
            with open(output_json, "r") as f:
                decompressed = json.load(f)
            load_weights_from_bundle(model, decompressed)
            del decompressed
        
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
| **Compression** | **{compression:.2f}x** |
| Baseline PPL | {baseline_ppl:.4f} |
| Quantized PPL | {quantized_ppl:.4f} |
| **PPL Delta** | **{ppl_delta:+.2f}%** |
| **Status** | **{status}** |

### Details
- Original size (FP32): {original_size / 1e9:.2f} GB
- Compressed size: {artifact_size / 1e6:.1f} MB
- Tensors quantized: {num_tensors}
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
    
    **Now using actual Rust tenpak binary** for compression - testing real codecs!
    
    ### Available Codecs
    - **Production**: `int4_residual_v1` (best quality), `int4_opt_llama_v1` (good balance)
    - **Experimental 10x**: `caldera_v1`, `aqlm_v1`, `pocketllm_v1/v2`, `tenpak_x_v1/v2`
    - **SpinQuant-inspired**: `int4_spin_v1`, `int4_hybrid_v1/v2`, `int4_mixed_v1`
    
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
            value="int4_residual_v1",
            label="Codec"
        )
        samples_input = gr.Slider(10, 50, value=20, step=5, label="Evaluation Samples")
    
    run_btn = gr.Button("Run Evaluation", variant="primary")
    output = gr.Markdown()
    
    run_btn.click(evaluate_model, inputs=[model_input, codec_input, samples_input], outputs=output)

if __name__ == "__main__":
    demo.launch()
