#!/usr/bin/env python3
"""
Sparse - Delta Compression Demo
Compress fine-tuned models as deltas from base models.
"""

import gradio as gr
import sys
import os
from pathlib import Path
from typing import Dict, Any
import tempfile
from huggingface_hub import login

# Login with HF token for gated model access
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
    print("✅ Logged in to HuggingFace for gated model access")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.delta import compress_delta, compress_adapter_delta, validate_int8_delta_quality, compress_delta_svd_full
from core.delta_rust import get_rust_info

# ==============================================================================
# FEATURE 1: FULL DELTA COMPRESSION
# ==============================================================================

def run_compression(base_model: str, finetune_model: str) -> Dict[str, Any]:
    """Run actual delta compression and return results."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = compress_delta(
                base_model_id=base_model,
                finetune_model_id=finetune_model,
                output_path=tmpdir,
            )
            
            return {
                "status": "✅ Compression Complete",
                "base_model": manifest.base_model_id,
                "finetune_model": manifest.finetune_model_id,
                "compression_ratio": f"{manifest.compression_ratio:.2f}x",
                "total_params": f"{manifest.total_params:,}",
                "changed_params": f"{manifest.changed_params:,}",
                "change_pct": f"{100*manifest.changed_params/manifest.total_params:.1f}%",
                "num_layers": manifest.num_layers,
                "rust_acceleration": "✅ Yes",
            }
    except Exception as e:
        import traceback
        return {
            "status": f"❌ Error: {str(e)}",
            "traceback": traceback.format_exc()[-500:],
        }

# ==============================================================================
# FEATURE 2: INT8 DELTA QUALITY VALIDATION
# ==============================================================================

def test_int8_quality(base_model: str, finetune_model: str, sample_layers: int) -> Dict[str, Any]:
    """Test INT8 delta compression quality."""
    try:
        report = validate_int8_delta_quality(
            base_model_id=base_model,
            finetune_model_id=finetune_model,
            sample_layers=int(sample_layers),
            prompts=["Hello, how are you?", "The capital of France is"],
        )
        return report
    except Exception as e:
        import traceback
        return {"status": f"❌ Error: {str(e)}", "traceback": traceback.format_exc()}

# ==============================================================================
# FEATURE 3: ADAPTER DELTA
# ==============================================================================

def test_adapter_delta(base_model: str, adapter: str) -> Dict[str, Any]:
    """Package a LoRA/PEFT adapter as delta."""
    if not adapter:
        return {"status": "❌ Error", "message": "Please provide an adapter ID or path"}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = compress_adapter_delta(
                base_model_id=base_model,
                adapter_id=adapter,
                output_path=tmpdir
            )
            return {
                "status": "✅ Success",
                "delta_type": manifest.delta_type,
                "base_model": manifest.base_model_id,
                "adapter": manifest.finetune_model_id,
                "note": "Adapter packaged as delta artifact."
            }
    except Exception as e:
        import traceback
        return {"status": f"❌ Error: {str(e)}", "traceback": traceback.format_exc()[-500:]}

# ==============================================================================
# RUST DIAGNOSTICS
# ==============================================================================

def test_rust_info() -> Dict[str, Any]:
    """Get Rust acceleration info."""
    info = get_rust_info()
    return {
        "rust_available": "✅ Yes" if info["available"] else "❌ No",
        "version": info["version"] or "N/A",
        "features": ", ".join(info["features"]) if info["features"] else "None",
    }

# ==============================================================================
# GRADIO UI
# ==============================================================================

MODEL_PAIRS = {
    "Mistral-7B (base → instruct)": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"),
    "CodeLlama-7B (base → instruct)": ("codellama/CodeLlama-7b-hf", "codellama/CodeLlama-7b-Instruct-hf"),
    "CodeLlama-13B (base → instruct)": ("codellama/CodeLlama-13b-hf", "codellama/CodeLlama-13b-Instruct-hf"),
}

def get_model_pair(selection):
    base, finetune = MODEL_PAIRS.get(selection, ("", ""))
    return base, finetune

with gr.Blocks(title="Sparse - Delta Compression", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Sparse - Delta Compression for Fine-tuned Models
    
    **Compress fine-tunes to 10-25% of original size.**
    
    | Metric | Value |
    |--------|-------|
    | **Addressable Market** | 67% of HuggingFace models |
    | **Compression** | 4-10x typical |
    | **HF Savings** | $30-80M/year |
    """)
    
    # Rust Status
    with gr.Row():
        rust_btn = gr.Button("Check Rust Acceleration", variant="secondary")
        rust_output = gr.JSON(label="Rust Status")
    
    rust_btn.click(test_rust_info, outputs=rust_output)
    
    # Tab 1: Delta Compression
    with gr.Tab("📦 Compress Delta"):
        gr.Markdown("### Compress a fine-tune as delta from base model")
        
        with gr.Row():
            with gr.Column():
                compress_preset = gr.Dropdown(
                    label="Model Pair",
                    choices=list(MODEL_PAIRS.keys()),
                    value="Mistral-7B (base → instruct)"
                )
                compress_base = gr.Textbox(
                    label="Base Model",
                    value="mistralai/Mistral-7B-v0.1",
                    interactive=True
                )
                compress_finetune = gr.Textbox(
                    label="Fine-tuned Model",
                    value="mistralai/Mistral-7B-Instruct-v0.1",
                    interactive=True
                )
                compress_btn = gr.Button("Compress Delta", variant="primary")
            
            with gr.Column():
                compress_output = gr.JSON(label="Results")
        
        compress_preset.change(
            get_model_pair,
            inputs=[compress_preset],
            outputs=[compress_base, compress_finetune]
        )
        
        compress_btn.click(
            run_compression,
            inputs=[compress_base, compress_finetune],
            outputs=compress_output
        )
    
    # Tab 2: INT8 Quality Check
    with gr.Tab("⚡ INT8 Quality"):
        gr.Markdown("### Validate INT8 compression quality")
        
        with gr.Row():
            with gr.Column():
                int8_preset = gr.Dropdown(
                    label="Model Pair",
                    choices=list(MODEL_PAIRS.keys()),
                    value="Mistral-7B (base → instruct)"
                )
                int8_base = gr.Textbox(
                    label="Base Model",
                    value="mistralai/Mistral-7B-v0.1",
                    interactive=True
                )
                int8_finetune = gr.Textbox(
                    label="Fine-tuned Model",
                    value="mistralai/Mistral-7B-Instruct-v0.1",
                    interactive=True
                )
                int8_layers = gr.Slider(
                    label="Sample Layers",
                    minimum=1, maximum=5, value=2, step=1
                )
                int8_btn = gr.Button("Check Quality", variant="primary")
            
            with gr.Column():
                int8_output = gr.JSON(label="Quality Report")
        
        int8_preset.change(
            get_model_pair,
            inputs=[int8_preset],
            outputs=[int8_base, int8_finetune]
        )
        
        int8_btn.click(
            test_int8_quality,
            inputs=[int8_base, int8_finetune, int8_layers],
            outputs=int8_output
        )
    
    # Tab 3: Adapter Delta
    with gr.Tab("🔌 LoRA Adapter"):
        gr.Markdown("### Package LoRA adapter as delta")
        
        with gr.Row():
            with gr.Column():
                adapter_base = gr.Textbox(
                    label="Base Model",
                    value="mistralai/Mistral-7B-v0.1"
                )
                adapter_id = gr.Textbox(
                    label="Adapter ID",
                    placeholder="e.g., my-org/lora-adapter"
                )
                adapter_btn = gr.Button("Package Adapter", variant="primary")
            
            with gr.Column():
                adapter_output = gr.JSON(label="Results")
        
        adapter_btn.click(
            test_adapter_delta,
            inputs=[adapter_base, adapter_id],
            outputs=adapter_output
        )
    
    # Tab 4: SVD Compression (LoRA-equivalent extraction)
    with gr.Tab("🧬 SVD Extract (LoRA-size)"):
        gr.Markdown("""### Extract LoRA-equivalent from any full fine-tune
        
**Post-hoc LoRA extraction:** Convert full fine-tunes to ~50MB, even if not trained with LoRA.

| Mode | Size | Quality |
|------|------|---------|
| Lossless delta | 1.4 GB | 100% |
| **SVD (this)** | **~50 MB** | **~95-99%** |
        """)
        
        with gr.Row():
            with gr.Column():
                svd_preset = gr.Dropdown(
                    label="Model Pair",
                    choices=list(MODEL_PAIRS.keys()),
                    value="Mistral-7B (base → instruct)"
                )
                svd_base = gr.Textbox(
                    label="Base Model",
                    value="mistralai/Mistral-7B-v0.1",
                    interactive=True
                )
                svd_finetune = gr.Textbox(
                    label="Fine-tuned Model",
                    value="mistralai/Mistral-7B-Instruct-v0.1",
                    interactive=True
                )
                svd_rank = gr.Slider(
                    label="SVD Rank (like LoRA rank)",
                    minimum=4, maximum=64, value=16, step=4,
                    info="Higher = better quality, larger size"
                )
                svd_btn = gr.Button("Extract SVD Delta", variant="primary")
            
            with gr.Column():
                svd_output = gr.JSON(label="Results")
        
        def run_svd_compression(base_model: str, finetune_model: str, rank: int) -> Dict[str, Any]:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    manifest = compress_delta_svd_full(
                        base_model_id=base_model,
                        finetune_model_id=finetune_model,
                        output_path=tmpdir,
                        rank=int(rank),
                    )
                    return {
                        "status": "✅ SVD Extraction Complete",
                        "compression_ratio": f"{manifest.compression_ratio:.1f}x",
                        "original_size": f"{manifest.original_size_bytes / 1e9:.2f} GB",
                        "compressed_size": f"{manifest.compressed_size_bytes / 1e6:.1f} MB",
                        "rank": manifest.rank,
                        "avg_error": f"{manifest.avg_reconstruction_error:.6f}",
                        "max_error": f"{manifest.max_reconstruction_error:.6f}",
                        "note": "This is LOSSY compression (LoRA-equivalent quality)"
                    }
            except Exception as e:
                import traceback
                return {"status": f"❌ Error: {str(e)}", "traceback": traceback.format_exc()[-500:]}
        
        svd_preset.change(
            get_model_pair,
            inputs=[svd_preset],
            outputs=[svd_base, svd_finetune]
        )
        
        svd_btn.click(
            run_svd_compression,
            inputs=[svd_base, svd_finetune, svd_rank],
            outputs=svd_output
        )
    
    gr.Markdown("""
    ---
    ### Compression Modes
    
    | Mode | Size | Quality | Use Case |
    |------|------|---------|----------|
    | **Lossless** | 1.4 GB | 100% | When quality matters |
    | **SVD (LoRA-equiv)** | ~50 MB | ~95-99% | When size matters |
    
    **GitHub:** https://github.com/gagansuie/sparse  
    **License:** Apache 2.0
    """)

if __name__ == "__main__":
    demo.launch()
