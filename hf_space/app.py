#!/usr/bin/env python3
"""
Sparse - Comprehensive Testing Space
TEMPORARY DEPLOYMENT FOR VALIDATION ONLY - WILL BE DELETED AFTER TESTING

Tests all features on HuggingFace infrastructure with 7B/70B models.
"""

import gradio as gr
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any
import time
from huggingface_hub import login

# Login with HF token for gated model access
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])
    print("✅ Logged in to HuggingFace for gated model access")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import QUANTIZATION_PRESETS, QuantizationWrapper
from core.delta import compute_layer_delta, compress_delta_sparse, decompress_delta_sparse, estimate_delta_savings, compress_adapter_delta
from core.delta_rust import is_rust_available, get_rust_info
from optimizer.routing import classify_request_complexity, suggest_optimal_model, estimate_routing_savings
from optimizer import generate_candidates, OptimizationConstraints

# ==============================================================================
# FEATURE 1: DELTA COMPRESSION (PRIMARY VALUE PROP)
# ==============================================================================

def test_delta_compression(base_model: str, finetune_model: str, threshold: float) -> Dict[str, Any]:
    """Test delta compression estimation with multi-strategy approach."""
    try:
        # Estimate savings between base and fine-tuned model
        start = time.time()
        result = estimate_delta_savings(base_model, finetune_model)
        estimate_time = time.time() - start
        
        # Extract values from result
        best_ratio = result.get('estimated_compression', 1.0)
        best_strategy = result.get('best_strategy', 'unknown')
        avg_sparsity = result.get('avg_sparsity', 0.0)
        
        # Estimate model size (7B model ~14GB in fp16)
        if '70b' in base_model.lower():
            original_size_gb = 140.0
        elif '13b' in base_model.lower():
            original_size_gb = 26.0
        elif '7b' in base_model.lower():
            original_size_gb = 14.0
        else:
            original_size_gb = 1.0
        
        # Calculate delta size based on compression
        delta_size_gb = original_size_gb / best_ratio if best_ratio > 0 else original_size_gb
        savings_pct = (1 - 1/best_ratio) * 100 if best_ratio > 1 else 0
        
        return {
            "status": "✅ Success",
            "base_model": base_model,
            "finetune_model": finetune_model,
            "original_size_gb": f"{original_size_gb:.2f} GB",
            "delta_size_gb": f"{delta_size_gb:.2f} GB",
            "best_strategy": best_strategy,
            "compression_ratio": f"{best_ratio:.2f}x",
            "savings_pct": f"{savings_pct:.1f}%",
            "compression_breakdown": {
                "sparse": f"{result.get('sparse_compression', 0):.2f}x",
                "int8": f"{result.get('int8_compression', 0):.2f}x",
                "sparse+int8": f"{result.get('sparse_int8_compression', 0):.2f}x",
            },
            "avg_sparsity": f"{avg_sparsity*100:.1f}%",
            "sample_layers": result.get('sample_layers', 0),
            "estimate_time_s": f"{estimate_time:.2f}s",
            "rust_acceleration": "✅ Available" if is_rust_available() else "⚠️ Python fallback",
        }
    except Exception as e:
        import traceback
        return {
            "status": f"❌ Error: {str(e)}",
            "base_model": base_model,
            "finetune_model": finetune_model,
            "traceback": traceback.format_exc()[-500:],
        }

# ==============================================================================
# FEATURE 2: QUANTIZATION
# ==============================================================================

def test_quantization(model_id: str, preset: str) -> Dict[str, Any]:
    """Test quantization size estimation."""
    try:
        config = QUANTIZATION_PRESETS[preset]
        size_info = QuantizationWrapper.estimate_size(model_id, config)
        
        # Handle method as string or enum
        method_name = config.method.value if hasattr(config.method, 'value') else config.method
        
        return {
            "status": "✅ Success",
            "model": model_id,
            "method": f"{method_name} {config.bits}-bit",
            "original_size_gb": f"{size_info['original_size_gb']:.2f} GB",
            "quantized_size_gb": f"{size_info['quantized_size_gb']:.2f} GB",
            "compression_ratio": f"{size_info['compression_ratio']:.2f}x",
            "savings_pct": f"{size_info['savings_pct']:.1f}%",
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "model": model_id,
        }

# ==============================================================================
# FEATURE 3: SMART ROUTING
# ==============================================================================

def test_routing(prompt: str, max_tokens: int, requested_model: str) -> Dict[str, Any]:
    """Test smart routing."""
    try:
        complexity = classify_request_complexity(prompt, max_tokens)
        decision = suggest_optimal_model(
            requested_model=requested_model,
            prompt=prompt,
            quality_threshold=0.85,
            cost_priority=True
        )
        
        return {
            "status": "✅ Success",
            "complexity": complexity.value,
            "requested_model": requested_model,
            "recommended_model": decision.recommended_model,
            "hardware": decision.recommended_hardware.hardware_name,
            "cost_per_1m_tokens": f"${decision.estimated_cost_per_1m_tokens:.2f}",
            "reasoning": decision.reasoning,
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
        }

# ==============================================================================
# FEATURE 4: COST OPTIMIZER
# ==============================================================================

def test_optimizer(max_ppl: float, max_latency: float, min_throughput: float) -> Dict[str, Any]:
    """Test cost optimizer."""
    try:
        candidates = generate_candidates(
            include_calibration=False,
            max_expected_ppl_delta=max_ppl,
            min_expected_compression=2.0
        )
        
        constraints = OptimizationConstraints(
            max_ppl_delta=max_ppl,
            max_latency_p99_ms=max_latency,
            min_throughput_tps=min_throughput
        )
        
        passing = [c for c in candidates if c.expected_ppl_delta <= constraints.max_ppl_delta]
        
        return {
            "status": "✅ Success",
            "total_candidates": len(candidates),
            "passing_candidates": len(passing),
            "top_5": [
                {
                    "name": c.name,
                    "compression": f"{c.expected_compression:.2f}x",
                    "ppl_delta": f"{c.expected_ppl_delta:.2f}%",
                }
                for c in passing[:5]
            ],
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
        }

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

with gr.Blocks(title="Sparse - Full Feature Testing", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 Sparse - Full Feature Validation
    
    **⚠️ TESTING ONLY - THIS SPACE WILL BE DELETED AFTER VALIDATION**
    
    Testing all proprietary features on HuggingFace infrastructure (7B/70B models).
    
    **Core Value Propositions:**
    1. 📦 **Delta Compression** - 90-96% savings on fine-tunes ($15-20M/year)
    2. 🎯 **Smart Routing** - Optimal model/hardware selection ($5-10M/year)
    3. 💰 **Cost Optimizer** - Auto-select best quantization method
    4. ⚡ **Rust Acceleration** - 10-20x faster compression
    """)
    
    # Rust Status
    with gr.Row():
        rust_btn = gr.Button("Check Rust Acceleration Status", variant="secondary")
        rust_output = gr.JSON(label="Rust Status")
    
    rust_btn.click(test_rust_info, outputs=rust_output)
    
    # Model pairs for delta compression testing
    # Note: RLHF-heavy models (chat) show low sparsity, code/instruct models show higher
    MODEL_PAIRS = {
        "Llama-2-7B (base → chat)": ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"),
        "Llama-2-13B (base → chat)": ("meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf"),
        "Llama-2-70B (base → chat) [RLHF-heavy]": ("meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf"),
        "CodeLlama-70B (base → instruct) [RECOMMENDED]": ("codellama/CodeLlama-70b-hf", "codellama/CodeLlama-70b-Instruct-hf"),
        "Llama-2-70B → CodeLlama-70B [code adaptation]": ("meta-llama/Llama-2-70b-hf", "codellama/CodeLlama-70b-hf"),
        "Mistral-7B (base → instruct)": ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.1"),
        "CodeLlama-7B (base → instruct)": ("codellama/CodeLlama-7b-hf", "codellama/CodeLlama-7b-Instruct-hf"),
    }
    
    def get_model_pair(selection):
        base, finetune = MODEL_PAIRS.get(selection, ("", ""))
        return base, finetune
    
    # Tab 1: Delta Compression
    with gr.Tab("📦 Delta Compression (PRIMARY)"):
        gr.Markdown("### Test delta compression between base and fine-tuned model")
        gr.Markdown("*Compare a base model to its fine-tune to see compression savings*")
        
        with gr.Row():
            with gr.Column():
                delta_preset = gr.Dropdown(
                    label="Model Pair",
                    choices=list(MODEL_PAIRS.keys()),
                    value="Llama-2-7B (base → chat)"
                )
                delta_base = gr.Textbox(
                    label="Base Model",
                    value="meta-llama/Llama-2-7b-hf",
                    interactive=True
                )
                delta_finetune = gr.Textbox(
                    label="Fine-tuned Model",
                    value="meta-llama/Llama-2-7b-chat-hf",
                    interactive=True
                )
                delta_threshold = gr.Slider(
                    label="Sparsity Threshold",
                    minimum=1e-8,
                    maximum=1e-4,
                    value=1e-6,
                    step=1e-8
                )
                delta_btn = gr.Button("Estimate Delta Savings", variant="primary")
            
            with gr.Column():
                delta_output = gr.JSON(label="Results")
        
        # Update text fields when dropdown changes
        delta_preset.change(
            get_model_pair,
            inputs=[delta_preset],
            outputs=[delta_base, delta_finetune]
        )
        
        delta_btn.click(
            test_delta_compression,
            inputs=[delta_base, delta_finetune, delta_threshold],
            outputs=delta_output
        )
    
    # Tab 1b: Adapter Delta (Optional)
    with gr.Tab("🔌 Adapter Delta (Optional)"):
        gr.Markdown("""### Package LoRA/PEFT Adapters as Delta Artifacts
        
*Adapters are treated as a delta type within the same framework. This is optional — full model deltas remain the primary feature.*
        """)
        
        with gr.Row():
            with gr.Column():
                adapter_base = gr.Textbox(
                    label="Base Model",
                    value="meta-llama/Llama-2-7b-hf",
                    info="The base model the adapter was trained on"
                )
                adapter_id = gr.Textbox(
                    label="Adapter (HF ID or local path)",
                    value="",
                    placeholder="e.g., my-org/llama-lora-adapter",
                    info="LoRA/PEFT adapter to package"
                )
                adapter_btn = gr.Button("Package Adapter Delta", variant="secondary")
            
            with gr.Column():
                adapter_output = gr.JSON(label="Results")
        
        def test_adapter_delta(base_model: str, adapter: str):
            if not adapter:
                return {"status": "❌ Error", "message": "Please provide an adapter ID or path"}
            try:
                import tempfile
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
                        "note": "Adapter packaged as delta artifact. Use reconstruct_from_delta() to apply."
                    }
            except Exception as e:
                import traceback
                return {"status": f"❌ Error: {str(e)}", "traceback": traceback.format_exc()[-500:]}
        
        adapter_btn.click(
            test_adapter_delta,
            inputs=[adapter_base, adapter_id],
            outputs=adapter_output
        )
    
    # Tab 2: Quantization
    with gr.Tab("🎯 Quantization"):
        gr.Markdown("### Test quantization on 70B models")
        
        with gr.Row():
            with gr.Column():
                quant_model = gr.Textbox(
                    label="Model",
                    value="meta-llama/Llama-2-70b-hf"
                )
                quant_preset = gr.Dropdown(
                    label="Preset",
                    choices=list(QUANTIZATION_PRESETS.keys()),
                    value="bnb_nf4"
                )
                quant_btn = gr.Button("Estimate Size", variant="primary")
            
            with gr.Column():
                quant_output = gr.JSON(label="Results")
        
        quant_btn.click(
            test_quantization,
            inputs=[quant_model, quant_preset],
            outputs=quant_output
        )
    
    # Tab 3: Smart Routing
    with gr.Tab("🧭 Smart Routing"):
        gr.Markdown("### Test routing recommendations")
        
        with gr.Row():
            with gr.Column():
                routing_prompt = gr.Textbox(
                    label="Prompt",
                    value="What is 2+2?",
                    lines=3
                )
                routing_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=10,
                    maximum=2000,
                    value=100
                )
                routing_model = gr.Textbox(
                    label="Requested Model",
                    value="meta-llama/Llama-2-70b-hf"
                )
                routing_btn = gr.Button("Get Routing", variant="primary")
            
            with gr.Column():
                routing_output = gr.JSON(label="Results")
        
        routing_btn.click(
            test_routing,
            inputs=[routing_prompt, routing_tokens, routing_model],
            outputs=routing_output
        )
    
    # Tab 4: Cost Optimizer
    with gr.Tab("💰 Cost Optimizer"):
        gr.Markdown("### Generate optimization candidates")
        
        with gr.Row():
            with gr.Column():
                opt_ppl = gr.Slider(
                    label="Max PPL Delta (%)",
                    minimum=0.5,
                    maximum=10.0,
                    value=2.0
                )
                opt_latency = gr.Slider(
                    label="Max Latency (ms)",
                    minimum=50,
                    maximum=500,
                    value=100
                )
                opt_throughput = gr.Slider(
                    label="Min Throughput (tok/s)",
                    minimum=100,
                    maximum=2000,
                    value=500
                )
                opt_btn = gr.Button("Generate Candidates", variant="primary")
            
            with gr.Column():
                opt_output = gr.JSON(label="Results")
        
        opt_btn.click(
            test_optimizer,
            inputs=[opt_ppl, opt_latency, opt_throughput],
            outputs=opt_output
        )
    
    gr.Markdown("""
    ---
    ### ⚠️ Important
    
    **This is a temporary testing deployment.**
    - Tests proprietary features on HF infrastructure
    - Will be deleted after validation testing
    - Results will be documented for acquisition pitch
    
    **For licensing:** gagan.suie@sparselabs.ai
    """)

if __name__ == "__main__":
    demo.launch()
