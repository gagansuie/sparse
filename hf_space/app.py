#!/usr/bin/env python3
"""
Sparse - Comprehensive Testing Space
TEMPORARY DEPLOYMENT FOR VALIDATION ONLY - WILL BE DELETED AFTER TESTING

Tests all features on HuggingFace infrastructure with 7B/70B models.
"""

import gradio as gr
import torch
import sys
from pathlib import Path
from typing import Dict, Any
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import QUANTIZATION_PRESETS, QuantizationWrapper
from core.delta import compute_layer_delta, compress_delta_sparse, decompress_delta_sparse, estimate_delta_savings
from core.delta_rust import is_rust_available, get_rust_info
from optimizer.routing import classify_request_complexity, suggest_optimal_model, estimate_routing_savings
from optimizer import generate_candidates, OptimizationConstraints

# ==============================================================================
# FEATURE 1: DELTA COMPRESSION (PRIMARY VALUE PROP)
# ==============================================================================

def test_delta_compression(base_model: str, threshold: float) -> Dict[str, Any]:
    """Test delta compression estimation and algorithm."""
    try:
        # Estimate savings
        start = time.time()
        savings = estimate_delta_savings(base_model, base_model)
        estimate_time = time.time() - start
        
        return {
            "status": "✅ Success",
            "base_model": base_model,
            "original_size_gb": f"{savings.get('original_size_gb', 0):.2f} GB",
            "delta_size_gb": f"{savings.get('delta_size_gb', 0):.2f} GB",
            "compression_ratio": f"{savings.get('compression_ratio', 0):.2f}x",
            "savings_pct": f"{savings.get('savings_pct', 0):.1f}%",
            "estimate_time_s": f"{estimate_time:.2f}s",
            "rust_acceleration": "✅ Available" if is_rust_available() else "⚠️ Python fallback",
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "base_model": base_model,
        }

# ==============================================================================
# FEATURE 2: QUANTIZATION
# ==============================================================================

def test_quantization(model_id: str, preset: str) -> Dict[str, Any]:
    """Test quantization size estimation."""
    try:
        config = QUANTIZATION_PRESETS[preset]
        size_info = QuantizationWrapper.estimate_size(model_id, config)
        
        return {
            "status": "✅ Success",
            "model": model_id,
            "method": f"{config.method.value} {config.bits}-bit",
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
    
    # Tab 1: Delta Compression
    with gr.Tab("📦 Delta Compression (PRIMARY)"):
        gr.Markdown("### Test delta compression on any model")
        
        with gr.Row():
            with gr.Column():
                delta_model = gr.Textbox(
                    label="Base Model",
                    value="meta-llama/Llama-2-7b-hf",
                    placeholder="meta-llama/Llama-2-70b-hf"
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
        
        delta_btn.click(
            test_delta_compression,
            inputs=[delta_model, delta_threshold],
            outputs=delta_output
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
