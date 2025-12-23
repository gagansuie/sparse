#!/usr/bin/env python3
"""
Sparse - HuggingFace Space Demo
Model Optimization Testing Interface for 70B Models

This Space allows testing Sparse's optimization features on large models
using HuggingFace's infrastructure (GPU access for 70B models).
"""

import gradio as gr
import torch
from typing import Dict, Any
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core import QUANTIZATION_PRESETS, QuantizationWrapper
from core.delta import compress_delta_sparse, decompress_delta_sparse, compute_layer_delta
from optimizer.routing import classify_request_complexity, suggest_optimal_model, estimate_routing_savings
from optimizer import generate_candidates, OptimizationConstraints

# ==============================================================================
# FEATURE 1: QUANTIZATION TESTING
# ==============================================================================

def test_quantization_estimation(model_id: str, preset: str) -> Dict[str, Any]:
    """Test quantization size estimation on any model."""
    try:
        config = QUANTIZATION_PRESETS[preset]
        size_info = QuantizationWrapper.estimate_size(model_id, config)
        
        return {
            "status": "✅ Success",
            "model": model_id,
            "method": f"{config.method.value} {config.bits}-bit",
            "original_size_gb": f"{size_info['original_size_gb']:.3f} GB",
            "quantized_size_gb": f"{size_info['quantized_size_gb']:.3f} GB",
            "compression_ratio": f"{size_info['compression_ratio']:.2f}x",
            "savings_gb": f"{size_info['savings_gb']:.3f} GB",
            "savings_pct": f"{size_info['savings_pct']:.1f}%",
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
            "model": model_id,
        }

# ==============================================================================
# FEATURE 2: SMART ROUTING
# ==============================================================================

def test_smart_routing(prompt: str, max_tokens: int, requested_model: str) -> Dict[str, Any]:
    """Test smart routing for a given prompt."""
    try:
        # Classify complexity
        complexity = classify_request_complexity(prompt, max_tokens)
        
        # Get routing suggestion
        decision = suggest_optimal_model(
            requested_model=requested_model,
            prompt=prompt,
            quality_threshold=0.85,
            cost_priority=True
        )
        
        return {
            "status": "✅ Success",
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt,
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
# FEATURE 3: COST OPTIMIZER
# ==============================================================================

def test_cost_optimizer(max_ppl_delta: float, max_latency: float, min_throughput: float) -> Dict[str, Any]:
    """Test cost optimizer candidate generation."""
    try:
        # Generate candidates
        candidates = generate_candidates(
            include_calibration=False,
            max_expected_ppl_delta=max_ppl_delta,
            min_expected_compression=2.0
        )
        
        # Apply constraints
        constraints = OptimizationConstraints(
            max_ppl_delta=max_ppl_delta,
            max_latency_p99_ms=max_latency,
            min_throughput_tps=min_throughput
        )
        
        passing = [c for c in candidates if c.expected_ppl_delta <= constraints.max_ppl_delta]
        
        candidate_info = []
        for c in passing[:5]:  # Top 5
            candidate_info.append({
                "name": c.name,
                "method": c.method.value,
                "compression": f"{c.expected_compression:.2f}x",
                "ppl_delta": f"{c.expected_ppl_delta:.2f}%",
            })
        
        return {
            "status": "✅ Success",
            "total_candidates": len(candidates),
            "passing_candidates": len(passing),
            "constraints": {
                "max_ppl_delta": f"{max_ppl_delta}%",
                "max_latency": f"{max_latency}ms",
                "min_throughput": f"{min_throughput} tok/s",
            },
            "top_candidates": candidate_info,
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
        }

# ==============================================================================
# FEATURE 4: SAVINGS ESTIMATION
# ==============================================================================

def test_savings_estimation(requests_per_day: int, cost_per_request: float, optimization_rate: float) -> Dict[str, Any]:
    """Estimate cost savings from optimization."""
    try:
        savings = estimate_routing_savings(
            current_requests_per_day=requests_per_day,
            avg_cost_per_request=cost_per_request,
            optimization_rate=optimization_rate
        )
        
        return {
            "status": "✅ Success",
            "annual_requests": f"{savings['annual_requests']:,}",
            "current_annual_cost": f"${savings['current_annual_cost_usd']:,.0f}",
            "optimizable_requests": f"{savings['optimizable_requests']:,}",
            "optimization_rate": f"{savings['optimization_rate']*100:.0f}%",
            "annual_savings": f"${savings['annual_savings_usd']:,.0f}",
            "monthly_savings": f"${savings['monthly_savings_usd']:,.0f}",
            "savings_pct": f"{savings['savings_pct']:.1f}%",
        }
    except Exception as e:
        return {
            "status": f"❌ Error: {str(e)}",
        }

# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================

def create_interface():
    """Create Gradio interface."""
    
    with gr.Blocks(title="Sparse - Model Optimization Testing", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Sparse - Model Optimization Testing
        
        Test Sparse's optimization features with large models (up to 70B parameters).
        This Space uses HuggingFace's infrastructure to test features that require significant compute.
        
        **Features tested:**
        1. ✅ Quantization size estimation (70B models)
        2. ✅ Smart routing recommendations
        3. ✅ Cost optimizer candidate generation
        4. ✅ Savings estimation
        """)
        
        # Tab 1: Quantization
        with gr.Tab("📦 Quantization"):
            gr.Markdown("### Test quantization size estimation on any HuggingFace model")
            
            with gr.Row():
                with gr.Column():
                    quant_model = gr.Textbox(
                        label="Model ID",
                        value="meta-llama/Llama-2-70b-hf",
                        placeholder="e.g., meta-llama/Llama-2-70b-hf"
                    )
                    quant_preset = gr.Dropdown(
                        label="Quantization Preset",
                        choices=list(QUANTIZATION_PRESETS.keys()),
                        value="bnb_nf4"
                    )
                    quant_btn = gr.Button("Estimate Size", variant="primary")
                
                with gr.Column():
                    quant_output = gr.JSON(label="Results")
            
            quant_btn.click(
                test_quantization_estimation,
                inputs=[quant_model, quant_preset],
                outputs=quant_output
            )
            
            gr.Markdown("""
            **Supported presets:**
            - `bnb_nf4`: 4-bit NormalFloat (best quality/size)
            - `bnb_int8`: 8-bit integer (balanced)
            - `gptq_quality`: GPTQ 4-bit (high quality)
            - `awq_balanced`: AWQ 4-bit (balanced)
            """)
        
        # Tab 2: Smart Routing
        with gr.Tab("🎯 Smart Routing"):
            gr.Markdown("### Test routing recommendations for different prompts")
            
            with gr.Row():
                with gr.Column():
                    routing_prompt = gr.Textbox(
                        label="Prompt",
                        value="What is 2+2?",
                        lines=3,
                        placeholder="Enter your prompt..."
                    )
                    routing_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=10,
                        maximum=2000,
                        value=100,
                        step=10
                    )
                    routing_model = gr.Textbox(
                        label="Requested Model",
                        value="meta-llama/Llama-2-70b-hf"
                    )
                    routing_btn = gr.Button("Get Routing Recommendation", variant="primary")
                
                with gr.Column():
                    routing_output = gr.JSON(label="Results")
            
            routing_btn.click(
                test_smart_routing,
                inputs=[routing_prompt, routing_tokens, routing_model],
                outputs=routing_output
            )
        
        # Tab 3: Cost Optimizer
        with gr.Tab("💰 Cost Optimizer"):
            gr.Markdown("### Generate optimization candidates based on constraints")
            
            with gr.Row():
                with gr.Column():
                    opt_ppl = gr.Slider(
                        label="Max PPL Delta (%)",
                        minimum=0.5,
                        maximum=10.0,
                        value=2.0,
                        step=0.5
                    )
                    opt_latency = gr.Slider(
                        label="Max Latency (ms)",
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=10
                    )
                    opt_throughput = gr.Slider(
                        label="Min Throughput (tokens/s)",
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=100
                    )
                    opt_btn = gr.Button("Generate Candidates", variant="primary")
                
                with gr.Column():
                    opt_output = gr.JSON(label="Results")
            
            opt_btn.click(
                test_cost_optimizer,
                inputs=[opt_ppl, opt_latency, opt_throughput],
                outputs=opt_output
            )
        
        # Tab 4: Savings Estimation
        with gr.Tab("💵 Savings Estimation"):
            gr.Markdown("### Estimate potential cost savings from optimization")
            
            with gr.Row():
                with gr.Column():
                    sav_requests = gr.Slider(
                        label="Requests per Day",
                        minimum=1000,
                        maximum=10000000,
                        value=1000000,
                        step=10000
                    )
                    sav_cost = gr.Slider(
                        label="Cost per Request ($)",
                        minimum=0.0001,
                        maximum=0.01,
                        value=0.002,
                        step=0.0001
                    )
                    sav_rate = gr.Slider(
                        label="Optimization Rate (%)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.05
                    )
                    sav_btn = gr.Button("Estimate Savings", variant="primary")
                
                with gr.Column():
                    sav_output = gr.JSON(label="Results")
            
            sav_btn.click(
                test_savings_estimation,
                inputs=[sav_requests, sav_cost, sav_rate],
                outputs=sav_output
            )
        
        gr.Markdown("""
        ---
        ### 📚 Learn More
        
        - **GitHub**: [Sparse Repository](https://github.com/yourusername/sparse)
        - **Docs**: Full API documentation and integration guides
        - **Benchmarks**: See `benchmarks/BENCHMARK_RESULTS.md` for full test results
        
        **Note**: This Space runs on HuggingFace's infrastructure to enable testing with 70B+ models.
        All computations are estimation-based and don't require downloading full models.
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
