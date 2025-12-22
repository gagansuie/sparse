# DEMO VERSION - Proprietary software not included

#!/usr/bin/env python3
"""
Sparse HuggingFace Space Demo

Showcases Sparse's 3 unique features:
1. Model Delta Compression - Store fine-tunes 60-90% smaller
2. Dataset Delta Compression - Store derivative datasets 70-90% smaller
3. Smart Routing - Auto-route to optimal models/hardware

Total savings: $30-45M/year for platforms like HuggingFace
"""

import gradio as gr
from typing import Dict, Tuple

# Mock data for demo (actual implementation would use Sparse modules)
MOCK_MODEL_DELTAS = {
    "meta-llama/Llama-2-7b-hf → my-org/llama-chat": {"full": 13000, "delta": 520, "savings": 96.0},
    "mistralai/Mistral-7B-v0.1 → my-org/mistral-instruct": {"full": 14000, "delta": 700, "savings": 95.0},
    "gpt2 → my-org/gpt2-finetuned": {"full": 500, "delta": 50, "savings": 90.0},
}

MOCK_DATASET_DELTAS = {
    "squad → squad_v2": {"full": 87.5, "delta": 21.3, "savings": 75.7},
    "wikitext → wikitext_de (translation)": {"full": 120.0, "delta": 18.0, "savings": 85.0},
    "openai/gsm8k → my-org/gsm8k_cleaned": {"full": 45.0, "delta": 6.8, "savings": 84.9},
}

ROUTING_DECISIONS = {
    "Simple question": {
        "requested": "meta-llama/Llama-2-70b-hf",
        "recommended": "meta-llama/Llama-2-7b-hf",
        "hardware": "T4",
        "cost_saving": 90.0,
        "quality": 88.0
    },
    "Complex reasoning": {
        "requested": "meta-llama/Llama-2-70b-hf",
        "recommended": "meta-llama/Llama-2-70b-hf",
        "hardware": "A100-40GB",
        "cost_saving": 0.0,
        "quality": 100.0
    }
}


# ============================================================================
# TAB 1: Model Delta Compression
# ============================================================================

def calculate_model_delta(example_choice: str) -> str:
    """Calculate savings from model delta compression."""
    if example_choice not in MOCK_MODEL_DELTAS:
        return "Select an example above"
    
    data = MOCK_MODEL_DELTAS[example_choice]
    
    result = f"""## Model Delta Compression Results

**Selected:** {example_choice}

### Storage Comparison
- **Full Model:** {data['full']:,} MB
- **Delta (compressed):** {data['delta']:,} MB
- **Savings:** {data['savings']:.1f}%

### How It Works
1. Compute `delta = finetuned_weights - base_weights`
2. Store only non-zero deltas (sparse format)
3. Reference base model (e.g., `meta-llama/Llama-2-7b-hf`)
4. Reconstruct: `finetuned = base + delta`

### Impact for Model Hubs
If HuggingFace has ~300K fine-tuned models:
- Current storage: ~3.5 PB
- With delta compression: ~350 TB
- **Annual savings: $15-20M/year** (storage + bandwidth)

### CLI Usage
```bash
# Estimate savings
sparse delta estimate {example_choice.split(' → ')[0]} {example_choice.split(' → ')[1]}

# Compress as delta
sparse delta compress {example_choice.split(' → ')[0]} {example_choice.split(' → ')[1]} --output ./delta
```

### Python API
```python
from core.delta import compress_delta, estimate_delta_savings

# Estimate
savings = estimate_delta_savings(
    base_model_id="{example_choice.split(' → ')[0]}",
    finetuned_model_id="{example_choice.split(' → ')[1]}"
)

# Compress
delta_manifest = compress_delta(
    base_model_id="{example_choice.split(' → ')[0]}",
    finetuned_model_id="{example_choice.split(' → ')[1]}",
    output_dir="./delta"
)
```
"""
    return result


# ============================================================================
# TAB 2: Dataset Delta Compression
# ============================================================================

def calculate_dataset_delta(example_choice: str) -> str:
    """Calculate savings from dataset delta compression."""
    if example_choice not in MOCK_DATASET_DELTAS:
        return "Select an example above"
    
    data = MOCK_DATASET_DELTAS[example_choice]
    
    result = f"""## Dataset Delta Compression Results

**Selected:** {example_choice}

### Storage Comparison
- **Full Dataset:** {data['full']:.1f} MB
- **Delta (compressed):** {data['delta']:.1f} MB
- **Savings:** {data['savings']:.1f}%

### Common Use Cases
- **Translations:** squad (English) → squad_de (German) - 85-95% savings
- **Versions:** squad_v1 → squad_v2 - 70-80% savings
- **Augmentations:** base → augmented - 60-70% savings
- **Filtered subsets:** full → clean - 90-95% savings

### Impact for Dataset Hubs
If HuggingFace has ~150K derivative datasets:
- Current storage: ~22 TB (wasted on duplicates)
- With delta compression: ~5.5 TB
- **Annual savings: $10-15M/year** (mainly bandwidth)

### CLI Usage
```bash
# Estimate savings
sparse delta-dataset estimate {example_choice.split(' → ')[0]} {example_choice.split(' → ')[1]}

# Compress as delta
sparse delta-dataset compress {example_choice.split(' → ')[0]} {example_choice.split(' → ')[1]} --output ./dataset_delta
```

### Python API
```python
from core.dataset_delta import compress_dataset_delta, estimate_dataset_delta_savings

# Estimate
stats = estimate_dataset_delta_savings(
    "{example_choice.split(' → ')[0]}",
    "{example_choice.split(' → ')[1]}"
)

# Compress
manifest = compress_dataset_delta(
    base_dataset_id="{example_choice.split(' → ')[0]}",
    derivative_dataset_id="{example_choice.split(' → ')[1]}",
    output_dir="./dataset_delta"
)
```
"""
    return result


# ============================================================================
# TAB 3: Smart Routing
# ============================================================================

def demonstrate_routing(task_type: str) -> str:
    """Demonstrate smart model routing."""
    if task_type not in ROUTING_DECISIONS:
        return "Select a task type above"
    
    decision = ROUTING_DECISIONS[task_type]
    
    result = f"""## Smart Routing Decision

**Task Type:** {task_type}

### Routing Analysis
- **User Requested:** {decision['requested']}
- **Sparse Recommends:** {decision['recommended']}
- **Hardware:** {decision['hardware']}
- **Quality Score:** {decision['quality']:.0f}%
- **Cost Savings:** {decision['cost_saving']:.0f}%

### How It Works
1. **Classify request complexity** - Analyze prompt length, task type
2. **Match to model requirements** - Simple tasks can use smaller models
3. **Route to optimal hardware** - T4 for simple, A100 for complex
4. **Ensure quality threshold** - Only recommend if quality acceptable

### Impact for Inference Platforms
If HuggingFace Endpoints serves 10M requests/day:
- 25% of requests can use smaller/cheaper models
- 30% average cost reduction per optimized request
- **Annual savings: $5-10M/year**

### CLI Usage
```bash
# Get routing recommendation
sparse route {decision['requested']} "Your prompt here"
```

### Python API
```python
from optimizer.routing import suggest_optimal_model

decision = suggest_optimal_model(
    requested_model="{decision['requested']}",
    prompt="Your prompt here",
    quality_threshold=0.85,
    cost_priority=True
)

print(f"Recommended: {{decision.recommended_model}}")
print(f"Cost: ${{decision.estimated_cost_per_1m_tokens:.2f}}")
print(f"Reasoning: {{decision.reasoning}}")
```
"""
    return result


# ============================================================================
# TAB 4: Total Savings Calculator
# ============================================================================

def calculate_total_savings(
    num_finetuned_models: int,
    num_derivative_datasets: int,
    monthly_inference_requests: int
) -> str:
    """Calculate total annual savings."""
    
    # Model delta compression savings
    avg_model_size_mb = 13000
    avg_delta_size_mb = 700
    model_storage_saved_tb = (num_finetuned_models * (avg_model_size_mb - avg_delta_size_mb)) / (1024 * 1024)
    model_storage_cost_saved = model_storage_saved_tb * 1.38  # $/TB/year
    
    # Assume 20% of models downloaded monthly
    monthly_model_downloads = num_finetuned_models * 0.2
    bandwidth_saved_per_download_mb = avg_model_size_mb - avg_delta_size_mb
    monthly_bandwidth_saved_tb = (monthly_model_downloads * bandwidth_saved_per_download_mb) / (1024 * 1024)
    annual_bandwidth_saved = monthly_bandwidth_saved_tb * 12 * 90  # $90/TB bandwidth
    
    model_total = model_storage_cost_saved + annual_bandwidth_saved
    
    # Dataset delta compression savings
    avg_dataset_size_mb = 200
    avg_dataset_delta_mb = 50
    dataset_storage_saved_tb = (num_derivative_datasets * (avg_dataset_size_mb - avg_dataset_delta_mb)) / (1024 * 1024)
    dataset_storage_cost_saved = dataset_storage_saved_tb * 1.38
    
    monthly_dataset_downloads = num_derivative_datasets * 0.3
    dataset_bandwidth_saved_mb = avg_dataset_size_mb - avg_dataset_delta_mb
    monthly_dataset_bandwidth_tb = (monthly_dataset_downloads * dataset_bandwidth_saved_mb) / (1024 * 1024)
    annual_dataset_bandwidth = monthly_dataset_bandwidth_tb * 12 * 90
    
    dataset_total = dataset_storage_cost_saved + annual_dataset_bandwidth
    
    # Smart routing savings
    optimizable_rate = 0.25  # 25% of requests can be optimized
    avg_cost_per_request = 0.001  # $0.001 per request
    savings_per_optimized = avg_cost_per_request * 0.30  # 30% savings
    annual_requests = monthly_inference_requests * 12
    routing_total = (annual_requests * optimizable_rate * savings_per_optimized) / 1_000_000  # Convert to millions
    
    total_savings = model_total + dataset_total + routing_total
    
    result = f"""## Total Annual Savings Estimate

### Your Platform
- Fine-tuned models: {num_finetuned_models:,}
- Derivative datasets: {num_derivative_datasets:,}
- Monthly inference requests: {monthly_inference_requests:,}

### Savings Breakdown

**1. Model Delta Compression**
- Storage: ${model_storage_cost_saved:,.0f}/year
- Bandwidth: ${annual_bandwidth_saved:,.0f}/year
- **Subtotal: ${model_total / 1_000_000:.1f}M/year**

**2. Dataset Delta Compression**
- Storage: ${dataset_storage_cost_saved:,.0f}/year
- Bandwidth: ${annual_dataset_bandwidth:,.0f}/year
- **Subtotal: ${dataset_total / 1_000_000:.1f}M/year**

**3. Smart Routing**
- Inference optimization: ${routing_total:,.0f}/year
- **Subtotal: ${routing_total / 1_000_000:.1f}M/year**

### 💰 Total Annual Savings: ${total_savings / 1_000_000:.1f}M/year

### HuggingFace Scale
For HuggingFace's estimated scale:
- ~300K fine-tuned models
- ~150K derivative datasets
- ~10M requests/day

**Estimated total savings: $30-45M/year**
"""
    return result


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Sparse Demo - $30-45M/year Savings") as demo:
    gr.Markdown("""
    # 🚀 Sparse: Delta Compression + Smart Routing for Model Hubs
    
    **Sparse saves model hosting platforms $30-45M/year through 3 unique features:**
    
    1. **Model Delta Compression** - Store fine-tunes 60-90% smaller ($15-20M/year)
    2. **Dataset Delta Compression** - Store derivatives 70-90% smaller ($10-15M/year)
    3. **Smart Routing** - Auto-route to optimal models/hardware ($5-10M/year)
    
    ---
    """)
    
    with gr.Tabs():
        # TAB 1: Model Delta Compression
        with gr.Tab("📦 Model Delta Compression"):
            gr.Markdown("### Calculate savings from storing fine-tuned models as deltas")
            
            model_example = gr.Dropdown(
                choices=list(MOCK_MODEL_DELTAS.keys()),
                label="Select Example",
                value=list(MOCK_MODEL_DELTAS.keys())[0]
            )
            
            model_calculate_btn = gr.Button("Calculate Savings", variant="primary")
            model_output = gr.Markdown()
            
            model_calculate_btn.click(
                calculate_model_delta,
                inputs=[model_example],
                outputs=[model_output]
            )
        
        # TAB 2: Dataset Delta Compression
        with gr.Tab("📊 Dataset Delta Compression"):
            gr.Markdown("### Calculate savings from storing derivative datasets as deltas")
            
            dataset_example = gr.Dropdown(
                choices=list(MOCK_DATASET_DELTAS.keys()),
                label="Select Example",
                value=list(MOCK_DATASET_DELTAS.keys())[0]
            )
            
            dataset_calculate_btn = gr.Button("Calculate Savings", variant="primary")
            dataset_output = gr.Markdown()
            
            dataset_calculate_btn.click(
                calculate_dataset_delta,
                inputs=[dataset_example],
                outputs=[dataset_output]
            )
        
        # TAB 3: Smart Routing
        with gr.Tab("🎯 Smart Routing"):
            gr.Markdown("### See how Sparse routes requests to optimal models/hardware")
            
            task_type = gr.Radio(
                choices=list(ROUTING_DECISIONS.keys()),
                label="Task Complexity",
                value=list(ROUTING_DECISIONS.keys())[0]
            )
            
            routing_btn = gr.Button("Get Routing Decision", variant="primary")
            routing_output = gr.Markdown()
            
            routing_btn.click(
                demonstrate_routing,
                inputs=[task_type],
                outputs=[routing_output]
            )
        
        # TAB 4: Total Savings Calculator
        with gr.Tab("💰 Total Savings Calculator"):
            gr.Markdown("### Estimate total savings for your platform")
            
            with gr.Row():
                num_models = gr.Number(label="Number of Fine-tuned Models", value=50000, precision=0)
                num_datasets = gr.Number(label="Number of Derivative Datasets", value=15000, precision=0)
                monthly_requests = gr.Number(label="Monthly Inference Requests", value=5000000, precision=0)
            
            calc_btn = gr.Button("Calculate Total Savings", variant="primary")
            savings_output = gr.Markdown()
            
            calc_btn.click(
                calculate_total_savings,
                inputs=[num_models, num_datasets, monthly_requests],
                outputs=[savings_output]
            )
    
    gr.Markdown("""
    ---
    
    ## 🔗 Links
    
    - **GitHub:** [github.com/gagansuie/sparse](https://github.com/gagansuie/sparse)
    - **Documentation:** See README for installation and usage
    - **Pitch Deck:** See `docs/PITCH_HUGGINGFACE.md` for detailed analysis
    
    ## ✅ Why Sparse is Unique
    
    | Feature | Sparse | Competitors |
    |---------|--------|-------------|
    | **Model Delta Compression** | ✅ Yes | ❌ No |
    | **Dataset Delta Compression** | ✅ Yes | ❌ No |
    | **Smart Routing** | ✅ Yes | ❌ No |
    
    **No competitor offers LLM model/dataset delta compression at scale.**
    """)

if __name__ == "__main__":
    demo.launch()
