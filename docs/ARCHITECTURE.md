# TenPak Architecture

## Overview

**TenPak solves two critical problems for model hosting platforms:**

### Primary: Delta Compression
Store fine-tuned models as 60-90% smaller deltas from base models.

### Secondary: Cost Optimizer
Auto-benchmark GPTQ/AWQ/bitsandbytes and select the cheapest method meeting constraints.

## Core Components

1. **Delta Compression** (`core/delta.py`) - Unique feature, 60-90% savings on fine-tunes
2. **Cost Optimizer** (`optimizer/`) - Auto-select best quantization method
3. **CLI** (`cli/`) - Delta compression + optimization commands

## Component Design

### Core: Quantization Wrapper (`core/quantization.py`)

**TenPak does NOT implement quantization** - it wraps industry-standard tools:

```python
from core import QuantizationWrapper, QUANTIZATION_PRESETS

# Unified API for all quantization methods
wrapper = QuantizationWrapper(method="gptq", bits=4, group_size=128)
quantized_model = wrapper.quantize("meta-llama/Llama-2-7b-hf")
```

**Supported Methods:**
- **AutoGPTQ** - 7-8x compression, <1% PPL, requires calibration
- **AutoAWQ** - 7-8x compression, <2% PPL, requires calibration
- **bitsandbytes NF4** - 6-7x compression, <1.5% PPL, no calibration
- **bitsandbytes INT8** - 2x compression, <0.5% PPL, conservative

**Presets:**
```python
QUANTIZATION_PRESETS = {
    "gptq_quality": QuantizationConfig(method="gptq", bits=4, group_size=128),
    "awq_balanced": QuantizationConfig(method="awq", bits=4, group_size=128),
    "bnb_nf4": QuantizationConfig(method="bitsandbytes", bits=4, quant_type="nf4"),
    # ... 8 total presets
}
```

### Core Modules (`core/`)

```
core/
├── __init__.py          # Public API exports
├── delta.py            # Delta compression (PRIMARY FEATURE)
├── calibration.py      # Stats collection for cost optimizer
└── quantization.py     # Minimal wrapper for optimizer
```

**Key APIs:**

1. **Delta Compression** (Primary)
```python
from core.delta import compress_delta, estimate_delta_savings

# Estimate savings
savings = estimate_delta_savings(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetuned_model_id="my-org/llama-chat"
)
print(f"Savings: {savings['savings_pct']:.1f}%")  # 60-90%

# Compress as delta
delta_manifest = compress_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetuned_model_id="my-org/llama-chat",
    output_dir="./delta"
)
```

2. **Cost Optimizer** (Secondary)
```python
from optimizer import optimize_model, OptimizationConstraints

constraints = OptimizationConstraints(
    max_ppl_delta=2.0,
    min_compression=5.0
)
result = optimize_model("meta-llama/Llama-2-7b-hf", constraints)
```

### Cost Optimizer (`optimizer/`)

**Secondary value proposition** - auto-select best quantization method:

```python
from optimizer import optimize_model, OptimizationConstraints

constraints = OptimizationConstraints(
    max_ppl_delta=2.0,      # <2% quality loss
    max_latency_ms=100,     # <100ms inference
    min_compression=5.0     # >5x compression
)

result = optimize_model(
    model_id="meta-llama/Llama-2-7b-hf",
    constraints=constraints,
    hardware_type="a10g"
)

print(f"Best method: {result.winner.method}")
print(f"Cost: ${result.winner.cost_per_1m_tokens}")
```

**Features:**
- Benchmark all quantization methods (GPTQ, AWQ, bitsandbytes)
- Select cheapest option meeting constraints
- Supports hardware-specific optimization (T4, A10G, A100)


### CLI (`cli/`)

**Commands:**
```bash
# PRIMARY: Delta compression for fine-tunes
tenpak delta compress meta-llama/Llama-2-7b my-org/llama-chat --output ./delta
tenpak delta estimate meta-llama/Llama-2-7b my-org/llama-chat

# SECONDARY: Cost optimizer
tenpak optimize meta-llama/Llama-2-7b --max-ppl-delta 2.0 --min-compression 5.0
```

## Data Flow

### Quantization Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   User Request                               │
│  wrapper = QuantizationWrapper(method="gptq", bits=4)       │
│  model = wrapper.quantize("meta-llama/Llama-2-7b-hf")       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Select Quantization Tool                     │
│  if method == "gptq": use AutoGPTQ                          │
│  if method == "awq": use AutoAWQ                            │
│  if method == "bitsandbytes": use bitsandbytes              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Prepare Calibration Data                     │
│  (if required by method)                                     │
│  Load dataset, tokenize, create DataLoader                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Call Quantization Tool                       │
│  AutoGPTQ.quantize() / AutoAWQ.quantize() / etc.            │
│  Tool handles all compression details                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              4. Return Quantized Model                       │
│  Model ready for inference or saving                         │
└─────────────────────────────────────────────────────────────┘
```

### Cost Optimization Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                   User Request                               │
│  result = optimize_model(model_id, constraints)              │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Generate Candidates                          │
│  candidates = [gptq_quality, awq_balanced, bnb_nf4, ...]    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Benchmark Each Candidate                     │
│  For each candidate:                                         │
│    - Quantize model                                          │
│    - Measure latency (p50, p95)                             │
│    - Measure throughput (tokens/sec)                        │
│    - Estimate cost per 1M tokens                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Select Winner                                │
│  Filter by constraints (PPL, latency, compression)           │
│  Pick cheapest option meeting all constraints                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              4. Return Optimization Result                   │
│  result.winner.method = "awq"                                │
│  result.winner.cost_per_1m_tokens = 0.15                     │
└─────────────────────────────────────────────────────────────┘
```

## Delta Compression Algorithm

**TenPak's unique feature** - compress fine-tunes as deltas:

### 1. Detect Changed Weights
```python
base_state = base_model.state_dict()
fine_state = finetuned_model.state_dict()

deltas = {}
for name in base_state:
    delta = fine_state[name] - base_state[name]
    if torch.abs(delta).max() > threshold:
        deltas[name] = delta  # Only store changed weights
```

### 2. Sparse Storage
```python
# Store only non-zero deltas
for name, delta in deltas.items():
    indices = torch.nonzero(torch.abs(delta) > threshold)
    values = delta[indices]
    # Store: (name, indices, values) instead of full tensor
```

### 3. Compression Result
```python
# Typical results:
# Full model: 13 GB
# Delta: 500 MB (96% reduction)
# Base model reference: meta-llama/Llama-2-7b-hf
```

### 4. Reconstruction
```python
def reconstruct_from_delta(base_model, delta_artifact):
    model = copy.deepcopy(base_model)
    for name, (indices, values) in delta_artifact.items():
        model.state_dict()[name][indices] += values
    return model
```


## Quantization Methods (For Cost Optimizer)

The cost optimizer benchmarks these industry-standard methods:

| Method | Compression | PPL Δ | Notes |
|--------|-------------|-------|-------|
| **GPTQ 4-bit** | 7-8x | <1% | Best quality (AutoGPTQ) |
| **AWQ 4-bit** | 7-8x | <2% | Best speed/quality (AutoAWQ) |
| **bitsandbytes NF4** | 6-7x | <1.5% | Fast, no calibration |
| **bitsandbytes INT8** | 2x | <0.5% | Conservative |

**Note:** TenPak doesn't implement these - it helps users choose between them.

## Key Design Decisions

1. **Delta Compression is Primary Value** - Unique feature, no competitor offers this (60-90% savings)
2. **Cost Optimizer is Secondary** - Helps users choose between GPTQ/AWQ/bitsandbytes
3. **Don't Compete on Infrastructure** - HF has CDN, TGI, etc. Focus on what they don't have
4. **Pure Python** - No Rust/C++ dependencies, easier integration
5. **Honest Positioning** - We're a model hub optimization tool, not a quantization library

## Performance Notes

- **Delta Compression**: CPU/PyTorch tensor ops, I/O bound (disk reading is bottleneck)
- **Cost Optimizer**: Benchmarking takes minutes (quantizes models), selection is instant

## Architecture Evolution

### v0.1.0 (Dec 2024) - Custom Rust Codecs
- Custom INT4 quantization in Rust
- FFI layer for Python integration
- Goal: 10x compression with <1% PPL
- **Result**: Achieved 4x without calibration, fragile for >6x

### v0.2.0 (Dec 2024) - Strategic Refocus
- **Pivoted** to focus on delta compression + cost optimizer
- Removed all Rust code
- Removed commodity features (HTTP streaming, inference integration, REST API)
- **Primary**: Delta compression (unique, $20-25M/year value for model hubs)
- **Secondary**: Cost optimizer (reduces user confusion)
- **Result**: Focused product with clear value proposition
