# TenPak Architecture

## Overview

TenPak is an **LLM quantization orchestration platform** built entirely in Python:

1. **Quantization Wrapper** (`core/quantization.py`) - Unified API for AutoGPTQ, AutoAWQ, bitsandbytes
2. **Delta Compression** (`core/delta.py`) - Efficient storage for fine-tunes
3. **Cost Optimizer** (`optimizer/`) - Auto-select best quantization method
4. **HTTP Streaming** (`artifact/http_streaming.py`) - CDN-friendly remote artifacts
5. **Inference Integration** (`inference/`) - vLLM/TGI deployment helpers
6. **REST API** (`studio/`) - Compression-as-a-service
7. **CLI** (`cli/`) - Command-line interface

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
├── quantization.py      # Wrapper for AutoGPTQ/AutoAWQ/bitsandbytes
├── delta.py            # Delta compression for fine-tunes
├── calibration.py      # Stats collection (Fisher, Hessian) - legacy
└── allocation.py       # Bit allocation strategies - legacy
```

**Key APIs:**

1. **Quantization Wrapper**
```python
wrapper = QuantizationWrapper(method="awq", bits=4)
model = wrapper.quantize("model_id", calibration_data=dataset)
```

2. **Delta Compression**
```python
from core.delta import compress_delta, estimate_delta_savings

# Estimate savings
savings = estimate_delta_savings(base_model, finetuned_model)
# 60-90% size reduction

# Compress as delta
delta_artifact = compress_delta(base_model, finetuned_model)
```

### Optimizer (`optimizer/`)

**TenPak's unique value** - auto-select best quantization method:

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

### Inference Integration (`inference/`)

```python
from inference.vllm_integration import TenPakVLLMLoader

# One-line vLLM deployment
loader = TenPakVLLMLoader(
    artifact_path="llama-7b.tnpk",
    tensor_parallel_size=4
)
engine = loader.create_engine()
```

**Supported Engines:**
- vLLM (tensor parallelism, paged attention)
- TGI (Text Generation Inference)

### Studio API (`studio/`)

REST API for compression-as-a-service.

```
studio/
├── api.py           # FastAPI endpoints
├── jobs.py          # Async job runner
└── storage.py       # Artifact storage
```

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compress` | POST | Quantize a model |
| `/optimize` | POST | **Auto-select best method** |
| `/delta/compress` | POST | **Compress fine-tune as delta** |
| `/status/{id}` | GET | Poll job progress |
| `/artifact/{id}` | GET | Download result |

### CLI (`cli/`)

**Commands:**
```bash
# Quantize with auto-optimization
tenpak optimize <model_id> --max-ppl-delta 2.0

# Quantize with specific method
tenpak quantize <model_id> --method gptq --bits 4

# Delta compression for fine-tunes
tenpak delta <finetuned_model> --base <base_model>

# Serve with vLLM
tenpak serve <artifact_path> --engine vllm
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

### Artifact Format (`artifact/`)

Chunked, signed, HTTP-streamable artifact format.

```
artifact/
├── format.py            # Artifact metadata and manifest
├── streaming.py         # Local streaming load/verify
├── signing.py           # HMAC + GPG signing
└── http_streaming.py    # HTTP streaming for remote artifacts
```

## Artifact Structure

```
<artifact_dir>/
├── manifest.json      # Metadata + quantization info
│   {
│     "version": "1.0",
│     "model_id": "meta-llama/Llama-2-7b-hf",
│     "quantization": {
│       "method": "awq",
│       "bits": 4,
│       "group_size": 128,
│       "model_path": "./quantized_model"
│     },
│     "delta": {  # Optional - if this is a fine-tune
│       "base_model_id": "meta-llama/Llama-2-7b-hf",
│       "delta_method": "sparse_int8",
│       "changed_layers": ["layers.10", "layers.11"]
│     },
│     "optimization": {  # Optional - if cost-optimized
│       "selected_method": "awq_balanced",
│       "cost_per_1m_tokens": 0.15,
│       "latency_p50_ms": 45.2
│     },
│     "chunks": [...],
│     "signature": {...}
│   }
├── quantized_model/   # Quantized model weights
│   ├── config.json
│   ├── model.safetensors
│   └── quantization_config.json
└── signature.sig      # Optional GPG signature
```

## Quantization Presets

| Preset | Method | Bits | Group Size | Compression | PPL Δ | Calibration |
|--------|--------|------|------------|-------------|-------|-------------|
| `gptq_quality` | GPTQ | 4 | 128 | 7-8x | <1% | Required |
| `awq_balanced` | AWQ | 4 | 128 | 7-8x | <2% | Required |
| `bnb_nf4` | bitsandbytes | 4 | - | 6-7x | <1.5% | Optional |
| `bnb_int8` | bitsandbytes | 8 | - | 2x | <0.5% | No |
| `gptq_aggressive` | GPTQ | 3 | 128 | 10-12x | 3-5% | Required |
| `awq_fast` | AWQ | 4 | 64 | 6-7x | <2% | Required |

## Key Design Decisions

1. **Wrapper Architecture** - Don't compete with AutoGPTQ/AutoAWQ, orchestrate them
2. **Delta Compression is Unique** - No one else offers this at scale (60-90% savings on fine-tunes)
3. **Cost Optimizer is Unique** - Auto-benchmark all methods, pick cheapest meeting constraints
4. **Pure Python** - No Rust/C++ dependencies, easier maintenance and deployment
5. **Focus on Orchestration** - HTTP streaming, vLLM/TGI integration, artifact format
6. **Tensor Parallelism** - Pass-through to wrapped tools and inference engines

## Performance Notes

- **Quantization**: Delegated to AutoGPTQ/AutoAWQ (highly optimized CUDA kernels)
- **Delta Compression**: CPU/PyTorch tensor ops, I/O bound (disk reading is bottleneck)
- **HTTP Streaming**: Network I/O bound, Python overhead negligible
- **Cost Optimizer**: Orchestration is <0.1s in 5-minute quantization workflow

## Architecture Evolution

### v0.1.0 (Dec 2024) - Custom Rust Codecs
- Custom INT4 quantization in Rust
- FFI layer for Python integration
- Goal: 10x compression with <1% PPL
- **Result**: Achieved 4x without calibration, fragile for >6x

### v0.2.0 (Dec 2024) - Wrapper Architecture
- **Pivoted** to wrap AutoGPTQ/AutoAWQ/bitsandbytes
- Removed all Rust code
- Added delta compression (unique)
- Added cost optimizer (unique)
- Added HTTP streaming and vLLM/TGI integration
- **Result**: Production-ready orchestration platform
