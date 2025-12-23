# Sparse Examples

Example scripts demonstrating Sparse's workflow with the wrapper architecture.

## Quick Start

### 1. Delta Compression for Fine-tunes

Store fine-tunes efficiently as deltas from base models:

```bash
python examples/delta_compression.py \
    mistralai/Mistral-7B-v0.1 \
    your-org/mistral-7b-finetuned \
    --output ./artifacts/delta \
    --quantization awq_balanced
```

Typical savings: **60-90%** storage reduction for fine-tunes!

### 2. Dataset Delta Compression

Compress derivative datasets as deltas:

```bash
python examples/dataset_delta_example.py
```

This demonstrates dataset delta compression with real examples and savings estimates.

## Available Quantization Presets

| Preset | Method | Bits | Group Size | Best For |
|--------|--------|------|------------|----------|
| `gptq_quality` | GPTQ | 4 | 128 | Best quality |
| `gptq_balanced` | GPTQ | 4 | 256 | Balanced |
| `gptq_size` | GPTQ | 4 | 512 | Max compression |
| `awq_quality` | AWQ | 4 | 128 | Best quality/ratio |
| `awq_balanced` | AWQ | 4 | 256 | Production default |
| `bnb_int8` | bitsandbytes | 8 | - | Fast, no calibration |
| `bnb_nf4` | bitsandbytes | 4 | - | Good quality |

## Python API Examples

### Basic Quantization

```python
from core import QuantizationWrapper, QUANTIZATION_PRESETS

# Use preset
config = QUANTIZATION_PRESETS["awq_balanced"]
model = QuantizationWrapper.quantize_model(
    model_id="mistralai/Mistral-7B-v0.1",
    config=config,
    device="cuda",
)

# Estimate size before quantizing
size_info = QuantizationWrapper.estimate_size("mistralai/Mistral-7B-v0.1", config)
print(f"Compression: {size_info['compression_ratio']:.2f}x")
```

### Delta Compression

```python
from core import compress_delta, reconstruct_from_delta

# Compress fine-tune as delta
delta_manifest = compress_delta(
    base_model_id="mistralai/Mistral-7B-v0.1",
    finetuned_model_id="your-org/mistral-7b-finetuned",
    output_dir="./artifacts/delta",
)

# Reconstruct full model from base + delta
model = reconstruct_from_delta(
    base_model_id="mistralai/Mistral-7B-v0.1",
    delta_path="./artifacts/delta",
)
```

## Output Structure

After running delta compression, you'll have:

```
artifacts/delta/
├── delta_manifest.json    # Delta metadata and savings
└── deltas/                # Changed layers only
    ├── layer.10.bin
    └── layer.11.bin
```

### Delta Manifest Example

```json
{
  "model_id": "your-org/mistral-7b-finetuned",
  "base_model_id": "mistralai/Mistral-7B-v0.1",
  "delta_method": "sparse_int8",
  "changed_layers": ["layer.10", "layer.11"],
  "quantization": {
    "method": "awq",
    "bits": 4
  },
  "savings": {
    "base_size_gb": 13.0,
    "finetuned_size_gb": 13.0,
    "delta_size_gb": 1.8,
    "savings_pct": 86.2,
    "compression_ratio": 7.2
  }
}
```

## What Sparse Provides

Sparse **doesn't replace** AutoGPTQ/AutoAWQ/bitsandbytes. It provides:

1. **Delta Compression** - Store fine-tunes 60-90% smaller as deltas from base models
2. **Dataset Delta Compression** - Store derivative datasets 70-90% smaller
3. **Smart Routing** - Auto-route requests to optimal models/hardware
4. **Cost Optimizer** - Generate and filter quantization candidates
5. **Unified Quantization API** - Single interface for GPTQ, AWQ, bitsandbytes

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# Quantization tool (choose one)
pip install auto-gptq     # GPTQ (CUDA required)
pip install autoawq       # AWQ (CUDA required)
pip install bitsandbytes  # Fast, CPU/CUDA

# Inference (optional)
pip install vllm          # vLLM inference
pip install accelerate    # Transformers inference
```

## Dataset Delta Compression

See `dataset_delta_example.py` for:
- Estimating savings for derivative datasets
- Compressing datasets as deltas
- Reconstructing datasets from deltas
- Real-world impact analysis

## Smart Routing & Cost Optimization

Use the Python API to test routing and optimization:

```python
from optimizer.routing import classify_request_complexity, suggest_optimal_model
from optimizer import generate_candidates, OptimizationConstraints

# Classify request complexity
complexity = classify_request_complexity("What is 2+2?", max_tokens=10)

# Get routing recommendation
decision = suggest_optimal_model(
    requested_model="meta-llama/Llama-2-70b-hf",
    prompt="What is 2+2?",
    quality_threshold=0.85,
    cost_priority=True
)

# Generate optimization candidates
candidates = generate_candidates(
    include_calibration=False,
    max_expected_ppl_delta=2.0,
    min_expected_compression=2.0
)
```

## Testing

Run comprehensive tests:

```bash
# Test with real HuggingFace models
python tests/test_real_models.py

# Test all individual features
python tests/test_individual_features.py

# Run all benchmarks
./benchmarks/run_benchmarks.sh
```

## Next Steps

- Try the examples with your own models
- Deploy to HuggingFace Spaces (see `hf_space/DEPLOYMENT.md`)
- Run the full benchmark suite
- See [../README.md](../README.md) for full documentation
