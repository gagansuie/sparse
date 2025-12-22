# TenPak Examples

Example scripts demonstrating TenPak's workflow with the wrapper architecture.

## Quick Start

### 1. Quantize and Serve a Model

```bash
# Install dependencies (choose your quantization tool)
pip install auto-gptq  # or autoawq, or bitsandbytes
pip install vllm  # optional, for serving

# Quantize Mistral-7B with AWQ
python examples/quantize_and_serve.py mistralai/Mistral-7B-v0.1 \
    --preset awq_balanced \
    --output ./artifacts

# Quantize and serve immediately
python examples/quantize_and_serve.py mistralai/Mistral-7B-v0.1 \
    --preset awq_balanced \
    --serve \
    --port 8000
```

### 2. Cost Optimization

Automatically test multiple quantization methods and select the best:

```bash
python examples/optimize_cost.py mistralai/Mistral-7B-v0.1 \
    --max-ppl-delta 2.0 \
    --min-compression 4.0 \
    --calibration \
    --output ./artifacts/optimized
```

This will:
- Generate candidates (GPTQ, AWQ, bitsandbytes variants)
- Benchmark each on your hardware
- Select the best method for your constraints
- Save optimization results in artifact manifest

### 3. Delta Compression for Fine-tunes

Store fine-tunes efficiently as deltas from base models:

```bash
python examples/delta_compression.py \
    mistralai/Mistral-7B-v0.1 \
    your-org/mistral-7b-finetuned \
    --output ./artifacts/delta \
    --quantization awq_balanced
```

Typical savings: **60-90%** storage reduction for fine-tunes!

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

### HTTP Streaming

```python
from artifact.http_streaming import HTTPArtifactStreamer

# Stream from CDN
streamer = HTTPArtifactStreamer("https://cdn.example.com/artifacts/model-123")
manifest = streamer.load_manifest()

# Get specific chunks
chunk_data = streamer.get_chunk_by_name("layer.0.weight")

# Stream all chunks
for chunk_meta, chunk_data in streamer.stream_all_chunks():
    print(f"Loading {chunk_meta.name}: {chunk_meta.size} bytes")
```

### vLLM Integration

```python
from inference.vllm_integration import TenPakVLLMLoader

# Create vLLM engine from artifact
engine = TenPakVLLMLoader.create_vllm_engine(
    artifact_path="./artifacts/mistral-7b",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
)

# Generate
outputs = engine.generate(["Hello, how are you?"], max_tokens=50)
print(outputs[0].outputs[0].text)

# Or serve OpenAI-compatible API
TenPakVLLMLoader.serve_with_vllm(
    artifact_path="./artifacts/mistral-7b",
    host="0.0.0.0",
    port=8000,
)
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

## Artifact Structure

After running these examples, you'll have artifacts with this structure:

```
artifacts/
├── mistral-7b/
│   ├── manifest.json          # TenPak metadata
│   ├── quantization_config.json  # Tool-specific config
│   └── model_weights/         # Quantized weights
│       ├── model.safetensors
│       └── ...
└── delta/
    ├── manifest.json          # Includes delta metadata
    └── deltas/
        ├── layer.10.bin       # Changed layers only
        └── layer.11.bin
```

### Manifest Structure

```json
{
  "version": "1.0",
  "model_id": "mistralai/Mistral-7B-v0.1",
  "quantization": {
    "method": "awq",
    "bits": 4,
    "group_size": 256,
    "zero_point": true
  },
  "compression_ratio": 7.8,
  "delta": {
    "base_model_id": "mistralai/Mistral-7B-v0.1",
    "changed_layers": ["layer.10", "layer.11"],
    "savings_pct": 85.0
  },
  "optimization": {
    "selected_method": "awq_balanced",
    "candidates_tested": ["gptq_quality", "awq_balanced", "bnb_nf4"],
    "latency_p50_ms": 45.2,
    "throughput_tps": 120.5
  }
}
```

## What TenPak Adds

TenPak **doesn't replace** AutoGPTQ/AutoAWQ/bitsandbytes. It adds:

1. **Delta Compression** - Efficient fine-tune storage
2. **Cost Optimizer** - Auto-benchmark and select best method
3. **Streaming Artifacts** - HTTP-streamable, chunked format
4. **Inference Integration** - vLLM/TGI helpers
5. **Enterprise Features** - Signing, verification, monitoring

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

## Next Steps

- See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for architecture details
- See [../docs/ROADMAP.md](../docs/ROADMAP.md) for future plans
- See [../README.md](../README.md) for full documentation
