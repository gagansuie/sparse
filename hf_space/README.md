# Sparse - Model Optimization Testing Space

Test Sparse's model optimization features with large language models (up to 70B parameters).

## Features

This Space allows you to test:

### 📦 Quantization Size Estimation
- Test any HuggingFace model (including 70B models)
- Multiple quantization presets (GPTQ, AWQ, bitsandbytes)
- Accurate compression ratio predictions
- Works without downloading full models

### 🎯 Smart Routing
- Classify request complexity (SIMPLE, MODERATE, COMPLEX, EXTREME)
- Get model recommendations based on prompt
- Optimize cost while maintaining quality
- Hardware routing suggestions

### 💰 Cost Optimizer
- Generate optimization candidates
- Apply quality/latency/throughput constraints
- Compare different quantization methods
- Find optimal configuration for your use case

### 💵 Savings Estimation
- Estimate annual/monthly cost savings
- Based on your request volume
- Configurable optimization rate
- Real-world ROI calculations

## How It Works

Sparse uses advanced techniques to optimize LLM inference:

1. **Delta Compression**: Store fine-tuned models as sparse deltas from base models (up to 95% smaller)
2. **Smart Routing**: Automatically route requests to optimal model/hardware combinations
3. **Quantization**: Reduce model size with minimal quality loss (3-8x compression)
4. **Cost Optimization**: Find the best configuration for your constraints

## Running Locally

To run this Space locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/sparse
cd sparse

# Install dependencies
pip install -r requirements-space.txt

# Run the app
python app.py
```

## Testing with 70B Models

This Space is designed to work with HuggingFace's infrastructure, which provides:
- GPU access for larger models
- Fast model metadata loading
- Accurate size estimation without full downloads

Try these models:
- `meta-llama/Llama-2-70b-hf` (70B parameters)
- `meta-llama/Llama-2-13b-hf` (13B parameters)
- `meta-llama/Llama-2-7b-hf` (7B parameters)
- `mistralai/Mistral-7B-v0.1` (7B parameters)

## Learn More

- **GitHub**: [Sparse Repository](https://github.com/yourusername/sparse)
- **Documentation**: Full API reference and integration guides
- **Benchmarks**: See benchmarks/BENCHMARK_RESULTS.md for test results
- **Paper**: Read about the techniques behind Sparse

## License

**Demo Only** - This Space demonstrates proprietary software capabilities.  
Full software is licensed under a proprietary license. Contact gagan.suie@sparselabs.ai for licensing.
