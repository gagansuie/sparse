# Sparse API Reference

**Version:** 0.2.0  
**License:** Proprietary

Complete API reference for Sparse integration.

⚡ **Performance Note:** Operations marked with ⚡ are Rust-accelerated (10-20x faster when Rust extension is installed).

---

## Table of Contents

1. [Model Delta Compression](#model-delta-compression)
2. [Dataset Delta Compression](#dataset-delta-compression)
3. [Smart Routing](#smart-routing)
4. [Cost Optimizer](#cost-optimizer)
5. [Data Types](#data-types)

---

## Model Delta Compression

### `estimate_delta_savings()`

Estimate storage savings from delta compression.

**Signature:**
```python
def estimate_delta_savings(
    base_model_id: str,
    finetune_model_id: str,
    compare_method: str = "state_dict"
) -> Dict[str, float]
```

**Parameters:**
- `base_model_id` (str): Base model identifier (e.g., "meta-llama/Llama-2-7b-hf")
- `finetune_model_id` (str): Fine-tuned model identifier
- `compare_method` (str): Method to compare models ("state_dict" or "safetensors")

**Returns:**
```python
{
    "base_size_mb": float,           # Base model size in MB
    "finetune_size_mb": float,       # Fine-tuned model size in MB
    "delta_size_mb": float,          # Estimated delta size in MB
    "savings_pct": float,            # Percentage saved (0-100)
    "estimated_compression": float,  # Compression ratio (e.g., 15.2x)
    "changed_layers": List[str],     # List of changed layer names
    "num_changed_params": int,       # Number of changed parameters
    "total_params": int              # Total parameters
}
```

**Example:**
```python
from core.delta import estimate_delta_savings

savings = estimate_delta_savings(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="my-org/llama-chat"
)

print(f"Savings: {savings['savings_pct']:.1f}%")
print(f"Compression: {savings['estimated_compression']:.2f}x")
# Output:
# Savings: 96.2%
# Compression: 26.3x
```

---

### `compress_delta()` ⚡

Compress a fine-tuned model as a delta from base model.

**⚡ Rust-Accelerated:** This function automatically uses Rust implementation when available for 10-20x speedup.

**Signature:
```python
def compress_delta(
    base_model_id: str,
    finetune_model_id: str,
    output_path: str,
    sparsity_threshold: float = 1e-6,
    compression_format: str = "safetensors"
) -> DeltaManifest
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `finetune_model_id` (str): Fine-tuned model identifier
- `output_path` (str): Output directory for delta files
- `sparsity_threshold` (float): Threshold for considering weights as "unchanged"
- `compression_format` (str): Output format ("safetensors" or "torch")

**Returns:** `DeltaManifest` object with delta metadata

**Example:**
```python
from core.delta import compress_delta

manifest = compress_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    finetune_model_id="my-org/llama-chat",
    output_path="s3://my-bucket/deltas/llama-chat/"
)

print(f"Delta size: {manifest.delta_size_mb:.1f} MB")
print(f"Compression: {manifest.compression_ratio:.2f}x")
```

---

### `reconstruct_from_delta()` ⚡

Reconstruct full model from base + delta.

**⚡ Rust-Accelerated:** Decompression is 10-15x faster with Rust extension.

**Signature:
```python
def reconstruct_from_delta(
    base_model_id: str,
    delta_path: str,
    output_path: str = None,
    device: str = "cpu"
) -> torch.nn.Module
```

**Parameters:**
- `base_model_id` (str): Base model identifier
- `delta_path` (str): Path to delta manifest
- `output_path` (str, optional): Save reconstructed model to this path
- `device` (str): Device for loading ("cpu" or "cuda")

**Returns:** Reconstructed model

**Example:**
```python
from core.delta import reconstruct_from_delta

model = reconstruct_from_delta(
    base_model_id="meta-llama/Llama-2-7b-hf",
    delta_path="s3://my-bucket/deltas/llama-chat/manifest.json",
    output_path="./reconstructed_model/"
)

# Use model for inference
outputs = model.generate(...)
```

---

## Dataset Delta Compression

### `estimate_dataset_delta_savings()`

Estimate storage savings for dataset delta compression.

**Signature:**
```python
def estimate_dataset_delta_savings(
    base_dataset_id: str,
    derivative_dataset_id: str,
    split: str = "train",
    sample_size: int = 1000
) -> DatasetDeltaStats
```

**Parameters:**
- `base_dataset_id` (str): Base dataset identifier
- `derivative_dataset_id` (str): Derivative dataset identifier
- `split` (str): Dataset split to analyze
- `sample_size` (int): Number of samples to analyze (for speed)

**Returns:** `DatasetDeltaStats` object with savings estimates

**Example:**
```python
from core.dataset_delta import estimate_dataset_delta_savings

stats = estimate_dataset_delta_savings(
    base_dataset_id="squad",
    derivative_dataset_id="squad_v2",
    sample_size=1000
)

print(f"Savings: {stats.savings_pct:.1f}%")
print(f"Shared samples: {stats.num_shared_samples}")
print(f"New samples: {stats.num_new_samples}")
```

---

### `compress_dataset_delta()`

Compress a derivative dataset as delta from base dataset.

**Signature:**
```python
def compress_dataset_delta(
    base_dataset_id: str,
    derivative_dataset_id: str,
    output_dir: str,
    splits: List[str] = None,
    compression: str = "gzip"
) -> Dict
```

**Parameters:**
- `base_dataset_id` (str): Base dataset identifier
- `derivative_dataset_id` (str): Derivative dataset identifier
- `output_dir` (str): Output directory for delta files
- `splits` (List[str], optional): Specific splits to compress (default: all)
- `compression` (str): Compression format ("gzip" or "none")

**Returns:** Manifest dict with delta metadata

**Example:**
```python
from core.dataset_delta import compress_dataset_delta

manifest = compress_dataset_delta(
    base_dataset_id="squad",
    derivative_dataset_id="squad_v2",
    output_dir="s3://my-bucket/dataset-deltas/squad_v2/"
)

print(f"Compressed splits: {list(manifest['splits'].keys())}")
print(f"Total savings: {manifest['size_stats']['savings_pct']:.1f}%")
```

---

### `reconstruct_from_dataset_delta()`

Reconstruct full dataset from base + delta.

**Signature:**
```python
def reconstruct_from_dataset_delta(
    base_dataset_id: str,
    delta_path: str,
    cache_dir: str = None
) -> datasets.Dataset
```

**Parameters:**
- `base_dataset_id` (str): Base dataset identifier
- `delta_path` (str): Path to delta manifest
- `cache_dir` (str, optional): Cache directory for reconstructed dataset

**Returns:** Reconstructed `datasets.Dataset`

**Example:**
```python
from core.dataset_delta import reconstruct_from_dataset_delta

dataset = reconstruct_from_dataset_delta(
    base_dataset_id="squad",
    delta_path="s3://my-bucket/dataset-deltas/squad_v2/manifest.json"
)

print(f"Rows: {len(dataset)}")
```

---

## Smart Routing

### `classify_request_complexity()`

Classify inference request complexity.

**Signature:**
```python
def classify_request_complexity(
    prompt: str,
    max_tokens: int = 100,
    context_length: int = 0
) -> TaskComplexity
```

**Parameters:**
- `prompt` (str): User prompt text
- `max_tokens` (int): Maximum tokens to generate
- `context_length` (int): Length of context (if any)

**Returns:** `TaskComplexity` enum (SIMPLE, MODERATE, COMPLEX, EXTREME)

**Example:**
```python
from optimizer.routing import classify_request_complexity, TaskComplexity

complexity = classify_request_complexity(
    prompt="What is 2+2?",
    max_tokens=10
)

if complexity == TaskComplexity.SIMPLE:
    print("Can use smaller model")
```

---

### `suggest_optimal_model()`

Suggest optimal model and hardware for inference request.

**Signature:**
```python
def suggest_optimal_model(
    requested_model: str,
    prompt: str,
    quality_threshold: float = 0.85,
    cost_priority: bool = True,
    max_tokens: int = 100
) -> RoutingDecision
```

**Parameters:**
- `requested_model` (str): Model user requested
- `prompt` (str): User prompt
- `quality_threshold` (float): Minimum acceptable quality (0-1)
- `cost_priority` (bool): Prioritize cost over latency
- `max_tokens` (int): Maximum tokens to generate

**Returns:** `RoutingDecision` object with recommendation

**Example:**
```python
from optimizer.routing import suggest_optimal_model

decision = suggest_optimal_model(
    requested_model="meta-llama/Llama-2-70b-hf",
    prompt="What is the capital of France?",
    quality_threshold=0.85,
    cost_priority=True
)

print(f"Recommended: {decision.recommended_model}")
print(f"Hardware: {decision.recommended_hardware}")
print(f"Cost: ${decision.estimated_cost_per_1m_tokens:.4f}/1M tokens")
print(f"Quality: {decision.quality_score:.2%}")
print(f"Reasoning: {decision.reasoning}")

# Output:
# Recommended: meta-llama/Llama-2-7b-hf
# Hardware: T4
# Cost: $0.1500/1M tokens
# Quality: 88.0%
# Reasoning: Simple question can use smaller model. Saves 90% cost.
```

---

### `estimate_routing_savings()`

Estimate annual savings from smart routing.

**Signature:**
```python
def estimate_routing_savings(
    current_requests_per_day: int,
    avg_cost_per_request: float = 0.001,
    optimization_rate: float = 0.25,
    avg_savings_per_optimized: float = 0.30
) -> Dict[str, float]
```

**Parameters:**
- `current_requests_per_day` (int): Current daily inference requests
- `avg_cost_per_request` (float): Average cost per request ($)
- `optimization_rate` (float): % of requests that can be optimized (0-1)
- `avg_savings_per_optimized` (float): Average savings per optimized request (0-1)

**Returns:**
```python
{
    "annual_requests": int,
    "current_annual_cost_usd": float,
    "annual_savings_usd": float,
    "savings_pct": float,
    "optimization_rate": float
}
```

**Example:**
```python
from optimizer.routing import estimate_routing_savings

savings = estimate_routing_savings(
    current_requests_per_day=10_000_000,
    avg_cost_per_request=0.001,
    optimization_rate=0.25
)

print(f"Annual savings: ${savings['annual_savings_usd'] / 1_000_000:.1f}M")
# Output: Annual savings: $6.8M
```

---

## Cost Optimizer

### `optimize_model()`

Find optimal quantization method for constraints.

**Signature:**
```python
def optimize_model(
    model_id: str,
    constraints: OptimizationConstraints,
    calibration_samples: int = 128
) -> OptimizationResult
```

**Parameters:**
- `model_id` (str): Model to optimize
- `constraints` (OptimizationConstraints): Quality/latency constraints
- `calibration_samples` (int): Samples for calibration

**Returns:** `OptimizationResult` with best candidate

**Example:**
```python
from optimizer import optimize_model, OptimizationConstraints

constraints = OptimizationConstraints(
    max_ppl_delta=2.0,
    max_latency_p99_ms=100
)

result = optimize_model(
    model_id="meta-llama/Llama-2-7b-hf",
    constraints=constraints
)

if result.winner:
    print(f"Best: {result.winner.candidate_name}")
    print(f"Compression: {result.winner.compression_ratio:.2f}x")
    print(f"Cost: ${result.winner.cost_per_1m_tokens:.4f}/1M tokens")
```

---

## Data Types

### `DeltaManifest`

```python
@dataclass
class DeltaManifest:
    base_model_id: str
    finetune_model_id: str
    delta_size_mb: float
    compression_ratio: float
    changed_layers: List[str]
    sparsity: float
    format: str
    created_at: str
```

### `DatasetDeltaStats`

```python
@dataclass
class DatasetDeltaStats:
    base_dataset_id: str
    derivative_dataset_id: str
    base_size_mb: float
    derivative_size_mb: float
    delta_size_mb: float
    savings_pct: float
    num_shared_samples: int
    num_new_samples: int
    num_modified_samples: int
```

### `RoutingDecision`

```python
@dataclass
class RoutingDecision:
    recommended_model: str
    recommended_hardware: HardwareType
    estimated_cost_per_1m_tokens: float
    estimated_latency_p99_ms: float
    quality_score: float
    reasoning: str
    cost_savings_pct: float
    alternatives: List[ModelSpec]
```

### `TaskComplexity`

```python
class TaskComplexity(Enum):
    SIMPLE = "simple"       # <50 tokens, simple question
    MODERATE = "moderate"   # 50-200 tokens, moderate complexity
    COMPLEX = "complex"     # 200-500 tokens, complex task
    EXTREME = "extreme"     # >500 tokens, very complex
```

### `HardwareType`

```python
class HardwareType(Enum):
    T4 = "t4"                    # $0.50/hour
    A10G = "a10g"                # $1.00/hour
    A100_40GB = "a100-40gb"      # $3.00/hour
    A100_80GB = "a100-80gb"      # $5.00/hour
    H100 = "h100"                # $8.00/hour
```

---

## Error Handling

All functions raise appropriate exceptions:

```python
from core.delta import compress_delta, DeltaCompressionError

try:
    manifest = compress_delta(base_id, finetune_id, output_path)
except DeltaCompressionError as e:
    print(f"Compression failed: {e}")
except FileNotFoundError:
    print("Model files not found")
except MemoryError:
    print("Insufficient memory")
```

---

## Best Practices

1. **Always estimate before compressing**
   ```python
   savings = estimate_delta_savings(base, finetune)
   if savings['savings_pct'] > 60:
       compress_delta(base, finetune, output)
   ```

2. **Use appropriate quality thresholds**
   ```python
   # Strict quality for production
   decision = suggest_optimal_model(..., quality_threshold=0.90)
   
   # More aggressive for dev/testing
   decision = suggest_optimal_model(..., quality_threshold=0.80)
   ```

3. **Monitor savings in production**
   ```python
   # Track actual vs estimated
   estimated = estimate_delta_savings(...)
   manifest = compress_delta(...)
   actual_ratio = manifest.compression_ratio
   
   if abs(actual_ratio - estimated['estimated_compression']) > 2.0:
       alert_team("Estimation error")
   ```

---

## Performance Optimization

### Rust Acceleration

For 10-20x faster delta compression operations, install the Rust extension:

```bash
cd rust/
bash build.sh
```

**Accelerated Operations:**
- ⚡ `compress_delta()` - 10-20x faster sparse compression
- ⚡ `reconstruct_from_delta()` - 10-15x faster decompression  
- ⚡ INT8 quantization - 5-10x faster

**Check if Rust is available:**

```python
from core.delta_rust import is_rust_available, get_rust_info

if is_rust_available():
    print("✓ Rust acceleration enabled")
    info = get_rust_info()
    print(f"Features: {', '.join(info['features'])}")
else:
    print("Using Python fallback (still works!)")
```

**No code changes needed** - your existing API calls automatically use Rust when available.

To install Rust acceleration: `cd rust/ && bash build.sh`

---

**Questions?** See `docs/INTEGRATION_GUIDE.md` or contact support.
