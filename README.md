tenpak – Model Artifact Compressor
====================================

tenpak is a Rust library and CLI for compressing and storing model artifacts
(tensors) in a versioned, tensor-aware binary format.

The goal is to provide a small, focused component that a platform (Azure AI,
NVIDIA NGC, etc.) could embed in its model storage layer to shrink checkpoint
storage and distribution costs using simple, fast quantization-based
compression for model weights.

## Features

- **Versioned artifact format**
  - Each artifact carries a `version` and `codec` string so the format can
    evolve safely.
- **Multiple codecs**
  - `int8_sym_v1`: symmetric per-tensor int8.
  - `int4_sym_v1`: symmetric per-tensor int4, packing two 4-bit values per
    byte (up to ~8x smaller than FP32 per weight).
- **Shape validation and safety**
  - Validates that `prod(shape) == data.len()` for all tensors.
  - Returns structured errors instead of panicking.
- **CLI with compression, planning, and quality metrics**
  - `compress` supports `--codec` to choose a codec.
  - Prints input vs artifact size and compression ratio.
  - Computes reconstruction error metrics (MSE, MAE, max abs error) between
    original and decompressed tensors.
  - `plan` compares codecs under a max-MAE constraint and writes a JSON plan.
  - `bench` benchmarks codecs and prints a size/error table.
  - `inspect` prints a human-readable summary of an artifact.
- **Base + delta storage**
  - `delta` builds a delta artifact relative to a base artifact using an L1
    threshold.
  - `materialize` reconstructs a full artifact from a base and delta artifact.
- **Unit tests**
  - Round-trip tests for simple bundles.
  - Error-path tests for shape mismatch.

## Concepts

tenpak works with **bundles** of named float32 tensors:

- Each tensor has a `name`, `shape` and flat `data` array of `f32` values.
- A bundle is a JSON file containing an array of such tensors.

tenpak converts these into an **artifact**:

- Each tensor is quantized to **int8** with a **per-tensor scale**:
  - Find `max_abs = max(|x|)` over the tensor.
  - `scale = max_abs / 127.0` (or `1.0` if `max_abs == 0`).
  - Store `q = clamp(round(x / scale), -127, 127)` as `i8`.
- The artifact is a binary blob (bincode-serialized) containing:
  - A top-level `version` and `codec` field
  - Tensor names
  - Shapes
  - Scales
  - Quantized data

## JSON bundle format

Input bundles are JSON files of the form:

```json
{
  "tensors": [
    {
      "name": "dense.weight",
      "shape": [2, 2],
      "data": [0.1, 0.2, -0.3, 0.4]
    },
    {
      "name": "dense.bias",
      "shape": [2],
      "data": [0.01, -0.02]
    }
  ]
}
```

All tensor data is flat; the length of `data` should equal the product of
`shape`.

## Building

From the `tenpak` directory:

```bash
cargo build
```

This builds both the library and the `tenpak` CLI binary.

## CLI usage

### Compress a JSON bundle into an artifact

```bash
cargo run -- \
  compress \
  --input examples/simple_bundle.json \
  --output examples/simple_bundle.tenpak \
  --codec int8_sym_v1
```

- Reads `examples/simple_bundle.json` (a float32 bundle).
- Writes a binary artifact `examples/simple_bundle.tenpak` containing the
  quantized tensors.

To use the 4-bit codec instead:

```bash
cargo run -- \
  compress \
  --input examples/simple_bundle.json \
  --output examples/simple_bundle_int4.tenpak \
  --codec int4_sym_v1
```

### Decompress an artifact back into JSON

```bash
cargo run -- \
  decompress \
  --input examples/simple_bundle.tenpak \
  --output examples/restored_bundle.json
```

- Reads a binary artifact.
- Writes a JSON bundle with reconstructed float32 values.

### Plan compression (choose codec under error constraint)

```bash
cargo run -- \
  plan \
  --input examples/simple_bundle.json \
  --output examples/simple_bundle.plan.json \
  --max-mae 0.01
```

This evaluates built-in codecs (currently `int8_sym_v1` and `int4_sym_v1`)
against the input bundle, measuring size and reconstruction error metrics, and
chooses the smallest artifact that satisfies the MAE constraint (or the lowest
MAE if none satisfy it).

### Inspect an artifact

```bash
cargo run -- \
  inspect \
  --input examples/simple_bundle.tenpak
```

This prints version, codec, number of tensors, and a short per-tensor summary.

### Create a delta artifact

```bash
cargo run -- \
  delta \
  --base examples/base.tenpak \
  --variant examples/ft_bundle.json \
  --output examples/ft_delta.tenpak \
  --epsilon 0.001
```

This command:

- Decompresses `base.tenpak`.
- Compares it with `ft_bundle.json` (a variant bundle) using an L1 threshold
  `epsilon` per tensor.
- Produces a delta artifact containing only tensors that differ materially
  from the base.

### Materialize a full artifact from base + delta

```bash
cargo run -- \
  materialize \
  --base examples/base.tenpak \
  --delta examples/ft_delta.tenpak \
  --output examples/ft_full.tenpak
```

This reconstructs a full artifact where tensors from the delta override those
in the base (and new tensors are added).

### Benchmark codecs on a bundle

```bash
cargo run -- \
  bench \
  --input examples/simple_bundle.json
```

This prints a small table showing, for each codec, the artifact size and
reconstruction error metrics.

## How this maps to a real platform

In a production setting (e.g., Azure AI or NVIDIA NGC), a system like tenpak
would be integrated into the **model artifact storage and loading layer**:

- **Storage:**
  - Compress checkpoints and fine-tune variants before storing them.
  - Use delta formats to store many variants efficiently.
- **Distribution:**
  - Ship compressed artifacts to regions/clusters.
  - Decode on load when bringing models into GPU memory, or run partially in
    quantized form when supported.

This repository intentionally keeps the codec simple and readable so you can
walk through the code and discuss how more advanced techniques (4-bit
quantization, structured sparsity, deltas, streaming decode) would extend the
same basic idea.

## FFI and integration surface (C/C++/Python)

tenpak exposes a small C ABI so other languages can call the Rust core
without knowing Rust:

- **Compression entrypoint:**
  - `tenpak_compress_json_bundle(json_ptr, json_len, codec_ptr, out_artifact_ptr, out_artifact_len, out_err_ptr)`
  - Takes a JSON bundle (as described above) plus an optional codec string.
  - Returns a pointer + length for the serialized artifact or an error string.
- **Decompression entrypoint:**
  - `tenpak_decompress_artifact_to_json(artifact_ptr, artifact_len, out_json_ptr, out_json_len, out_err_ptr)`
  - Takes an artifact blob and returns a JSON bundle.
- **Memory management helpers:**
  - `tenpak_free_buffer(ptr, len)` to free buffers returned by tenpak.
  - `tenpak_free_cstring(ptr)` to free error strings.

This interface is designed so that C, C++, Python (via `ctypes`/`cffi`), or
other languages can integrate tenpak as a drop-in component.

### Example Python wrapper design (PyTorch / HF)

A thin Python layer can sit on top of this ABI to integrate with existing
model tooling:

- **State dict to bundle:**
  - Walk a PyTorch `state_dict` and build a Python dict matching the JSON bundle
    schema (`{"tensors": [{"name": ..., "shape": ..., "data": [...]}, ...]}`).
  - Serialize this dict with `json.dumps` and pass the bytes into
    `tenpak_compress_json_bundle` via `ctypes`.
- **Artifact storage:**
  - Store the returned artifact blob as the checkpoint representation (optionally
    alongside metadata describing which codec/epsilon was used).
- **Evaluation harness:**
  - For accuracy experiments, decompress with
    `tenpak_decompress_artifact_to_json`, reconstruct tensors back into a
    PyTorch `state_dict`, and run the standard evaluation loop (perplexity,
    downstream tasks, etc.).
- **Delta workflow:**
  - Use the CLI or Rust APIs to create base + delta artifacts for fine-tunes,
    then load materialized artifacts into PyTorch for inference or evaluation.

This design keeps tenpak as a self-contained Rust engine, but makes it easy
for a platform team (Azure AI, NVIDIA NGC, Meta) to plug it into their existing
PyTorch/C++/Python infrastructure without adopting Rust across the stack.

### Example storage economics for fine-tunes (base + delta)

tenpak is most powerful when you have one large base model and many
fine-tuned variants. Instead of storing full checkpoints for each variant, you
store a compressed base artifact plus small deltas per fine-tune.

Conceptually, you can compare:

| Variant                     | Files stored                         | Total on-disk size             | Notes                                      |
|-----------------------------|--------------------------------------|--------------------------------|--------------------------------------------|
| Full FP fine-tune           | `base_fp.pt` + `ft_fp.pt`           | `S_base_fp + S_ft_fp`          | Two full-precision checkpoints.           |
| Full tenpak fine-tune        | `base_fp.pt` + `ft_int4.tenpak`  | `S_base_fp + S_ft_int4`        | Compress the fine-tune only.              |
| tenpak base + delta (A)      | `base_int4.tenpak` + `ft_delta`  | `S_base_int4 + S_delta`        | Compressed base + small variant delta.    |

For a fleet of `N` fine-tunes, the comparison becomes:

- **Full FP:** `N * S_ft_fp` additional bytes beyond the base model.
- **tenpak base + delta:** `S_base_int4 + N * S_delta` for the entire
  family, where `S_delta` is often only a small fraction of a full model.

This is the kind of scaling story that matters to large platforms: as the
number of variants grows (thousands of fine-tunes per base), the storage and
replication savings from base+delta artifacts compound, while decode remains a
simple, local operation at load time.

## Run the full evaluation pipeline

### Using the CLI (recommended)

The easiest way to run the full evaluation pipeline is using the built-in `runeval` command:

```bash
# Build the CLI
cargo build --release

# Run evaluation with default settings (gpt2, 128 samples)
./target/release/tenpak runeval

# Or customize the model and sample count
./target/release/tenpak runeval --model gpt2 --samples 256
```

The `runeval` command will:
- Check for Python installation
- Download the specified model automatically
- Compress the model with int8 and int4 codecs
- Compute perplexity metrics
- Create simulated fine-tune and delta artifacts
- **Print the updated README to stdout** (copy and paste into README.md)

### Using the shell script (alternative)

You can also run the evaluation using the orchestration script:

```bash
cd tenpak

# Make orchestration script executable
chmod +x scripts/run_full_eval.sh

# Optional: choose model (default is gpt2)
export TENPAK_EVAL_MODEL="gpt2"

# Run the full pipeline
./scripts/run_full_eval.sh
```

### Running on EC2

#### Option 1: Using packaged binary (from CI/CD)

```bash
# Copy tarball to EC2 (must include scripts/ directory - see CICD_PACKAGING.md)
scp tenpak.tar.gz ubuntu@your-ec2-instance:/mnt/

# SSH to EC2 and extract
ssh ubuntu@your-ec2-instance
cd /mnt
tar -xzf tenpak.tar.gz

# Run evaluation
cd tenpak
./bin/tenpak runeval

# The updated README will be printed - copy and paste it back
```

#### Option 2: From source

```bash
# Copy the entire project to EC2
scp -r tenpak ubuntu@your-ec2-instance:/home/ubuntu/

# SSH to EC2 and run
ssh ubuntu@your-ec2-instance
cd /home/ubuntu/tenpak

# Build and run
cargo build --release
./target/release/tenpak runeval
```

**Note:** The packaged binary must include the `scripts/` directory. See `CICD_PACKAGING.md` for details.

## Results (fill in with your own evals)

Once you have run the evaluation plan (e.g., TinyLlama + Wikitext-2), you can
summarize the tradeoffs in a buyer-friendly way.

### Codec vs. quality for a single model

Example table structure:

| Variant           | On-disk size (GB) | Compression vs FP | Perplexity (↓) | Δ Perplexity | Task accuracy | Δ Accuracy |
|-------------------|-------------------|-------------------|----------------|--------------|---------------|-----------|
| FP baseline       |                   | 1.0×              |                |              |               |           |
| tenpak int8        |                   |                   |                |              |               |           |
| tenpak int4        |                   |                   |                |              |               |           |

You would fill in the sizes, perplexity and any downstream metrics from your
Python evaluation harness.

### Base + delta fine-tune storage

Example table structure for one base model + one fine-tune:

| Variant                     | Files stored                         | Total on-disk size             | Notes                                      |
|-----------------------------|--------------------------------------|--------------------------------|--------------------------------------------|
| Full FP fine-tune           | `base_fp.pt` + `ft_fp.pt`           | `S_base_fp + S_ft_fp`          | Two full-precision checkpoints.           |
| Full tenpak fine-tune        | `base_fp.pt` + `ft_int4.tenpak`  | `S_base_fp + S_ft_int4`        | Compress the fine-tune only.              |
| tenpak base + delta (A)      | `base_int4.tenpak` + `ft_delta`  | `S_base_int4 + S_delta`        | Compressed base + small variant delta.    |

For a fleet of many fine-tunes, you can extend this to show how the base+delta
layout scales versus storing full checkpoints per variant.
