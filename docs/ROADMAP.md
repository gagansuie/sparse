# TenPak Studio Roadmap

**Vision:** Operationalize quantization for Hugging Face — potential **$700M-$2B/yr** in savings and competitive advantage.

This document tracks the three major features that would make TenPak a compelling acquisition target for HF.

---

## Value Breakdown

| Feature | Conservative | Aggressive | Notes |
|---------|--------------|------------|-------|
| Cost Optimizer | $300M/yr | $1B/yr | Biggest lever - inference costs |
| Delta Compression | $100M/yr | $300M/yr | Fine-tune storage savings |
| Streaming Artifact | $100M ARR | $500M ARR | Efficient distribution |
| **Total** | **$700M/yr** | **$2B+/yr** | |

---

## Status Overview

| Feature | Status | Priority | Est. Effort |
|---------|--------|----------|-------------|
| [1. Cost Optimizer](#1-automatic-cost-per-token-optimizer) | 🟢 **Complete** | 🥇 **Highest** | 4-6 weeks |
| [2. Delta Compression](#2-delta-compression-for-fine-tunes) | � **Complete** | 🥈 High | 2-3 weeks |
| [3. Streamable Artifact](#3-hub-native-streamable-serving-artifact) | � **Complete** | 🥉 High | 3-4 weeks |

---

## 1. Delta Compression for Fine-tunes

### The Gap

HF Hub dedupe is currently "identical file/object" style (Git/LFS-like). It doesn't natively store:
> "This fine-tune is 98% the same as base weights — store the delta efficiently"

in a first-class, **quantization-aware** way.

### What We'll Build

- [ ] **Delta detection**: Compare fine-tune to base model, identify changed layers
- [ ] **Quantization-aware delta**: Store deltas at appropriate precision (INT8 for small deltas, INT4 for large)
- [ ] **Tensor-block level dedupe**: Content-address at tensor block level, not file level
- [ ] **Reconstruction API**: Load base + delta → full model seamlessly

### Technical Approach

```
Fine-tune storage:
1. Load base model weights
2. For each layer:
   - Compute delta = fine_tune_weight - base_weight
   - If ||delta|| < threshold: store as sparse INT8
   - If ||delta|| > threshold: store as full INT4
3. Generate manifest: {base_model, layer_deltas[], compression_ratio}

Reconstruction:
1. Load base model
2. Apply deltas in-place
3. Return ready-to-use model
```

### Success Metrics

| Metric | Target |
|--------|--------|
| Storage reduction for LoRA fine-tunes | 95%+ vs full copy |
| Storage reduction for full fine-tunes | 50-80% vs full copy |
| Reconstruction time | <5s for 7B model |
| Quality loss | 0% (lossless delta) |

### Files to Create

- [ ] `tenpak/core/delta.py` - Delta compression/decompression
- [ ] `tenpak/studio/api.py` - Add `/delta/compress`, `/delta/reconstruct` endpoints
- [ ] `tenpak/cli/main.py` - Add `tenpak delta` command

---

## 2. Hub-Native Streamable Serving Artifact

### The Gap

There isn't one HF-blessed container that is simultaneously:
- ✅ Chunked for partial fetch
- ✅ Content-addressed for dedupe
- ✅ Quantization-native
- ✅ Signed/attestable for enterprise
- ✅ Directly consumable by serving stack without bespoke conversions

### What We'll Build

- [ ] **TenPak Artifact Format (.tnpk)**: Single container format
- [ ] **Chunked storage**: Stream layers on demand
- [ ] **Content addressing**: SHA256 per chunk for dedupe
- [ ] **Signing**: GPG/Sigstore attestation support
- [ ] **Direct inference**: Zero-copy load into vLLM/TGI

### Artifact Format Spec

```
tenpak_artifact_v1/
├── manifest.json           # Metadata, checksums, layer index
│   {
│     "version": "1.0",
│     "model_id": "mistralai/Mistral-7B-v0.1",
│     "codec": "int4_awq_v1",
│     "chunks": [
│       {"name": "embed", "sha256": "abc...", "offset": 0, "size": 1024},
│       {"name": "layers.0", "sha256": "def...", "offset": 1024, "size": 4096},
│       ...
│     ],
│     "signature": "...",
│     "attestation": {...}
│   }
├── chunks/
│   ├── 0000.bin            # Embeddings
│   ├── 0001.bin            # Layer 0
│   ├── ...
│   └── 00XX.bin            # LM head
└── signature.sig           # GPG/Sigstore signature
```

### Streaming API

```python
# Stream only the layers you need
from tenpak import TenPakArtifact

artifact = TenPakArtifact.from_hub("tenpak/mistral-7b-int4")

# Partial load (e.g., for layer-wise inference)
layer_5 = artifact.load_chunk("layers.5")

# Full streaming load
for layer in artifact.stream_layers():
    process(layer)

# Verify integrity
assert artifact.verify_signature()
```

### Success Metrics

| Metric | Target |
|--------|--------|
| Time to first token (streaming) | <2s for 7B on slow connection |
| Dedupe savings across similar models | 60-90% |
| Integration with vLLM | Direct load, no conversion |
| Enterprise attestation | Sigstore + GPG support |

### Files to Create

- [ ] `tenpak/artifact/format.py` - Artifact format spec
- [ ] `tenpak/artifact/streaming.py` - Chunked streaming
- [ ] `tenpak/artifact/signing.py` - Signature/attestation
- [ ] `tenpak/artifact/inference.py` - Direct inference integration

---

## 3. Automatic Cost-per-Token Optimizer

### The Gap

HF Endpoints don't (publicly) operate like:
> "Generate 6 candidates (AWQ/GPTQ/INT4+INT2/hybrid), benchmark on target HW, pick cheapest meeting PPL/latency constraints, then re-tune periodically."

### What We'll Build

- [ ] **Candidate generation**: Auto-generate N compression variants
- [ ] **Hardware-aware benchmarking**: Measure latency/throughput on target HW
- [ ] **Quality gates**: PPL/accuracy thresholds
- [ ] **Cost optimization**: Pick cheapest config meeting constraints
- [ ] **Periodic re-tuning**: Monitor and re-optimize as traffic changes

### Optimization Pipeline

```
Input:
  - model_id: "mistralai/Mistral-7B-v0.1"
  - hardware: "A10G"
  - constraints:
      max_ppl_delta: 2%
      max_latency_p99: 100ms
      target_throughput: 1000 tok/s

Pipeline:
1. Generate candidates:
   - AWQ INT4 g=128
   - AWQ INT4 g=256
   - GPTQ INT4
   - INT4+INT2 residual
   - INT4+Sparse 25%
   - FP16 (baseline)

2. For each candidate:
   - Compress model
   - Measure PPL on eval set
   - Benchmark on target HW:
     - Latency (p50, p95, p99)
     - Throughput (tok/s)
     - Memory usage
   - Calculate cost ($/1M tokens)

3. Filter by constraints:
   - PPL delta < 2%
   - Latency p99 < 100ms
   - Throughput > 1000 tok/s

4. Select cheapest passing candidate

5. Deploy + monitor

6. Re-tune trigger:
   - Weekly schedule
   - PPL drift detected
   - New codec available
   - Traffic pattern change
```

### API Design

```python
# POST /optimize
{
  "model_id": "mistralai/Mistral-7B-v0.1",
  "hardware": "a10g",
  "constraints": {
    "max_ppl_delta": 2.0,
    "max_latency_p99_ms": 100,
    "min_throughput_tps": 1000
  },
  "candidates": ["awq_g128", "awq_g256", "gptq", "int4_residual"],
  "eval_samples": 100
}

# Response
{
  "job_id": "opt_abc123",
  "status": "running",
  "candidates_evaluated": 2,
  "candidates_total": 4
}

# GET /optimize/opt_abc123
{
  "status": "completed",
  "winner": {
    "codec": "awq_g256",
    "compression": 7.42,
    "ppl_delta": 1.47,
    "latency_p99_ms": 45,
    "throughput_tps": 2100,
    "cost_per_1m_tokens": "$0.12"
  },
  "all_candidates": [...]
}
```

### Success Metrics

| Metric | Target |
|--------|--------|
| Cost reduction vs FP16 | 50-70% |
| Time to optimize 7B model | <1 hour |
| Quality guarantee | PPL within user-specified threshold |
| Throughput improvement | 2-4x vs FP16 |

### Files to Create

- [ ] `tenpak/optimizer/candidates.py` - Candidate generation
- [ ] `tenpak/optimizer/benchmark.py` - HW benchmarking
- [ ] `tenpak/optimizer/selector.py` - Constraint-based selection
- [ ] `tenpak/optimizer/monitor.py` - Drift detection + re-tuning
- [ ] `tenpak/studio/api.py` - Add `/optimize` endpoint

---

## Implementation Order

### Phase 1: Foundation ✅
- [x] Core compression codecs
- [x] Calibration pipeline
- [x] REST API
- [x] CLI

### Phase 2: Cost Optimizer ✅
- [x] Candidate generation (`optimizer/candidates.py`)
- [x] Hardware benchmarking (`optimizer/benchmark.py`)
- [x] Constraint-based selection (`optimizer/selector.py`)
- [x] `/optimize` endpoint
- [x] `tenpak optimize` CLI command
- [ ] Monitoring + re-tuning (future)

### Phase 3: Delta Compression ✅
- [x] Delta detection + storage (`core/delta.py`)
- [x] Sparse + INT8 compression methods
- [x] Reconstruction API
- [x] CLI integration (`tenpak delta`)
- [x] API endpoints (`/delta/compress`, `/delta/estimate`)

### Phase 4: Streaming Artifact ✅
- [x] Chunked format (.tnpk) - `artifact/format.py`
- [x] Content addressing (SHA256 per chunk)
- [x] Signing (HMAC + GPG support) - `artifact/signing.py`
- [x] Streaming support - `artifact/streaming.py`
- [x] CLI commands (`tenpak artifact`)
- [ ] Remote HTTP streaming (future)
- [ ] vLLM/TGI integration (future)

---

## HF Integration Points

| TenPak Feature | HF Integration |
|----------------|----------------|
| Delta compression | Hub storage backend |
| Streaming artifact | Hub model cards, safetensors replacement |
| Cost optimizer | Inference Endpoints auto-scaling |

---

## Resources

- [AWQ Paper](https://arxiv.org/abs/2306.00978)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AQLM Paper](https://arxiv.org/abs/2401.06118)
- [Safetensors Format](https://github.com/huggingface/safetensors)
- [Sigstore](https://www.sigstore.dev/)
