# Removed Features

These features were removed during the strategic refocus to concentrate on TenPak's core value propositions:
1. Delta compression for fine-tuned models
2. Cross-tool cost optimizer

## Why These Were Removed

### artifact/
- **Reason:** HuggingFace already has superior CDN infrastructure (Cloudflare + git-lfs)
- **Content:** HTTP streaming, artifact signing, chunked formats
- **Status:** Solving a problem HF already solved better

### inference/
- **Reason:** HuggingFace built TGI (Text Generation Inference) themselves
- **Content:** vLLM integration, inference benchmarking
- **Status:** HF doesn't need integration help for their own tool

### studio/
- **Reason:** Not core to HF value proposition
- **Content:** REST API for compression-as-a-service
- **Status:** If needed, HF would build their own API

### deploy/
- **Reason:** Deployment is not TenPak's core competency
- **Content:** Backend deployment configurations
- **Status:** Not essential to delta compression or cost optimization

## Date Removed
December 22, 2024

## Can Be Restored
If a specific feature is needed later, it can be restored from this archive.
