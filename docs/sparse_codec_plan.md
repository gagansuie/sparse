# Sparse Codec Plan (int4_perchannel_sparse_block_v1)

## Motivation

The current `int4_perchannel_sparse50_v1` codec stores every surviving weight with a 32-bit absolute index plus a byte containing two 4-bit values. With 50% sparsity this metadata dominates the payload and the artifact becomes *larger* than the dense int4 versions.

## Goals

1. **Shrink metadata dramatically** so 50% sparsity yields at least ~1.8× over dense int4.
2. **Preserve per-channel scaling** for quality.
3. **Keep decoding simple** (no heavy decompression kernels) but allow vectorized packing.
4. **Stay deterministic** with round-trippable structure.

## Proposed Format Overview

```
Artifact header (unchanged)
└─ For each tensor:
   ├─ name, shape, per-channel scales (same as dense per-channel)
   ├─ blocks: Vec<BlockHeader>
   └─ block_payload: packed i4 values (same as dense coder)
```

### Block definition
- Fixed block size: **32 elements** along the flattened tensor.
- Each block stores:
  - `u32 base_index`: index of the first element in the block (multiple of 32).
  - `u32 mask_low` + `u32 mask_high`: 64-bit bitmask covering the 32 positions (2×u32 keeps alignment simple). A bit of `1` means *value present*.
  - `u16 channel_id`: since blocks never cross per-channel boundaries, this identifies which scale to use.
  - `u16 payload_offset`: pointer into the shared `block_payload` byte array.

### Payload packing
- Present values are written sequentially using the existing symmetrical int4 packing (2 values per byte).
- Because each block knows how many 1 bits it has (popcount), the decoder can read exactly `ceil(nnz / 2)` bytes starting at `payload_offset`.

### Encoding steps
1. **Per-channel scale** identical to dense `int4_perchannel_v1`.
2. **Pruning**: keep top-50% magnitudes per channel (threshold as today).
3. **Blockization**: iterate through channel slice in chunks of 32, create `BlockHeader` only if at least one element survives.
4. **Payload write**: append packed nibbles to a global payload vector; store offset in header.

### Decoding steps
1. Initialize tensor buffer to zeros.
2. For each block header:
   - Determine scale from `channel_id`.
   - Read `nnz = popcount(mask_low, mask_high)`.
   - Slice `ceil(nnz/2)` bytes starting at `payload_offset`, decode sequentially, and scatter values into the global tensor using the bitmask order.

### Benefits
- Metadata per non-zero drops from 4 bytes (index) to ~0.125 bytes (bitmask amortized) + tiny header.
- Block-locality ensures better cache use and makes future SIMD implementations easier.
- Compatible with delta storage (block headers can be diff-compressed later).

## Follow-up Tasks
1. Implement new codec structs in Rust (`BlockHeader`, etc.).
2. Migrate sparse encoder/decoder to block layout.
3. Update Python script to compute artifact size & PPL for the new codec.
4. Retain legacy codec behind a feature flag for experimentation.
