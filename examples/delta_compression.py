#!/usr/bin/env python3
"""
Example: Delta Compression for Fine-tunes

Compress a fine-tuned model as a delta from its base model.
This is one of Sparse's unique features.
"""

import argparse
from pathlib import Path
import json

from core import compress_delta, estimate_delta_savings, DeltaManifest
from core import QuantizationWrapper, QUANTIZATION_PRESETS


def main():
    parser = argparse.ArgumentParser(description="Compress fine-tune as delta")
    parser.add_argument("base_model", help="Base model ID")
    parser.add_argument("fine_tuned_model", help="Fine-tuned model ID")
    parser.add_argument("--output", default="./artifacts/delta",
                        help="Output directory")
    parser.add_argument("--quantization", default="awq_balanced",
                        help="Quantization preset for both models")
    
    args = parser.parse_args()
    
    print(f"🔄 Delta Compression")
    print(f"   Base: {args.base_model}")
    print(f"   Fine-tune: {args.fine_tuned_model}")
    
    # Step 1: Estimate savings
    print(f"\n📊 Estimating delta savings...")
    savings = estimate_delta_savings(
        base_model_id=args.base_model,
        finetune_model_id=args.fine_tuned_model,
    )
    
    print(f"   Best strategy: {savings['best_strategy']}")
    print(f"   Compression ratio: {savings['estimated_compression']:.2f}x")
    print(f"   Average sparsity: {savings['avg_sparsity']*100:.1f}%")
    print(f"   Breakdown: sparse={savings['sparse_compression']:.2f}x, int8={savings['int8_compression']:.2f}x, sparse+int8={savings['sparse_int8_compression']:.2f}x")
    
    # Step 2: Compress delta
    print(f"\n🗜️  Compressing delta...")
    delta_manifest = compress_delta(
        base_model_id=args.base_model,
        finetune_model_id=args.fine_tuned_model,
        output_path=args.output,
    )
    
    print(f"✓ Delta compressed")
    print(f"   Compression ratio: {delta_manifest.compression_ratio:.2f}x")
    print(f"   Changed params: {delta_manifest.changed_params:,} / {delta_manifest.total_params:,}")
    
    # Step 3: Quantize base model (optional but recommended)
    print(f"\n📦 Quantizing base model...")
    config = QUANTIZATION_PRESETS.get(args.quantization)
    if config:
        base_model = QuantizationWrapper.quantize_model(
            model_id=args.base_model,
            config=config,
        )
        print(f"✓ Base model quantized with {config.method}")
    
    # Step 4: Save delta manifest
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_data = {
        "model_id": args.fine_tuned_model,
        "base_model_id": args.base_model,
        "compression_ratio": delta_manifest.compression_ratio,
        "num_layers": delta_manifest.num_layers,
        "quantization": {
            "method": config.method.value if config else "none",
            "bits": config.bits if config else 16,
        },
        "savings": {
            "best_strategy": savings["best_strategy"],
            "estimated_compression": savings["estimated_compression"],
            "avg_sparsity": savings["avg_sparsity"],
        }
    }
    
    manifest_path = output_dir / "delta_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    print(f"\n✓ Delta saved to: {output_dir}")
    print(f"   Manifest: {manifest_path}")
    
    print(f"\n💡 To load this fine-tune:")
    print(f"   1. Load base model: {args.base_model}")
    print(f"   2. Apply delta from: {output_dir}")
    savings_pct = (1 - 1/savings['estimated_compression']) * 100
    print(f"   3. Saves {savings_pct:.1f}% storage vs full model")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
