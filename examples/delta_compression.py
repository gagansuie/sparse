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
from artifact.format import ArtifactManifest


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
        finetuned_model_id=args.fine_tuned_model,
    )
    
    print(f"   Base size: {savings['base_size_gb']:.2f} GB")
    print(f"   Fine-tune size: {savings['finetuned_size_gb']:.2f} GB")
    print(f"   Delta size: {savings['delta_size_gb']:.2f} GB")
    print(f"   Savings: {savings['savings_pct']:.1f}%")
    
    # Step 2: Compress delta
    print(f"\n🗜️  Compressing delta...")
    delta_manifest = compress_delta(
        base_model_id=args.base_model,
        finetuned_model_id=args.fine_tuned_model,
        output_dir=args.output,
    )
    
    print(f"✓ Delta compressed")
    print(f"   Changed layers: {len(delta_manifest.changed_layers)}")
    print(f"   Method: {delta_manifest.delta_method}")
    
    # Step 3: Quantize base model (optional but recommended)
    print(f"\n📦 Quantizing base model...")
    config = QUANTIZATION_PRESETS.get(args.quantization)
    if config:
        base_model = QuantizationWrapper.quantize_model(
            model_id=args.base_model,
            config=config,
        )
        print(f"✓ Base model quantized with {config.method}")
    
    # Step 4: Create Sparse artifact
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = ArtifactManifest(
        model_id=args.fine_tuned_model,
        quantization={
            "method": config.method if config else "none",
            "bits": config.bits if config else 16,
        },
        delta={
            "base_model_id": args.base_model,
            "delta_method": delta_manifest.delta_method,
            "changed_layers": delta_manifest.changed_layers,
            "delta_size_gb": savings["delta_size_gb"],
            "savings_pct": savings["savings_pct"],
        },
        compression_ratio=savings["finetuned_size_gb"] / savings["delta_size_gb"],
        original_size_bytes=int(savings["finetuned_size_gb"] * 1024**3),
        compressed_size_bytes=int(savings["delta_size_gb"] * 1024**3),
    )
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    print(f"\n✓ Delta artifact created: {output_dir}")
    print(f"   Manifest: {manifest_path}")
    
    print(f"\n💡 To load this fine-tune:")
    print(f"   1. Load base model: {args.base_model}")
    print(f"   2. Apply delta from: {output_dir}")
    print(f"   3. Saves {savings['savings_pct']:.1f}% storage vs full model")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
