#!/usr/bin/env python3
"""
Example: Quantize a model and serve with vLLM

This script demonstrates the complete TenPak workflow:
1. Quantize a model using AutoGPTQ/AutoAWQ/bitsandbytes
2. Create a TenPak artifact with metadata
3. Serve the model with vLLM
"""

import argparse
from pathlib import Path

from core import QuantizationWrapper, QuantizationConfig, QUANTIZATION_PRESETS
from artifact.format import ArtifactManifest
from inference.vllm_integration import TenPakVLLMLoader
import json


def main():
    parser = argparse.ArgumentParser(description="Quantize and serve a model")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--method", choices=["gptq", "awq", "bitsandbytes"], default="awq",
                        help="Quantization method")
    parser.add_argument("--preset", default="awq_balanced",
                        help="Preset config (e.g., gptq_quality, awq_balanced, bnb_nf4)")
    parser.add_argument("--output", default="./artifacts",
                        help="Output directory for artifacts")
    parser.add_argument("--serve", action="store_true",
                        help="Serve with vLLM after quantization")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for vLLM server")
    
    args = parser.parse_args()
    
    # Step 1: Get quantization config
    if args.preset in QUANTIZATION_PRESETS:
        config = QUANTIZATION_PRESETS[args.preset]
        print(f"✓ Using preset: {args.preset}")
    else:
        config = QuantizationConfig(method=args.method, bits=4, group_size=128)
        print(f"✓ Using custom config: {config}")
    
    # Step 2: Quantize model
    print(f"\n📦 Quantizing {args.model_id}...")
    print(f"   Method: {config.method}")
    print(f"   Bits: {config.bits}")
    print(f"   Group size: {config.group_size}")
    
    model = QuantizationWrapper.quantize_model(
        model_id=args.model_id,
        config=config,
        calibration_data=None,  # Could add calibration data here
        device="cuda",
    )
    
    print("✓ Quantization complete")
    
    # Step 3: Estimate size
    size_info = QuantizationWrapper.estimate_size(args.model_id, config)
    print(f"\n📊 Size estimates:")
    print(f"   Original: {size_info['original_size_gb']:.2f} GB")
    print(f"   Quantized: {size_info['quantized_size_gb']:.2f} GB")
    print(f"   Compression: {size_info['compression_ratio']:.2f}x")
    
    # Step 4: Create TenPak artifact manifest
    output_dir = Path(args.output) / args.model_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = ArtifactManifest(
        model_id=args.model_id,
        quantization={
            "method": config.method,
            "bits": config.bits,
            "group_size": config.group_size,
            "desc_act": getattr(config, "desc_act", None),
            "zero_point": getattr(config, "zero_point", None),
        },
        compression_ratio=size_info["compression_ratio"],
        original_size_bytes=int(size_info["original_size_gb"] * 1024**3),
        compressed_size_bytes=int(size_info["quantized_size_gb"] * 1024**3),
    )
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    print(f"\n✓ Artifact created: {output_dir}")
    print(f"   Manifest: {manifest_path}")
    
    # Step 5: Serve with vLLM (optional)
    if args.serve:
        print(f"\n🚀 Starting vLLM server on port {args.port}...")
        print(f"   Model: {args.model_id}")
        print(f"   Quantization: {config.method}")
        
        TenPakVLLMLoader.serve_with_vllm(
            artifact_path=str(output_dir),
            host="0.0.0.0",
            port=args.port,
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
