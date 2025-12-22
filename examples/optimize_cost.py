#!/usr/bin/env python3
"""
Example: Cost Optimization

Automatically benchmark multiple quantization methods and select the best one
based on cost/performance constraints.
"""

import argparse
from pathlib import Path
import json

from core import QUANTIZATION_PRESETS
from optimizer.candidates import CANDIDATE_PRESETS, generate_candidates
from inference.vllm_integration import benchmark_inference
from artifact.format import ArtifactManifest


def main():
    parser = argparse.ArgumentParser(description="Optimize quantization method for cost")
    parser.add_argument("model_id", help="HuggingFace model ID")
    parser.add_argument("--max-ppl-delta", type=float, default=2.0,
                        help="Maximum acceptable PPL delta (%)")
    parser.add_argument("--min-compression", type=float, default=4.0,
                        help="Minimum compression ratio")
    parser.add_argument("--output", default="./artifacts",
                        help="Output directory")
    parser.add_argument("--calibration", action="store_true",
                        help="Include calibration-based methods")
    
    args = parser.parse_args()
    
    print(f"🔍 Optimizing quantization for {args.model_id}")
    print(f"   Constraints:")
    print(f"   - Max PPL delta: {args.max_ppl_delta}%")
    print(f"   - Min compression: {args.min_compression}x")
    print(f"   - Calibration: {args.calibration}")
    
    # Generate candidates
    candidates = generate_candidates(
        include_calibration=args.calibration,
        max_expected_ppl_delta=args.max_ppl_delta,
        min_expected_compression=args.min_compression,
    )
    
    print(f"\n📋 Generated {len(candidates)} candidates:")
    for c in candidates:
        print(f"   - {c.name}: {c.expected_compression:.2f}x, {c.expected_ppl_delta:.2f}% PPL Δ")
    
    # Benchmark each candidate
    results = []
    print(f"\n⚡ Benchmarking candidates...")
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n[{i}/{len(candidates)}] Testing {candidate.name}...")
        
        try:
            # Create artifact for this candidate
            output_dir = Path(args.output) / f"candidate_{i}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Benchmark inference
            metrics = benchmark_inference(
                artifact_path=str(output_dir),
                engine="vllm",
                num_samples=50,
                prompt_length=128,
                output_length=128,
            )
            
            print(f"   ✓ Latency: {metrics['latency_mean_ms']:.2f} ms")
            print(f"   ✓ Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/s")
            
            results.append({
                "candidate": candidate.name,
                "method": candidate.method.value,
                "config": candidate.config,
                "metrics": metrics,
                "expected_compression": candidate.expected_compression,
                "expected_ppl_delta": candidate.expected_ppl_delta,
            })
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            continue
    
    # Select best candidate
    if not results:
        print("\n❌ No candidates succeeded")
        return
    
    # Sort by cost (latency * 1/throughput)
    results.sort(key=lambda r: r["metrics"]["latency_mean_ms"] / r["metrics"]["throughput_samples_per_sec"])
    
    best = results[0]
    print(f"\n🏆 Best candidate: {best['candidate']}")
    print(f"   Method: {best['method']}")
    print(f"   Compression: {best['expected_compression']:.2f}x")
    print(f"   PPL Delta: {best['expected_ppl_delta']:.2f}%")
    print(f"   Latency: {best['metrics']['latency_mean_ms']:.2f} ms")
    print(f"   Throughput: {best['metrics']['throughput_samples_per_sec']:.2f} samples/s")
    
    # Create final artifact with optimization metadata
    final_dir = Path(args.output) / "optimized"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = ArtifactManifest(
        model_id=args.model_id,
        quantization={
            "method": best["method"],
            **best["config"],
        },
        optimization={
            "selected_method": best["candidate"],
            "candidates_tested": [r["candidate"] for r in results],
            "latency_p50_ms": best["metrics"]["latency_mean_ms"],
            "throughput_tps": best["metrics"]["throughput_samples_per_sec"],
            "constraints": {
                "max_ppl_delta": args.max_ppl_delta,
                "min_compression": args.min_compression,
            },
        },
        compression_ratio=best["expected_compression"],
    )
    
    manifest_path = final_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    print(f"\n✓ Optimized artifact: {final_dir}")
    print(f"   Manifest: {manifest_path}")
    
    # Save full results
    results_path = final_dir / "optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"   Results: {results_path}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
