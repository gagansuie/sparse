#!/usr/bin/env python3
"""Test INT8 delta quality validation on GPT-2 (fast download)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.delta import validate_int8_delta_quality

def main():
    print("=" * 70)
    print("INT8 Delta Quality Validation - GPT-2 (Quick Test)")
    print("=" * 70)
    
    # Test with GPT-2 base -> medium (small, fast download)
    report = validate_int8_delta_quality(
        base_model_id="gpt2",
        finetune_model_id="gpt2-medium",
        sample_layers=2,
        prompts=[
            "Hello, how are you today?",
            "The capital of France is",
        ],
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nStatus: {report['status']}")
    print(f"Base Model: {report['base_model']}")
    print(f"Fine-tuned Model: {report['finetune_model']}")
    print(f"Rust Acceleration: {report['rust_acceleration']}")
    
    if report.get("timings"):
        print(f"\nTimings:")
        for k, v in report["timings"].items():
            print(f"  {k}: {v:.2f}s")
    
    if report.get("layer_metrics"):
        print(f"\nLayer Metrics ({len(report['layer_metrics'])} layers sampled):")
        for layer in report["layer_metrics"]:
            print(f"  {layer['name']}:")
            print(f"    Shape: {layer['shape']}, Numel: {layer['numel']:,}")
            print(f"    Scale: {layer['scale']:.6f}")
            print(f"    Compression Ratio: {layer['compression_ratio']:.2f}x")
            print(f"    Max Abs Error: {layer['max_abs_error']:.6f}")
            print(f"    Mean Abs Error: {layer['mean_abs_error']:.8f}")
    
    if report.get("logits_metrics"):
        print(f"\nLogits Comparison ({len(report['logits_metrics'])} prompts):")
        for logit in report["logits_metrics"]:
            print(f"  Prompt: \"{logit['prompt'][:40]}\"")
            print(f"    Max Logit Diff: {logit['max_logit_diff']:.4f}")
            print(f"    Mean Logit Diff: {logit['mean_logit_diff']:.6f}")
    
    # Save full report
    output_path = Path(__file__).parent / "int8_quality_report_gpt2.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {output_path}")
    
    return 0 if "✅" in report["status"] else 1

if __name__ == "__main__":
    sys.exit(main())
