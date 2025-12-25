"""
Sparse CLI - Main entry point

Command-line tool for delta compression and smart routing.
"""

import argparse
import sys
import os
import json
from typing import Optional


def cmd_pack(args):
    """Quantize a model using QuantizationWrapper, then package as artifact."""
    from core import QuantizationWrapper
    from artifact.format import create_artifact
    
    print(f"Sparse Pack - Quantizing {args.model_id}")
    print(f"  Preset: {args.preset}")
    print(f"  Output: {args.output or 'auto'}")
    print()
    
    # Auto-generate output path if not provided
    if args.output:
        output_path = args.output
    else:
        model_name = args.model_id.replace("/", "_")
        output_path = f"sparse_{model_name}_{args.preset}.sparse"
    
    try:
        # Step 1: Quantize using wrapper
        print("[1/2] Quantizing model...")
        wrapper = QuantizationWrapper.from_preset(args.preset)
        quantized_model_path = wrapper.quantize(
            model_id=args.model_id,
            device="cuda" if args.device == "auto" else args.device
        )
        
        # Step 2: Package as artifact
        print("[2/2] Creating artifact...")
        artifact = create_artifact(
            model_id=quantized_model_path,
            output_path=output_path,
            quantization_preset=args.preset,
        )
        
        print()
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Model:             {args.model_id}")
        print(f"  Preset:            {args.preset}")
        print(f"  Output:            {output_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


def cmd_pack_legacy(args):
    """DEPRECATED: Legacy compression command removed. Use QuantizationWrapper instead."""
    print("❌ ERROR: Legacy compression command is no longer supported.")
    print()
    print("The custom Rust compression codecs have been removed in Sparse v0.2.0.")
    print("Sparse now wraps industry-standard quantization tools instead.")
    print()
    print("Please use one of these commands instead:")
    print()
    print("  # Quantize with a preset:")
    print("  sparse pack <model> --preset awq_balanced")
    print()
    print("  # Available presets:")
    print("  - gptq_quality, gptq_balanced, gptq_aggressive")
    print("  - awq_quality, awq_balanced, awq_fast")
    print("  - bnb_nf4, bnb_int8")
    print()
    print("  # Or use the optimizer to auto-select:")
    print("  sparse optimize <model> --max-ppl-delta 2.0")
    print()
    return 1


def cmd_eval(args):
    """Evaluate a model's perplexity."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    from core.calibration import compute_ppl
    
    print(f"Sparse Eval - Evaluating {args.model_id}")
    print(f"  Samples: {args.samples}")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[INFO] Using device: {device}")
    
    # Load model
    print(f"[1/2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Load eval data
    print(f"[2/2] Evaluating...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = (item.get("text", "") for item in dataset if len(item.get("text", "")) > 100)
    
    ppl = compute_ppl(
        model,
        tokenizer,
        texts,
        device,
        max_samples=args.samples,
        streaming=args.streaming,
        max_length=args.max_length,
        stride=args.stride,
    )
    
    print()
    print("=" * 40)
    print(f"  Model:      {args.model_id}")
    print(f"  PPL:        {ppl:.4f}")
    print(f"  Samples:    {args.samples}")
    print("=" * 40)
    
    return 0


def cmd_info(args):
    """Show info about a compressed artifact."""
    from studio.storage import get_storage
    
    if not os.path.exists(args.artifact_path):
        print(f"Error: Artifact not found: {args.artifact_path}")
        return 1
    
    # Try to load manifest
    manifest_path = os.path.join(args.artifact_path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Not a valid Sparse artifact (no manifest.json)")
        return 1
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print()
    print("=" * 60)
    print("ARTIFACT INFO")
    print("=" * 60)
    print(f"  Model:             {manifest.get('model_id', 'unknown')}")
    print(f"  Codec:             {manifest.get('codec', 'unknown')}")
    print(f"  Created:           {manifest.get('created_at', 'unknown')}")
    print(f"  Compression:       {manifest.get('compression_ratio', 0):.2f}x")
    print(f"  Baseline PPL:      {manifest.get('baseline_ppl', 0):.4f}")
    print(f"  Compressed PPL:    {manifest.get('compressed_ppl', 0):.4f}")
    print(f"  PPL Delta:         {manifest.get('ppl_delta', 0):+.2f}%")
    print(f"  Layers:            {manifest.get('num_layers', 0)}")
    print(f"  Parameters:        {manifest.get('total_params', 0):,}")
    print(f"  Shards:            {manifest.get('num_shards', 0)}")
    print("=" * 60)
    
    return 0


def cmd_optimize(args):
    """Find optimal compression config."""
    from optimizer import optimize_model, OptimizationConstraints, CANDIDATE_PRESETS
    
    print(f"Sparse Optimize - Finding optimal config for {args.model_id}")
    print(f"  Hardware: {args.hardware}")
    print(f"  Constraints:")
    print(f"    Max PPL Delta: {args.max_ppl_delta}%")
    print(f"    Max Latency (p99): {args.max_latency}ms")
    print(f"    Min Throughput: {args.min_throughput} tps")
    print(f"  Samples:")
    print(f"    Eval:      {args.eval_samples}")
    print(f"    Benchmark: {args.benchmark_samples}")
    print()
    
    if args.candidates:
        print(f"  Candidates: {', '.join(args.candidates)}")
    else:
        print(f"  Candidates: all ({len(CANDIDATE_PRESETS)} available)")
    print()
    
    constraints = OptimizationConstraints(
        max_ppl_delta=args.max_ppl_delta,
        max_latency_p99_ms=args.max_latency,
        min_throughput_tps=args.min_throughput,
    )
    
    def progress_callback(msg, progress):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% - {msg[:50]:<50}", end="", flush=True)
        if progress >= 1.0:
            print()
    
    result = optimize_model(
        model_id=args.model_id,
        hardware=args.hardware,
        constraints=constraints,
        candidates=args.candidates,
        include_calibration=not args.no_calibration,
        num_eval_samples=args.eval_samples,
        num_benchmark_samples=args.benchmark_samples,
        progress_callback=progress_callback,
    )
    
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"  Model:              {result.model_id}")
    print(f"  Hardware:           {result.hardware}")
    print(f"  Candidates tested:  {result.candidates_evaluated}/{result.total_candidates}")
    print(f"  Candidates passed:  {len(result.passing_results)}")
    print(f"  Time:               {result.optimization_time_s:.1f}s")
    print()
    
    if result.winner:
        print("WINNER:")
        print(f"  Codec:              {result.winner.candidate_name}")
        print(f"  Compression:        {result.winner.compression_ratio:.2f}x")
        print(f"  PPL Delta:          {result.winner.ppl_delta_pct:+.2f}%")
        print(f"  Latency (p99):      {result.winner.latency_p99_ms:.1f}ms")
        print(f"  Throughput:         {result.winner.throughput_tps:.0f} tps")
        print(f"  Cost:               ${result.winner.cost_per_1m_tokens:.4f}/1M tokens")
        print(f"  Cost Savings:       {result.cost_savings_pct:.1f}% vs FP16")
    else:
        print("NO WINNER - No candidates passed all constraints")
        print()
        print("Try relaxing constraints:")
        print("  --max-ppl-delta 5.0")
        print("  --max-latency 200.0")
        print("  --min-throughput 500.0")
    
    print("=" * 70)
    
    # Save results to JSON if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
    
    return 0 if result.winner else 1


def cmd_deploy(args):
    from optimizer import optimize_model, OptimizationConstraints, CANDIDATE_PRESETS
    from datetime import datetime
    from pathlib import Path
    from deploy import default_backend_for_engine, get_backend

    backend_id = args.backend
    if backend_id is None:
        try:
            backend_id = default_backend_for_engine(args.engine)
        except Exception as e:
            print(f"[WARN] No default backend for engine '{args.engine}': {e}")
            backend_id = None

    backend = None
    backend_fragment = None
    backend_availability = None
    if backend_id is not None:
        try:
            backend = get_backend(backend_id)
            backend_availability = backend.availability(hardware=args.hardware)
            backend_fragment = backend.recipe_fragment(model_id=args.model_id, hardware=args.hardware)
            if not backend_availability.available:
                print(f"[WARN] Backend '{backend_id}' not available for hardware '{args.hardware}': {backend_availability.reason}")
        except Exception as e:
            print(f"[WARN] Failed to load backend '{backend_id}': {e}")
            backend = None
            backend_fragment = None
            backend_availability = None

    engine = backend.engine if backend else args.engine

    print(f"Sparse Deploy - Generating deployment package for {args.model_id}")
    print(f"  Hardware: {args.hardware}")
    print(f"  Engine:   {engine}")
    print(f"  Backend:  {backend_id or 'none'}")
    print(f"  Output:   {args.output or 'auto'}")
    print(f"  Constraints:")
    print(f"    Max PPL Delta: {args.max_ppl_delta}%")
    print(f"    Max Latency (p99): {args.max_latency}ms")
    print(f"    Min Throughput: {args.min_throughput} tps")
    print(f"  Samples:")
    print(f"    Eval:      {args.eval_samples}")
    print(f"    Benchmark: {args.benchmark_samples}")
    print()

    if args.candidates:
        print(f"  Candidates: {', '.join(args.candidates)}")
    else:
        print(f"  Candidates: all ({len(CANDIDATE_PRESETS)} available)")
    print()

    constraints = OptimizationConstraints(
        max_ppl_delta=args.max_ppl_delta,
        max_latency_p99_ms=args.max_latency,
        min_throughput_tps=args.min_throughput,
    )

    def progress_callback(msg, progress):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% - {msg[:50]:<50}", end="", flush=True)
        if progress >= 1.0:
            print()

    result = optimize_model(
        model_id=args.model_id,
        hardware=args.hardware,
        constraints=constraints,
        candidates=args.candidates,
        include_calibration=not args.no_calibration,
        num_eval_samples=args.eval_samples,
        num_benchmark_samples=args.benchmark_samples,
        progress_callback=progress_callback,
    )

    if args.output:
        output_dir = Path(args.output)
    else:
        model_name = args.model_id.replace("/", "_")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"sparse_deploy_{model_name}_{ts}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    winner_candidate = result.winner_candidate
    winner_preset = None
    if winner_candidate:
        for key, preset in CANDIDATE_PRESETS.items():
            if preset.name == winner_candidate.name and preset.method == winner_candidate.method:
                winner_preset = key
                break

    recipe = {
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "model_id": result.model_id,
        "hardware": result.hardware,
        "engine": engine,
        "constraints": result.constraints.to_dict(),
        "winner": result.winner.to_dict() if result.winner else None,
        "winner_candidate": {
            "preset": winner_preset,
            "name": winner_candidate.name,
            "method": winner_candidate.method.value,
            "config": winner_candidate.config,
        } if winner_candidate else None,
        "package": {
            "results_json": "results.json",
        },
        "deployment": {
            "backend_id": backend_id,
            "availability": backend_availability.to_dict() if backend_availability else None,
            "engine_config": backend_fragment,
        },
        "artifacts": {},
    }

    if args.with_artifact:
        try:
            from artifact import create_artifact

            artifact_path = output_dir / "artifact.sparse"

            def artifact_progress_callback(msg, progress):
                bar_width = 30
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"\r[{bar}] {progress*100:5.1f}% - {msg[:50]:<50}", end="", flush=True)
                if progress >= 1.0:
                    print()

            create_artifact(
                model_id=args.model_id,
                output_path=str(artifact_path),
                codec=args.artifact_codec,
                progress_callback=artifact_progress_callback,
            )

            recipe["artifacts"]["sparse"] = {
                "path": "artifact.sparse",
                "codec": args.artifact_codec,
            }
        except Exception as e:
            print(f"[WARN] Failed to create .sparse artifact: {e}")

    recipe_path = output_dir / "recipe.json"
    with open(recipe_path, "w") as f:
        json.dump(recipe, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("DEPLOYMENT PACKAGE")
    print("=" * 70)
    print(f"  Output dir:         {output_dir}")
    print(f"  Recipe:             {recipe_path}")
    print(f"  Results:            {results_path}")
    if recipe.get("artifacts", {}).get("tnpk"):
        print(f"  Artifact:           {output_dir / recipe['artifacts']['tnpk']['path']}")
    print("=" * 70)

    if result.winner:
        return 0
    return 1


def cmd_delta_model(args):
    """Handle delta compression commands."""
    if args.delta_command == "compress":
        return cmd_delta_compress(args)
    elif args.delta_command == "reconstruct":
        return cmd_delta_reconstruct(args)
    elif args.delta_command == "estimate":
        return cmd_delta_estimate(args)
    else:
        print("Usage: sparse delta {compress|reconstruct|estimate}")
        return 1


def cmd_delta_compress(args):
    """Compress fine-tune as delta from base model."""
    from core.delta import compress_delta
    
    print(f"Sparse Delta Compress")
    print(f"  Base model:     {args.base_model}")
    print(f"  Fine-tune:      {args.finetune_model}")
    print(f"  Output:         {args.output}")
    print()
    
    def progress_callback(msg, progress):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% - {msg[:50]:<50}", end="", flush=True)
        if progress >= 1.0:
            print()
    
    manifest = compress_delta(
        base_model_id=args.base_model,
        finetune_model_id=args.finetune_model,
        output_path=args.output,
        progress_callback=progress_callback,
    )
    
    print()
    print("=" * 60)
    print("DELTA COMPRESSION RESULTS")
    print("=" * 60)
    print(f"  Base model:       {manifest.base_model_id}")
    print(f"  Fine-tune:        {manifest.finetune_model_id}")
    print(f"  Compression:      {manifest.compression_ratio:.2f}x")
    print(f"  Total params:     {manifest.total_params:,}")
    print(f"  Changed params:   {manifest.changed_params:,} ({100*manifest.changed_params/manifest.total_params:.1f}%)")
    print(f"  Layers:           {manifest.num_layers}")
    print(f"  Output:           {args.output}")
    print("=" * 60)
    
    return 0


def cmd_delta_reconstruct(args):
    """Reconstruct model from base + delta."""
    from core.delta import reconstruct_from_delta
    
    print(f"Sparse Delta Reconstruct")
    print(f"  Base model:     {args.base_model}")
    print(f"  Delta path:     {args.delta_path}")
    print()
    
    def progress_callback(msg, progress):
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% - {msg[:50]:<50}", end="", flush=True)
        if progress >= 1.0:
            print()
    
    model = reconstruct_from_delta(
        base_model_id=args.base_model,
        delta_path=args.delta_path,
        progress_callback=progress_callback,
    )
    
    print()
    print("=" * 60)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Model loaded and ready for inference")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.output:
        print(f"  Saving to: {args.output}")
        model.save_pretrained(args.output)
        print(f"  Saved!")
    
    print("=" * 60)
    
    return 0


def cmd_delta_estimate(args):
    """Estimate delta compression savings."""
    from core.delta import estimate_delta_savings
    
    print(f"Sparse Delta Estimate")
    print(f"  Base model:     {args.base_model}")
    print(f"  Fine-tune:      {args.finetune_model}")
    print()
    print("Analyzing layers (this may take a minute)...")
    
    result = estimate_delta_savings(
        base_model_id=args.base_model,
        finetune_model_id=args.finetune_model,
    )
    
    print()
    print("=" * 60)
    print("DELTA ESTIMATE")
    print("=" * 60)
    print(f"  Estimated compression:  {result['estimated_compression']:.2f}x")
    print(f"  Average sparsity:       {result['avg_sparsity']*100:.1f}%")
    print(f"  Layers sampled:         {result['sample_layers']}")
    print()
    print(f"  This means storing the fine-tune as a delta would use")
    print(f"  ~{100/result['estimated_compression']:.0f}% of the original storage.")
    print("=" * 60)
    
    return 0


def cmd_artifact(args):
    """Handle artifact commands."""
    if args.artifact_command == "create":
        return cmd_artifact_create(args)
    elif args.artifact_command == "info":
        return cmd_artifact_info(args)
    elif args.artifact_command == "verify":
        return cmd_artifact_verify(args)
    elif args.artifact_command == "sign":
        return cmd_artifact_sign(args)
    else:
        print("Usage: sparse artifact {create|info|verify|sign}")
        return 1


def cmd_artifact_create(args):
    """Create .sparse artifact from quantized model."""
    from artifact import create_artifact
    from core import QuantizationWrapper
    
    print(f"Sparse Artifact Create")
    print(f"  Model:  {args.model_id}")
    print(f"  Output: {args.output}")
    print(f"  Preset: {args.preset}")
    print()
    
    try:
        # Step 1: Quantize
        print("[1/2] Quantizing model...")
        wrapper = QuantizationWrapper.from_preset(args.preset)
        quantized_path = wrapper.quantize(model_id=args.model_id)
        
        # Step 2: Create artifact
        print("[2/2] Creating artifact...")
        artifact = create_artifact(
            model_id=quantized_path,
            output_path=args.output,
            quantization_preset=args.preset,
        )
        
        print()
        print("=" * 60)
        print("ARTIFACT CREATED")
        print("=" * 60)
        print(f"  Model:       {args.model_id}")
        print(f"  Preset:      {args.preset}")
        print(f"  Chunks:      {len(artifact.manifest.chunks)}")
        print(f"  Size:        {artifact.manifest.compressed_size_bytes / 1e6:.1f} MB")
        print(f"  Path:        {args.output}")
        print("=" * 60)
        
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


def cmd_artifact_info(args):
    """Show artifact info."""
    from artifact import load_artifact
    
    artifact = load_artifact(args.artifact_path)
    m = artifact.manifest
    
    print()
    print("=" * 60)
    print("ARTIFACT INFO")
    print("=" * 60)
    print(f"  Model:         {m.model_id}")
    print(f"  Architecture:  {m.architecture}")
    print(f"  Codec:         {m.codec}")
    print(f"  Layers:        {m.num_layers}")
    print(f"  Chunks:        {len(m.chunks)}")
    print(f"  Compression:   {m.compression_ratio:.2f}x")
    print(f"  Original:      {m.original_size_bytes / 1e9:.2f} GB")
    print(f"  Compressed:    {m.compressed_size_bytes / 1e6:.1f} MB")
    print(f"  Created:       {m.created_at}")
    print(f"  Signed:        {'Yes' if m.signature else 'No'}")
    if m.signer:
        print(f"  Signer:        {m.signer}")
    print("=" * 60)
    
    return 0


def cmd_artifact_verify(args):
    """Verify artifact integrity."""
    from artifact import load_artifact
    
    print(f"Verifying artifact: {args.artifact_path}")
    print()
    
    artifact = load_artifact(args.artifact_path)
    
    print(f"Checking {len(artifact.manifest.chunks)} chunks...")
    
    if artifact.verify():
        print("✅ All chunks verified successfully!")
        return 0
    else:
        print("❌ Verification failed - some chunks are corrupted or missing")
        return 1


def cmd_artifact_sign(args):
    """Sign artifact."""
    from artifact import sign_artifact
    
    print(f"Signing artifact: {args.artifact_path}")
    print(f"  Signer: {args.signer}")
    print()
    
    sig_info = sign_artifact(
        artifact_path=args.artifact_path,
        signer=args.signer,
        secret_key=args.key,
    )
    
    print("=" * 60)
    print("ARTIFACT SIGNED")
    print("=" * 60)
    print(f"  Algorithm:  {sig_info.algorithm}")
    print(f"  Signer:     {sig_info.signer}")
    print(f"  Signed at:  {sig_info.signed_at}")
    print(f"  Hash:       {sig_info.manifest_hash[:16]}...")
    print("=" * 60)
    
    return 0


def cmd_native(args):
    from core.native import NativeSparseNotFoundError, run_native_sparse

    native_args = list(args.native_args or [])
    if native_args and native_args[0] == "--":
        native_args = native_args[1:]
    if not native_args:
        native_args = ["--help"]

    try:
        proc = run_native_sparse(native_args, bin_path=args.native_bin)
        return proc.returncode
    except NativeSparseNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_delta_dataset(args):
    """Dataset delta compression commands."""
    from core.dataset_delta import (
        estimate_dataset_delta_savings,
        compress_dataset_delta,
        reconstruct_from_dataset_delta
    )
    
    if args.delta_dataset_cmd == "estimate":
        print(f"Estimating dataset delta savings...")
        print(f"  Base: {args.base}")
        print(f"  Derivative: {args.derivative}")
        print()
        
        stats = estimate_dataset_delta_savings(
            base_dataset_id=args.base,
            derivative_dataset_id=args.derivative,
            sample_size=args.sample_size or 1000
        )
        
        print("Results:")
        print(f"  Base size: {stats.base_size_mb:.1f} MB")
        print(f"  Derivative size: {stats.derivative_size_mb:.1f} MB")
        print(f"  Delta size: {stats.delta_size_mb:.1f} MB")
        print(f"  Savings: {stats.savings_pct:.1f}%")
        print()
        print(f"  Shared samples: {stats.num_shared_samples}")
        print(f"  New samples: {stats.num_new_samples}")
        
    elif args.delta_dataset_cmd == "compress":
        print(f"Compressing dataset as delta...")
        print(f"  Base: {args.base}")
        print(f"  Derivative: {args.derivative}")
        print(f"  Output: {args.output}")
        print()
        
        manifest = compress_dataset_delta(
            base_dataset_id=args.base,
            derivative_dataset_id=args.derivative,
            output_dir=args.output
        )
        
        print("\nDelta compression complete!")
        print(f"  Savings: {manifest['size_stats']['savings_pct']:.1f}%")
        
    elif args.delta_dataset_cmd == "reconstruct":
        print(f"Reconstructing dataset from delta...")
        print(f"  Delta dir: {args.delta_dir}")
        print()
        
        dataset = reconstruct_from_dataset_delta(args.delta_dir)
        
        print("Reconstruction complete!")
        for split in dataset.keys():
            print(f"  {split}: {len(dataset[split])} samples")
    
    return 0


def cmd_route(args):
    """Smart model routing and recommendation."""
    from optimizer.routing import suggest_optimal_model
    
    print(f"Analyzing request for optimal routing...")
    print(f"  Requested model: {args.model}")
    print(f"  Prompt: {args.prompt[:100]}..." if len(args.prompt) > 100 else f"  Prompt: {args.prompt}")
    print()
    
    decision = suggest_optimal_model(
        requested_model=args.model,
        prompt=args.prompt,
        quality_threshold=args.quality_threshold or 0.85,
        cost_priority=not args.prioritize_latency
    )
    
    print("Routing Decision:")
    print(f"  Recommended model: {decision.recommended_model}")
    print(f"  Hardware: {decision.recommended_hardware.hardware_name}")
    print(f"  Estimated cost: ${decision.estimated_cost_per_1m_tokens:.2f} per 1M tokens")
    print(f"  Estimated latency (p99): {decision.estimated_latency_p99_ms:.0f}ms")
    print(f"  Quality score: {decision.quality_score:.2%}")
    print(f"\n  Reasoning: {decision.reasoning}")
    
    if decision.alternatives:
        print("\nAlternatives:")
        for i, alt in enumerate(decision.alternatives[:3], 1):
            print(f"  {i}. {alt['model']} on {alt['hardware']}")
            print(f"     Cost: ${alt['cost']:.2f}/1M tokens, Quality: {alt['quality']:.2%}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sparse",
        description="Sparse - Delta Compression + Smart Routing for Model Hubs ($30-45M/year savings)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # delta-dataset command (NEW)
    delta_dataset_parser = subparsers.add_parser(
        "delta-dataset",
        help="Dataset delta compression"
    )
    delta_dataset_subparsers = delta_dataset_parser.add_subparsers(
        dest="delta_dataset_cmd",
        help="Dataset delta commands"
    )
    
    # delta-dataset estimate
    ds_estimate = delta_dataset_subparsers.add_parser(
        "estimate",
        help="Estimate savings from dataset delta compression"
    )
    ds_estimate.add_argument("base", help="Base dataset ID")
    ds_estimate.add_argument("derivative", help="Derivative dataset ID")
    ds_estimate.add_argument("--sample-size", type=int, help="Number of samples to analyze")
    
    # delta-dataset compress
    ds_compress = delta_dataset_subparsers.add_parser(
        "compress",
        help="Compress derivative dataset as delta"
    )
    ds_compress.add_argument("base", help="Base dataset ID")
    ds_compress.add_argument("derivative", help="Derivative dataset ID")
    ds_compress.add_argument("--output", "-o", required=True, help="Output directory")
    
    # delta-dataset reconstruct
    ds_reconstruct = delta_dataset_subparsers.add_parser(
        "reconstruct",
        help="Reconstruct dataset from delta"
    )
    ds_reconstruct.add_argument("delta_dir", help="Delta directory")
    
    # route command (NEW)
    route_parser = subparsers.add_parser(
        "route",
        help="Smart model routing and recommendation"
    )
    route_parser.add_argument("model", help="Requested model ID")
    route_parser.add_argument("prompt", help="User prompt")
    route_parser.add_argument("--quality-threshold", type=float, help="Minimum quality (0-1)")
    route_parser.add_argument("--prioritize-latency", action="store_true", help="Prioritize latency over cost")
    
    # pack command
    pack_parser = subparsers.add_parser("pack", help="Compress a model")
    pack_parser.add_argument("model_id", help="HuggingFace model ID or local path")
    pack_parser.add_argument(
        "--target", "-t",
        choices=["quality", "balanced", "size"],
        default="balanced",
        help="Target preset mapping: quality=gptq_quality, balanced=awq_balanced, size=gptq_aggressive (deprecated, use --preset instead)"
    )
    pack_parser.add_argument(
        "--output", "-o",
        help="Output path for artifact"
    )
    pack_parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of samples for PPL evaluation (default: 50)"
    )
    pack_parser.add_argument(
        "--streaming-eval",
        action="store_true",
        help="Use streaming/chunked PPL evaluation"
    )
    pack_parser.add_argument(
        "--eval-max-length",
        type=int,
        default=512,
        help="Max sequence length for PPL evaluation (default: 512)"
    )
    pack_parser.add_argument(
        "--eval-stride",
        type=int,
        default=512,
        help="Stride for streaming PPL evaluation (default: 512)"
    )
    pack_parser.add_argument(
        "--preset",
        choices=[
            "gptq_quality",
            "gptq_balanced",
            "gptq_aggressive",
            "awq_quality",
            "awq_balanced",
            "awq_fast",
            "bnb_nf4",
            "bnb_int8",
            "fp16",
        ],
        default="awq_balanced",
        help="Quantization preset (default: awq_balanced). Use 'gptq_quality' for best compression, 'bnb_nf4' for fast no-calibration"
    )
    pack_parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)"
    )
    
    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate model perplexity")
    eval_parser.add_argument("model_id", help="HuggingFace model ID or local path")
    eval_parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Number of samples (default: 50)"
    )
    eval_parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming/chunked PPL evaluation"
    )
    eval_parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for PPL evaluation (default: 512)"
    )
    eval_parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for streaming PPL evaluation (default: 512)"
    )
    
    # info command
    info_parser = subparsers.add_parser("info", help="Show artifact info")
    info_parser.add_argument("artifact_path", help="Path to artifact directory")
    
    # optimize command
    opt_parser = subparsers.add_parser("optimize", help="Find optimal compression config")
    opt_parser.add_argument("model_id", help="HuggingFace model ID or local path")
    opt_parser.add_argument(
        "--hardware", "-hw",
        default="cuda",
        help="Target hardware (a10g, t4, h100, cuda, cpu)"
    )
    opt_parser.add_argument(
        "--max-ppl-delta",
        type=float,
        default=2.0,
        help="Maximum PPL delta in %% (default: 2.0)"
    )
    opt_parser.add_argument(
        "--max-latency",
        type=float,
        default=100.0,
        help="Maximum p99 latency in ms (default: 100.0)"
    )
    opt_parser.add_argument(
        "--min-throughput",
        type=float,
        default=1000.0,
        help="Minimum throughput in tokens/sec (default: 1000.0)"
    )
    opt_parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of samples for PPL evaluation (default: 50)"
    )
    opt_parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=30,
        help="Number of samples for latency/throughput benchmark (default: 30)"
    )
    opt_parser.add_argument(
        "--candidates",
        nargs="+",
        help="Specific candidates to test (default: all)"
    )
    opt_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Exclude candidates requiring calibration"
    )
    opt_parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results"
    )

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Generate deployment package (recipe + results)")
    deploy_parser.add_argument("model_id", help="HuggingFace model ID or local path")
    deploy_parser.add_argument(
        "--engine",
        default="tgi",
        help="Target serving engine (tgi, vllm, sglang)"
    )
    deploy_parser.add_argument(
        "--backend",
        help="Deployment backend id (e.g. tgi.bitsandbytes-nf4)"
    )
    deploy_parser.add_argument(
        "--hardware", "-hw",
        default="cuda",
        help="Target hardware (a10g, t4, h100, cuda, cpu)"
    )
    deploy_parser.add_argument(
        "--max-ppl-delta",
        type=float,
        default=2.0,
        help="Maximum PPL delta in %% (default: 2.0)"
    )
    deploy_parser.add_argument(
        "--max-latency",
        type=float,
        default=100.0,
        help="Maximum p99 latency in ms (default: 100.0)"
    )
    deploy_parser.add_argument(
        "--min-throughput",
        type=float,
        default=1000.0,
        help="Minimum throughput in tokens/sec (default: 1000.0)"
    )
    deploy_parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of samples for PPL evaluation (default: 50)"
    )
    deploy_parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=30,
        help="Number of samples for latency/throughput benchmark (default: 30)"
    )
    deploy_parser.add_argument(
        "--candidates",
        nargs="+",
        help="Specific candidates to test (default: all)"
    )
    deploy_parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Exclude candidates requiring calibration"
    )
    deploy_parser.add_argument(
        "--output", "-o",
        help="Output directory for deployment package"
    )
    deploy_parser.add_argument(
        "--with-artifact",
        action="store_true",
        help="Also create a .sparse artifact in the package"
    )
    deploy_parser.add_argument(
        "--artifact-preset",
        default="awq_balanced",
        help="Quantization preset to use when creating a .sparse artifact (default: awq_balanced)"
    )

    # native command
    native_parser = subparsers.add_parser("native", help="Run native sparse binary")
    native_parser.add_argument(
        "--bin",
        dest="native_bin",
        help="Path to the native sparse binary (overrides SPARSE_NATIVE_BIN)"
    )
    native_parser.add_argument(
        "native_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to native sparse (e.g. compress -i in.json -o out.bin)"
    )
    
    # delta command
    delta_parser = subparsers.add_parser("delta", help="Delta compression for fine-tunes")
    delta_subparsers = delta_parser.add_subparsers(dest="delta_command", help="Delta subcommands")
    
    # delta compress
    delta_compress = delta_subparsers.add_parser("compress", help="Compress fine-tune as delta")
    delta_compress.add_argument("base_model", help="Base model HuggingFace ID")
    delta_compress.add_argument("finetune_model", help="Fine-tuned model HuggingFace ID or path")
    delta_compress.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for delta artifact"
    )
    
    # delta reconstruct
    delta_reconstruct = delta_subparsers.add_parser("reconstruct", help="Reconstruct model from delta")
    delta_reconstruct.add_argument("base_model", help="Base model HuggingFace ID")
    delta_reconstruct.add_argument("delta_path", help="Path to delta artifact")
    delta_reconstruct.add_argument(
        "--output", "-o",
        help="Output path for reconstructed model"
    )
    
    # delta estimate
    delta_estimate = delta_subparsers.add_parser("estimate", help="Estimate delta savings")
    delta_estimate.add_argument("base_model", help="Base model HuggingFace ID")
    delta_estimate.add_argument("finetune_model", help="Fine-tuned model HuggingFace ID or path")
    
    # artifact command
    artifact_parser = subparsers.add_parser("artifact", help="Streamable artifact format (.sparse)")
    artifact_subparsers = artifact_parser.add_subparsers(dest="artifact_command", help="Artifact subcommands")
    
    # artifact create
    artifact_create = artifact_subparsers.add_parser("create", help="Create .sparse artifact from model")
    artifact_create.add_argument("model_id", help="HuggingFace model ID")
    artifact_create.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for .sparse artifact"
    )
    artifact_create.add_argument(
        "--preset",
        default="awq_balanced",
        help="Quantization preset (default: awq_balanced)"
    )
    
    # artifact info
    artifact_info = artifact_subparsers.add_parser("info", help="Show artifact info")
    artifact_info.add_argument("artifact_path", help="Path to .sparse artifact")
    
    # artifact verify
    artifact_verify = artifact_subparsers.add_parser("verify", help="Verify artifact integrity")
    artifact_verify.add_argument("artifact_path", help="Path to .sparse artifact")
    
    # artifact sign
    artifact_sign = artifact_subparsers.add_parser("sign", help="Sign artifact")
    artifact_sign.add_argument("artifact_path", help="Path to .sparse artifact")
    artifact_sign.add_argument("--signer", required=True, help="Signer identity")
    artifact_sign.add_argument("--key", help="Secret key for HMAC signing")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "pack":
        return cmd_pack(args)
    elif args.command == "eval":
        return cmd_eval(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "deploy":
        return cmd_deploy(args)
    elif args.command == "delta":
        return cmd_delta_model(args)
    elif args.command == "delta-dataset":
        return cmd_delta_dataset(args)
    elif args.command == "route":
        return cmd_route(args)
    elif args.command == "artifact":
        return cmd_artifact(args)
    elif args.command == "native":
        return cmd_native(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
