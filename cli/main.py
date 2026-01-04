"""
Sparse CLI - Delta Compression for Models and Datasets

Compress fine-tuned models and derivative datasets as deltas from base versions.
"""

import argparse
import sys
import json
from tqdm import tqdm


def cmd_delta_compress(args):
    """Compress fine-tune as delta from base model."""
    from core.delta import compress_delta
    
    print("\n" + "="*70)
    print("🗜️  SPARSE COMPRESS (Lossless Delta)")
    print("="*70)
    print(f"  Base model:     {args.base_model}")
    print(f"  Fine-tune:      {args.finetune_model}")
    print(f"  Output:         {args.output}")
    print("="*70 + "\n")
    
    # Create tqdm progress bar
    pbar = tqdm(total=100, desc="Compressing", bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]")
    last_progress = 0
    
    def progress_callback(msg, progress):
        nonlocal last_progress
        delta = int((progress - last_progress) * 100)
        if delta > 0:
            pbar.update(delta)
            last_progress = progress
        pbar.set_postfix_str(msg[:40])
    
    try:
        manifest = compress_delta(
            base_model_id=args.base_model,
            finetune_model_id=args.finetune_model,
            output_path=args.output,
            progress_callback=progress_callback,
        )
    finally:
        pbar.close()
    
    print("\n" + "="*70)
    print("✅ COMPRESSION COMPLETE")
    print("="*70)
    print(f"  Base model:       {manifest.base_model_id}")
    print(f"  Fine-tune:        {manifest.finetune_model_id}")
    print(f"  Compression:      {manifest.compression_ratio:.1f}x")
    print(f"  Changed params:   {manifest.changed_params:,} / {manifest.total_params:,}")
    print(f"  Output:           {args.output}")
    print("="*70)
    print(f"\n💾 Delta saved to: {args.output}\n")
    
    return 0



def cmd_delta_reconstruct(args):
    """Reconstruct model from base + delta."""
    from core.delta import reconstruct_from_delta
    
    print("\n" + "="*70)
    print("🔄 SPARSE RECONSTRUCT")
    print("="*70)
    print(f"  Base model:     {args.base_model}")
    print(f"  Delta path:     {args.delta_path}")
    print(f"  Output:         {args.output}")
    print("="*70 + "\n")
    
    pbar = tqdm(total=100, desc="Reconstructing", bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]")
    last_progress = 0
    
    def progress_callback(msg, progress):
        nonlocal last_progress
        delta = int((progress - last_progress) * 100)
        if delta > 0:
            pbar.update(delta)
            last_progress = progress
        pbar.set_postfix_str(msg[:40])
    
    try:
        model = reconstruct_from_delta(
            base_model_id=args.base_model,
            delta_path=args.delta_path,
            progress_callback=progress_callback,
        )
    finally:
        pbar.close()
    
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


def cmd_compress_lossy(args):
    """Compress fine-tune with lossy compression (LoRA-equivalent quality)."""
    from core.delta import compress_delta_svd_full
    
    print("\n" + "="*70)
    print("🎯 SPARSE COMPRESS (Lossy - LoRA Quality)")
    print("="*70)
    print(f"  Base model:     {args.base_model}")
    print(f"  Fine-tune:      {args.finetune_model}")
    print(f"  Rank:           {args.rank}")
    print(f"  Output:         {args.output}")
    print("="*70 + "\n")
    
    pbar = tqdm(total=100, desc="Compressing (SVD)", bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}]")
    last_progress = 0
    
    def progress_callback(msg, progress):
        nonlocal last_progress
        delta = int((progress - last_progress) * 100)
        if delta > 0:
            pbar.update(delta)
            last_progress = progress
        pbar.set_postfix_str(msg[:40])
    
    try:
        manifest = compress_delta_svd_full(
            base_model_id=args.base_model,
            finetune_model_id=args.finetune_model,
            output_path=args.output,
            rank=args.rank,
            progress_callback=progress_callback,
        )
    finally:
        pbar.close()
    
    print("\n" + "="*70)
    print("✅ LOSSY COMPRESSION COMPLETE")
    print("="*70)
    print(f"  Base model:       {manifest.base_model_id}")
    print(f"  Fine-tune:        {manifest.finetune_model_id}")
    print(f"  Rank:             {manifest.rank}")
    print(f"  Compression:      {manifest.compression_ratio:.1f}x")
    print(f"  Original size:    {manifest.original_size_bytes / 1e9:.2f} GB")
    print(f"  Compressed size:  {manifest.compressed_size_bytes / 1e6:.1f} MB")
    print(f"  Avg error:        {manifest.avg_reconstruction_error:.6f}")
    print(f"  Max error:        {manifest.max_reconstruction_error:.6f}")
    print(f"  Output:           {args.output}")
    print("="*70)
    print(f"\n💾 Delta saved to: {args.output}\n")
    
    return 0


def cmd_reconstruct_lossy(args):
    """Reconstruct model from lossy-compressed delta."""
    from core.delta import reconstruct_from_svd_delta
    
    print(f"Sparse Reconstruct (Lossy)")
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
    
    model = reconstruct_from_svd_delta(
        base_model_id=args.base_model,
        delta_path=args.delta_path,
        progress_callback=progress_callback,
    )
    
    print()
    print("=" * 60)
    print("LOSSY RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"  Model loaded and ready for inference")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Note: This is a lossy reconstruction (LoRA-equivalent quality)")
    
    if args.output:
        print(f"  Saving to: {args.output}")
        model.save_pretrained(args.output)
        print(f"  Saved!")
    
    print("=" * 60)
    
    return 0


def cmd_dataset_compress(args):
    """Compress derivative dataset as delta from base."""
    from core.dataset_delta import compress_dataset_delta
    
    print(f"Sparse Dataset Compress")
    print(f"  Base:        {args.base}")
    print(f"  Derivative:  {args.derivative}")
    print(f"  Output:      {args.output}")
    print()
    
    manifest = compress_dataset_delta(
        base_dataset_id=args.base,
        derivative_dataset_id=args.derivative,
        output_dir=args.output
    )
    
    print()
    print("=" * 60)
    print("DATASET DELTA COMPRESSION RESULTS")
    print("=" * 60)
    print(f"  Base dataset:       {args.base}")
    print(f"  Derivative:         {args.derivative}")
    print(f"  Savings:            {manifest['size_stats']['savings_pct']:.1f}%")
    print(f"  Output:             {args.output}")
    print("=" * 60)
    
    return 0


def cmd_dataset_reconstruct(args):
    """Reconstruct dataset from base + delta."""
    from core.dataset_delta import reconstruct_from_dataset_delta
    
    print(f"Sparse Dataset Reconstruct")
    print(f"  Delta dir:  {args.delta_dir}")
    print()
    
    dataset = reconstruct_from_dataset_delta(args.delta_dir)
    
    print()
    print("=" * 60)
    print("DATASET RECONSTRUCTION COMPLETE")
    print("=" * 60)
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} samples")
    print("=" * 60)
    
    return 0


def cmd_dataset_estimate(args):
    """Estimate dataset delta savings."""
    from core.dataset_delta import estimate_dataset_delta_savings
    
    print(f"Sparse Dataset Estimate")
    print(f"  Base:        {args.base}")
    print(f"  Derivative:  {args.derivative}")
    print()
    
    stats = estimate_dataset_delta_savings(
        base_dataset_id=args.base,
        derivative_dataset_id=args.derivative,
        sample_size=args.sample_size or 1000
    )
    
    print()
    print("=" * 60)
    print("DATASET DELTA ESTIMATE")
    print("=" * 60)
    print(f"  Base size:       {stats.base_size_mb:.1f} MB")
    print(f"  Derivative size: {stats.derivative_size_mb:.1f} MB")
    print(f"  Delta size:      {stats.delta_size_mb:.1f} MB")
    print(f"  Savings:         {stats.savings_pct:.1f}%")
    print()
    print(f"  Shared samples:  {stats.num_shared_samples}")
    print(f"  New samples:     {stats.num_new_samples}")
    print("=" * 60)
    
    return 0


def cmd_info(args):
    """Show info about a delta artifact."""
    import os
    
    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}")
        return 1
    
    manifest_path = os.path.join(args.path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Not a valid delta artifact (no manifest.json)")
        return 1
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print()
    print("=" * 60)
    print("DELTA ARTIFACT INFO")
    print("=" * 60)
    
    if manifest.get("type") == "dataset_delta":
        print(f"  Type:              Dataset Delta")
        print(f"  Base dataset:      {manifest.get('base_dataset_id', 'unknown')}")
        print(f"  Derivative:        {manifest.get('derivative_dataset_id', 'unknown')}")
        if "size_stats" in manifest:
            stats = manifest["size_stats"]
            print(f"  Savings:           {stats.get('savings_pct', 0):.1f}%")
    else:
        print(f"  Type:              Model Delta")
        print(f"  Base model:        {manifest.get('base_model_id', 'unknown')}")
        print(f"  Fine-tune:         {manifest.get('finetune_model_id', 'unknown')}")
        print(f"  Compression:       {manifest.get('compression_ratio', 0):.2f}x")
        print(f"  Layers:            {manifest.get('num_layers', 0)}")
    
    print(f"  Created:           {manifest.get('created_at', 'unknown')}")
    print("=" * 60)
    
    return 0


def cmd_help(args):
    """Show detailed help and examples."""
    help_text = """
================================================================================
                    SPARSE - Delta Compression for Fine-tuned Models
================================================================================

Sparse compresses fine-tuned models by storing only the differences (deltas)
from base models. Instead of 14GB, share 50-500MB.

--------------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------------

  # Compress a fine-tune
  sparse compress meta-llama/Llama-3.1-8B ./my-finetune -o ./my-delta

  # Reconstruct from delta
  sparse reconstruct meta-llama/Llama-3.1-8B ./my-delta -o ./reconstructed

  # Lossy compression (~50MB output, LoRA-equivalent quality)
  sparse compress-lossy meta-llama/Llama-3.1-8B ./my-finetune -o ./lossy-delta --rank 64

--------------------------------------------------------------------------------
COMMANDS
--------------------------------------------------------------------------------

  MODEL COMPRESSION:
    compress          Compress fine-tune as delta from base (lossless)
    compress-lossy    Compress with lossy compression (~50MB output)

  RECONSTRUCTION:
    reconstruct       Reconstruct model from lossless delta
    reconstruct-lossy Reconstruct model from lossy delta

  DATASET COMPRESSION:
    dataset-compress     Compress derivative dataset as delta
    dataset-reconstruct  Reconstruct dataset from delta
    dataset-estimate     Estimate compression savings

  UTILITIES:
    info              Show info about a delta artifact
    help              Show this help message

--------------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------------

  # Compress Llama fine-tune
  sparse compress meta-llama/Llama-3.1-8B ./my-llama-finetune -o ./llama-delta

  # Lossy compression (smaller, ~95-99% quality)
  sparse compress-lossy meta-llama/Llama-3.1-8B ./my-finetune -o ./lossy-delta --rank 64

  # Check delta info
  sparse info ./llama-delta

  # Reconstruct for use
  sparse reconstruct meta-llama/Llama-3.1-8B ./llama-delta -o ./model-ready

--------------------------------------------------------------------------------
MORE HELP
--------------------------------------------------------------------------------

  sparse <command> --help    Show help for specific command
  sparse compress --help     Show compress options
  sparse compress-lossy --help  Show lossy compression options

================================================================================
    GitHub: https://github.com/gagansuie/sparse
    License: Apache 2.0
================================================================================
"""
    print(help_text)
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sparse",
        description="Sparse - Delta Compression for Fine-tuned Models and Datasets"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ==========================================================================
    # MODEL DELTA COMMANDS
    # ==========================================================================
    
    # sparse compress <base> <finetune> -o <output>
    compress_parser = subparsers.add_parser(
        "compress",
        help="Compress fine-tuned model as delta from base"
    )
    compress_parser.add_argument("base_model", help="Base model ID (e.g., meta-llama/Llama-2-7b-hf)")
    compress_parser.add_argument("finetune_model", help="Fine-tuned model ID or path")
    compress_parser.add_argument("--output", "-o", required=True, help="Output directory for delta")
    
    # sparse reconstruct <base> <delta> [-o <output>]
    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        help="Reconstruct model from base + delta"
    )
    reconstruct_parser.add_argument("base_model", help="Base model ID")
    reconstruct_parser.add_argument("delta_path", help="Path to delta artifact")
    reconstruct_parser.add_argument("--output", "-o", help="Save reconstructed model to path")
    
    # ==========================================================================
    # LOSSY COMPRESSION COMMANDS (~50MB output, LoRA-equivalent quality)
    # ==========================================================================
    
    # sparse compress-lossy <base> <finetune> -o <output> [--rank N]
    compress_lossy_parser = subparsers.add_parser(
        "compress-lossy",
        help="Compress fine-tune with lossy compression (~50MB, LoRA-equivalent quality)"
    )
    compress_lossy_parser.add_argument("base_model", help="Base model ID")
    compress_lossy_parser.add_argument("finetune_model", help="Fine-tuned model ID or path")
    compress_lossy_parser.add_argument("--output", "-o", required=True, help="Output directory")
    compress_lossy_parser.add_argument("--rank", "-r", type=int, default=16, help="Compression rank (default 16, higher = better quality)")
    
    # sparse reconstruct-lossy <base> <delta> [-o <output>]
    reconstruct_lossy_parser = subparsers.add_parser(
        "reconstruct-lossy",
        help="Reconstruct model from lossy-compressed delta"
    )
    reconstruct_lossy_parser.add_argument("base_model", help="Base model ID")
    reconstruct_lossy_parser.add_argument("delta_path", help="Path to lossy delta artifact")
    reconstruct_lossy_parser.add_argument("--output", "-o", help="Save reconstructed model to path")
    
    # ==========================================================================
    # DATASET DELTA COMMANDS
    # ==========================================================================
    
    # sparse dataset compress <base> <derivative> -o <output>
    dataset_compress_parser = subparsers.add_parser(
        "dataset-compress",
        help="Compress derivative dataset as delta from base"
    )
    dataset_compress_parser.add_argument("base", help="Base dataset ID (e.g., squad)")
    dataset_compress_parser.add_argument("derivative", help="Derivative dataset ID (e.g., squad_v2)")
    dataset_compress_parser.add_argument("--output", "-o", required=True, help="Output directory")
    
    # sparse dataset reconstruct <delta_dir>
    dataset_reconstruct_parser = subparsers.add_parser(
        "dataset-reconstruct",
        help="Reconstruct dataset from base + delta"
    )
    dataset_reconstruct_parser.add_argument("delta_dir", help="Path to dataset delta")
    
    # sparse dataset estimate <base> <derivative>
    dataset_estimate_parser = subparsers.add_parser(
        "dataset-estimate",
        help="Estimate dataset compression savings"
    )
    dataset_estimate_parser.add_argument("base", help="Base dataset ID")
    dataset_estimate_parser.add_argument("derivative", help="Derivative dataset ID")
    dataset_estimate_parser.add_argument("--sample-size", type=int, default=1000, help="Samples to analyze")
    
    # ==========================================================================
    # INFO COMMAND
    # ==========================================================================
    
    info_parser = subparsers.add_parser(
        "info",
        help="Show info about a delta artifact"
    )
    info_parser.add_argument("path", help="Path to delta artifact")
    
    # ==========================================================================
    # HELP COMMAND
    # ==========================================================================
    
    help_parser = subparsers.add_parser(
        "help",
        help="Show detailed help and examples"
    )
    
    # ==========================================================================
    # PARSE AND DISPATCH
    # ==========================================================================
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "compress": cmd_delta_compress,
        "reconstruct": cmd_delta_reconstruct,
        "compress-lossy": cmd_compress_lossy,
        "reconstruct-lossy": cmd_reconstruct_lossy,
        "dataset-compress": cmd_dataset_compress,
        "dataset-reconstruct": cmd_dataset_reconstruct,
        "dataset-estimate": cmd_dataset_estimate,
        "info": cmd_info,
        "help": cmd_help,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
