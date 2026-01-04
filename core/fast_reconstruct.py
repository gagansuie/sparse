"""
Fast Reconstruction + Auto-Caching for Sparse Delta Compression.

This module provides:
- DeltaCache: Manages reconstructed model caching and background reconstruction
- from_pretrained_with_delta: Drop-in replacement for HF's from_pretrained
- Rust-accelerated delta application for <10 second reconstruction

Key Design Decision: INTEGRATE with HuggingFace's existing cache, don't duplicate.
- Base models: Use HF Hub's cache (huggingface_hub already caches downloads)
- Reconstructed models: Cache in ~/.cache/sparse/reconstructed/
- Deltas: Could be hosted on HF Hub, benefit from Cloudflare CDN

Why this works with Cloudflare CDN + git-lfs:
- CDN makes downloads faster (edge caching)
- Sparse makes downloads SMALLER (500MB delta vs 14GB model)
- These are complementary, not competing

Timing targets:
- First download: ~same as LoRA (base from HF cache + delta)
- Subsequent downloads (same base): Delta only + <10s reconstruct
- Multiple fine-tunes: Pre-reconstruct all in background
"""

import gc
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    from core.delta import (
        DeltaManifest,
        decompress_delta_int8,
        decompress_delta_sparse,
    )
except ImportError:
    from delta import (
        DeltaManifest,
        decompress_delta_int8,
        decompress_delta_sparse,
    )

# Rust acceleration is REQUIRED
try:
    import sparse_core
except ImportError as e:
    raise ImportError(
        "Rust acceleration (sparse_core) is required but not installed. "
        "Install with: pip install sparse-llm"
    ) from e


def benchmark_reconstruction(tensor_size: int = 1_000_000, iterations: int = 10) -> Dict[str, Any]:
    """
    Benchmark Rust-accelerated delta application.
    
    Returns timing info and estimated reconstruction time for various model sizes.
    """
    ms_per_iter = sparse_core.benchmark_int8_apply(tensor_size, iterations)
    
    return {
        "rust_available": True,
        "tensor_size": tensor_size,
        "iterations": iterations,
        "ms_per_iteration": ms_per_iter,
        "estimated_times": {
            "1B_model": f"{(1_000_000_000 / tensor_size) * ms_per_iter / 1000:.2f}s",
            "7B_model": f"{(7_000_000_000 / tensor_size) * ms_per_iter / 1000:.2f}s",
            "13B_model": f"{(13_000_000_000 / tensor_size) * ms_per_iter / 1000:.2f}s",
            "70B_model": f"{(70_000_000_000 / tensor_size) * ms_per_iter / 1000:.2f}s",
        }
    }


@dataclass
class ReconstructionJob:
    """Tracks a background reconstruction job."""
    base_model_id: str
    delta_path: str
    output_path: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class CacheStats:
    """Statistics for the delta cache."""
    base_models_cached: int = 0
    deltas_reconstructed: int = 0
    total_cache_size_gb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_reconstruction_time_s: float = 0.0
    reconstruction_times: List[float] = field(default_factory=list)


class DeltaCache:
    """
    Manages base model caching and fast delta reconstruction.
    
    Key features:
    - Caches base models to avoid re-downloading
    - Reconstructs deltas in background threads
    - Rust-accelerated delta application
    - Tracks reconstruction jobs and progress
    
    Usage:
        cache = DeltaCache()
        model_path = cache.get_or_reconstruct(
            base_model_id="meta-llama/Llama-2-7b-hf",
            delta_path="./my_delta",
            background=False  # Wait for completion
        )
        model = AutoModelForCausalLM.from_pretrained(model_path)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size_gb: float = 100.0,
        max_workers: int = 2,
    ):
        """
        Initialize the delta cache.
        
        Args:
            cache_dir: Directory for cached models (default: ~/.cache/sparse)
            max_cache_size_gb: Maximum cache size in GB
            max_workers: Max concurrent reconstruction jobs
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/sparse")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NOTE: We don't cache base models ourselves!
        # HuggingFace Hub already caches downloads in ~/.cache/huggingface/hub/
        # We only cache RECONSTRUCTED models (the result of base + delta)
        self.reconstructed_dir = self.cache_dir / "reconstructed"
        self.reconstructed_dir.mkdir(exist_ok=True)
        
        self.max_cache_size_gb = max_cache_size_gb
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Track active jobs
        self.jobs: Dict[str, ReconstructionJob] = {}
        self.jobs_lock = threading.Lock()
        
        # Statistics
        self.stats = CacheStats()
        
        # Load existing cache index
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load the cache index from disk."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                data = json.load(f)
                self.stats.base_models_cached = data.get("base_models_cached", 0)
                self.stats.deltas_reconstructed = data.get("deltas_reconstructed", 0)
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        index_path = self.cache_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump({
                "base_models_cached": self.stats.base_models_cached,
                "deltas_reconstructed": self.stats.deltas_reconstructed,
            }, f)
    
    def _get_delta_id(self, base_model_id: str, delta_path: str) -> str:
        """Generate a unique ID for a delta reconstruction."""
        key = f"{base_model_id}:{delta_path}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _get_base_model_id_hash(self, base_model_id: str) -> str:
        """Generate a safe directory name for a base model."""
        return hashlib.md5(base_model_id.encode()).hexdigest()[:16]
    
    def is_base_cached_in_hf(self, base_model_id: str) -> bool:
        """
        Check if a base model is cached in HuggingFace's cache.
        
        Uses huggingface_hub to check if the model is already downloaded.
        This avoids duplicating HF's cache in our own directory.
        """
        try:
            from huggingface_hub import try_to_load_from_cache, HfFileSystem
            
            # Check if config.json is cached (indicates model is downloaded)
            cached_path = try_to_load_from_cache(
                repo_id=base_model_id,
                filename="config.json",
            )
            return cached_path is not None
        except Exception:
            # Fallback: try to check if model files exist in default cache
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            model_cache_name = f"models--{base_model_id.replace('/', '--')}"
            model_path = hf_cache / model_cache_name
            return model_path.exists()
    
    def is_reconstructed(self, base_model_id: str, delta_path: str) -> bool:
        """Check if a delta has already been reconstructed."""
        delta_id = self._get_delta_id(base_model_id, delta_path)
        output_path = self.reconstructed_dir / delta_id
        return output_path.exists() and (output_path / "config.json").exists()
    
    def get_reconstructed_path(self, base_model_id: str, delta_path: str) -> Optional[str]:
        """Get the path to a reconstructed model if it exists."""
        if self.is_reconstructed(base_model_id, delta_path):
            delta_id = self._get_delta_id(base_model_id, delta_path)
            self.stats.cache_hits += 1
            return str(self.reconstructed_dir / delta_id)
        self.stats.cache_misses += 1
        return None
    
    def ensure_base_in_hf_cache(
        self,
        base_model_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> bool:
        """
        Ensure a base model is in HuggingFace's cache.
        
        We don't maintain our own cache of base models - we use HF's cache.
        This just triggers a download if the model isn't already cached.
        
        Args:
            base_model_id: HuggingFace model ID
            progress_callback: Optional callback(msg, progress)
            
        Returns:
            True if model is now in cache
        """
        from huggingface_hub import snapshot_download
        
        def log(msg: str, progress: float = 0.0):
            if progress_callback:
                progress_callback(msg, progress)
            print(f"[DeltaCache] {msg}")
        
        # Already cached?
        if self.is_base_cached_in_hf(base_model_id):
            log(f"Base model already in HF cache: {base_model_id}", 1.0)
            return True
        
        log(f"Downloading base model to HF cache: {base_model_id}", 0.1)
        
        # Use HF's snapshot_download to cache the model
        # This downloads to ~/.cache/huggingface/hub/ (standard HF cache)
        try:
            snapshot_download(
                repo_id=base_model_id,
                repo_type="model",
            )
            log(f"Base model downloaded to HF cache: {base_model_id}", 1.0)
            return True
        except Exception as e:
            log(f"Failed to download base model: {e}", 0.0)
            return False
    
    def reconstruct_fast(
        self,
        base_model_id: str,
        delta_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        use_rust: bool = True,
    ) -> str:
        """
        Fast delta reconstruction using Rust acceleration.
        
        Target: <10 seconds for 7B models.
        
        Args:
            base_model_id: HuggingFace model ID for base
            delta_path: Path to delta artifact
            output_path: Where to save reconstructed model
            progress_callback: Optional callback(msg, progress)
            use_rust: Whether to use Rust acceleration
            
        Returns:
            Path to reconstructed model
        """
        from transformers import AutoModelForCausalLM, AutoConfig
        
        def log(msg: str, progress: float = 0.0):
            if progress_callback:
                progress_callback(msg, progress)
            print(f"[FastReconstruct] {msg}")
        
        start_time = time.time()
        delta_path = Path(delta_path)
        
        # Generate output path if not provided
        if output_path is None:
            delta_id = self._get_delta_id(base_model_id, str(delta_path))
            output_path = self.reconstructed_dir / delta_id
        output_path = Path(output_path)
        
        # Already reconstructed?
        if output_path.exists() and (output_path / "config.json").exists():
            log("Already reconstructed, returning cached version", 1.0)
            return str(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        log("Loading delta manifest...", 0.05)
        with open(delta_path / "manifest.json", "r") as f:
            manifest = DeltaManifest.from_dict(json.load(f))
        
        # Load base model from HF cache (or download if not cached)
        # NOTE: We use HF's cache directly - no duplication!
        log("Loading base model...", 0.1)
        if self.is_base_cached_in_hf(base_model_id):
            log("Base model in HF cache, loading...", 0.15)
        else:
            log("Base model not in cache, will download from HF Hub (CDN-accelerated)", 0.15)
        
        # Always load from model_id - transformers handles caching automatically
        base_path = base_model_id
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        load_time = time.time() - start_time
        log(f"Base model loaded in {load_time:.1f}s", 0.3)
        
        # Apply deltas
        reconstruct_start = time.time()
        state_dict = model.state_dict()
        
        total_layers = len(manifest.layer_deltas)
        for i, layer_info in enumerate(manifest.layer_deltas):
            layer_name = layer_info["name"]
            method = layer_info.get("method", "sparse")
            
            if method == "zero":
                continue
            
            progress = 0.3 + (0.6 * (i / total_layers))
            log(f"Applying delta to {layer_name}...", progress)
            
            if layer_name not in state_dict:
                continue
            
            base_weight = state_dict[layer_name]
            
            # Load and decompress delta
            if method == "sparse":
                delta_file = delta_path / f"{layer_name.replace('.', '_')}_delta.pt"
                if delta_file.exists():
                    delta_data = torch.load(delta_file, map_location="cpu")
                    indices = delta_data["indices"]
                    values = delta_data["values"]
                    
                    # Rust-accelerated decompression
                    delta = decompress_delta_sparse(
                        indices, values, tuple(base_weight.shape), base_weight.dtype
                    )
                    
                    state_dict[layer_name] = base_weight + delta.to(base_weight.device)
            
            elif method == "int8":
                delta_file = delta_path / f"{layer_name.replace('.', '_')}_delta_int8.bin"
                scale_file = delta_path / f"{layer_name.replace('.', '_')}_scale.pt"
                
                if delta_file.exists() and scale_file.exists():
                    with open(delta_file, "rb") as f:
                        quantized_bytes = f.read()
                    scale = torch.load(scale_file, map_location="cpu").item()
                    
                    # GPU-accelerated INT8 delta application (if CUDA available)
                    if torch.cuda.is_available() and base_weight.is_cuda:
                        try:
                            # Use GPU-optimized ops for faster reconstruction
                            gpu_ops = sparse_core.GpuOptimizedOps(tile_size=256, use_fma=True)
                            
                            # Decompress to quantized tensor
                            import numpy as np
                            quantized = np.frombuffer(quantized_bytes, dtype=np.int8)
                            quantized = torch.from_numpy(quantized).reshape(base_weight.shape)
                            
                            # Apply with GPU-optimized tiled processing
                            base_flat = base_weight.flatten().cpu().numpy()
                            quant_flat = quantized.flatten().cpu().numpy()
                            result = gpu_ops.apply_int8_delta_tiled(base_flat, quant_flat, scale)
                            
                            # Reshape and move back to GPU
                            delta_result = torch.from_numpy(np.array(result)).reshape(base_weight.shape)
                            state_dict[layer_name] = delta_result.to(base_weight.device, dtype=base_weight.dtype)
                        except Exception:
                            # Fallback to standard Rust decompression
                            delta = decompress_delta_int8(
                                quantized_bytes, scale, tuple(base_weight.shape), base_weight.dtype
                            )
                            state_dict[layer_name] = base_weight + delta.to(base_weight.device)
                    else:
                        # Standard Rust-accelerated decompression
                        delta = decompress_delta_int8(
                            quantized_bytes, scale, tuple(base_weight.shape), base_weight.dtype
                        )
                        state_dict[layer_name] = base_weight + delta.to(base_weight.device)
        
        # Load state dict back
        model.load_state_dict(state_dict)
        
        reconstruct_time = time.time() - reconstruct_start
        log(f"Deltas applied in {reconstruct_time:.1f}s", 0.9)
        
        # Save reconstructed model
        log("Saving reconstructed model...", 0.95)
        model.save_pretrained(output_path, safe_serialization=True)
        
        # Copy tokenizer if exists in delta
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
        for tf in tokenizer_files:
            src = delta_path / tf
            if src.exists():
                shutil.copy(src, output_path / tf)
        
        # Save reconstruction metadata
        total_time = time.time() - start_time
        meta = {
            "base_model_id": base_model_id,
            "delta_path": str(delta_path),
            "reconstructed_at": time.time(),
            "total_time_s": total_time,
            "load_time_s": load_time,
            "reconstruct_time_s": reconstruct_time,
            "rust_acceleration": use_rust,
        }
        with open(output_path / "sparse_reconstruct_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        # Update stats
        self.stats.deltas_reconstructed += 1
        self.stats.reconstruction_times.append(reconstruct_time)
        self.stats.avg_reconstruction_time_s = sum(self.stats.reconstruction_times) / len(self.stats.reconstruction_times)
        self._save_cache_index()
        
        # Cleanup
        del model, state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        log(f"Reconstruction complete in {total_time:.1f}s (delta application: {reconstruct_time:.1f}s)", 1.0)
        return str(output_path)
    
    def get_or_reconstruct(
        self,
        base_model_id: str,
        delta_path: str,
        background: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[str]:
        """
        Get reconstructed model path, reconstructing if necessary.
        
        Args:
            base_model_id: HuggingFace model ID for base
            delta_path: Path to delta artifact
            background: If True, start reconstruction in background thread
            progress_callback: Optional callback(msg, progress)
            
        Returns:
            Path to reconstructed model, or None if background=True and not yet ready
        """
        # Already reconstructed?
        cached_path = self.get_reconstructed_path(base_model_id, delta_path)
        if cached_path:
            return cached_path
        
        delta_id = self._get_delta_id(base_model_id, delta_path)
        
        # Check if job already running
        with self.jobs_lock:
            if delta_id in self.jobs:
                job = self.jobs[delta_id]
                if job.status == "completed":
                    return job.output_path
                elif job.status == "running":
                    if background:
                        return None  # Still running
                    else:
                        # Wait for completion (poll)
                        while job.status == "running":
                            time.sleep(0.5)
                        return job.output_path if job.status == "completed" else None
        
        # Start reconstruction
        output_path = str(self.reconstructed_dir / delta_id)
        job = ReconstructionJob(
            base_model_id=base_model_id,
            delta_path=delta_path,
            output_path=output_path,
            status="pending",
            started_at=time.time(),
        )
        
        with self.jobs_lock:
            self.jobs[delta_id] = job
        
        if background:
            # Submit to thread pool
            self.executor.submit(self._run_reconstruction_job, delta_id, progress_callback)
            return None
        else:
            # Run synchronously
            self._run_reconstruction_job(delta_id, progress_callback)
            return job.output_path if job.status == "completed" else None
    
    def _run_reconstruction_job(
        self,
        delta_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """Run a reconstruction job."""
        with self.jobs_lock:
            job = self.jobs[delta_id]
            job.status = "running"
        
        try:
            result_path = self.reconstruct_fast(
                base_model_id=job.base_model_id,
                delta_path=job.delta_path,
                output_path=job.output_path,
                progress_callback=progress_callback,
            )
            
            with self.jobs_lock:
                job.status = "completed"
                job.output_path = result_path
                job.completed_at = time.time()
        
        except Exception as e:
            with self.jobs_lock:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = time.time()
    
    def prefetch_deltas(
        self,
        base_model_id: str,
        delta_paths: List[str],
    ) -> Dict[str, str]:
        """
        Pre-fetch and reconstruct multiple deltas in background.
        
        This is the key to beating LoRA: if a user wants to try 10 fine-tunes
        from the same base, we can reconstruct all of them in parallel while
        they're using the first one.
        
        Args:
            base_model_id: Base model ID
            delta_paths: List of delta paths to prefetch
            
        Returns:
            Dict mapping delta_path -> job_id for tracking
        """
        job_ids = {}
        
        # First, ensure base model is in HF's cache
        # This triggers a download if not already cached (CDN-accelerated)
        if not self.is_base_cached_in_hf(base_model_id):
            self.ensure_base_in_hf_cache(base_model_id)
        
        # Start background jobs for each delta
        for delta_path in delta_paths:
            delta_id = self._get_delta_id(base_model_id, delta_path)
            
            # Skip if already reconstructed
            if self.is_reconstructed(base_model_id, delta_path):
                job_ids[delta_path] = delta_id
                continue
            
            # Start background reconstruction
            self.get_or_reconstruct(
                base_model_id=base_model_id,
                delta_path=delta_path,
                background=True,
            )
            job_ids[delta_path] = delta_id
        
        return job_ids
    
    def get_job_status(self, delta_id: str) -> Optional[ReconstructionJob]:
        """Get the status of a reconstruction job."""
        with self.jobs_lock:
            return self.jobs.get(delta_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "base_models_cached": self.stats.base_models_cached,
            "deltas_reconstructed": self.stats.deltas_reconstructed,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "avg_reconstruction_time_s": self.stats.avg_reconstruction_time_s,
            "cache_dir": str(self.cache_dir),
        }


def from_pretrained_with_delta(
    model_id_or_delta: str,
    base_model_id: Optional[str] = None,
    delta_mode: str = "auto",
    cache: Optional[DeltaCache] = None,
    **kwargs,
):
    """
    Drop-in replacement for AutoModelForCausalLM.from_pretrained()
    that automatically uses deltas when available.
    
    Args:
        model_id_or_delta: HF model ID or path to delta artifact
        base_model_id: Base model ID (required if model_id_or_delta is a delta)
        delta_mode: "auto", "prefer_delta", "full_only"
        cache: Optional DeltaCache instance
        **kwargs: Passed to from_pretrained
        
    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM
    
    if cache is None:
        cache = DeltaCache()
    
    model_path = Path(model_id_or_delta)
    
    # Check if it's a delta artifact
    is_delta = model_path.exists() and (model_path / "manifest.json").exists()
    
    if is_delta and delta_mode != "full_only":
        if base_model_id is None:
            # Try to read base_model_id from manifest
            with open(model_path / "manifest.json", "r") as f:
                manifest = json.load(f)
                base_model_id = manifest.get("base_model_id")
        
        if base_model_id is None:
            raise ValueError("base_model_id required for delta artifacts")
        
        # Reconstruct
        reconstructed_path = cache.get_or_reconstruct(
            base_model_id=base_model_id,
            delta_path=str(model_path),
            background=False,
        )
        
        return AutoModelForCausalLM.from_pretrained(reconstructed_path, **kwargs)
    
    else:
        # Standard loading
        return AutoModelForCausalLM.from_pretrained(model_id_or_delta, **kwargs)


# Global cache instance for convenience
_global_cache: Optional[DeltaCache] = None


def get_global_cache() -> DeltaCache:
    """Get or create the global DeltaCache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DeltaCache()
    return _global_cache
