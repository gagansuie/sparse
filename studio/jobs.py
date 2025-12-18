"""
TenPak Studio - Job Management

Handles compression job lifecycle: create, run, monitor, complete.
Jobs run asynchronously and can be polled for status.
"""

import uuid
import time
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime

import torch
import torch.nn as nn


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CompressionJob:
    """Represents a compression job."""
    id: str
    model_id: str
    target: str  # "quality", "balanced", "size"
    hardware: str  # "a10g", "t4", "cpu"
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results (populated on completion)
    compression_ratio: Optional[float] = None
    baseline_ppl: Optional[float] = None
    compressed_ppl: Optional[float] = None
    ppl_delta: Optional[float] = None
    artifact_path: Optional[str] = None
    error: Optional[str] = None
    
    # Metrics tracking
    layers_processed: int = 0
    total_layers: int = 0
    original_size_mb: float = 0.0
    compressed_size_mb: float = 0.0


# In-memory job store (replace with Redis/DB for production)
_jobs: Dict[str, CompressionJob] = {}
_jobs_lock = threading.Lock()


def create_job(
    model_id: str,
    target: str = "balanced",
    hardware: str = "cuda"
) -> CompressionJob:
    """Create a new compression job.
    
    Args:
        model_id: HuggingFace model ID or local path
        target: Compression target ("quality", "balanced", "size")
        hardware: Target hardware ("a10g", "t4", "cpu", "cuda")
        
    Returns:
        CompressionJob instance
    """
    job_id = str(uuid.uuid4())[:8]
    job = CompressionJob(
        id=job_id,
        model_id=model_id,
        target=target,
        hardware=hardware
    )
    
    with _jobs_lock:
        _jobs[job_id] = job
    
    return job


def get_job(job_id: str) -> Optional[CompressionJob]:
    """Get a job by ID."""
    with _jobs_lock:
        return _jobs.get(job_id)


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status as a dictionary."""
    job = get_job(job_id)
    if job is None:
        return None
    
    return {
        "id": job.id,
        "model_id": job.model_id,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "compression_ratio": job.compression_ratio,
        "baseline_ppl": job.baseline_ppl,
        "compressed_ppl": job.compressed_ppl,
        "ppl_delta": job.ppl_delta,
        "artifact_path": job.artifact_path,
        "error": job.error,
        "layers_processed": job.layers_processed,
        "total_layers": job.total_layers,
    }


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """List recent jobs."""
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [get_job_status(j.id) for j in jobs[:limit]]


def run_compression_job(job: CompressionJob) -> None:
    """Execute a compression job (runs in background thread).
    
    This is the main compression pipeline:
    1. Load model
    2. Collect calibration stats
    3. Allocate bits per layer
    4. Compress each layer
    5. Evaluate PPL
    6. Save artifact
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    from core.calibration import collect_calibration_stats, compute_ppl
    from core.allocation import allocate_bits
    from core.codecs import compress_int4_awq
    
    try:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.progress = 0.05
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() and job.hardware != "cpu" else "cpu"
        
        # Load model
        print(f"[JOB {job.id}] Loading model {job.model_id}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(job.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            job.model_id,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        job.progress = 0.15
        
        # Load calibration data
        print(f"[JOB {job.id}] Loading calibration data...", flush=True)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100]
        calibration_texts = texts[:128]
        eval_texts = texts[:50]
        job.progress = 0.20
        
        # Baseline PPL
        print(f"[JOB {job.id}] Computing baseline PPL...", flush=True)
        job.baseline_ppl = compute_ppl(model, tokenizer, eval_texts, device, max_samples=50)
        print(f"[JOB {job.id}] Baseline PPL: {job.baseline_ppl:.4f}", flush=True)
        job.progress = 0.25
        
        # Calibration
        print(f"[JOB {job.id}] Collecting calibration stats...", flush=True)
        fisher_scores, activation_scales, hessian_diags, input_samples = \
            collect_calibration_stats(model, tokenizer, calibration_texts, num_samples=64, device=device)
        job.progress = 0.35
        
        # Allocation
        print(f"[JOB {job.id}] Allocating bits...", flush=True)
        allocations = allocate_bits(model, fisher_scores, target=job.target)
        job.total_layers = len(allocations)
        job.progress = 0.40
        
        # Compression
        print(f"[JOB {job.id}] Compressing {job.total_layers} layers...", flush=True)
        total_original = 0
        total_compressed = 0
        
        for i, (name, alloc) in enumerate(allocations.items()):
            # Find module
            module = None
            for n, m in model.named_modules():
                if n == name and hasattr(m, 'weight'):
                    module = m
                    break
            
            if module is None:
                continue
            
            weight = module.weight.data
            orig_size = weight.numel() * 4  # FP32 equivalent
            total_original += orig_size
            
            # Get calibration data for this layer
            act_scale = activation_scales.get(name, None)
            
            # Compress
            deq_weight, comp_ratio = compress_int4_awq(
                weight,
                group_size=alloc.group_size,
                act_scale=act_scale,
                outlier_pct=0.5
            )
            
            # Update weight in-place
            module.weight.data = deq_weight.to(weight.dtype)
            
            compressed_size = orig_size / comp_ratio
            total_compressed += compressed_size
            
            job.layers_processed = i + 1
            job.progress = 0.40 + 0.50 * (i + 1) / job.total_layers
        
        job.original_size_mb = total_original / 1e6
        job.compressed_size_mb = total_compressed / 1e6
        job.compression_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
        job.progress = 0.90
        
        # Evaluate compressed model
        print(f"[JOB {job.id}] Evaluating compressed model...", flush=True)
        job.compressed_ppl = compute_ppl(model, tokenizer, eval_texts, device, max_samples=50)
        job.ppl_delta = ((job.compressed_ppl - job.baseline_ppl) / job.baseline_ppl) * 100
        
        print(f"[JOB {job.id}] Results: {job.compression_ratio:.2f}x compression, {job.ppl_delta:+.2f}% PPL delta", flush=True)
        
        # TODO: Save artifact to storage
        job.artifact_path = f"/tmp/tenpak_artifacts/{job.id}"
        
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.progress = 1.0
        
        print(f"[JOB {job.id}] Completed successfully!", flush=True)
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.utcnow()
        print(f"[JOB {job.id}] Failed: {e}", flush=True)
        traceback.print_exc()


def start_job_async(job: CompressionJob) -> None:
    """Start a job in a background thread."""
    thread = threading.Thread(target=run_compression_job, args=(job,), daemon=True)
    thread.start()
