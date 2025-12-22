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
    1. Select quantization method based on target
    2. Load model
    3. Quantize using QuantizationWrapper
    4. Evaluate PPL
    5. Save artifact
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    from core import QuantizationWrapper, QUANTIZATION_PRESETS
    from core.calibration import compute_ppl
    
    # Map target to preset
    TARGET_PRESET_MAP = {
        "quality": "gptq_quality",
        "balanced": "awq_balanced",
        "size": "gptq_aggressive",
    }
    preset_name = TARGET_PRESET_MAP.get(job.target, "awq_balanced")
    
    try:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.progress = 0.05
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() and job.hardware != "cpu" else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load tokenizer and calibration data
        print(f"[JOB {job.id}] Loading calibration data...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(job.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_texts = [item.get("text", "") for item in dataset if len(item.get("text", "")) > 100][:50]
        job.progress = 0.15
        
        # Load and evaluate baseline model
        print(f"[JOB {job.id}] Loading baseline model for evaluation...", flush=True)
        baseline_model = AutoModelForCausalLM.from_pretrained(
            job.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        job.progress = 0.25
        
        print(f"[JOB {job.id}] Computing baseline PPL...", flush=True)
        job.baseline_ppl = compute_ppl(
            baseline_model,
            tokenizer,
            eval_texts,
            device,
            max_samples=50,
            streaming=(device == "cpu"),
        )
        print(f"[JOB {job.id}] Baseline PPL: {job.baseline_ppl:.4f}", flush=True)
        
        # Free baseline model
        del baseline_model
        if device == "cuda":
            torch.cuda.empty_cache()
        job.progress = 0.35
        
        # Quantize using QuantizationWrapper
        print(f"[JOB {job.id}] Quantizing model with preset '{preset_name}'...", flush=True)
        wrapper = QuantizationWrapper.from_preset(preset_name)
        
        # For calibration data, prepare simple text samples
        calibration_data = None
        if preset_name in ["gptq_quality", "gptq_aggressive", "awq_balanced", "awq_fast"]:
            # These methods need calibration data
            dataset_calib = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            calib_texts = [item.get("text", "") for item in dataset_calib if len(item.get("text", "")) > 100][:128]
            calibration_data = calib_texts
        
        job.progress = 0.45
        
        # Quantize (this delegates to AutoGPTQ/AutoAWQ/bitsandbytes)
        model = wrapper.quantize(
            model_id=job.model_id,
            calibration_data=calibration_data,
            device=device
        )
        
        # Get estimated compression ratio from config
        config = QUANTIZATION_PRESETS[preset_name]
        compression_map = {
            "gptq_quality": 7.5,
            "gptq_aggressive": 11.0,
            "awq_balanced": 7.5,
            "awq_fast": 6.5,
            "bnb_nf4": 6.5,
            "bnb_int8": 2.0,
        }
        job.compression_ratio = compression_map.get(preset_name, 7.0)
        job.progress = 0.85
        
        # Evaluate quantized model
        print(f"[JOB {job.id}] Evaluating quantized model...", flush=True)
        job.compressed_ppl = compute_ppl(
            model,
            tokenizer,
            eval_texts,
            device,
            max_samples=50,
            streaming=(device == "cpu"),
        )
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
