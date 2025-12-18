"""
TenPak Studio - REST API

FastAPI endpoints for compression jobs and artifact management.

Endpoints:
    POST /compress     - Start a compression job
    GET  /status/{id}  - Get job status
    GET  /artifact/{id} - Download compressed artifact
    POST /evaluate     - Evaluate a model (baseline or compressed)
    GET  /jobs         - List recent jobs

Usage:
    uvicorn tenpak.studio.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os

from .jobs import (
    CompressionJob,
    JobStatus,
    create_job,
    get_job,
    get_job_status,
    list_jobs,
    start_job_async,
)

app = FastAPI(
    title="TenPak Studio",
    description="Model compression API for LLMs. Achieve 7x+ compression with <2% quality loss.",
    version="0.1.0",
)


# ============================================================================
# Request/Response Models
# ============================================================================

class CompressRequest(BaseModel):
    """Request to start a compression job."""
    model_id: str  # HuggingFace model ID or local path
    target: str = "balanced"  # "quality", "balanced", "size"
    hardware: str = "cuda"  # "a10g", "t4", "cpu", "cuda"
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "target": "balanced",
                "hardware": "cuda"
            }
        }
    }


class CompressResponse(BaseModel):
    """Response from starting a compression job."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response."""
    id: str
    model_id: str
    status: str
    progress: float
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    compression_ratio: Optional[float]
    baseline_ppl: Optional[float]
    compressed_ppl: Optional[float]
    ppl_delta: Optional[float]
    artifact_path: Optional[str]
    error: Optional[str]
    layers_processed: int
    total_layers: int


class EvaluateRequest(BaseModel):
    """Request to evaluate a model."""
    model_id: str
    artifact_id: Optional[str] = None  # If provided, compare baseline vs compressed
    num_samples: int = 50
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "num_samples": 50
            }
        }
    }


class EvaluateResponse(BaseModel):
    """Evaluation response."""
    model_id: str
    ppl: float
    num_samples: int
    artifact_id: Optional[str] = None
    compressed_ppl: Optional[float] = None
    ppl_delta: Optional[float] = None


class OptimizeRequest(BaseModel):
    """Request to start an optimization job."""
    model_id: str
    hardware: str = "cuda"
    max_ppl_delta: float = 2.0
    max_latency_p99_ms: float = 100.0
    min_throughput_tps: float = 1000.0
    candidates: Optional[List[str]] = None
    include_calibration: bool = True
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "hardware": "a10g",
                "max_ppl_delta": 2.0,
                "max_latency_p99_ms": 100.0,
                "min_throughput_tps": 1000.0,
            }
        }
    }


class OptimizeResponse(BaseModel):
    """Response from optimization."""
    status: str
    model_id: str
    hardware: str
    winner: Optional[dict] = None
    cost_savings_pct: float = 0.0
    all_candidates: List[dict] = []
    optimization_time_s: float = 0.0
    error: Optional[str] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root - health check and info."""
    return {
        "service": "TenPak Studio",
        "version": "0.1.0",
        "status": "healthy",
        "docs": "/docs",
        "endpoints": {
            "compress": "POST /compress",
            "status": "GET /status/{job_id}",
            "artifact": "GET /artifact/{job_id}",
            "evaluate": "POST /evaluate",
            "jobs": "GET /jobs",
        }
    }


@app.post("/compress", response_model=CompressResponse)
async def compress(request: CompressRequest, background_tasks: BackgroundTasks):
    """Start a model compression job.
    
    Creates a new job and starts compression in the background.
    Poll /status/{job_id} to monitor progress.
    
    Targets:
    - **quality**: Conservative compression, best PPL (~5x compression)
    - **balanced**: v10 config, good balance (~7x compression, <2% PPL)
    - **size**: Aggressive compression, may have higher PPL (~8x+ compression)
    """
    # Validate target
    valid_targets = ["quality", "balanced", "size"]
    if request.target not in valid_targets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target '{request.target}'. Must be one of: {valid_targets}"
        )
    
    # Create job
    job = create_job(
        model_id=request.model_id,
        target=request.target,
        hardware=request.hardware
    )
    
    # Start in background
    background_tasks.add_task(start_job_async, job)
    
    return CompressResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Compression job started. Poll /status/{job.id} for progress."
    )


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Get the status of a compression job.
    
    Returns current progress, metrics, and results when complete.
    """
    job_status = get_job_status(job_id)
    
    if job_status is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return job_status


@app.get("/artifact/{job_id}")
async def artifact(job_id: str):
    """Download the compressed artifact for a completed job.
    
    Returns the compressed model artifact (weights + metadata).
    """
    job = get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not completed (status: {job.status.value})"
        )
    
    if job.artifact_path is None or not os.path.exists(job.artifact_path):
        raise HTTPException(
            status_code=404,
            detail=f"Artifact for job '{job_id}' not found"
        )
    
    return FileResponse(
        job.artifact_path,
        media_type="application/octet-stream",
        filename=f"tenpak_{job_id}.bin"
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    """Evaluate a model's perplexity.
    
    If artifact_id is provided, also evaluates the compressed version
    and returns the PPL delta.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from core.calibration import compute_ppl
    
    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(request.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            request.model_id,
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
        )
        
        # Load eval data
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100]
        
        # Compute PPL
        ppl = compute_ppl(model, tokenizer, texts, device, max_samples=request.num_samples)
        
        return EvaluateResponse(
            model_id=request.model_id,
            ppl=ppl,
            num_samples=request.num_samples
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
async def get_jobs(limit: int = 10):
    """List recent compression jobs.
    
    Returns the most recent jobs, sorted by creation time (newest first).
    """
    return {"jobs": list_jobs(limit=limit)}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """Find optimal compression config for a model.
    
    Benchmarks multiple compression candidates and selects the cheapest
    one that meets the specified constraints (PPL, latency, throughput).
    
    This is a synchronous operation that may take several minutes.
    """
    from optimizer import optimize_model, OptimizationConstraints
    
    try:
        constraints = OptimizationConstraints(
            max_ppl_delta=request.max_ppl_delta,
            max_latency_p99_ms=request.max_latency_p99_ms,
            min_throughput_tps=request.min_throughput_tps,
        )
        
        result = optimize_model(
            model_id=request.model_id,
            hardware=request.hardware,
            constraints=constraints,
            candidates=request.candidates,
            include_calibration=request.include_calibration,
        )
        
        return OptimizeResponse(
            status="completed",
            model_id=result.model_id,
            hardware=result.hardware,
            winner=result.winner.to_dict() if result.winner else None,
            cost_savings_pct=result.cost_savings_pct,
            all_candidates=[r.to_dict() for r in result.all_results],
            optimization_time_s=result.optimization_time_s,
        )
        
    except Exception as e:
        return OptimizeResponse(
            status="failed",
            model_id=request.model_id,
            hardware=request.hardware,
            error=str(e),
        )


@app.get("/optimize/candidates")
async def list_candidates():
    """List available compression candidates."""
    from optimizer import CANDIDATE_PRESETS
    
    return {
        "candidates": [
            {
                "name": name,
                "method": c.method.value,
                "expected_compression": c.expected_compression,
                "expected_ppl_delta": c.expected_ppl_delta,
                "requires_calibration": c.requires_calibration,
            }
            for name, c in CANDIDATE_PRESETS.items()
        ]
    }


# ============================================================================
# Delta Compression Endpoints
# ============================================================================

class DeltaCompressRequest(BaseModel):
    """Request to compress a fine-tune as delta."""
    base_model_id: str
    finetune_model_id: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "base_model_id": "meta-llama/Llama-2-7b-hf",
                "finetune_model_id": "my-org/llama-2-7b-finetuned"
            }
        }
    }


class DeltaCompressResponse(BaseModel):
    """Response from delta compression."""
    status: str
    base_model_id: str
    finetune_model_id: str
    compression_ratio: float = 0.0
    changed_params_pct: float = 0.0
    artifact_path: Optional[str] = None
    error: Optional[str] = None


class DeltaEstimateResponse(BaseModel):
    """Response from delta estimation."""
    base_model_id: str
    finetune_model_id: str
    estimated_compression: float
    avg_sparsity: float
    sample_layers: int


@app.post("/delta/compress", response_model=DeltaCompressResponse)
async def delta_compress(request: DeltaCompressRequest):
    """Compress a fine-tuned model as delta from base model.
    
    Stores only the differences between base and fine-tuned model,
    achieving 80-95% storage reduction for typical fine-tunes.
    """
    from core.delta import compress_delta
    import tempfile
    
    try:
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="tenpak_delta_")
        
        manifest = compress_delta(
            base_model_id=request.base_model_id,
            finetune_model_id=request.finetune_model_id,
            output_path=output_dir,
        )
        
        changed_pct = (manifest.changed_params / manifest.total_params * 100) if manifest.total_params > 0 else 0
        
        return DeltaCompressResponse(
            status="completed",
            base_model_id=manifest.base_model_id,
            finetune_model_id=manifest.finetune_model_id,
            compression_ratio=manifest.compression_ratio,
            changed_params_pct=changed_pct,
            artifact_path=output_dir,
        )
        
    except Exception as e:
        return DeltaCompressResponse(
            status="failed",
            base_model_id=request.base_model_id,
            finetune_model_id=request.finetune_model_id,
            error=str(e),
        )


@app.post("/delta/estimate", response_model=DeltaEstimateResponse)
async def delta_estimate(request: DeltaCompressRequest):
    """Estimate delta compression savings without full compression.
    
    Quick analysis by sampling a few layers to estimate storage savings.
    """
    from core.delta import estimate_delta_savings
    
    result = estimate_delta_savings(
        base_model_id=request.base_model_id,
        finetune_model_id=request.finetune_model_id,
    )
    
    return DeltaEstimateResponse(
        base_model_id=request.base_model_id,
        finetune_model_id=request.finetune_model_id,
        estimated_compression=result["estimated_compression"],
        avg_sparsity=result["avg_sparsity"],
        sample_layers=result["sample_layers"],
    )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
