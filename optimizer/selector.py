"""
TenPak Optimizer - Constraint-Based Selection

Selects the optimal compression candidate based on constraints.
"""

import gc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

import torch

from .candidates import CompressionCandidate, generate_candidates, CompressionMethod
from .benchmark import BenchmarkResult, benchmark_candidate, estimate_hardware_type


@dataclass
class OptimizationConstraints:
    """Constraints for optimization."""
    max_ppl_delta: float = 2.0  # Maximum PPL delta in %
    max_latency_p99_ms: float = 100.0  # Maximum p99 latency in ms
    min_throughput_tps: float = 1000.0  # Minimum throughput in tokens/sec
    max_memory_gb: Optional[float] = None  # Maximum memory usage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_ppl_delta": self.max_ppl_delta,
            "max_latency_p99_ms": self.max_latency_p99_ms,
            "min_throughput_tps": self.min_throughput_tps,
            "max_memory_gb": self.max_memory_gb,
        }


@dataclass
class OptimizationResult:
    """Result of the optimization process."""
    model_id: str
    hardware: str
    constraints: OptimizationConstraints
    
    # Winner
    winner: Optional[BenchmarkResult] = None
    winner_candidate: Optional[CompressionCandidate] = None
    
    # All results
    all_results: List[BenchmarkResult] = field(default_factory=list)
    passing_results: List[BenchmarkResult] = field(default_factory=list)
    
    # Metadata
    total_candidates: int = 0
    candidates_evaluated: int = 0
    optimization_time_s: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Savings
    cost_savings_pct: float = 0.0  # vs FP16 baseline
    compression_vs_baseline: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hardware": self.hardware,
            "constraints": self.constraints.to_dict(),
            "winner": self.winner.to_dict() if self.winner else None,
            "total_candidates": self.total_candidates,
            "candidates_evaluated": self.candidates_evaluated,
            "optimization_time_s": self.optimization_time_s,
            "cost_savings_pct": self.cost_savings_pct,
            "compression_vs_baseline": self.compression_vs_baseline,
            "all_results": [r.to_dict() for r in self.all_results],
        }


def select_optimal(
    results: List[BenchmarkResult],
    constraints: OptimizationConstraints,
) -> Optional[BenchmarkResult]:
    """Select the optimal result from benchmark results.
    
    Selection criteria:
    1. Must pass all constraints
    2. Among passing candidates, select lowest cost_per_1m_tokens
    
    Args:
        results: List of benchmark results
        constraints: Optimization constraints
        
    Returns:
        Best BenchmarkResult or None if no candidates pass
    """
    passing = []
    
    for result in results:
        # Check PPL constraint
        if result.ppl_delta_pct > constraints.max_ppl_delta:
            continue
            
        # Check latency constraint
        if result.latency_p99_ms > constraints.max_latency_p99_ms:
            continue
            
        # Check throughput constraint
        if result.throughput_tps < constraints.min_throughput_tps:
            continue
            
        # Check memory constraint (if specified)
        if constraints.max_memory_gb and result.memory_peak_gb > constraints.max_memory_gb:
            continue
            
        passing.append(result)
    
    if not passing:
        return None
    
    # Select lowest cost
    return min(passing, key=lambda r: r.cost_per_1m_tokens)


def optimize_model(
    model_id: str,
    hardware: Optional[str] = None,
    constraints: Optional[OptimizationConstraints] = None,
    candidates: Optional[List[str]] = None,
    include_calibration: bool = True,
    num_eval_samples: int = 50,
    num_benchmark_samples: int = 30,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> OptimizationResult:
    """Run full optimization pipeline for a model.
    
    Pipeline:
    1. Generate candidates based on constraints
    2. For each candidate:
       a. Compress model
       b. Evaluate PPL
       c. Benchmark latency/throughput
    3. Select optimal candidate
    
    Args:
        model_id: HuggingFace model ID
        hardware: Target hardware (auto-detected if None)
        constraints: Optimization constraints
        candidates: Specific candidate names to try (None = all)
        include_calibration: Include candidates requiring calibration
        num_eval_samples: Samples for PPL evaluation
        num_benchmark_samples: Samples for latency benchmark
        progress_callback: Optional callback(status, progress)
        
    Returns:
        OptimizationResult with winner and all benchmark results
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from core.calibration import compute_ppl, collect_calibration_stats
    from core.allocation import allocate_bits
    
    # Codec constant
    CODEC_INT4_RESIDUAL = "int4_residual_v1"
    
    start_time = datetime.utcnow()
    
    if constraints is None:
        constraints = OptimizationConstraints()
    
    if hardware is None:
        hardware = estimate_hardware_type()
    
    def log(msg: str, progress: float = 0.0):
        if progress_callback:
            progress_callback(msg, progress)
        print(f"[Optimizer] {msg}")
    
    log(f"Starting optimization for {model_id}", 0.0)
    log(f"Hardware: {hardware}", 0.01)
    log(f"Constraints: PPL<={constraints.max_ppl_delta}%, "
        f"Latency<={constraints.max_latency_p99_ms}ms, "
        f"Throughput>={constraints.min_throughput_tps} tps", 0.02)
    
    # Generate candidates
    candidate_list = generate_candidates(
        include_calibration=include_calibration,
        max_expected_ppl_delta=constraints.max_ppl_delta * 2,  # Allow some margin
        specific_candidates=candidates,
    )
    
    result = OptimizationResult(
        model_id=model_id,
        hardware=hardware,
        constraints=constraints,
        total_candidates=len(candidate_list),
    )
    
    log(f"Generated {len(candidate_list)} candidates", 0.05)
    
    # Load tokenizer
    log("Loading tokenizer...", 0.06)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load calibration data
    log("Loading calibration data...", 0.08)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    need = max(128, num_eval_samples)
    for item in dataset:
        text = item.get("text", "")
        if len(text) > 100:
            texts.append(text)
            if len(texts) >= need:
                break
    eval_texts = texts[:num_eval_samples]
    
    # Load baseline model and compute baseline PPL
    log("Loading baseline model...", 0.10)
    device = "cpu" if hardware == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    log("Computing baseline PPL...", 0.15)
    baseline_ppl = compute_ppl(
        baseline_model,
        tokenizer,
        eval_texts,
        device=device,
        max_samples=num_eval_samples,
        streaming=(device == "cpu"),
    )
    log(f"Baseline PPL: {baseline_ppl:.4f}", 0.20)
    
    # Collect calibration stats if needed
    fisher_scores = {}
    activation_scales = {}
    hessian_diags = {}
    if include_calibration:
        log("Collecting calibration stats...", 0.22)
        fisher_scores, activation_scales, hessian_diags, _ = collect_calibration_stats(
            baseline_model,
            tokenizer,
            texts[:128],
            num_samples=64,
            device=device,
        )
        log("Calibration complete", 0.28)
    
    # Evaluate each candidate
    all_results = []
    progress_per_candidate = 0.6 / len(candidate_list)
    
    for i, candidate in enumerate(candidate_list):
        progress = 0.30 + (i * progress_per_candidate)
        log(f"Evaluating candidate {i+1}/{len(candidate_list)}: {candidate.name}", progress)
        
        try:
            # Reload model fresh for each candidate
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            
            # Compress based on method
            compression_ratio = 1.0
            
            if candidate.method == CompressionMethod.FP16:
                # Baseline - no compression
                pass
            elif candidate.method == CompressionMethod.INT4_AWQ:
                # AWQ compression
                act_scales = activation_scales
                total_original = 0
                total_compressed = 0
                
                for name, module in model.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        if len(module.weight.shape) == 2:  # Linear/Conv1D layers
                            weight = module.weight.data
                            is_conv1d = module.__class__.__name__ == "Conv1D"
                            codec_weight = weight.T.contiguous() if is_conv1d else weight
                            act_scale = act_scales.get(name)
                            
                            deq_weight, comp = compress_int4_awq(
                                codec_weight,
                                group_size=candidate.config.get("group_size", 256),
                                act_scale=act_scale,
                                outlier_pct=candidate.config.get("outlier_pct", 0.5),
                            )
 
                            if is_conv1d:
                                deq_weight = deq_weight.T
 
                            module.weight.data = deq_weight.to(weight.dtype)
                            total_original += weight.numel() * 2
                            total_compressed += weight.numel() * 2 / comp
                
                if total_compressed > 0:
                    compression_ratio = total_original / total_compressed
                    
            elif candidate.method == CompressionMethod.INT4_RESIDUAL:
                # INT4+Residual compression
                total_original = 0
                total_compressed = 0
                
                for name, module in model.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        if len(module.weight.shape) == 2:
                            weight = module.weight.data
                            is_conv1d = module.__class__.__name__ == "Conv1D"
                            codec_weight = weight.T.contiguous() if is_conv1d else weight
                            
                            deq_weight, comp = compress_int4_residual(
                                codec_weight,
                                group_size=candidate.config.get("group_size", 16),
                                residual_group=candidate.config.get("residual_group", 16),
                            )
 
                            if is_conv1d:
                                deq_weight = deq_weight.T
 
                            module.weight.data = deq_weight.to(weight.dtype)
                            total_original += weight.numel() * 2
                            total_compressed += weight.numel() * 2 / comp
                
                if total_compressed > 0:
                    compression_ratio = total_original / total_compressed

            elif candidate.method == CompressionMethod.HYBRID_AWQ_VQ:
                if not fisher_scores:
                    raise RuntimeError("Hybrid candidate requires calibration stats (fisher_scores)")

                allocations = allocate_bits(baseline_model, fisher_scores, target="v11")
                modules_by_name = {n: m for n, m in model.named_modules()}
                total_original = 0
                total_compressed = 0

                for layer_name, alloc in allocations.items():
                    module = modules_by_name.get(layer_name)
                    if module is None or not hasattr(module, "weight") or module.weight is None:
                        continue

                    weight = module.weight.data
                    is_conv1d = module.__class__.__name__ == "Conv1D"
                    codec_weight = weight.T.contiguous() if is_conv1d else weight
                    if weight.dim() != 2:
                        continue

                    orig_size = weight.numel() * 2
                    act_scale = activation_scales.get(layer_name)
                    hessian_diag = hessian_diags.get(layer_name)

                    if alloc.method == "calibrated_vq":
                        deq_weight, comp = compress_calibrated_vq(
                            codec_weight,
                            vec_dim=alloc.vq_vec_dim,
                            codebook_size=alloc.vq_codebook_size,
                            hessian_diag=hessian_diag,
                            act_scale=act_scale,
                        )
                    else:
                        deq_weight, comp = compress_int4_awq(
                            codec_weight,
                            group_size=alloc.group_size,
                            act_scale=act_scale,
                            outlier_pct=candidate.config.get("awq_outlier_pct", 0.5),
                        )

                    if is_conv1d:
                        deq_weight = deq_weight.T

                    module.weight.data = deq_weight.to(weight.dtype)
                    total_original += orig_size
                    total_compressed += orig_size / comp

                if total_compressed > 0:
                    compression_ratio = total_original / total_compressed
            
            # Compute compressed PPL
            log(f"  Computing PPL for {candidate.name}...", progress + 0.3 * progress_per_candidate)
            compressed_ppl = compute_ppl(
                model,
                tokenizer,
                eval_texts,
                device=device,
                max_samples=num_eval_samples,
                streaming=(device == "cpu"),
            )
            
            # Run benchmark
            log(f"  Benchmarking {candidate.name}...", progress + 0.6 * progress_per_candidate)
            benchmark_result = benchmark_candidate(
                model=model,
                tokenizer=tokenizer,
                candidate_name=candidate.name,
                compression_ratio=compression_ratio,
                ppl_baseline=baseline_ppl,
                ppl_compressed=compressed_ppl,
                hardware=hardware,
                num_samples=num_benchmark_samples,
                warmup_samples=5,
            )
            
            all_results.append(benchmark_result)
            result.candidates_evaluated += 1
            
            log(f"  {candidate.name}: {compression_ratio:.2f}x, "
                f"PPL Δ={benchmark_result.ppl_delta_pct:+.2f}%, "
                f"Cost=${benchmark_result.cost_per_1m_tokens:.4f}/1M tokens",
                progress + progress_per_candidate)
            
            # Update candidate with actual results
            candidate.actual_compression = compression_ratio
            candidate.actual_ppl_delta = benchmark_result.ppl_delta_pct
            candidate.cost_per_1m_tokens = benchmark_result.cost_per_1m_tokens
            
            # Cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            log(f"  Error with {candidate.name}: {e}", progress + progress_per_candidate)
            result.candidates_evaluated += 1
            continue
    
    # Select winner
    log("Selecting optimal candidate...", 0.92)
    result.all_results = all_results
    result.passing_results = [
        r for r in all_results 
        if r.passes_constraints(
            constraints.max_ppl_delta,
            constraints.max_latency_p99_ms,
            constraints.min_throughput_tps,
        )
    ]
    
    winner = select_optimal(all_results, constraints)
    result.winner = winner
    
    if winner:
        # Find the winning candidate
        for c in candidate_list:
            if c.name == winner.candidate_name:
                result.winner_candidate = c
                break
        
        # Calculate savings vs baseline
        baseline_result = next((r for r in all_results if r.candidate_name == "FP16 Baseline"), None)
        if baseline_result:
            result.cost_savings_pct = (
                (baseline_result.cost_per_1m_tokens - winner.cost_per_1m_tokens) /
                baseline_result.cost_per_1m_tokens * 100
            )
        result.compression_vs_baseline = winner.compression_ratio
        
        log(f"Winner: {winner.candidate_name}", 0.95)
        log(f"  Compression: {winner.compression_ratio:.2f}x", 0.96)
        log(f"  PPL Delta: {winner.ppl_delta_pct:+.2f}%", 0.97)
        log(f"  Cost: ${winner.cost_per_1m_tokens:.4f}/1M tokens", 0.98)
        log(f"  Cost Savings: {result.cost_savings_pct:.1f}% vs FP16", 0.99)
    else:
        log("No candidates passed all constraints", 0.95)
    
    # Finalize
    result.completed_at = datetime.utcnow()
    result.optimization_time_s = (result.completed_at - start_time).total_seconds()
    
    log(f"Optimization complete in {result.optimization_time_s:.1f}s", 1.0)
    
    # Cleanup baseline model
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result
