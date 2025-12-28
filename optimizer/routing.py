"""
Smart Model Routing & Recommendation System

Automatically route inference requests to optimal models/hardware.
Suggests smaller models when quality is acceptable.

Estimated savings: $5-10M/year for inference platforms
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class HardwareType(Enum):
    """Available hardware types with cost per hour"""
    T4 = ("T4", 0.50)
    A10G = ("A10G", 1.00)
    A100_40GB = ("A100-40GB", 3.50)
    A100_80GB = ("A100-80GB", 5.00)
    H100 = ("H100", 8.00)
    
    def __init__(self, name: str, cost_per_hour: float):
        self.hardware_name = name
        self.cost_per_hour = cost_per_hour


class TaskComplexity(Enum):
    """Request complexity levels"""
    SIMPLE = "simple"  # Short prompts, simple tasks
    MODERATE = "moderate"  # Medium prompts, reasoning
    COMPLEX = "complex"  # Long context, complex reasoning
    EXTREME = "extreme"  # Very long context, multi-step


@dataclass
class ModelSpec:
    """Model specification with performance characteristics"""
    model_id: str
    size_b: float  # Model size in billions of parameters
    min_memory_gb: int
    recommended_hardware: List[HardwareType]
    avg_latency_ms: Dict[HardwareType, float]
    quality_score: float  # 0-1, relative to largest model


@dataclass
class RoutingDecision:
    """Result of routing optimization"""
    recommended_model: str
    recommended_hardware: HardwareType
    estimated_cost_per_1m_tokens: float
    estimated_latency_p99_ms: float
    quality_score: float
    reasoning: str
    alternatives: List[Dict]


# Demo model database - DEFAULTS FOR TESTING ONLY
# In production, this should be populated dynamically from:
# - Customer's model registry
# - HuggingFace Hub API
# - Internal model catalog
# The routing logic is model-agnostic; these are just example specs
DEMO_MODEL_SPECS = {
    "codellama/CodeLlama-70b-hf": ModelSpec(
        model_id="codellama/CodeLlama-70b-hf",
        size_b=70.0,
        min_memory_gb=40,
        recommended_hardware=[HardwareType.A100_40GB, HardwareType.A100_80GB, HardwareType.H100],
        avg_latency_ms={
            HardwareType.A100_40GB: 120,
            HardwareType.A100_80GB: 80,
            HardwareType.H100: 50,
        },
        quality_score=1.0
    ),
    "codellama/CodeLlama-13b-hf": ModelSpec(
        model_id="codellama/CodeLlama-13b-hf",
        size_b=13.0,
        min_memory_gb=16,
        recommended_hardware=[HardwareType.A10G, HardwareType.A100_40GB],
        avg_latency_ms={
            HardwareType.A10G: 80,
            HardwareType.A100_40GB: 40,
        },
        quality_score=0.92
    ),
    "mistralai/Mistral-7B-v0.1": ModelSpec(
        model_id="mistralai/Mistral-7B-v0.1",
        size_b=7.0,
        min_memory_gb=8,
        recommended_hardware=[HardwareType.T4, HardwareType.A10G],
        avg_latency_ms={
            HardwareType.T4: 100,
            HardwareType.A10G: 45,
        },
        quality_score=0.88
    ),
}


def classify_request_complexity(
    prompt: str,
    max_tokens: int = 100,
    context_length: int = 0
) -> TaskComplexity:
    """
    Classify inference request complexity.
    
    Args:
        prompt: User prompt
        max_tokens: Max tokens to generate
        context_length: Length of context window used
    
    Returns:
        TaskComplexity level
    
    Example:
        >>> complexity = classify_request_complexity("What is 2+2?", max_tokens=10)
        >>> print(complexity)
        TaskComplexity.SIMPLE
    """
    prompt_tokens = len(prompt.split())
    total_tokens = prompt_tokens + max_tokens + context_length
    
    # Simple heuristics - prioritize simplicity for small prompts
    # Check prompt and output size first for quick classification
    if prompt_tokens <= 10 and max_tokens <= 20:
        return TaskComplexity.SIMPLE
    elif total_tokens < 50 and max_tokens < 30:
        return TaskComplexity.SIMPLE
    elif total_tokens < 150:
        return TaskComplexity.MODERATE
    elif total_tokens < 1000:
        return TaskComplexity.COMPLEX
    else:
        return TaskComplexity.EXTREME


def suggest_optimal_model(
    requested_model: str,
    prompt: str,
    quality_threshold: float = 0.85,
    latency_budget_ms: Optional[float] = None,
    cost_priority: bool = True
) -> RoutingDecision:
    """
    Suggest optimal model for given request.
    
    May recommend smaller/cheaper model if quality is acceptable.
    
    Args:
        requested_model: Model user requested
        prompt: User's prompt
        quality_threshold: Minimum acceptable quality (0-1)
        latency_budget_ms: Max acceptable latency
        cost_priority: Prioritize cost over latency
    
    Returns:
        RoutingDecision with recommendation
    
    Example:
        >>> decision = suggest_optimal_model(
        ...     "meta-llama/Llama-2-70b-hf",
        ...     "What is the capital of France?",
        ...     quality_threshold=0.85
        ... )
        >>> print(f"Recommended: {decision.recommended_model}")
        Recommended: meta-llama/Llama-2-7b-hf
        >>> print(f"Cost savings: {decision.estimated_cost_per_1m_tokens}")
    """
    # Classify request
    complexity = classify_request_complexity(prompt)
    
    # Get all models that meet quality threshold
    candidate_models = [
        spec for spec in DEMO_MODEL_SPECS.values()
        if spec.quality_score >= quality_threshold
    ]
    
    if not candidate_models:
        # No models meet threshold, use requested
        requested_spec = DEMO_MODEL_SPECS.get(requested_model)
        if not requested_spec:
            raise ValueError(f"Unknown model: {requested_model}")
        
        best_hardware = requested_spec.recommended_hardware[0]
        
        return RoutingDecision(
            recommended_model=requested_model,
            recommended_hardware=best_hardware,
            estimated_cost_per_1m_tokens=_calculate_cost(requested_spec, best_hardware),
            estimated_latency_p99_ms=requested_spec.avg_latency_ms[best_hardware] * 1.5,
            quality_score=requested_spec.quality_score,
            reasoning="No models meet quality threshold",
            alternatives=[]
        )
    
    # Score each candidate by cost/latency
    scored_candidates = []
    for spec in candidate_models:
        for hardware in spec.recommended_hardware:
            cost = _calculate_cost(spec, hardware)
            latency = spec.avg_latency_ms.get(hardware, float('inf'))
            
            # Skip if exceeds latency budget
            if latency_budget_ms and latency > latency_budget_ms:
                continue
            
            # Score based on priority
            if cost_priority:
                score = -cost  # Lower cost = higher score
            else:
                score = -latency  # Lower latency = higher score
            
            scored_candidates.append({
                "model": spec.model_id,
                "hardware": hardware,
                "cost": cost,
                "latency": latency,
                "quality": spec.quality_score,
                "score": score
            })
    
    if not scored_candidates:
        # Fallback to requested model
        requested_spec = DEMO_MODEL_SPECS.get(requested_model, list(DEMO_MODEL_SPECS.values())[0])
        best_hardware = requested_spec.recommended_hardware[0]
        
        return RoutingDecision(
            recommended_model=requested_model,
            recommended_hardware=best_hardware,
            estimated_cost_per_1m_tokens=_calculate_cost(requested_spec, best_hardware),
            estimated_latency_p99_ms=requested_spec.avg_latency_ms[best_hardware] * 1.5,
            quality_score=requested_spec.quality_score,
            reasoning="No candidates meet latency budget",
            alternatives=[]
        )
    
    # Sort by score (highest first)
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    best = scored_candidates[0]
    alternatives = scored_candidates[1:6]  # Top 5 alternatives
    
    # Build reasoning
    requested_spec = DEMO_MODEL_SPECS.get(requested_model)
    if requested_spec and best["model"] != requested_model:
        requested_cost = _calculate_cost(
            requested_spec,
            requested_spec.recommended_hardware[0]
        )
        savings_pct = ((requested_cost - best["cost"]) / requested_cost) * 100
        reasoning = (
            f"Recommended {best['model']} instead of {requested_model}. "
            f"Saves {savings_pct:.0f}% cost with {best['quality']*100:.0f}% quality."
        )
    else:
        reasoning = f"Optimal configuration for {requested_model}"
    
    return RoutingDecision(
        recommended_model=best["model"],
        recommended_hardware=best["hardware"],
        estimated_cost_per_1m_tokens=best["cost"],
        estimated_latency_p99_ms=best["latency"] * 1.5,  # P99 ~1.5x avg
        quality_score=best["quality"],
        reasoning=reasoning,
        alternatives=[
            {
                "model": alt["model"],
                "hardware": alt["hardware"].hardware_name,
                "cost": alt["cost"],
                "latency": alt["latency"],
                "quality": alt["quality"]
            }
            for alt in alternatives
        ]
    )


def _calculate_cost(spec: ModelSpec, hardware: HardwareType) -> float:
    """
    Calculate cost per 1M tokens.
    
    Args:
        spec: Model specification
        hardware: Hardware type
    
    Returns:
        Cost per 1M tokens in USD
    """
    # Assume average throughput (tokens/sec) based on model size
    if spec.size_b > 50:
        throughput = 20  # tokens/sec for 70B on A100
    elif spec.size_b > 10:
        throughput = 50  # tokens/sec for 13B
    else:
        throughput = 100  # tokens/sec for 7B
    
    # Adjust for hardware
    if hardware == HardwareType.H100:
        throughput *= 2.5
    elif hardware == HardwareType.A100_80GB:
        throughput *= 1.5
    elif hardware == HardwareType.T4:
        throughput *= 0.5
    
    # Cost per 1M tokens
    seconds_per_1m_tokens = 1_000_000 / throughput
    hours_per_1m_tokens = seconds_per_1m_tokens / 3600
    cost_per_1m_tokens = hours_per_1m_tokens * hardware.cost_per_hour
    
    return cost_per_1m_tokens


def batch_requests(
    requests: List[Dict],
    max_batch_size: int = 32,
    max_wait_ms: int = 50
) -> List[List[Dict]]:
    """
    Batch similar requests for efficient inference.
    
    Args:
        requests: List of inference requests
        max_batch_size: Maximum batch size
        max_wait_ms: Maximum wait time for batching
    
    Returns:
        List of batches
    
    Example:
        >>> requests = [
        ...     {"model": "llama-7b", "prompt": "Hello"},
        ...     {"model": "llama-7b", "prompt": "Hi there"}
        ... ]
        >>> batches = batch_requests(requests)
        >>> print(f"Created {len(batches)} batches")
    """
    # Group by model
    by_model = {}
    for req in requests:
        model = req.get("model", "default")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(req)
    
    # Create batches
    batches = []
    for model, model_requests in by_model.items():
        # Split into batches of max_batch_size
        for i in range(0, len(model_requests), max_batch_size):
            batch = model_requests[i:i + max_batch_size]
            batches.append(batch)
    
    return batches


def estimate_routing_savings(
    current_requests_per_day: int,
    avg_cost_per_request: float,
    optimization_rate: float = 0.25
) -> Dict:
    """
    Estimate annual savings from smart routing.
    
    Args:
        current_requests_per_day: Daily request volume
        avg_cost_per_request: Current average cost per request
        optimization_rate: % of requests that can be optimized (default 25%)
    
    Returns:
        Dict with savings estimates
    
    Example:
        >>> savings = estimate_routing_savings(
        ...     current_requests_per_day=10_000_000,
        ...     avg_cost_per_request=0.001,
        ...     optimization_rate=0.25
        ... )
        >>> print(f"Annual savings: ${savings['annual_savings_usd']:,.0f}")
        Annual savings: $2,737,500
    """
    # Annual request volume
    annual_requests = current_requests_per_day * 365
    
    # Current annual cost
    current_annual_cost = annual_requests * avg_cost_per_request
    
    # Optimizable requests
    optimizable_requests = annual_requests * optimization_rate
    
    # Average savings per optimized request (30% cost reduction)
    avg_savings_per_optimized = avg_cost_per_request * 0.30
    
    # Total annual savings
    annual_savings = optimizable_requests * avg_savings_per_optimized
    
    return {
        "annual_requests": annual_requests,
        "current_annual_cost_usd": current_annual_cost,
        "optimizable_requests": optimizable_requests,
        "optimization_rate": optimization_rate,
        "avg_savings_per_request": avg_savings_per_optimized,
        "annual_savings_usd": annual_savings,
        "monthly_savings_usd": annual_savings / 12,
        "savings_pct": (annual_savings / current_annual_cost) * 100
    }
