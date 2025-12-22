"""
Tests for smart routing functionality (NEW)
"""

import pytest
from unittest.mock import Mock, patch

from optimizer.routing import (
    HardwareType,
    TaskComplexity,
    ModelSpec,
    RoutingDecision,
    classify_request_complexity,
    suggest_optimal_model,
    batch_requests,
    estimate_routing_savings,
)


class TestTaskComplexityClassification:
    """Test request complexity classification."""
    
    def test_simple_task(self):
        """Test classification of simple tasks."""
        complexity = classify_request_complexity(
            prompt="What is 2+2?",
            max_tokens=10,
            context_length=0
        )
        assert complexity == TaskComplexity.SIMPLE
    
    def test_moderate_task(self):
        """Test classification of moderate tasks."""
        complexity = classify_request_complexity(
            prompt="Explain how photosynthesis works in plants.",
            max_tokens=150,
            context_length=100
        )
        assert complexity == TaskComplexity.MODERATE
    
    def test_complex_task(self):
        """Test classification of complex tasks."""
        complexity = classify_request_complexity(
            prompt="Write a detailed analysis of the economic impacts of climate change.",
            max_tokens=500,
            context_length=1000
        )
        assert complexity == TaskComplexity.COMPLEX
    
    def test_extreme_task(self):
        """Test classification of extreme tasks."""
        complexity = classify_request_complexity(
            prompt="Analyze this entire book: " + "text " * 5000,
            max_tokens=2000,
            context_length=10000
        )
        assert complexity == TaskComplexity.EXTREME


class TestSuggestOptimalModel:
    """Test model routing suggestions."""
    
    def test_simple_prompt_recommends_smaller_model(self):
        """Test that simple prompts recommend smaller models."""
        decision = suggest_optimal_model(
            requested_model="meta-llama/Llama-2-70b-hf",
            prompt="What is the capital of France?",
            quality_threshold=0.85,
            cost_priority=True
        )
        
        # Should recommend smaller model for simple task
        assert decision.recommended_model == "meta-llama/Llama-2-7b-hf"
        assert decision.quality_score >= 0.85
        assert "Saves" in decision.reasoning or "cheaper" in decision.reasoning.lower()
    
    def test_quality_threshold_respected(self):
        """Test that quality threshold is respected."""
        decision = suggest_optimal_model(
            requested_model="meta-llama/Llama-2-70b-hf",
            prompt="What is 2+2?",
            quality_threshold=0.95,  # High threshold
            cost_priority=True
        )
        
        # With high quality threshold, might not recommend smaller model
        assert decision.quality_score >= 0.95
    
    def test_latency_priority(self):
        """Test routing with latency priority."""
        decision = suggest_optimal_model(
            requested_model="meta-llama/Llama-2-70b-hf",
            prompt="Quick question",
            quality_threshold=0.85,
            cost_priority=False  # Prioritize latency
        )
        
        # Should optimize for latency, not cost
        assert decision.recommended_hardware is not None
        assert decision.estimated_latency_p99_ms > 0
    
    def test_alternatives_provided(self):
        """Test that alternatives are provided."""
        decision = suggest_optimal_model(
            requested_model="meta-llama/Llama-2-70b-hf",
            prompt="Hello",
            quality_threshold=0.80
        )
        
        # Should provide alternatives
        assert isinstance(decision.alternatives, list)
        # May have 0 or more alternatives depending on candidates


class TestBatchRequests:
    """Test request batching."""
    
    def test_batch_same_model(self):
        """Test batching requests for same model."""
        requests = [
            {"model": "llama-7b", "prompt": "Hello"},
            {"model": "llama-7b", "prompt": "Hi"},
            {"model": "llama-7b", "prompt": "Hey"},
        ]
        
        batches = batch_requests(requests, max_batch_size=32)
        
        assert len(batches) == 1  # All in one batch
        assert len(batches[0]) == 3
    
    def test_batch_different_models(self):
        """Test batching requests for different models."""
        requests = [
            {"model": "llama-7b", "prompt": "Hello"},
            {"model": "llama-70b", "prompt": "Hi"},
            {"model": "llama-7b", "prompt": "Hey"},
        ]
        
        batches = batch_requests(requests, max_batch_size=32)
        
        assert len(batches) == 2  # Separate batches per model
    
    def test_batch_size_limit(self):
        """Test that batch size limit is respected."""
        requests = [
            {"model": "llama-7b", "prompt": f"Prompt {i}"}
            for i in range(100)
        ]
        
        batches = batch_requests(requests, max_batch_size=32)
        
        # Should split into multiple batches
        assert len(batches) > 1
        for batch in batches:
            assert len(batch) <= 32


class TestEstimateRoutingSavings:
    """Test routing savings estimation."""
    
    def test_savings_calculation(self):
        """Test savings calculation logic."""
        savings = estimate_routing_savings(
            current_requests_per_day=1_000_000,
            avg_cost_per_request=0.001,
            optimization_rate=0.25
        )
        
        assert "annual_requests" in savings
        assert "current_annual_cost_usd" in savings
        assert "annual_savings_usd" in savings
        assert "savings_pct" in savings
        
        # Verify calculations
        assert savings["annual_requests"] == 1_000_000 * 365
        assert savings["optimization_rate"] == 0.25
        assert savings["annual_savings_usd"] > 0
        assert savings["savings_pct"] > 0
    
    def test_zero_optimization_rate(self):
        """Test with zero optimization rate."""
        savings = estimate_routing_savings(
            current_requests_per_day=1_000_000,
            avg_cost_per_request=0.001,
            optimization_rate=0.0
        )
        
        assert savings["annual_savings_usd"] == 0
        assert savings["savings_pct"] == 0
    
    def test_high_optimization_rate(self):
        """Test with high optimization rate."""
        savings = estimate_routing_savings(
            current_requests_per_day=10_000_000,
            avg_cost_per_request=0.001,
            optimization_rate=0.5  # 50% optimizable
        )
        
        # Should show significant savings
        assert savings["annual_savings_usd"] > 1_000_000  # >$1M
        assert savings["savings_pct"] > 10  # >10% total savings


class TestHardwareType:
    """Test HardwareType enum."""
    
    def test_hardware_types_exist(self):
        """Test that hardware types are defined."""
        assert HardwareType.T4 is not None
        assert HardwareType.A10G is not None
        assert HardwareType.A100_40GB is not None
        assert HardwareType.A100_80GB is not None
        assert HardwareType.H100 is not None
    
    def test_hardware_costs(self):
        """Test that hardware has cost information."""
        assert HardwareType.T4.cost_per_hour == 0.50
        assert HardwareType.A10G.cost_per_hour == 1.00
        assert HardwareType.H100.cost_per_hour == 8.00
        
        # T4 should be cheapest
        assert HardwareType.T4.cost_per_hour < HardwareType.A10G.cost_per_hour
        assert HardwareType.A10G.cost_per_hour < HardwareType.H100.cost_per_hour


class TestRoutingIntegration:
    """Integration tests for routing workflow."""
    
    def test_complete_routing_workflow(self):
        """Test complete routing decision workflow."""
        # User makes request
        requested = "meta-llama/Llama-2-70b-hf"
        prompt = "What is machine learning?"
        
        # Classify complexity
        complexity = classify_request_complexity(prompt, max_tokens=100)
        assert complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]
        
        # Get routing decision
        decision = suggest_optimal_model(
            requested_model=requested,
            prompt=prompt,
            quality_threshold=0.85,
            cost_priority=True
        )
        
        # Verify decision structure
        assert decision.recommended_model is not None
        assert decision.recommended_hardware is not None
        assert decision.estimated_cost_per_1m_tokens > 0
        assert decision.quality_score >= 0.85
        assert len(decision.reasoning) > 0
    
    def test_savings_realistic(self):
        """Test that estimated savings are realistic."""
        # HuggingFace scale: 10M requests/day
        savings = estimate_routing_savings(
            current_requests_per_day=10_000_000,
            avg_cost_per_request=0.001,
            optimization_rate=0.25
        )
        
        # Annual savings should be in $5-10M range for this scale
        annual_savings_m = savings["annual_savings_usd"] / 1_000_000
        assert 2 < annual_savings_m < 15  # Realistic range
