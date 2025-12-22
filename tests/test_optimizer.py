"""
Tests for cost optimizer functionality
"""

import pytest
from unittest.mock import Mock, patch

from optimizer.candidates import (
    CompressionCandidate,
    QuantizationMethod,
    CANDIDATE_PRESETS,
    generate_candidates,
)


class TestCompressionCandidate:
    """Test CompressionCandidate dataclass."""
    
    def test_candidate_creation(self):
        """Test candidate creation."""
        candidate = CompressionCandidate(
            name="Test Candidate",
            method=QuantizationMethod.GPTQ,
            config={"bits": 4, "group_size": 128},
            expected_compression=7.5,
            expected_ppl_delta=1.0,
        )
        
        assert candidate.name == "Test Candidate"
        assert candidate.method == QuantizationMethod.GPTQ
        assert candidate.config["bits"] == 4
        assert candidate.expected_compression == 7.5
    
    def test_candidate_repr(self):
        """Test candidate string representation."""
        candidate = CompressionCandidate(
            name="GPTQ",
            method=QuantizationMethod.GPTQ,
            expected_compression=7.5,
        )
        
        repr_str = repr(candidate)
        assert "GPTQ" in repr_str
        assert "7.5" in repr_str


class TestCandidatePresets:
    """Test candidate presets."""
    
    def test_all_presets_exist(self):
        """Test that all expected presets exist."""
        expected_presets = [
            "fp16",
            "gptq_quality",
            "gptq_balanced",
            "gptq_size",
            "awq_quality",
            "awq_balanced",
            "bnb_int8",
            "bnb_nf4",
        ]
        
        for preset in expected_presets:
            assert preset in CANDIDATE_PRESETS, f"Missing preset: {preset}"
    
    def test_fp16_baseline(self):
        """Test FP16 baseline candidate."""
        fp16 = CANDIDATE_PRESETS["fp16"]
        
        assert fp16.method == QuantizationMethod.FP16
        assert fp16.expected_compression == 1.0
        assert fp16.expected_ppl_delta == 0.0
        assert fp16.requires_calibration is False
    
    def test_gptq_candidates(self):
        """Test GPTQ candidates."""
        gptq_quality = CANDIDATE_PRESETS["gptq_quality"]
        gptq_balanced = CANDIDATE_PRESETS["gptq_balanced"]
        gptq_size = CANDIDATE_PRESETS["gptq_size"]
        
        # All should be GPTQ
        assert gptq_quality.method == QuantizationMethod.GPTQ
        assert gptq_balanced.method == QuantizationMethod.GPTQ
        assert gptq_size.method == QuantizationMethod.GPTQ
        
        # All require calibration
        assert gptq_quality.requires_calibration is True
        assert gptq_balanced.requires_calibration is True
        assert gptq_size.requires_calibration is True
        
        # Quality < Balanced < Size in compression
        assert gptq_quality.expected_compression < gptq_balanced.expected_compression
        assert gptq_balanced.expected_compression < gptq_size.expected_compression
        
        # Quality < Balanced < Size in PPL delta
        assert gptq_quality.expected_ppl_delta < gptq_balanced.expected_ppl_delta
        assert gptq_balanced.expected_ppl_delta < gptq_size.expected_ppl_delta
    
    def test_awq_candidates(self):
        """Test AWQ candidates."""
        awq_quality = CANDIDATE_PRESETS["awq_quality"]
        awq_balanced = CANDIDATE_PRESETS["awq_balanced"]
        
        assert awq_quality.method == QuantizationMethod.AWQ
        assert awq_balanced.method == QuantizationMethod.AWQ
        
        assert awq_quality.requires_calibration is True
        assert awq_balanced.requires_calibration is True
        
        # Quality should have better PPL delta than balanced
        assert awq_quality.expected_ppl_delta < awq_balanced.expected_ppl_delta
    
    def test_bitsandbytes_candidates(self):
        """Test bitsandbytes candidates."""
        bnb_int8 = CANDIDATE_PRESETS["bnb_int8"]
        bnb_nf4 = CANDIDATE_PRESETS["bnb_nf4"]
        
        assert bnb_int8.method == QuantizationMethod.BITSANDBYTES
        assert bnb_nf4.method == QuantizationMethod.BITSANDBYTES
        
        # bitsandbytes doesn't require calibration
        assert bnb_int8.requires_calibration is False
        assert bnb_nf4.requires_calibration is False
        
        # INT8 should have lower compression than NF4
        assert bnb_int8.expected_compression < bnb_nf4.expected_compression


class TestGenerateCandidates:
    """Test candidate generation."""
    
    def test_generate_all_candidates(self):
        """Test generating all candidates."""
        candidates = generate_candidates(
            include_calibration=True,
            max_expected_ppl_delta=10.0,
            min_expected_compression=1.0,
        )
        
        assert len(candidates) > 0
        assert all(isinstance(c, CompressionCandidate) for c in candidates)
    
    def test_generate_no_calibration(self):
        """Test generating only non-calibration candidates."""
        candidates = generate_candidates(
            include_calibration=False,
            max_expected_ppl_delta=10.0,
            min_expected_compression=1.0,
        )
        
        assert len(candidates) > 0
        assert all(not c.requires_calibration for c in candidates)
    
    def test_generate_ppl_delta_filter(self):
        """Test PPL delta filtering."""
        candidates = generate_candidates(
            include_calibration=True,
            max_expected_ppl_delta=0.5,
            min_expected_compression=1.0,
        )
        
        assert all(c.expected_ppl_delta <= 0.5 for c in candidates)
    
    def test_generate_compression_filter(self):
        """Test compression ratio filtering."""
        candidates = generate_candidates(
            include_calibration=True,
            max_expected_ppl_delta=10.0,
            min_expected_compression=5.0,
        )
        
        assert all(c.expected_compression >= 5.0 for c in candidates)
    
    def test_generate_specific_candidates(self):
        """Test generating specific candidates."""
        candidates = generate_candidates(
            specific_candidates=["gptq_quality", "awq_balanced"],
        )
        
        assert len(candidates) == 2
        names = [c.name for c in candidates]
        assert any("GPTQ" in n and "Quality" in n for n in names)
        assert any("AWQ" in n and "Balanced" in n for n in names)
    
    def test_generate_invalid_specific_candidate(self):
        """Test that invalid specific candidate is ignored."""
        candidates = generate_candidates(
            specific_candidates=["invalid_candidate"],
        )
        
        # Should return empty list if no valid candidates
        assert len(candidates) == 0
    
    def test_generate_mixed_filters(self):
        """Test combining multiple filters."""
        candidates = generate_candidates(
            include_calibration=False,
            max_expected_ppl_delta=1.0,
            min_expected_compression=2.0,
        )
        
        # All candidates should meet all criteria
        for c in candidates:
            assert not c.requires_calibration
            assert c.expected_ppl_delta <= 1.0
            assert c.expected_compression >= 2.0


class TestCandidateResults:
    """Test candidate with benchmark results."""
    
    def test_candidate_with_results(self):
        """Test candidate with populated results."""
        candidate = CompressionCandidate(
            name="Test",
            method=QuantizationMethod.GPTQ,
            expected_compression=7.5,
        )
        
        # Initially None
        assert candidate.actual_compression is None
        assert candidate.latency_p50_ms is None
        
        # Set results
        candidate.actual_compression = 7.3
        candidate.latency_p50_ms = 45.2
        candidate.throughput_tps = 120.5
        candidate.memory_gb = 3.5
        candidate.cost_per_1m_tokens = 0.15
        
        assert candidate.actual_compression == 7.3
        assert candidate.latency_p50_ms == 45.2
        assert candidate.throughput_tps == 120.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
