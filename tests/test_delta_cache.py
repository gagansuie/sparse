"""
Tests for DeltaCache and fast reconstruction functionality
"""

import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoConfig

from core.fast_reconstruct import (
    DeltaCache,
    from_pretrained_with_delta,
    benchmark_reconstruction,
    get_global_cache,
)
from core.delta import compress_delta, DeltaManifest


class TestDeltaCache:
    """Test DeltaCache functionality."""
    
    @pytest.fixture
    def cache_dir(self):
        """Create temporary cache directory."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    @pytest.fixture
    def cache(self, cache_dir):
        """Create DeltaCache instance."""
        return DeltaCache(cache_dir=cache_dir, max_workers=2)
    
    def test_cache_initialization(self, cache):
        """Test cache initializes correctly."""
        assert cache.cache_dir.exists()
        assert cache.reconstructed_dir.exists()
        assert cache.stats.cache_hits == 0
        assert cache.stats.cache_misses == 0
    
    def test_delta_id_generation(self, cache):
        """Test unique delta ID generation."""
        delta_id_1 = cache._get_delta_id("meta-llama/Llama-2-7b-hf", "/path/to/delta1")
        delta_id_2 = cache._get_delta_id("meta-llama/Llama-2-7b-hf", "/path/to/delta2")
        delta_id_3 = cache._get_delta_id("meta-llama/Llama-2-7b-hf", "/path/to/delta1")
        
        # Same inputs should give same ID
        assert delta_id_1 == delta_id_3
        
        # Different inputs should give different IDs
        assert delta_id_1 != delta_id_2
    
    def test_cache_hit_miss_tracking(self, cache, cache_dir):
        """Test cache hit/miss statistics."""
        base_model = "gpt2"
        delta_path = "/fake/delta"
        
        # First check - should be a miss
        result = cache.get_reconstructed_path(base_model, delta_path)
        assert result is None
        assert cache.stats.cache_misses == 1
        assert cache.stats.cache_hits == 0
        
        # Create fake reconstructed model
        delta_id = cache._get_delta_id(base_model, delta_path)
        fake_model_dir = cache.reconstructed_dir / delta_id
        fake_model_dir.mkdir(parents=True)
        (fake_model_dir / "config.json").write_text("{}")
        
        # Second check - should be a hit
        result = cache.get_reconstructed_path(base_model, delta_path)
        assert result is not None
        assert cache.stats.cache_hits == 1
        assert cache.stats.cache_misses == 1
    
    def test_is_reconstructed(self, cache, cache_dir):
        """Test checking if delta is reconstructed."""
        base_model = "gpt2"
        delta_path = "/fake/delta"
        
        # Should not be reconstructed initially
        assert not cache.is_reconstructed(base_model, delta_path)
        
        # Create fake reconstructed model
        delta_id = cache._get_delta_id(base_model, delta_path)
        fake_model_dir = cache.reconstructed_dir / delta_id
        fake_model_dir.mkdir(parents=True)
        (fake_model_dir / "config.json").write_text("{}")
        
        # Should now be reconstructed
        assert cache.is_reconstructed(base_model, delta_path)
    
    def test_cache_index_persistence(self, cache_dir):
        """Test cache index saves and loads."""
        cache1 = DeltaCache(cache_dir=cache_dir)
        cache1.stats.deltas_reconstructed = 5
        cache1._save_cache_index()
        
        # Create new cache instance - should load index
        cache2 = DeltaCache(cache_dir=cache_dir)
        assert cache2.stats.deltas_reconstructed == 5
    
    @patch('transformers.AutoModelForCausalLM')
    def test_reconstruct_fast_basic(self, mock_model_class, cache, cache_dir):
        """Test basic reconstruction flow."""
        # Create fake delta directory
        delta_dir = Path(cache_dir) / "test_delta"
        delta_dir.mkdir()
        
        # Create minimal manifest
        manifest = {
            "version": "1.0",
            "base_model_id": "gpt2",
            "finetune_model_id": "gpt2-finetuned",
            "compression_ratio": 10.0,
            "total_params": 1000000,
            "changed_params": 100000,
            "num_layers": 2,
            "layer_deltas": [
                {"name": "layer1", "method": "zero"},
                {"name": "layer2", "method": "zero"},
            ]
        }
        
        with open(delta_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        # Mock model
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.save_pretrained = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Run reconstruction
        output_path = cache.reconstruct_fast(
            base_model_id="gpt2",
            delta_path=str(delta_dir),
            use_rust=True,
        )
        
        assert Path(output_path).exists()
        assert cache.stats.deltas_reconstructed == 1
        assert len(cache.stats.reconstruction_times) == 1
    
    def test_get_stats(self, cache):
        """Test statistics retrieval."""
        cache.stats.cache_hits = 10
        cache.stats.cache_misses = 5
        cache.stats.deltas_reconstructed = 3
        
        stats = cache.get_stats()
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["deltas_reconstructed"] == 3
        assert "cache_dir" in stats


class TestBenchmarking:
    """Test reconstruction benchmarking."""
    
    def test_benchmark_reconstruction(self):
        """Test Rust benchmark function."""
        result = benchmark_reconstruction(tensor_size=100000, iterations=5)
        
        assert result["rust_available"] is True
        assert result["tensor_size"] == 100000
        assert result["iterations"] == 5
        assert "ms_per_iteration" in result
        assert "estimated_times" in result
        assert "7B_model" in result["estimated_times"]


class TestGlobalCache:
    """Test global cache functionality."""
    
    def test_global_cache_singleton(self):
        """Test global cache is a singleton."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        
        # Should be the same instance
        assert cache1 is cache2


class TestFromPretrainedWithDelta:
    """Test drop-in replacement for from_pretrained."""
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('core.fast_reconstruct.DeltaCache')
    def test_from_pretrained_standard_model(self, mock_cache_class, mock_model_class):
        """Test loading a standard model (not a delta)."""
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load standard model
        model = from_pretrained_with_delta("gpt2")
        
        # Should use standard loading
        mock_model_class.from_pretrained.assert_called_once_with("gpt2")
        assert model is mock_model
    
    @patch('transformers.AutoModelForCausalLM')
    @patch('core.fast_reconstruct.DeltaCache')
    def test_from_pretrained_with_delta_artifact(self, mock_cache_class, mock_model_class, tmp_path):
        """Test loading from a delta artifact."""
        # Create fake delta directory
        delta_dir = tmp_path / "my_delta"
        delta_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "base_model_id": "gpt2",
            "finetune_model_id": "gpt2-finetuned",
        }
        
        with open(delta_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get_or_reconstruct.return_value = "/path/to/reconstructed"
        mock_cache_class.return_value = mock_cache
        
        # Mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Load from delta
        model = from_pretrained_with_delta(
            str(delta_dir),
            base_model_id="gpt2"
        )
        
        # Should reconstruct and load
        mock_cache.get_or_reconstruct.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once_with("/path/to/reconstructed")


def test_rust_acceleration_available():
    """Test that Rust acceleration is available."""
    import sparse_core
    
    # Should be able to call benchmark
    result = sparse_core.benchmark_int8_apply(1000, 5)
    assert isinstance(result, float)
    assert result >= 0  # Can be 0 for very small/fast operations
