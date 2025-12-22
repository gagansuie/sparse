"""
Tests for vLLM integration functionality
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from inference.vllm_integration import (
    TenPakVLLMLoader,
    TenPakTGILoader,
    benchmark_inference,
)


class TestTenPakVLLMLoader:
    """Test TenPakVLLMLoader functionality."""
    
    @pytest.fixture
    def mock_artifact_dir(self, tmp_path):
        """Create mock artifact directory."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "gpt2",
            "quantization": {
                "method": "awq",
                "bits": 4,
                "group_size": 128,
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        return artifact_dir
    
    def test_load_for_vllm(self, mock_artifact_dir):
        """Test vLLM configuration generation."""
        config = TenPakVLLMLoader.load_for_vllm(
            artifact_path=str(mock_artifact_dir),
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
        )
        
        assert config["model"] == "gpt2"
        assert config["tensor_parallel_size"] == 2
        assert config["quantization"] == "awq"
        assert config["gpu_memory_utilization"] == 0.9
    
    def test_load_for_vllm_gptq(self, tmp_path):
        """Test vLLM config for GPTQ."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "mistral-7b",
            "quantization": {
                "method": "gptq",
                "bits": 4,
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        config = TenPakVLLMLoader.load_for_vllm(str(artifact_dir))
        
        assert config["quantization"] == "gptq"
    
    def test_load_for_vllm_no_quantization(self, tmp_path):
        """Test vLLM config with no quantization."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "gpt2",
            "quantization": {
                "method": "none",
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        config = TenPakVLLMLoader.load_for_vllm(str(artifact_dir))
        
        assert config["quantization"] is None
    
    @patch("inference.vllm_integration.LLM")
    def test_create_vllm_engine(self, mock_llm_class, mock_artifact_dir):
        """Test vLLM engine creation."""
        mock_engine = Mock()
        mock_llm_class.return_value = mock_engine
        
        engine = TenPakVLLMLoader.create_vllm_engine(
            artifact_path=str(mock_artifact_dir),
            tensor_parallel_size=1,
        )
        
        assert engine == mock_engine
        mock_llm_class.assert_called_once()
        
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "gpt2"
        assert call_kwargs["quantization"] == "awq"
    
    @patch("inference.vllm_integration.subprocess.run")
    def test_serve_with_vllm(self, mock_subprocess, mock_artifact_dir):
        """Test vLLM server launching."""
        TenPakVLLMLoader.serve_with_vllm(
            artifact_path=str(mock_artifact_dir),
            host="0.0.0.0",
            port=8000,
        )
        
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        
        assert "-m" in cmd
        assert "vllm.entrypoints.openai.api_server" in cmd
        assert "--model" in cmd
        assert "--quantization" in cmd
        assert "awq" in cmd


class TestTenPakTGILoader:
    """Test TenPakTGILoader functionality."""
    
    @pytest.fixture
    def mock_artifact_dir(self, tmp_path):
        """Create mock artifact directory."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "mistral-7b",
            "quantization": {
                "method": "gptq",
                "bits": 4,
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        return artifact_dir
    
    def test_create_tgi_config(self, mock_artifact_dir):
        """Test TGI configuration creation."""
        config = TenPakTGILoader.create_tgi_config(
            artifact_path=str(mock_artifact_dir),
            max_batch_prefill_tokens=4096,
            max_total_tokens=2048,
        )
        
        assert config["model_id"] == "mistral-7b"
        assert config["quantize"] == "gptq"
        assert config["max_batch_prefill_tokens"] == 4096
        assert config["max_total_tokens"] == 2048
    
    def test_create_tgi_config_bitsandbytes(self, tmp_path):
        """Test TGI config for bitsandbytes."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "gpt2",
            "quantization": {
                "method": "bitsandbytes",
                "bits": 8,
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        config = TenPakTGILoader.create_tgi_config(str(artifact_dir))
        
        assert config["quantize"] == "bitsandbytes"
    
    @patch("inference.vllm_integration.subprocess.run")
    def test_serve_with_tgi(self, mock_subprocess, mock_artifact_dir):
        """Test TGI server launching."""
        TenPakTGILoader.serve_with_tgi(
            artifact_path=str(mock_artifact_dir),
            port=8080,
        )
        
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        
        assert "docker" in cmd
        assert "run" in cmd
        assert "--model-id" in cmd
        assert "--quantize" in cmd
        assert "gptq" in cmd


class TestBenchmarkInference:
    """Test inference benchmarking."""
    
    @pytest.fixture
    def mock_artifact_dir(self, tmp_path):
        """Create mock artifact directory."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        
        manifest = {
            "version": "1.0",
            "model_id": "gpt2",
            "quantization": {
                "method": "awq",
                "bits": 4,
            },
        }
        
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)
        
        return artifact_dir
    
    @patch("inference.vllm_integration.TenPakVLLMLoader.create_vllm_engine")
    def test_benchmark_vllm(self, mock_create_engine, mock_artifact_dir):
        """Test vLLM benchmarking."""
        mock_engine = Mock()
        mock_output = Mock()
        mock_output.outputs = [Mock(text="test output")]
        mock_engine.generate.return_value = [mock_output]
        mock_create_engine.return_value = mock_engine
        
        metrics = benchmark_inference(
            artifact_path=str(mock_artifact_dir),
            engine="vllm",
            num_samples=10,
            prompt_length=64,
            output_length=64,
        )
        
        assert "latency_mean_ms" in metrics
        assert "throughput_samples_per_sec" in metrics
        assert "total_time_sec" in metrics
        assert metrics["latency_mean_ms"] > 0
        assert metrics["throughput_samples_per_sec"] > 0
    
    @patch("inference.vllm_integration.AutoModelForCausalLM")
    @patch("inference.vllm_integration.AutoTokenizer")
    def test_benchmark_transformers(self, mock_tokenizer, mock_model_class, mock_artifact_dir):
        """Test Transformers benchmarking."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": Mock()}
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_model.generate.return_value = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        metrics = benchmark_inference(
            artifact_path=str(mock_artifact_dir),
            engine="transformers",
            num_samples=5,
            prompt_length=32,
            output_length=32,
        )
        
        assert "latency_mean_ms" in metrics
        assert "latency_p50_ms" in metrics
        assert "latency_p95_ms" in metrics
        assert "latency_p99_ms" in metrics
        assert "throughput_samples_per_sec" in metrics
    
    def test_benchmark_invalid_engine(self, mock_artifact_dir):
        """Test that invalid engine raises error."""
        with pytest.raises(ValueError, match="Unknown engine"):
            benchmark_inference(
                artifact_path=str(mock_artifact_dir),
                engine="invalid_engine",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
