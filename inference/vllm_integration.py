"""
vLLM Integration for TenPak Artifacts

Helpers for loading TenPak artifacts directly into vLLM.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import torch


class TenPakVLLMLoader:
    """Load TenPak artifacts for vLLM inference."""
    
    @staticmethod
    def load_for_vllm(
        artifact_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Prepare a TenPak artifact for vLLM loading.
        
        Args:
            artifact_path: Path to TenPak artifact directory
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory utilization (0-1)
            
        Returns:
            Dict with vLLM configuration and model path
        """
        artifact_path = Path(artifact_path)
        
        # Load manifest
        with open(artifact_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        
        model_id = manifest.get("model_id")
        quantization_method = manifest.get("quantization", {}).get("method", "none")
        
        # Determine vLLM quantization parameter
        if quantization_method == "gptq":
            quantization = "gptq"
        elif quantization_method == "awq":
            quantization = "awq"
        else:
            quantization = None
        
        vllm_config = {
            "model": model_id or str(artifact_path),
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "quantization": quantization,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        
        if max_model_len is not None:
            vllm_config["max_model_len"] = max_model_len
        
        return vllm_config
    
    @staticmethod
    def create_vllm_engine(artifact_path: str, **kwargs):
        """
        Create a vLLM LLM engine from a TenPak artifact.
        
        Args:
            artifact_path: Path to TenPak artifact
            **kwargs: Additional vLLM engine arguments
            
        Returns:
            vLLM LLM engine
        """
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            )
        
        config = TenPakVLLMLoader.load_for_vllm(artifact_path, **kwargs)
        
        # Merge user kwargs
        config.update(kwargs)
        
        return LLM(**config)
    
    @staticmethod
    def serve_with_vllm(
        artifact_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs,
    ):
        """
        Serve a TenPak artifact via vLLM OpenAI-compatible API.
        
        Args:
            artifact_path: Path to TenPak artifact
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional vLLM arguments
        """
        import subprocess
        import sys
        
        artifact_path = Path(artifact_path)
        
        # Load manifest to get model ID
        with open(artifact_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        
        model_id = manifest.get("model_id")
        quantization_method = manifest.get("quantization", {}).get("method", "none")
        
        # Build vLLM command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_id or str(artifact_path),
            "--host", host,
            "--port", str(port),
        ]
        
        if quantization_method == "gptq":
            cmd.extend(["--quantization", "gptq"])
        elif quantization_method == "awq":
            cmd.extend(["--quantization", "awq"])
        
        # Add user kwargs
        for key, value in kwargs.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        print(f"Starting vLLM server: {' '.join(cmd)}")
        subprocess.run(cmd)


class TenPakTGILoader:
    """Load TenPak artifacts for Text Generation Inference (TGI)."""
    
    @staticmethod
    def create_tgi_config(
        artifact_path: str,
        max_batch_prefill_tokens: int = 4096,
        max_batch_total_tokens: int = 8192,
        max_input_length: int = 1024,
        max_total_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Create TGI configuration for a TenPak artifact.
        
        Args:
            artifact_path: Path to TenPak artifact
            max_batch_prefill_tokens: Maximum prefill batch size
            max_batch_total_tokens: Maximum total batch size
            max_input_length: Maximum input length
            max_total_tokens: Maximum total tokens
            
        Returns:
            TGI configuration dict
        """
        artifact_path = Path(artifact_path)
        
        with open(artifact_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        
        model_id = manifest.get("model_id")
        quantization_method = manifest.get("quantization", {}).get("method", "none")
        
        # Determine TGI quantization
        if quantization_method == "gptq":
            quantize = "gptq"
        elif quantization_method == "awq":
            quantize = "awq"
        elif quantization_method == "bitsandbytes":
            quantize = "bitsandbytes"
        else:
            quantize = None
        
        tgi_config = {
            "model_id": model_id or str(artifact_path),
            "max_batch_prefill_tokens": max_batch_prefill_tokens,
            "max_batch_total_tokens": max_batch_total_tokens,
            "max_input_length": max_input_length,
            "max_total_tokens": max_total_tokens,
        }
        
        if quantize:
            tgi_config["quantize"] = quantize
        
        return tgi_config
    
    @staticmethod
    def serve_with_tgi(
        artifact_path: str,
        port: int = 8080,
        **kwargs,
    ):
        """
        Serve a TenPak artifact via TGI.
        
        Args:
            artifact_path: Path to TenPak artifact
            port: Port to bind to
            **kwargs: Additional TGI arguments
        """
        import subprocess
        
        config = TenPakTGILoader.create_tgi_config(artifact_path, **kwargs)
        
        # Build TGI docker command
        cmd = [
            "docker", "run", "--gpus", "all",
            "-p", f"{port}:80",
            "-v", f"{artifact_path}:/data",
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id", config["model_id"],
            "--max-batch-prefill-tokens", str(config["max_batch_prefill_tokens"]),
            "--max-batch-total-tokens", str(config["max_batch_total_tokens"]),
            "--max-input-length", str(config["max_input_length"]),
            "--max-total-tokens", str(config["max_total_tokens"]),
        ]
        
        if "quantize" in config:
            cmd.extend(["--quantize", config["quantize"]])
        
        print(f"Starting TGI server: {' '.join(cmd)}")
        subprocess.run(cmd)


def benchmark_inference(
    artifact_path: str,
    engine: str = "vllm",
    num_samples: int = 100,
    prompt_length: int = 128,
    output_length: int = 128,
) -> Dict[str, float]:
    """
    Benchmark inference performance of a TenPak artifact.
    
    Args:
        artifact_path: Path to artifact
        engine: Inference engine ("vllm" or "transformers")
        num_samples: Number of inference samples
        prompt_length: Prompt length in tokens
        output_length: Output length in tokens
        
    Returns:
        Dict with latency and throughput metrics
    """
    import time
    import numpy as np
    
    if engine == "vllm":
        llm = TenPakVLLMLoader.create_vllm_engine(artifact_path)
        
        # Warm up
        llm.generate(["test" * 10], max_tokens=10)
        
        # Benchmark
        prompts = ["test " * prompt_length] * num_samples
        
        start = time.time()
        outputs = llm.generate(prompts, max_tokens=output_length)
        end = time.time()
        
        total_time = end - start
        latency_per_sample = total_time / num_samples
        throughput = num_samples / total_time
        
    elif engine == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        artifact_path = Path(artifact_path)
        with open(artifact_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        
        model_id = manifest.get("model_id")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Warm up
        inputs = tokenizer("test" * 10, return_tensors="pt").to(model.device)
        model.generate(**inputs, max_new_tokens=10)
        
        # Benchmark
        prompt = "test " * prompt_length
        latencies = []
        
        for _ in range(num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            model.generate(**inputs, max_new_tokens=output_length)
            end = time.time()
            
            latencies.append(end - start)
        
        total_time = sum(latencies)
        latency_per_sample = np.mean(latencies)
        throughput = num_samples / total_time
    
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
    return {
        "latency_mean_ms": latency_per_sample * 1000,
        "latency_p50_ms": np.percentile(latencies, 50) * 1000 if engine == "transformers" else latency_per_sample * 1000,
        "latency_p95_ms": np.percentile(latencies, 95) * 1000 if engine == "transformers" else latency_per_sample * 1000,
        "latency_p99_ms": np.percentile(latencies, 99) * 1000 if engine == "transformers" else latency_per_sample * 1000,
        "throughput_samples_per_sec": throughput,
        "total_time_sec": total_time,
    }
