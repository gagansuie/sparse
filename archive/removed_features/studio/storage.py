"""
TenPak Studio - Artifact Storage

Handles packaging, storage, and retrieval of compressed model artifacts.
Designed for integration with HF Hub and cloud storage.

Artifact Format:
    tenpak_artifact/
    ├── manifest.json      # Metadata, codec info, layer allocations
    ├── weights/           # Compressed weight shards
    │   ├── shard_0.bin
    │   └── ...
    └── codebook.bin       # Shared codebooks (if using VQ)
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import torch


@dataclass
class ArtifactManifest:
    """Metadata for a compressed model artifact."""
    version: str = "1.0"
    created_at: str = ""
    
    # Source model
    model_id: str = ""
    model_hash: str = ""  # SHA256 of original weights
    
    # Compression settings
    quantization_method: str = "awq"  # gptq, awq, bitsandbytes
    quantization_bits: int = 4
    target: str = "balanced"
    
    # Results
    compression_ratio: float = 1.0
    baseline_ppl: float = 0.0
    compressed_ppl: float = 0.0
    ppl_delta: float = 0.0
    
    # Layer info
    num_layers: int = 0
    total_params: int = 0
    
    # Shard info
    num_shards: int = 1
    shard_size_bytes: int = 0
    
    # Layer allocations
    allocations: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at == "":
            self.created_at = datetime.utcnow().isoformat()
        if self.allocations is None:
            self.allocations = {}


class ArtifactStorage:
    """Manages compressed model artifacts."""
    
    def __init__(self, base_path: str = "/tmp/tenpak_artifacts"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_artifact(
        self,
        job_id: str,
        model_id: str,
        quantized_model_path: str,
        metrics: Dict[str, float],
        quantization_method: str = "awq",
        quantization_bits: int = 4
    ) -> str:
        """Save a quantized model artifact.
        
        Args:
            job_id: Unique job identifier
            model_id: Original model ID
            quantized_model_path: Path to quantized model
            metrics: Compression metrics (compression_ratio, ppl, etc.)
            quantization_method: Method used (gptq, awq, bitsandbytes)
            quantization_bits: Bit width (3, 4, 8)
            
        Returns:
            Path to saved artifact
        """
        artifact_path = os.path.join(self.base_path, job_id)
        os.makedirs(artifact_path, exist_ok=True)
        
        # Create manifest
        manifest = ArtifactManifest(
            model_id=model_id,
            quantization_method=quantization_method,
            quantization_bits=quantization_bits,
            compression_ratio=metrics.get("compression_ratio", 1.0),
            baseline_ppl=metrics.get("baseline_ppl", 0.0),
            compressed_ppl=metrics.get("compressed_ppl", 0.0),
            ppl_delta=metrics.get("ppl_delta", 0.0),
        )
        
        # Copy quantized model to artifact directory
        import shutil
        model_dest = os.path.join(artifact_path, "quantized_model")
        if os.path.isdir(quantized_model_path):
            shutil.copytree(quantized_model_path, model_dest, dirs_exist_ok=True)
        else:
            # Single file - create directory and copy
            os.makedirs(model_dest, exist_ok=True)
            shutil.copy(quantized_model_path, os.path.join(model_dest, "model.bin"))
        
        # Save manifest
        manifest.created_at = datetime.utcnow().isoformat()
        manifest_path = os.path.join(artifact_path, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)
        
        return artifact_path
    
    def load_artifact(self, artifact_path: str) -> Dict[str, Any]:
        """Load a compressed model artifact.
        
        Args:
            artifact_path: Path to artifact directory
            
        Returns:
            Dict with 'manifest' and 'weights'
        """
        manifest_path = os.path.join(artifact_path, "manifest.json")
        weights_path = os.path.join(artifact_path, "weights", "shard_0.bin")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        
        manifest = ArtifactManifest(**manifest_data)
        
        weights = {}
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location="cpu")
        
        return {
            "manifest": manifest,
            "weights": weights
        }
    
    def list_artifacts(self) -> List[Dict[str, Any]]:
        """List all stored artifacts."""
        artifacts = []
        
        for job_id in os.listdir(self.base_path):
            artifact_path = os.path.join(self.base_path, job_id)
            if os.path.isdir(artifact_path):
                manifest_path = os.path.join(artifact_path, "manifest.json")
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                    artifacts.append({
                        "job_id": job_id,
                        "path": artifact_path,
                        **manifest
                    })
        
        return artifacts
    
    def delete_artifact(self, job_id: str) -> bool:
        """Delete an artifact."""
        import shutil
        artifact_path = os.path.join(self.base_path, job_id)
        
        if os.path.exists(artifact_path):
            shutil.rmtree(artifact_path)
            return True
        return False
    
    def get_artifact_size(self, job_id: str) -> int:
        """Get total size of an artifact in bytes."""
        artifact_path = os.path.join(self.base_path, job_id)
        total_size = 0
        
        for dirpath, dirnames, filenames in os.walk(artifact_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        return total_size


# Global storage instance
_storage = None


def get_storage(base_path: str = "/tmp/tenpak_artifacts") -> ArtifactStorage:
    """Get the global storage instance."""
    global _storage
    if _storage is None:
        _storage = ArtifactStorage(base_path)
    return _storage
