"""
Tests for artifact format functionality
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from artifact.format import (
    ArtifactManifest,
    ChunkInfo,
    TNPK_VERSION,
)


class TestChunkInfo:
    """Test ChunkInfo dataclass."""
    
    def test_chunk_creation(self):
        """Test chunk info creation."""
        chunk = ChunkInfo(
            name="layer.0.weight",
            sha256="abc123",
            size=1024,
            offset=0,
            layer_type="attn",
            layer_index=0,
            compression="awq",
        )
        
        assert chunk.name == "layer.0.weight"
        assert chunk.sha256 == "abc123"
        assert chunk.size == 1024
        assert chunk.layer_type == "attn"
    
    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = ChunkInfo(
            name="test",
            sha256="hash",
            size=100,
            offset=50,
        )
        
        d = chunk.to_dict()
        
        assert d["name"] == "test"
        assert d["sha256"] == "hash"
        assert d["size"] == 100
        assert d["offset"] == 50
    
    def test_chunk_from_dict(self):
        """Test chunk deserialization."""
        d = {
            "name": "test",
            "sha256": "hash",
            "size": 100,
            "offset": 50,
            "layer_type": None,
            "layer_index": None,
            "compression": None,
        }
        
        chunk = ChunkInfo.from_dict(d)
        
        assert chunk.name == "test"
        assert chunk.sha256 == "hash"


class TestArtifactManifest:
    """Test ArtifactManifest dataclass."""
    
    def test_default_manifest(self):
        """Test default manifest creation."""
        manifest = ArtifactManifest()
        
        assert manifest.version == TNPK_VERSION
        assert manifest.model_id == ""
        assert manifest.compression_ratio == 1.0
        assert len(manifest.chunks) == 0
    
    def test_manifest_with_quantization(self):
        """Test manifest with quantization metadata."""
        manifest = ArtifactManifest(
            model_id="gpt2",
            quantization={
                "method": "awq",
                "bits": 4,
                "group_size": 128,
            },
            compression_ratio=7.5,
        )
        
        assert manifest.quantization["method"] == "awq"
        assert manifest.quantization["bits"] == 4
        assert manifest.compression_ratio == 7.5
    
    def test_manifest_with_delta(self):
        """Test manifest with delta compression."""
        manifest = ArtifactManifest(
            model_id="fine-tuned-model",
            delta={
                "base_model_id": "base-model",
                "changed_layers": ["layer.10", "layer.11"],
                "savings_pct": 85.0,
            },
        )
        
        assert manifest.delta["base_model_id"] == "base-model"
        assert len(manifest.delta["changed_layers"]) == 2
        assert manifest.delta["savings_pct"] == 85.0
    
    def test_manifest_with_optimization(self):
        """Test manifest with optimization results."""
        manifest = ArtifactManifest(
            model_id="optimized-model",
            optimization={
                "selected_method": "awq_balanced",
                "candidates_tested": ["gptq", "awq", "bnb"],
                "latency_p50_ms": 45.2,
                "throughput_tps": 120.5,
            },
        )
        
        assert manifest.optimization["selected_method"] == "awq_balanced"
        assert len(manifest.optimization["candidates_tested"]) == 3
        assert manifest.optimization["latency_p50_ms"] == 45.2
    
    def test_manifest_to_dict(self):
        """Test manifest serialization."""
        manifest = ArtifactManifest(
            model_id="test-model",
            architecture="llama",
            num_layers=32,
            hidden_size=4096,
            quantization={"method": "gptq", "bits": 4},
        )
        
        d = manifest.to_dict()
        
        assert d["model_id"] == "test-model"
        assert d["architecture"] == "llama"
        assert d["quantization"]["method"] == "gptq"
        assert "created_at" in d
    
    def test_manifest_from_dict(self):
        """Test manifest deserialization."""
        d = {
            "version": "1.0",
            "model_id": "test",
            "created_at": "2024-01-01T00:00:00",
            "architecture": "gpt2",
            "num_layers": 12,
            "hidden_size": 768,
            "vocab_size": 50257,
            "total_params": 124000000,
            "quantization": {"method": "awq"},
            "compression_ratio": 7.5,
            "original_size_bytes": 1000000,
            "compressed_size_bytes": 133333,
            "delta": None,
            "optimization": None,
            "chunks": [],
            "signature": None,
            "signer": None,
            "signed_at": None,
        }
        
        manifest = ArtifactManifest.from_dict(d)
        
        assert manifest.model_id == "test"
        assert manifest.architecture == "gpt2"
        assert manifest.quantization["method"] == "awq"
    
    def test_manifest_with_chunks(self):
        """Test manifest with chunk list."""
        chunks = [
            ChunkInfo(name="chunk1", sha256="hash1", size=100, offset=0),
            ChunkInfo(name="chunk2", sha256="hash2", size=200, offset=100),
        ]
        
        manifest = ArtifactManifest(
            model_id="test",
            chunks=chunks,
        )
        
        assert len(manifest.chunks) == 2
        assert manifest.chunks[0].name == "chunk1"
        assert manifest.chunks[1].name == "chunk2"
    
    def test_manifest_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = ArtifactManifest(
            model_id="test-model",
            architecture="llama",
            num_layers=32,
            quantization={"method": "awq", "bits": 4},
            delta={"base_model_id": "base"},
            optimization={"selected_method": "awq"},
            chunks=[
                ChunkInfo(name="c1", sha256="h1", size=100, offset=0),
            ],
        )
        
        # Serialize
        d = original.to_dict()
        json_str = json.dumps(d)
        
        # Deserialize
        d2 = json.loads(json_str)
        restored = ArtifactManifest.from_dict(d2)
        
        assert restored.model_id == original.model_id
        assert restored.architecture == original.architecture
        assert restored.quantization == original.quantization
        assert restored.delta == original.delta
        assert len(restored.chunks) == len(original.chunks)


class TestArtifactFormatIntegration:
    """Integration tests for artifact format."""
    
    def test_save_and_load_manifest(self, tmp_path):
        """Test saving and loading manifest file."""
        manifest = ArtifactManifest(
            model_id="test-model",
            quantization={"method": "awq", "bits": 4},
            compression_ratio=7.5,
        )
        
        # Save
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        # Load
        with open(manifest_path, "r") as f:
            loaded_dict = json.load(f)
        
        loaded_manifest = ArtifactManifest.from_dict(loaded_dict)
        
        assert loaded_manifest.model_id == manifest.model_id
        assert loaded_manifest.quantization == manifest.quantization
    
    def test_complete_artifact_structure(self, tmp_path):
        """Test creating complete artifact structure."""
        artifact_dir = tmp_path / "artifact"
        artifact_dir.mkdir()
        chunks_dir = artifact_dir / "chunks"
        chunks_dir.mkdir()
        
        # Create manifest
        manifest = ArtifactManifest(
            model_id="gpt2",
            quantization={"method": "awq", "bits": 4},
            chunks=[
                ChunkInfo(
                    name="layer.0.weight",
                    sha256="abc123",
                    size=1024,
                    offset=0,
                    file="chunks/0000.bin",
                ),
            ],
        )
        
        # Save manifest
        with open(artifact_dir / "manifest.json", "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        
        # Create chunk file
        chunk_file = chunks_dir / "0000.bin"
        chunk_file.write_bytes(b"fake_chunk_data")
        
        # Verify structure
        assert (artifact_dir / "manifest.json").exists()
        assert (artifact_dir / "chunks" / "0000.bin").exists()
        
        # Load and verify
        with open(artifact_dir / "manifest.json", "r") as f:
            loaded = json.load(f)
        
        assert loaded["model_id"] == "gpt2"
        assert len(loaded["chunks"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
