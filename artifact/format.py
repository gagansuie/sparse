"""
TenPak Artifact Format (.tnpk)

Defines the streamable artifact format for compressed models.
"""

import os
import json
import hashlib
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
from pathlib import Path


TNPK_VERSION = "1.0"
TNPK_MAGIC = b"TNPK"


@dataclass
class ChunkInfo:
    """Information about a single chunk in the artifact."""
    name: str  # e.g., "embed", "layers.0.attn", "layers.0.mlp"
    sha256: str  # Content hash for deduplication
    size: int  # Size in bytes
    offset: int  # Offset in consolidated file (if using single-file mode)
    
    # Optional metadata
    layer_type: Optional[str] = None  # "embed", "attn", "mlp", "norm", "lm_head"
    layer_index: Optional[int] = None
    compression: Optional[str] = None  # "int4_awq", "int4_residual", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "size": self.size,
            "offset": self.offset,
            "layer_type": self.layer_type,
            "layer_index": self.layer_index,
            "compression": self.compression,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkInfo":
        return cls(**d)


@dataclass
class ArtifactManifest:
    """Manifest for a .tnpk artifact."""
    version: str = TNPK_VERSION
    model_id: str = ""
    codec: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Model info
    architecture: str = ""  # "llama", "mistral", "gpt2", etc.
    num_layers: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    total_params: int = 0
    
    # Compression info
    compression_ratio: float = 1.0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    # Chunks
    chunks: List[ChunkInfo] = field(default_factory=list)
    
    # Signing (optional)
    signature: Optional[str] = None
    signer: Optional[str] = None
    signed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "codec": self.codec,
            "created_at": self.created_at,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "total_params": self.total_params,
            "compression_ratio": self.compression_ratio,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "chunks": [c.to_dict() for c in self.chunks],
            "signature": self.signature,
            "signer": self.signer,
            "signed_at": self.signed_at,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactManifest":
        chunks = [ChunkInfo.from_dict(c) for c in d.pop("chunks", [])]
        return cls(chunks=chunks, **d)


class TenPakArtifact:
    """A .tnpk artifact for streaming model distribution."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self.manifest: Optional[ArtifactManifest] = None
        self._loaded = False
    
    def load(self) -> "TenPakArtifact":
        """Load the artifact manifest."""
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json found in {self.path}")
        
        with open(manifest_path, "r") as f:
            self.manifest = ArtifactManifest.from_dict(json.load(f))
        
        self._loaded = True
        return self
    
    def get_chunk(self, name: str) -> Optional[ChunkInfo]:
        """Get info about a specific chunk."""
        if not self._loaded:
            self.load()
        
        for chunk in self.manifest.chunks:
            if chunk.name == name:
                return chunk
        return None
    
    def read_chunk(self, name: str) -> bytes:
        """Read a specific chunk's data."""
        chunk = self.get_chunk(name)
        if chunk is None:
            raise KeyError(f"Chunk not found: {name}")
        
        chunk_path = self.path / "chunks" / f"{name.replace('.', '_')}.bin"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        
        with open(chunk_path, "rb") as f:
            data = f.read()
        
        # Verify hash
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != chunk.sha256:
            raise ValueError(f"Chunk hash mismatch for {name}: expected {chunk.sha256}, got {actual_hash}")
        
        return data
    
    def stream_chunks(self) -> Iterator[tuple]:
        """Stream all chunks as (name, data) tuples."""
        if not self._loaded:
            self.load()
        
        for chunk in self.manifest.chunks:
            yield chunk.name, self.read_chunk(chunk.name)
    
    def verify(self) -> bool:
        """Verify all chunk hashes."""
        if not self._loaded:
            self.load()
        
        for chunk in self.manifest.chunks:
            try:
                self.read_chunk(chunk.name)  # This verifies the hash
            except (ValueError, FileNotFoundError):
                return False
        return True
    
    @property
    def is_signed(self) -> bool:
        """Check if artifact is signed."""
        if not self._loaded:
            self.load()
        return self.manifest.signature is not None
    
    def __repr__(self):
        if self._loaded:
            return f"TenPakArtifact({self.manifest.model_id}, {len(self.manifest.chunks)} chunks)"
        return f"TenPakArtifact({self.path}, not loaded)"


def compute_chunk_hash(data: bytes) -> str:
    """Compute SHA256 hash of chunk data."""
    return hashlib.sha256(data).hexdigest()


def create_artifact(
    model_id: str,
    output_path: str,
    codec: str = "int4_awq",
    chunk_size_mb: int = 64,
    progress_callback: Optional[callable] = None,
) -> TenPakArtifact:
    """Create a .tnpk artifact from a HuggingFace model.
    
    Args:
        model_id: HuggingFace model ID
        output_path: Output path for .tnpk artifact
        codec: Compression codec to use
        chunk_size_mb: Target chunk size in MB
        progress_callback: Optional callback(msg, progress)
        
    Returns:
        TenPakArtifact instance
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from core.codecs import compress_int4_awq
    
    def log(msg: str, progress: float = 0.0):
        if progress_callback:
            progress_callback(msg, progress)
        print(f"[Artifact] {msg}")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    chunks_path = output_path / "chunks"
    chunks_path.mkdir(exist_ok=True)
    
    log(f"Loading model: {model_id}", 0.0)
    
    # Load config first
    config = AutoConfig.from_pretrained(model_id)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    log("Creating chunks...", 0.20)
    
    # Create manifest
    manifest = ArtifactManifest(
        model_id=model_id,
        codec=codec,
        architecture=config.model_type,
        num_layers=getattr(config, "num_hidden_layers", 0),
        hidden_size=getattr(config, "hidden_size", 0),
        vocab_size=getattr(config, "vocab_size", 0),
    )
    
    # Process each layer into chunks
    chunks = []
    total_original = 0
    total_compressed = 0
    total_params = 0
    
    state_dict = model.state_dict()
    param_names = list(state_dict.keys())
    
    for i, name in enumerate(param_names):
        progress = 0.20 + (i / len(param_names)) * 0.70
        
        tensor = state_dict[name]
        total_params += tensor.numel()
        original_size = tensor.numel() * tensor.element_size()
        total_original += original_size
        
        # Determine layer type
        layer_type = None
        layer_index = None
        
        if "embed" in name:
            layer_type = "embed"
        elif "lm_head" in name or "output" in name:
            layer_type = "lm_head"
        elif "layers." in name or "h." in name:
            # Extract layer index
            parts = name.split(".")
            for j, p in enumerate(parts):
                if p.isdigit():
                    layer_index = int(p)
                    break
            if "attn" in name or "attention" in name:
                layer_type = "attn"
            elif "mlp" in name or "fc" in name:
                layer_type = "mlp"
            elif "norm" in name or "ln" in name:
                layer_type = "norm"
        
        # Compress if it's a weight matrix
        if len(tensor.shape) == 2 and tensor.numel() > 1000:
            # Apply compression
            if codec == "int4_awq":
                deq_weight, comp_ratio = compress_int4_awq(tensor, group_size=256)
                compressed_data = deq_weight.numpy().tobytes()
            else:
                compressed_data = tensor.numpy().tobytes()
                comp_ratio = 1.0
        else:
            compressed_data = tensor.numpy().tobytes()
            comp_ratio = 1.0
        
        compressed_size = len(compressed_data)
        total_compressed += compressed_size
        
        # Compute hash
        chunk_hash = compute_chunk_hash(compressed_data)
        
        # Save chunk
        chunk_filename = name.replace(".", "_") + ".bin"
        with open(chunks_path / chunk_filename, "wb") as f:
            f.write(compressed_data)
        
        # Add to manifest
        chunk_info = ChunkInfo(
            name=name,
            sha256=chunk_hash,
            size=compressed_size,
            offset=0,  # Would be set in single-file mode
            layer_type=layer_type,
            layer_index=layer_index,
            compression=codec if comp_ratio > 1 else None,
        )
        chunks.append(chunk_info)
        
        if (i + 1) % 50 == 0:
            log(f"Processed {i+1}/{len(param_names)} parameters", progress)
    
    log("Finalizing artifact...", 0.95)
    
    # Update manifest
    manifest.chunks = chunks
    manifest.total_params = total_params
    manifest.original_size_bytes = total_original
    manifest.compressed_size_bytes = total_compressed
    manifest.compression_ratio = total_original / max(total_compressed, 1)
    
    # Save manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    
    log(f"Artifact created: {output_path}", 1.0)
    log(f"  Chunks: {len(chunks)}")
    log(f"  Compression: {manifest.compression_ratio:.2f}x")
    log(f"  Size: {total_compressed / 1e6:.1f} MB")
    
    # Cleanup
    del model, state_dict
    
    artifact = TenPakArtifact(str(output_path))
    artifact.load()
    return artifact


def load_artifact(path: str) -> TenPakArtifact:
    """Load an existing .tnpk artifact."""
    artifact = TenPakArtifact(path)
    artifact.load()
    return artifact
