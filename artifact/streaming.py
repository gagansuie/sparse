"""
TenPak Artifact - Streaming Support

Enables partial downloads and streaming inference.
"""

import os
import hashlib
from dataclasses import dataclass
from typing import Iterator, Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ChunkIterator:
    """Iterator for streaming chunks from an artifact."""
    artifact_path: Path
    chunk_names: List[str]
    current_index: int = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple:
        if self.current_index >= len(self.chunk_names):
            raise StopIteration
        
        name = self.chunk_names[self.current_index]
        self.current_index += 1
        
        chunk_path = self.artifact_path / "chunks" / f"{name.replace('.', '_')}.bin"
        with open(chunk_path, "rb") as f:
            data = f.read()
        
        return name, data
    
    def __len__(self):
        return len(self.chunk_names)
    
    @property
    def progress(self) -> float:
        return self.current_index / len(self.chunk_names) if self.chunk_names else 1.0


def stream_chunks(
    artifact_path: str,
    layer_types: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> ChunkIterator:
    """Create a chunk iterator for streaming.
    
    Args:
        artifact_path: Path to .tnpk artifact
        layer_types: Filter by layer types (e.g., ["attn", "mlp"])
        layer_indices: Filter by layer indices (e.g., [0, 1, 2])
        
    Returns:
        ChunkIterator for streaming chunks
    """
    import json
    
    artifact_path = Path(artifact_path)
    manifest_path = artifact_path / "manifest.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    chunk_names = []
    for chunk in manifest["chunks"]:
        # Apply filters
        if layer_types and chunk.get("layer_type") not in layer_types:
            continue
        if layer_indices and chunk.get("layer_index") not in layer_indices:
            continue
        chunk_names.append(chunk["name"])
    
    return ChunkIterator(
        artifact_path=artifact_path,
        chunk_names=chunk_names,
    )


def fetch_chunk(
    artifact_path: str,
    chunk_name: str,
    verify: bool = True,
) -> bytes:
    """Fetch a single chunk from an artifact.
    
    Args:
        artifact_path: Path to .tnpk artifact
        chunk_name: Name of the chunk to fetch
        verify: Whether to verify the hash
        
    Returns:
        Chunk data as bytes
    """
    import json
    
    artifact_path = Path(artifact_path)
    
    # Load manifest to get expected hash
    if verify:
        with open(artifact_path / "manifest.json", "r") as f:
            manifest = json.load(f)
        
        expected_hash = None
        for chunk in manifest["chunks"]:
            if chunk["name"] == chunk_name:
                expected_hash = chunk["sha256"]
                break
    
    # Read chunk
    chunk_path = artifact_path / "chunks" / f"{chunk_name.replace('.', '_')}.bin"
    with open(chunk_path, "rb") as f:
        data = f.read()
    
    # Verify
    if verify and expected_hash:
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Hash mismatch for {chunk_name}")
    
    return data


def estimate_download_size(
    artifact_path: str,
    layer_types: Optional[List[str]] = None,
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Estimate download size for partial fetch.
    
    Args:
        artifact_path: Path to .tnpk artifact
        layer_types: Filter by layer types
        layer_indices: Filter by layer indices
        
    Returns:
        Dict with size estimates
    """
    import json
    
    artifact_path = Path(artifact_path)
    
    with open(artifact_path / "manifest.json", "r") as f:
        manifest = json.load(f)
    
    total_size = 0
    selected_size = 0
    selected_chunks = 0
    
    for chunk in manifest["chunks"]:
        total_size += chunk["size"]
        
        # Check if selected
        selected = True
        if layer_types and chunk.get("layer_type") not in layer_types:
            selected = False
        if layer_indices and chunk.get("layer_index") not in layer_indices:
            selected = False
        
        if selected:
            selected_size += chunk["size"]
            selected_chunks += 1
    
    return {
        "total_size_bytes": total_size,
        "selected_size_bytes": selected_size,
        "selected_chunks": selected_chunks,
        "total_chunks": len(manifest["chunks"]),
        "savings_pct": (1 - selected_size / total_size) * 100 if total_size > 0 else 0,
    }


class RemoteChunkFetcher:
    """Fetcher for streaming chunks from a remote URL.
    
    This is a placeholder for future HTTP streaming support.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.manifest = None
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load manifest from remote."""
        # Placeholder - would use httpx or aiohttp
        raise NotImplementedError("Remote fetching not yet implemented")
    
    def fetch_chunk(self, chunk_name: str) -> bytes:
        """Fetch a chunk from remote."""
        # Placeholder - would use HTTP range requests
        raise NotImplementedError("Remote fetching not yet implemented")
    
    def stream_chunks(self, **filters) -> Iterator[tuple]:
        """Stream chunks from remote."""
        raise NotImplementedError("Remote fetching not yet implemented")
