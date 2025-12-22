"""
HTTP Streaming for TenPak Artifacts

Enables remote loading of artifacts via HTTP range requests.
Supports CDN distribution and lazy loading of model chunks.
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, List
from pathlib import Path
import requests
from dataclasses import dataclass


@dataclass
class ChunkMetadata:
    """Metadata for a single chunk."""
    name: str
    sha256: str
    offset: int
    size: int
    file: str


class HTTPArtifactStreamer:
    """Stream TenPak artifacts from HTTP endpoints."""
    
    def __init__(
        self,
        base_url: str,
        cache_dir: Optional[str] = None,
        verify_checksums: bool = True,
    ):
        """
        Initialize HTTP artifact streamer.
        
        Args:
            base_url: Base URL of the artifact (e.g., https://cdn.example.com/artifacts/model-123/)
            cache_dir: Local cache directory (default: ~/.cache/tenpak)
            verify_checksums: Whether to verify SHA256 checksums
        """
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/tenpak"))
        self.verify_checksums = verify_checksums
        self.manifest: Optional[Dict[str, Any]] = None
        self.chunks: List[ChunkMetadata] = []
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load and cache the artifact manifest."""
        if self.manifest is not None:
            return self.manifest
        
        manifest_url = f"{self.base_url}/manifest.json"
        
        # Check cache first
        manifest_cache_path = self._get_cache_path("manifest.json")
        if manifest_cache_path.exists():
            try:
                with open(manifest_cache_path, "r") as f:
                    self.manifest = json.load(f)
                    self._parse_chunks()
                    return self.manifest
            except Exception:
                pass  # Fall through to HTTP fetch
        
        # Fetch from HTTP
        response = requests.get(manifest_url)
        response.raise_for_status()
        
        self.manifest = response.json()
        
        # Cache manifest
        with open(manifest_cache_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
        
        self._parse_chunks()
        return self.manifest
    
    def _parse_chunks(self):
        """Parse chunk metadata from manifest."""
        if not self.manifest or "chunks" not in self.manifest:
            return
        
        self.chunks = [
            ChunkMetadata(**chunk)
            for chunk in self.manifest["chunks"]
        ]
    
    def get_chunk(self, chunk_index: int) -> bytes:
        """
        Fetch a specific chunk by index.
        
        Args:
            chunk_index: Index of the chunk to fetch
            
        Returns:
            Chunk data as bytes
        """
        if not self.chunks:
            self.load_manifest()
        
        if chunk_index >= len(self.chunks):
            raise IndexError(f"Chunk index {chunk_index} out of range (max: {len(self.chunks) - 1})")
        
        chunk = self.chunks[chunk_index]
        
        # Check cache first
        chunk_cache_path = self._get_cache_path(chunk.file)
        if chunk_cache_path.exists():
            data = chunk_cache_path.read_bytes()
            if self._verify_chunk(data, chunk):
                return data
        
        # Fetch from HTTP
        chunk_url = f"{self.base_url}/{chunk.file}"
        response = requests.get(chunk_url)
        response.raise_for_status()
        
        data = response.content
        
        # Verify checksum
        if self.verify_checksums and not self._verify_chunk(data, chunk):
            raise ValueError(f"Checksum mismatch for chunk {chunk.name}")
        
        # Cache chunk
        chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_cache_path.write_bytes(data)
        
        return data
    
    def get_chunk_by_name(self, name: str) -> bytes:
        """
        Fetch a chunk by name.
        
        Args:
            name: Name of the chunk (e.g., "layer.0.weight")
            
        Returns:
            Chunk data as bytes
        """
        if not self.chunks:
            self.load_manifest()
        
        for i, chunk in enumerate(self.chunks):
            if chunk.name == name:
                return self.get_chunk(i)
        
        raise KeyError(f"Chunk not found: {name}")
    
    def stream_all_chunks(self):
        """
        Generator that yields all chunks in order.
        
        Yields:
            Tuple of (chunk_metadata, chunk_data)
        """
        if not self.chunks:
            self.load_manifest()
        
        for i, chunk in enumerate(self.chunks):
            data = self.get_chunk(i)
            yield chunk, data
    
    def prefetch_chunks(self, chunk_indices: List[int]):
        """
        Prefetch multiple chunks in parallel.
        
        Args:
            chunk_indices: List of chunk indices to prefetch
        """
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.get_chunk, idx)
                for idx in chunk_indices
            ]
            concurrent.futures.wait(futures)
    
    def _verify_chunk(self, data: bytes, chunk: ChunkMetadata) -> bool:
        """Verify chunk checksum."""
        if not self.verify_checksums:
            return True
        
        computed_hash = hashlib.sha256(data).hexdigest()
        return computed_hash == chunk.sha256
    
    def _get_cache_path(self, relative_path: str) -> Path:
        """Get cache path for a relative path."""
        # Create a unique cache directory for this artifact
        artifact_id = hashlib.sha256(self.base_url.encode()).hexdigest()[:16]
        return self.cache_dir / artifact_id / relative_path
    
    def clear_cache(self):
        """Clear cached artifacts for this base URL."""
        artifact_id = hashlib.sha256(self.base_url.encode()).hexdigest()[:16]
        cache_path = self.cache_dir / artifact_id
        
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)


class ArtifactServer:
    """Simple HTTP server for serving TenPak artifacts."""
    
    @staticmethod
    def serve(
        artifact_dir: str,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """
        Serve an artifact directory via HTTP.
        
        Args:
            artifact_dir: Path to artifact directory
            host: Host to bind to
            port: Port to bind to
        """
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import os
        
        # Change to artifact directory
        os.chdir(artifact_dir)
        
        server = HTTPServer((host, port), SimpleHTTPRequestHandler)
        print(f"Serving artifact at http://{host}:{port}")
        print(f"Artifact directory: {artifact_dir}")
        print("Press Ctrl+C to stop")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()


def download_artifact(
    url: str,
    output_dir: str,
    verify_checksums: bool = True,
    show_progress: bool = True,
) -> str:
    """
    Download a complete artifact from HTTP to local directory.
    
    Args:
        url: Base URL of the artifact
        output_dir: Output directory
        verify_checksums: Whether to verify checksums
        show_progress: Whether to show progress bar
        
    Returns:
        Path to downloaded artifact directory
    """
    streamer = HTTPArtifactStreamer(url, verify_checksums=verify_checksums)
    manifest = streamer.load_manifest()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Download all chunks
    if show_progress:
        try:
            from tqdm import tqdm
            chunks_iter = tqdm(streamer.chunks, desc="Downloading chunks")
        except ImportError:
            chunks_iter = streamer.chunks
    else:
        chunks_iter = streamer.chunks
    
    for i, chunk in enumerate(chunks_iter):
        data = streamer.get_chunk(i)
        
        chunk_path = output_path / chunk.file
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_bytes(data)
    
    return str(output_path)
