"""
Tests for HTTP streaming functionality
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import hashlib

from artifact.http_streaming import (
    HTTPArtifactStreamer,
    ChunkMetadata,
    ArtifactServer,
    download_artifact,
)


class TestChunkMetadata:
    """Test ChunkMetadata dataclass."""
    
    def test_chunk_metadata_creation(self):
        """Test chunk metadata creation."""
        chunk = ChunkMetadata(
            name="layer.0.weight",
            sha256="abc123",
            offset=0,
            size=1024,
            file="chunks/0000.bin",
        )
        
        assert chunk.name == "layer.0.weight"
        assert chunk.sha256 == "abc123"
        assert chunk.size == 1024


class TestHTTPArtifactStreamer:
    """Test HTTPArtifactStreamer functionality."""
    
    @pytest.fixture
    def mock_manifest(self):
        """Create a mock manifest."""
        return {
            "version": "1.0",
            "model_id": "gpt2",
            "chunks": [
                {
                    "name": "layer.0.weight",
                    "sha256": "abc123",
                    "offset": 0,
                    "size": 1024,
                    "file": "chunks/0000.bin",
                },
                {
                    "name": "layer.1.weight",
                    "sha256": "def456",
                    "offset": 1024,
                    "size": 2048,
                    "file": "chunks/0001.bin",
                },
            ],
        }
    
    @pytest.fixture
    def streamer(self, tmp_path):
        """Create a streamer instance."""
        return HTTPArtifactStreamer(
            base_url="https://cdn.example.com/artifacts/model-123",
            cache_dir=str(tmp_path / "cache"),
            verify_checksums=True,
        )
    
    def test_init(self, streamer):
        """Test streamer initialization."""
        assert streamer.base_url == "https://cdn.example.com/artifacts/model-123"
        assert streamer.verify_checksums is True
        assert streamer.manifest is None
    
    @patch("artifact.http_streaming.requests.get")
    def test_load_manifest(self, mock_get, streamer, mock_manifest):
        """Test manifest loading."""
        mock_response = Mock()
        mock_response.json.return_value = mock_manifest
        mock_get.return_value = mock_response
        
        manifest = streamer.load_manifest()
        
        assert manifest == mock_manifest
        assert len(streamer.chunks) == 2
        mock_get.assert_called_once_with(
            "https://cdn.example.com/artifacts/model-123/manifest.json"
        )
    
    @patch("artifact.http_streaming.requests.get")
    def test_load_manifest_caching(self, mock_get, streamer, mock_manifest, tmp_path):
        """Test that manifest is cached."""
        # First load
        mock_response = Mock()
        mock_response.json.return_value = mock_manifest
        mock_get.return_value = mock_response
        
        manifest1 = streamer.load_manifest()
        
        # Second load should not call HTTP
        manifest2 = streamer.load_manifest()
        
        assert manifest1 == manifest2
        assert mock_get.call_count == 1
    
    @patch("artifact.http_streaming.requests.get")
    def test_get_chunk(self, mock_get, streamer, mock_manifest):
        """Test chunk fetching."""
        # Load manifest first
        mock_manifest_response = Mock()
        mock_manifest_response.json.return_value = mock_manifest
        
        # Mock chunk data
        chunk_data = b"fake_chunk_data"
        chunk_sha = hashlib.sha256(chunk_data).hexdigest()
        
        # Update manifest with correct hash
        mock_manifest["chunks"][0]["sha256"] = chunk_sha
        
        mock_chunk_response = Mock()
        mock_chunk_response.content = chunk_data
        
        def mock_get_fn(url):
            if "manifest.json" in url:
                return mock_manifest_response
            else:
                return mock_chunk_response
        
        mock_get.side_effect = mock_get_fn
        
        data = streamer.get_chunk(0)
        
        assert data == chunk_data
    
    @patch("artifact.http_streaming.requests.get")
    def test_get_chunk_by_name(self, mock_get, streamer, mock_manifest):
        """Test fetching chunk by name."""
        mock_manifest_response = Mock()
        mock_manifest_response.json.return_value = mock_manifest
        
        chunk_data = b"fake_chunk_data"
        chunk_sha = hashlib.sha256(chunk_data).hexdigest()
        mock_manifest["chunks"][0]["sha256"] = chunk_sha
        
        mock_chunk_response = Mock()
        mock_chunk_response.content = chunk_data
        
        def mock_get_fn(url):
            if "manifest.json" in url:
                return mock_manifest_response
            else:
                return mock_chunk_response
        
        mock_get.side_effect = mock_get_fn
        
        data = streamer.get_chunk_by_name("layer.0.weight")
        
        assert data == chunk_data
    
    def test_get_chunk_invalid_index(self, streamer, mock_manifest):
        """Test that invalid chunk index raises error."""
        with patch("artifact.http_streaming.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_manifest
            mock_get.return_value = mock_response
            
            streamer.load_manifest()
            
            with pytest.raises(IndexError, match="out of range"):
                streamer.get_chunk(999)
    
    def test_get_chunk_by_name_not_found(self, streamer, mock_manifest):
        """Test that missing chunk name raises error."""
        with patch("artifact.http_streaming.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = mock_manifest
            mock_get.return_value = mock_response
            
            streamer.load_manifest()
            
            with pytest.raises(KeyError, match="Chunk not found"):
                streamer.get_chunk_by_name("nonexistent.weight")
    
    @patch("artifact.http_streaming.requests.get")
    def test_stream_all_chunks(self, mock_get, streamer, mock_manifest):
        """Test streaming all chunks."""
        mock_manifest_response = Mock()
        mock_manifest_response.json.return_value = mock_manifest
        
        chunk_data_0 = b"chunk_0_data"
        chunk_data_1 = b"chunk_1_data"
        
        # Update manifest with correct hashes
        mock_manifest["chunks"][0]["sha256"] = hashlib.sha256(chunk_data_0).hexdigest()
        mock_manifest["chunks"][1]["sha256"] = hashlib.sha256(chunk_data_1).hexdigest()
        
        def mock_get_fn(url):
            if "manifest.json" in url:
                return mock_manifest_response
            elif "0000.bin" in url:
                resp = Mock()
                resp.content = chunk_data_0
                return resp
            elif "0001.bin" in url:
                resp = Mock()
                resp.content = chunk_data_1
                return resp
        
        mock_get.side_effect = mock_get_fn
        
        chunks = list(streamer.stream_all_chunks())
        
        assert len(chunks) == 2
        assert chunks[0][1] == chunk_data_0
        assert chunks[1][1] == chunk_data_1
    
    def test_verify_chunk_success(self, streamer):
        """Test successful chunk verification."""
        data = b"test_data"
        chunk = ChunkMetadata(
            name="test",
            sha256=hashlib.sha256(data).hexdigest(),
            offset=0,
            size=len(data),
            file="test.bin",
        )
        
        assert streamer._verify_chunk(data, chunk) is True
    
    def test_verify_chunk_failure(self, streamer):
        """Test failed chunk verification."""
        data = b"test_data"
        chunk = ChunkMetadata(
            name="test",
            sha256="wrong_hash",
            offset=0,
            size=len(data),
            file="test.bin",
        )
        
        assert streamer._verify_chunk(data, chunk) is False


class TestDownloadArtifact:
    """Test download_artifact function."""
    
    @patch("artifact.http_streaming.HTTPArtifactStreamer")
    def test_download_artifact(self, mock_streamer_class, tmp_path):
        """Test artifact download."""
        mock_streamer = Mock()
        mock_manifest = {
            "version": "1.0",
            "model_id": "gpt2",
            "chunks": [
                {
                    "name": "layer.0",
                    "sha256": "abc123",
                    "offset": 0,
                    "size": 1024,
                    "file": "chunks/0000.bin",
                },
            ],
        }
        
        mock_streamer.load_manifest.return_value = mock_manifest
        mock_streamer.chunks = [
            ChunkMetadata(
                name="layer.0",
                sha256="abc123",
                offset=0,
                size=1024,
                file="chunks/0000.bin",
            )
        ]
        mock_streamer.get_chunk.return_value = b"fake_data"
        
        mock_streamer_class.return_value = mock_streamer
        
        output_dir = tmp_path / "output"
        result = download_artifact(
            url="https://cdn.example.com/artifacts/model-123",
            output_dir=str(output_dir),
            verify_checksums=True,
            show_progress=False,
        )
        
        assert Path(result).exists()
        assert (Path(result) / "manifest.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
