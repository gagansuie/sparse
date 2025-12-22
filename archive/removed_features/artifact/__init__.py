"""
TenPak Artifact - Streamable Model Format (.tnpk)

Open, efficient format for distributing compressed LLMs.

Features:
- Chunked storage for partial/streaming downloads
- Content addressing (SHA256) for deduplication
- Optional signing for enterprise verification
- Direct inference integration

Usage:
    from artifact import TenPakArtifact, create_artifact, load_artifact
    
    # Create artifact from model
    artifact = create_artifact(
        model_id="mistralai/Mistral-7B-v0.1",
        output_path="./model.tnpk",
        codec="int4_awq"
    )
    
    # Load and stream
    artifact = load_artifact("./model.tnpk")
    for chunk in artifact.stream_chunks():
        process(chunk)
"""

from .format import (
    TenPakArtifact,
    ArtifactManifest,
    ChunkInfo,
    create_artifact,
    load_artifact,
)

from .streaming import (
    stream_chunks,
    fetch_chunk,
    ChunkIterator,
)

from .signing import (
    sign_artifact,
    verify_signature,
    SignatureInfo,
)

__all__ = [
    "TenPakArtifact",
    "ArtifactManifest", 
    "ChunkInfo",
    "create_artifact",
    "load_artifact",
    "stream_chunks",
    "fetch_chunk",
    "ChunkIterator",
    "sign_artifact",
    "verify_signature",
    "SignatureInfo",
]
