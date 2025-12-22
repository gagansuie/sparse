"""
TenPak Artifact - Signing and Verification

Provides cryptographic signing for enterprise verification.
Supports GPG and basic HMAC signing.
"""

import os
import json
import hashlib
import hmac
import base64
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class SignatureInfo:
    """Information about an artifact signature."""
    algorithm: str  # "sha256-hmac", "gpg", "sigstore"
    signature: str  # Base64-encoded signature
    signer: str  # Signer identity
    signed_at: str  # ISO timestamp
    manifest_hash: str  # Hash of the manifest that was signed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "signature": self.signature,
            "signer": self.signer,
            "signed_at": self.signed_at,
            "manifest_hash": self.manifest_hash,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignatureInfo":
        return cls(**d)


def compute_manifest_hash(manifest_path: str) -> str:
    """Compute SHA256 hash of manifest file."""
    with open(manifest_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sign_artifact(
    artifact_path: str,
    signer: str,
    secret_key: Optional[str] = None,
    algorithm: str = "sha256-hmac",
) -> SignatureInfo:
    """Sign a .tnpk artifact.
    
    Args:
        artifact_path: Path to .tnpk artifact
        signer: Signer identity (e.g., email or org name)
        secret_key: Secret key for HMAC signing (required for sha256-hmac)
        algorithm: Signing algorithm ("sha256-hmac" or "gpg")
        
    Returns:
        SignatureInfo with signature details
    """
    artifact_path = Path(artifact_path)
    manifest_path = artifact_path / "manifest.json"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {artifact_path}")
    
    # Compute manifest hash
    manifest_hash = compute_manifest_hash(str(manifest_path))
    
    if algorithm == "sha256-hmac":
        if not secret_key:
            raise ValueError("secret_key required for sha256-hmac signing")
        
        # Create HMAC signature
        signature_bytes = hmac.new(
            secret_key.encode(),
            manifest_hash.encode(),
            hashlib.sha256
        ).digest()
        signature = base64.b64encode(signature_bytes).decode()
        
    elif algorithm == "gpg":
        # GPG signing (requires gpg to be installed)
        try:
            import subprocess
            
            result = subprocess.run(
                ["gpg", "--detach-sign", "--armor", "-o", "-", str(manifest_path)],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"GPG signing failed: {result.stderr}")
            
            signature = base64.b64encode(result.stdout.encode()).decode()
            
        except FileNotFoundError:
            raise RuntimeError("GPG not installed. Use sha256-hmac instead.")
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create signature info
    sig_info = SignatureInfo(
        algorithm=algorithm,
        signature=signature,
        signer=signer,
        signed_at=datetime.utcnow().isoformat(),
        manifest_hash=manifest_hash,
    )
    
    # Save signature file
    sig_path = artifact_path / "signature.json"
    with open(sig_path, "w") as f:
        json.dump(sig_info.to_dict(), f, indent=2)
    
    # Also update manifest with signature info
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    manifest["signature"] = signature
    manifest["signer"] = signer
    manifest["signed_at"] = sig_info.signed_at
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return sig_info


def verify_signature(
    artifact_path: str,
    secret_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a .tnpk artifact signature.
    
    Args:
        artifact_path: Path to .tnpk artifact
        secret_key: Secret key for HMAC verification (required for sha256-hmac)
        
    Returns:
        Dict with verification results
    """
    artifact_path = Path(artifact_path)
    manifest_path = artifact_path / "manifest.json"
    sig_path = artifact_path / "signature.json"
    
    if not sig_path.exists():
        return {
            "verified": False,
            "error": "No signature found",
            "signed": False,
        }
    
    # Load signature info
    with open(sig_path, "r") as f:
        sig_info = SignatureInfo.from_dict(json.load(f))
    
    # Compute current manifest hash
    current_hash = compute_manifest_hash(str(manifest_path))
    
    # Check if manifest has been modified
    if current_hash != sig_info.manifest_hash:
        return {
            "verified": False,
            "error": "Manifest has been modified since signing",
            "signed": True,
            "signer": sig_info.signer,
            "signed_at": sig_info.signed_at,
        }
    
    # Verify signature based on algorithm
    if sig_info.algorithm == "sha256-hmac":
        if not secret_key:
            return {
                "verified": False,
                "error": "secret_key required for HMAC verification",
                "signed": True,
                "signer": sig_info.signer,
            }
        
        # Recompute HMAC
        expected_bytes = hmac.new(
            secret_key.encode(),
            sig_info.manifest_hash.encode(),
            hashlib.sha256
        ).digest()
        expected = base64.b64encode(expected_bytes).decode()
        
        if hmac.compare_digest(sig_info.signature, expected):
            return {
                "verified": True,
                "signed": True,
                "signer": sig_info.signer,
                "signed_at": sig_info.signed_at,
                "algorithm": sig_info.algorithm,
            }
        else:
            return {
                "verified": False,
                "error": "Signature verification failed",
                "signed": True,
                "signer": sig_info.signer,
            }
    
    elif sig_info.algorithm == "gpg":
        # GPG verification would go here
        return {
            "verified": False,
            "error": "GPG verification not yet implemented",
            "signed": True,
            "signer": sig_info.signer,
        }
    
    else:
        return {
            "verified": False,
            "error": f"Unknown algorithm: {sig_info.algorithm}",
            "signed": True,
        }


def get_signature_info(artifact_path: str) -> Optional[SignatureInfo]:
    """Get signature info for an artifact without verification."""
    artifact_path = Path(artifact_path)
    sig_path = artifact_path / "signature.json"
    
    if not sig_path.exists():
        return None
    
    with open(sig_path, "r") as f:
        return SignatureInfo.from_dict(json.load(f))
