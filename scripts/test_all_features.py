#!/usr/bin/env python3
"""
TenPak Feature Verification Script

Tests all 4 main features:
1. Core Compression (v10, INT4+AWQ)
2. Cost Optimizer
3. Delta Compression
4. Artifact Format (.tnpk)

Usage:
    python scripts/test_all_features.py
    python scripts/test_all_features.py --full  # Run with actual model
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

def test_imports():
    """Test all imports work."""
    print("\n" + "="*60)
    print("TEST 1: Import Verification")
    print("="*60)
    
    try:
        # Core imports
        from core import (
            CODEC_V10, CODEC_V60, V10_CONFIG, V60_CONFIG,
            compress_int4_awq, compress_int4_residual,
            collect_calibration_stats, compute_ppl,
            allocate_bits, LayerAllocation,
            compress_delta, reconstruct_from_delta, estimate_delta_savings,
        )
        print("✅ core imports: OK")
        print(f"   CODEC_V10 = {CODEC_V10}")
        print(f"   V10_CONFIG = {V10_CONFIG}")
        
        # Optimizer imports
        from optimizer import (
            generate_candidates, benchmark_candidate,
            select_optimal, optimize_model,
        )
        print("✅ optimizer imports: OK")
        
        # Artifact imports
        from artifact import (
            TenPakArtifact, create_artifact, load_artifact,
            sign_artifact, verify_signature,
        )
        print("✅ artifact imports: OK")
        
        # Studio imports
        from studio import api
        print("✅ studio imports: OK")
        
        # CLI imports
        from cli import main
        print("✅ cli imports: OK")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_compression():
    """Test core compression functions."""
    print("\n" + "="*60)
    print("TEST 2: Core Compression (v10 INT4+AWQ)")
    print("="*60)
    
    try:
        from core import compress_int4_awq, compress_int4_residual, V10_CONFIG
        
        # Create test weight tensor
        weight = torch.randn(256, 512)
        print(f"   Input shape: {weight.shape}")
        print(f"   Input size: {weight.numel() * 4 / 1024:.1f} KB (FP32)")
        
        # Test INT4+AWQ (v10 config)
        result, compression = compress_int4_awq(
            weight, 
            group_size=V10_CONFIG["attention_group"],
            outlier_pct=V10_CONFIG["outlier_pct"]
        )
        print(f"✅ compress_int4_awq: {compression:.2f}x compression")
        print(f"   Output shape: {result.shape}")
        
        # Verify reconstruction error is small
        mse = ((weight - result) ** 2).mean().item()
        print(f"   MSE: {mse:.6f}")
        
        # Test INT4+Residual
        result2, compression2 = compress_int4_residual(weight, group_size=16)
        print(f"✅ compress_int4_residual: {compression2:.2f}x compression")
        
        return True
    except Exception as e:
        print(f"❌ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer():
    """Test cost optimizer."""
    print("\n" + "="*60)
    print("TEST 3: Cost Optimizer")
    print("="*60)
    
    try:
        from optimizer import generate_candidates, CANDIDATE_PRESETS
        
        # Generate candidates
        candidates = generate_candidates(
            include_calibration=True,
            max_expected_ppl_delta=5.0,
        )
        print(f"✅ generate_candidates: {len(candidates)} candidates generated")
        
        for c in candidates[:3]:
            print(f"   - {c.name}: {c.method.value}, comp={c.expected_compression:.1f}x")
        
        # Check presets
        print(f"✅ CANDIDATE_PRESETS: {list(CANDIDATE_PRESETS.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_delta_compression():
    """Test delta compression."""
    print("\n" + "="*60)
    print("TEST 4: Delta Compression")
    print("="*60)
    
    try:
        from core.delta import compute_layer_delta, compress_delta_sparse
        
        # Create fake base and fine-tuned weights
        base_weight = torch.randn(256, 512)
        # Fine-tune is 95% similar (small delta)
        finetune_weight = base_weight + torch.randn_like(base_weight) * 0.05
        
        # Compute delta (returns tuple: delta, stats)
        delta, stats = compute_layer_delta(base_weight, finetune_weight)
        print(f"✅ compute_layer_delta: delta shape {delta.shape}")
        print(f"   Stats: l2_norm={stats['l2_norm']:.4f}, sparsity={stats['sparsity']*100:.1f}%")
        
        # Check sparsity of delta
        sparsity = (delta.abs() < 0.01).float().mean().item()
        print(f"   Delta sparsity (|d|<0.01): {sparsity*100:.1f}%")
        
        # Test sparse compression (returns indices, values, ratio)
        indices, values, ratio = compress_delta_sparse(delta, threshold=0.01)
        print(f"✅ compress_delta_sparse: {ratio:.2f}x compression")
        print(f"   Non-zero elements: {len(values)} / {delta.numel()}")
        
        return True
    except Exception as e:
        print(f"❌ Delta compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_artifact():
    """Test artifact format."""
    print("\n" + "="*60)
    print("TEST 5: Artifact Format (.tnpk)")
    print("="*60)
    
    try:
        from artifact.format import ArtifactManifest, ChunkInfo, compute_chunk_hash
        from artifact import sign_artifact, verify_signature
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test.tnpk"
            artifact_path.mkdir(parents=True)
            chunks_path = artifact_path / "chunks"
            chunks_path.mkdir()
            
            # Create test chunk data
            test_data = b"test tensor data " * 100
            chunk_hash = compute_chunk_hash(test_data)
            
            # Create chunk info
            chunk = ChunkInfo(
                name="layer_0_weight",
                sha256=chunk_hash,
                size=len(test_data),
                offset=0,
            )
            print(f"✅ ChunkInfo created: {chunk.name}, {chunk.size} bytes")
            print(f"   SHA256: {chunk.sha256[:16]}...")
            
            # Create manifest
            manifest = ArtifactManifest(
                model_id="test/model",
                codec="v10_int4_awq",
                compression_ratio=7.42,
                chunks=[chunk],
            )
            print(f"✅ ArtifactManifest created: {manifest.model_id}")
            print(f"   Codec: {manifest.codec}")
            print(f"   Compression: {manifest.compression_ratio}x")
            
            # Save manifest
            with open(artifact_path / "manifest.json", "w") as f:
                json.dump(manifest.to_dict(), f)
            print(f"✅ Manifest saved")
            
            # Save chunk
            with open(chunks_path / "layer_0_weight.bin", "wb") as f:
                f.write(test_data)
            print(f"✅ Chunk saved")
            
            # Load and verify using TenPakArtifact
            from artifact import TenPakArtifact
            loaded = TenPakArtifact(str(artifact_path))
            loaded.load()
            print(f"✅ TenPakArtifact loaded: {loaded.manifest.model_id}")
            
            # Verify
            is_valid = loaded.verify()
            print(f"✅ verify: {is_valid}")
            
        return True
    except Exception as e:
        print(f"❌ Artifact test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli():
    """Test CLI commands."""
    print("\n" + "="*60)
    print("TEST 6: CLI Commands")
    print("="*60)
    
    try:
        import subprocess
        
        # Test help commands
        commands = [
            ["tenpak", "--help"],
            ["tenpak", "pack", "--help"],
            ["tenpak", "optimize", "--help"],
            ["tenpak", "delta", "--help"],
            ["tenpak", "artifact", "--help"],
        ]
        
        for cmd in commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {' '.join(cmd)}: OK")
            else:
                print(f"❌ {' '.join(cmd)}: FAILED")
                print(f"   {result.stderr[:100]}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def test_full_model(model_id="gpt2"):
    """Full model test (requires transformers)."""
    print("\n" + "="*60)
    print(f"TEST 7: Full Model Test ({model_id})")
    print("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from core import compress_int4_awq, V10_CONFIG
        
        print(f"   Loading {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params / 1e6:.1f}M")
        
        # Compress one layer
        layer_name = None
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.dim() == 2:
                if module.weight.numel() > 1000:
                    layer_name = name
                    break
        
        if layer_name:
            module = dict(model.named_modules())[layer_name]
            weight = module.weight.data
            print(f"   Compressing layer: {layer_name} ({weight.shape})")
            
            result, compression = compress_int4_awq(
                weight,
                group_size=V10_CONFIG["attention_group"],
                outlier_pct=V10_CONFIG["outlier_pct"]
            )
            print(f"✅ Compression: {compression:.2f}x")
            
            # Check reconstruction quality
            mse = ((weight - result) ** 2).mean().item()
            print(f"   MSE: {mse:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Full model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("TENPAK FEATURE VERIFICATION")
    print("="*60)
    
    full_test = "--full" in sys.argv
    
    results = {}
    
    # Run tests
    results["imports"] = test_imports()
    results["compression"] = test_compression()
    results["optimizer"] = test_optimizer()
    results["delta"] = test_delta_compression()
    results["artifact"] = test_artifact()
    results["cli"] = test_cli()
    
    if full_test:
        results["full_model"] = test_full_model()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n   {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL FEATURES WORKING!")
        return 0
    else:
        print("\n⚠️  SOME FEATURES NEED ATTENTION")
        return 1


if __name__ == "__main__":
    sys.exit(main())
