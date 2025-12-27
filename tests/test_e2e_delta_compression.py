#!/usr/bin/env python3
"""
End-to-End Delta Compression Validation

This test validates that our delta compression implementation:
1. Actually compresses deltas (not just estimates)
2. Can reconstruct models from base + compressed delta
3. Preserves model quality (inference outputs match)

Uses GPT-2 for fast local testing.
"""

import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


def compress_delta_int8(delta: torch.Tensor) -> Tuple[bytes, float, tuple]:
    """
    Actually compress a delta tensor to int8.
    
    Returns:
        Tuple of (compressed_bytes, scale, original_shape)
    """
    original_shape = delta.shape
    flat_delta = delta.flatten().float()
    
    # Find scale (max abs value)
    max_abs = torch.abs(flat_delta).max().item()
    if max_abs < 1e-10:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    
    # Quantize to int8
    quantized = (flat_delta / scale).round().clamp(-127, 127).to(torch.int8)
    
    # Convert to bytes
    compressed_bytes = quantized.numpy().tobytes()
    
    return compressed_bytes, scale, original_shape


def decompress_delta_int8(compressed_bytes: bytes, scale: float, shape: tuple, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Decompress int8 bytes back to delta tensor.
    """
    # Convert bytes back to int8 tensor
    quantized = torch.from_numpy(np.frombuffer(compressed_bytes, dtype=np.int8))
    
    # Dequantize
    delta = (quantized.float() * scale).reshape(shape).to(dtype)
    
    return delta


def test_compression_roundtrip():
    """Test that compression/decompression preserves values."""
    print("\n" + "="*60)
    print("TEST 1: Compression Roundtrip")
    print("="*60)
    
    # Create random delta tensor
    original = torch.randn(1000, 1000, dtype=torch.float16)
    
    # Compress
    compressed, scale, shape = compress_delta_int8(original)
    
    # Decompress
    reconstructed = decompress_delta_int8(compressed, scale, shape, torch.float16)
    
    # Calculate error
    error = torch.abs(original - reconstructed)
    max_error = error.max().item()
    mean_error = error.mean().item()
    relative_error = mean_error / torch.abs(original).mean().item()
    
    # Calculate actual compression
    original_bytes = original.numel() * 2  # fp16
    compressed_bytes = len(compressed) + 4  # int8 + scale
    compression_ratio = original_bytes / compressed_bytes
    
    print(f"  Original size: {original_bytes / 1024:.2f} KB")
    print(f"  Compressed size: {compressed_bytes / 1024:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Relative error: {relative_error*100:.3f}%")
    
    assert compression_ratio > 1.9, f"Compression ratio too low: {compression_ratio}"
    assert relative_error < 0.01, f"Relative error too high: {relative_error}"
    
    print("  ✅ PASSED: Compression roundtrip works correctly")
    return True


def test_model_reconstruction():
    """Test that we can reconstruct a model from base + compressed delta."""
    print("\n" + "="*60)
    print("TEST 2: Model Reconstruction")
    print("="*60)
    
    # Load base model
    print("  Loading GPT-2 base model...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    
    # Create a "fine-tuned" version by adding small deltas
    print("  Creating simulated fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    
    # Add small random deltas to simulate fine-tuning
    with torch.no_grad():
        for name, param in finetuned_model.named_parameters():
            if 'weight' in name:
                delta = torch.randn_like(param) * 0.01  # Small perturbation
                param.add_(delta)
    
    # Now test delta compression
    print("  Computing and compressing deltas...")
    compressed_deltas = {}
    total_original = 0
    total_compressed = 0
    
    base_params = dict(base_model.named_parameters())
    ft_params = dict(finetuned_model.named_parameters())
    
    for name in base_params:
        if 'weight' in name:
            base_weight = base_params[name].data
            ft_weight = ft_params[name].data
            
            # Compute delta
            delta = ft_weight - base_weight
            
            # Compress
            compressed, scale, shape = compress_delta_int8(delta)
            compressed_deltas[name] = (compressed, scale, shape)
            
            # Track sizes
            total_original += delta.numel() * 4  # float32
            total_compressed += len(compressed) + 4
    
    compression_ratio = total_original / total_compressed
    print(f"  Total original: {total_original / 1024 / 1024:.2f} MB")
    print(f"  Total compressed: {total_compressed / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Reconstruct model from base + compressed deltas
    print("  Reconstructing model from base + compressed deltas...")
    reconstructed_model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    
    with torch.no_grad():
        for name, param in reconstructed_model.named_parameters():
            if name in compressed_deltas:
                compressed, scale, shape = compressed_deltas[name]
                delta = decompress_delta_int8(compressed, scale, shape, torch.float32)
                param.add_(delta)
    
    # Compare weights
    print("  Comparing reconstructed vs original fine-tuned model...")
    max_diff = 0
    mean_diff = 0
    count = 0
    
    recon_params = dict(reconstructed_model.named_parameters())
    for name in ft_params:
        if 'weight' in name:
            ft_weight = ft_params[name].data
            recon_weight = recon_params[name].data
            diff = torch.abs(ft_weight - recon_weight)
            max_diff = max(max_diff, diff.max().item())
            mean_diff += diff.mean().item()
            count += 1
    
    mean_diff /= count
    print(f"  Max weight difference: {max_diff:.6f}")
    print(f"  Mean weight difference: {mean_diff:.6f}")
    
    assert max_diff < 0.1, f"Max weight difference too high: {max_diff}"
    
    print("  ✅ PASSED: Model reconstruction works correctly")
    return base_model, finetuned_model, reconstructed_model, compression_ratio


def test_inference_equivalence(base_model, finetuned_model, reconstructed_model):
    """Test that inference outputs match between original and reconstructed."""
    print("\n" + "="*60)
    print("TEST 3: Inference Equivalence")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    test_prompts = [
        "The quick brown fox",
        "In the beginning",
        "Machine learning is",
    ]
    
    base_model.eval()
    finetuned_model.eval()
    reconstructed_model.eval()
    
    print("  Comparing outputs for test prompts...")
    
    all_match = True
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # Get logits from all three models
            base_logits = base_model(**inputs).logits
            ft_logits = finetuned_model(**inputs).logits
            recon_logits = reconstructed_model(**inputs).logits
        
        # Compare fine-tuned vs reconstructed
        diff = torch.abs(ft_logits - recon_logits)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Get predicted tokens
        ft_pred = ft_logits.argmax(dim=-1)
        recon_pred = recon_logits.argmax(dim=-1)
        tokens_match = torch.all(ft_pred == recon_pred).item()
        
        print(f"  Prompt: '{prompt}'")
        print(f"    Max logit diff: {max_diff:.4f}")
        print(f"    Mean logit diff: {mean_diff:.4f}")
        print(f"    Predictions match: {'✅' if tokens_match else '❌'}")
        
        if not tokens_match:
            all_match = False
    
    # Also test generation
    print("\n  Testing text generation...")
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        ft_output = finetuned_model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        recon_output = reconstructed_model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    ft_text = tokenizer.decode(ft_output[0], skip_special_tokens=True)
    recon_text = tokenizer.decode(recon_output[0], skip_special_tokens=True)
    
    print(f"  Fine-tuned output: '{ft_text}'")
    print(f"  Reconstructed output: '{recon_text}'")
    print(f"  Outputs match: {'✅' if ft_text == recon_text else '❌'}")
    
    if ft_text == recon_text:
        print("  ✅ PASSED: Inference equivalence verified")
    else:
        print("  ⚠️ WARNING: Outputs differ slightly (expected due to quantization)")
    
    return all_match or (ft_text == recon_text)


def test_storage_savings():
    """Test actual storage savings with file I/O."""
    print("\n" + "="*60)
    print("TEST 4: Actual Storage Savings")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test tensor (simulating a layer)
        print("  Creating test layer (100M parameters)...")
        layer = torch.randn(10000, 10000, dtype=torch.float16)
        
        # Save as fp16
        fp16_path = tmpdir / "layer_fp16.pt"
        torch.save(layer, fp16_path)
        fp16_size = os.path.getsize(fp16_path)
        
        # Compress and save as int8
        compressed, scale, shape = compress_delta_int8(layer)
        int8_path = tmpdir / "layer_int8.bin"
        with open(int8_path, 'wb') as f:
            # Write scale (4 bytes) + shape info + compressed data
            f.write(np.array([scale], dtype=np.float32).tobytes())
            f.write(np.array(shape, dtype=np.int32).tobytes())
            f.write(compressed)
        int8_size = os.path.getsize(int8_path)
        
        compression_ratio = fp16_size / int8_size
        savings_pct = (1 - int8_size / fp16_size) * 100
        
        print(f"  FP16 file size: {fp16_size / 1024 / 1024:.2f} MB")
        print(f"  INT8 file size: {int8_size / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Storage savings: {savings_pct:.1f}%")
        
        # Verify we can load it back
        with open(int8_path, 'rb') as f:
            loaded_scale = np.frombuffer(f.read(4), dtype=np.float32)[0]
            loaded_shape = tuple(np.frombuffer(f.read(8), dtype=np.int32))
            loaded_compressed = f.read()
        
        reconstructed = decompress_delta_int8(loaded_compressed, loaded_scale, loaded_shape, torch.float16)
        
        error = torch.abs(layer - reconstructed).mean().item()
        print(f"  Reconstruction error: {error:.6f}")
        
        assert compression_ratio > 1.9, f"Compression ratio too low: {compression_ratio}"
        assert savings_pct > 45, f"Savings too low: {savings_pct}"
        
        print("  ✅ PASSED: Actual storage savings verified")
        return compression_ratio, savings_pct


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("END-TO-END DELTA COMPRESSION VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test 1: Compression roundtrip
    try:
        results['roundtrip'] = test_compression_roundtrip()
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results['roundtrip'] = False
    
    # Test 2: Model reconstruction
    try:
        base, ft, recon, ratio = test_model_reconstruction()
        results['reconstruction'] = True
        results['compression_ratio'] = ratio
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results['reconstruction'] = False
        base, ft, recon = None, None, None
    
    # Test 3: Inference equivalence
    if base is not None:
        try:
            results['inference'] = test_inference_equivalence(base, ft, recon)
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            results['inference'] = False
    
    # Test 4: Storage savings
    try:
        ratio, savings = test_storage_savings()
        results['storage'] = True
        results['storage_ratio'] = ratio
        results['storage_savings'] = savings
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        results['storage'] = False
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all([
        results.get('roundtrip', False),
        results.get('reconstruction', False),
        results.get('inference', False),
        results.get('storage', False),
    ])
    
    print(f"  Compression Roundtrip: {'✅' if results.get('roundtrip') else '❌'}")
    print(f"  Model Reconstruction: {'✅' if results.get('reconstruction') else '❌'}")
    print(f"  Inference Equivalence: {'✅' if results.get('inference') else '❌'}")
    print(f"  Storage Savings: {'✅' if results.get('storage') else '❌'}")
    
    if 'compression_ratio' in results:
        print(f"\n  Achieved compression: {results['compression_ratio']:.2f}x")
    if 'storage_savings' in results:
        print(f"  Storage savings: {results['storage_savings']:.1f}%")
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Implementation is validated!")
    else:
        print("❌ SOME TESTS FAILED - Review results above")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_all_tests()
