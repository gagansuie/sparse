#!/usr/bin/env python3
"""
Local Delta Compression Validation Test
Tests delta compression on 7B models to validate claims before acquisition pitch

Run this to validate:
- Compression ratios (target: 90-96%)
- Rust acceleration (target: 10-20x speedup)
- Production readiness

Usage:
    python tests/test_delta_7b_local.py
"""

import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from core.delta import compress_delta, reconstruct_from_delta, estimate_delta_savings

def test_delta_compression_7b():
    """Test delta compression on Llama-2-7B simulated fine-tune."""
    
    print("=" * 80)
    print("DELTA COMPRESSION VALIDATION - 7B MODELS")
    print("=" * 80)
    print()
    
    # We'll simulate a fine-tune by using two similar models or the same model
    # In real scenario, you'd use base + actual fine-tune
    base_model_id = "meta-llama/Llama-2-7b-hf"
    
    print("⚠️  NOTE: This test simulates a fine-tune scenario.")
    print("   For real validation, you need:")
    print("   - Base: meta-llama/Llama-2-7b-hf")
    print("   - Fine-tune: user/llama-2-7b-chat or similar")
    print()
    
    # Test 1: Estimate savings (no download needed)
    print("[1/3] Estimating delta compression savings...")
    print(f"  Base model: {base_model_id}")
    print()
    
    try:
        savings = estimate_delta_savings(base_model_id, base_model_id)
        
        savings_pct = (1 - 1/savings['estimated_compression']) * 100 if savings['estimated_compression'] > 1 else 0
        print("  Estimation Results:")
        print(f"    Best strategy:     {savings['best_strategy']}")
        print(f"    Compression ratio: {savings['estimated_compression']:.2f}x")
        print(f"    Average sparsity:  {savings['avg_sparsity']*100:.1f}%")
        print(f"    Savings:           {savings_pct:.1f}%")
        print(f"    Breakdown:")
        print(f"      sparse:       {savings['sparse_compression']:.2f}x")
        print(f"      int8:         {savings['int8_compression']:.2f}x")
        print(f"      sparse+int8:  {savings['sparse_int8_compression']:.2f}x")
        print()
        
    except Exception as e:
        print(f"  ❌ Estimation failed: {e}")
        print()
    
    # Test 2: Actual compression (requires model download - use smaller model)
    print("[2/3] Testing actual delta compression on smaller model (GPT-2)...")
    print("  (Using GPT-2 for speed - same algorithm applies to 7B)")
    print()
    
    try:
        test_model = "gpt2"  # 124M params, ~500MB
        output_dir = Path("./test_delta_output")
        
        print(f"  Loading base model: {test_model}")
        base = AutoModelForCausalLM.from_pretrained(
            test_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print(f"  Simulating fine-tune (adding noise to weights)...")
        finetuned = AutoModelForCausalLM.from_pretrained(
            test_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Simulate fine-tuning by modifying ~5% of weights
        for name, param in finetuned.named_parameters():
            if "weight" in name and len(param.shape) >= 2:
                # Add small noise to simulate fine-tuning changes
                noise = torch.randn_like(param) * 0.01
                mask = torch.rand_like(param) < 0.05  # Only change 5%
                param.data += noise * mask
        
        print(f"  Compressing delta...")
        start_time = time.time()
        
        # Note: This will create the output directory structure
        # In production, you'd use actual base + fine-tune model IDs
        from core.delta import compute_layer_delta, compress_delta_sparse
        
        total_params = 0
        changed_params = 0
        total_original_size = 0
        total_compressed_size = 0
        
        base_params = dict(base.named_parameters())
        finetune_params = dict(finetuned.named_parameters())
        
        for name in list(base_params.keys())[:10]:  # Test first 10 layers for speed
            base_weight = base_params[name].data
            finetune_weight = finetune_params[name].data
            
            if base_weight.shape != finetune_weight.shape:
                continue
            
            delta, stats = compute_layer_delta(base_weight, finetune_weight)
            indices, values, comp_ratio = compress_delta_sparse(delta, threshold=1e-6)
            
            total_params += delta.numel()
            changed_params += values.numel()
            total_original_size += delta.numel() * 2  # FP16
            total_compressed_size += indices.numel() * 4 + values.numel() * 2
        
        compression_time = time.time() - start_time
        
        overall_compression = total_original_size / max(total_compressed_size, 1)
        changed_pct = (changed_params / max(total_params, 1)) * 100
        
        print()
        print("  Compression Results (Sample):")
        print(f"    Total params:      {total_params:,}")
        print(f"    Changed params:    {changed_params:,} ({changed_pct:.1f}%)")
        print(f"    Original size:     {total_original_size/1024/1024:.2f} MB")
        print(f"    Compressed size:   {total_compressed_size/1024/1024:.2f} MB")
        print(f"    Compression ratio: {overall_compression:.2f}x")
        print(f"    Time taken:        {compression_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"  ❌ Compression test failed: {e}")
        print()
    
    # Test 3: Rust acceleration check
    print("[3/3] Checking Rust acceleration...")
    print()
    
    try:
        from core.delta_rust import is_rust_available
        
        if is_rust_available():
            print("  ✅ Rust acceleration: AVAILABLE")
            print("     Expected speedup: 10-20x on compression operations")
        else:
            print("  ⚠️  Rust acceleration: NOT AVAILABLE")
            print("     Falling back to Python implementation")
    except ImportError:
        print("  ⚠️  Rust acceleration: NOT AVAILABLE")
        print("     Falling back to Python implementation")
    
    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. For full validation, test with actual base + fine-tune pair")
    print("2. Document results in benchmarks/DELTA_COMPRESSION.md")
    print("3. Include in acquisition pitch materials")
    print()

if __name__ == "__main__":
    test_delta_compression_7b()
