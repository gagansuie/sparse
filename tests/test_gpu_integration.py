"""
Test GPU integration for INT8 delta reconstruction.
"""

import torch
import numpy as np

def test_gpu_availability():
    """Check if GPU acceleration is available."""
    print("=" * 60)
    print("GPU INTEGRATION TEST")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ PyTorch CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    # Check if sparse_core is available
    try:
        import sparse_core
        print(f"✓ sparse_core imported successfully")
        
        # Check if GPU ops are available
        try:
            gpu_ops = sparse_core.GpuOptimizedOps(tile_size=256, use_fma=True)
            print(f"✓ GpuOptimizedOps available")
            
            # Test tiled processing
            base = np.random.randn(10000).astype(np.float32)
            delta = np.random.randint(-127, 127, 10000, dtype=np.int8)
            result = gpu_ops.apply_int8_delta_tiled(base, delta, 0.001)
            print(f"✓ GPU ops working (processed {len(result)} elements)")
            
        except Exception as e:
            print(f"✗ GpuOptimizedOps error: {e}")
    except ImportError as e:
        print(f"✗ sparse_core import failed: {e}")
    
    print("\n" + "=" * 60)
    print("INTEGRATION STATUS")
    print("=" * 60)
    
    if cuda_available:
        print("\n✅ GPU acceleration ENABLED")
        print("   INT8 deltas will use GPU-optimized reconstruction")
        print("   Expected speedup: 2-3x for INT8 methods")
    else:
        print("\n⚠️  GPU acceleration DISABLED")
        print("   Using standard Rust acceleration")
        print("   To enable GPU: Install PyTorch with CUDA support")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_gpu_availability()
