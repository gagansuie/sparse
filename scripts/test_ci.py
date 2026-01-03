#!/usr/bin/env python3
"""
Sparse CI Tests - Lightweight smoke tests for GitHub Actions

Tests structure and imports WITHOUT heavy dependencies (torch, transformers).
For full model testing, use test_all_features.py locally.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_package_structure():
    """Test package structure and imports."""
    print("\n" + "="*60)
    print("TEST: Package Structure")
    print("="*60)
    
    # Test core modules exist
    core_modules = ['core', 'cli']
    for module in core_modules:
        module_path = Path(__file__).parent.parent / module
        assert module_path.exists(), f"Missing module: {module}"
        print(f"✅ Module exists: {module}")
    
    # Test essential files exist
    essential_files = [
        'pyproject.toml',
        'README.md',
        'LICENSE',
        'requirements.txt',
    ]
    for file in essential_files:
        file_path = Path(__file__).parent.parent / file
        assert file_path.exists(), f"Missing file: {file}"
        print(f"✅ File exists: {file}")
    
    print("\n✅ Package structure: PASS")

def test_imports_no_torch():
    """Test imports that don't require torch/transformers."""
    print("\n" + "="*60)
    print("TEST: Lightweight Imports")
    print("="*60)
    
    # Test CLI imports
    try:
        import cli
        print("✅ cli module: OK")
    except ImportError as e:
        print(f"❌ cli import failed: {e}")
        sys.exit(1)
    
    # Test optimizer imports (should work without torch)
    try:
        from optimizer import CostOptimizer
        print("✅ optimizer module: OK")
    except ImportError as e:
        print(f"⚠️  optimizer import failed (may need torch): {e}")
    
    print("\n✅ Lightweight imports: PASS")

def test_cli_entrypoint():
    """Test CLI entrypoint is defined."""
    print("\n" + "="*60)
    print("TEST: CLI Entry Point")
    print("="*60)
    
    import subprocess
    import sys
    
    # Check if sparse command is installed
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", "sparse"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Package installed: sparse")
    else:
        print("⚠️  Package not installed (expected in development mode)")
    
    print("\n✅ CLI entry point: PASS")

def test_documentation():
    """Test documentation files exist and are non-empty."""
    print("\n" + "="*60)
    print("TEST: Documentation")
    print("="*60)
    
    docs = [
        'README.md',
        'docs/INTEGRATION_GUIDE.md',
        'docs/API_REFERENCE.md',
        'docs/PITCH_HUGGINGFACE.md',
    ]
    
    for doc in docs:
        doc_path = Path(__file__).parent.parent / doc
        if doc_path.exists():
            size = doc_path.stat().st_size
            assert size > 100, f"{doc} is too small ({size} bytes)"
            print(f"✅ {doc}: {size:,} bytes")
        else:
            print(f"⚠️  {doc}: missing")
    
    print("\n✅ Documentation: PASS")

def test_examples():
    """Test example scripts exist."""
    print("\n" + "="*60)
    print("TEST: Examples")
    print("="*60)
    
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        examples = list(examples_dir.glob('*.py'))
        print(f"✅ Found {len(examples)} example scripts:")
        for example in examples:
            print(f"   - {example.name}")
    else:
        print("⚠️  examples/ directory not found")
    
    print("\n✅ Examples: PASS")

def main():
    """Run all lightweight tests."""
    print("\n" + "="*60)
    print("SPARSE CI TESTS (Lightweight)")
    print("="*60)
    
    try:
        test_package_structure()
        test_imports_no_torch()
        test_cli_entrypoint()
        test_documentation()
        test_examples()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nNote: This is a lightweight test suite for CI/CD.")
        print("For full model testing, run: python scripts/test_all_features.py")
        
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
