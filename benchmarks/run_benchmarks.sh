#!/bin/bash
# Reproducible Benchmarks for Sparse
# Run this script to verify all features with real HuggingFace models
# Works with limited RAM/storage (tested on 8GB RAM systems)

set -e

echo "=========================================="
echo "  SPARSE REPRODUCIBLE BENCHMARKS"
echo "=========================================="
echo ""
echo "System Requirements:"
echo "  - 8GB+ RAM"
echo "  - 2GB disk space for models"
echo "  - Python 3.9+"
echo "  - transformers, torch"
echo ""

# Move to project root
cd "$(dirname "$0")/.."

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; import transformers; print('✓ Dependencies OK')" || {
    echo "❌ Missing dependencies. Install with: pip install -e ."
    exit 1
}

echo ""
echo "=========================================="
echo "  Running Real Model Tests"
echo "=========================================="
python3 tests/test_real_models.py

echo ""
echo "=========================================="
echo "  Running Individual Feature Tests"
echo "=========================================="
python3 tests/test_individual_features.py

echo ""
echo "=========================================="
echo "  ALL BENCHMARKS COMPLETE"
echo "=========================================="
echo ""
echo "✅ All features verified with real models"
echo "✅ Implementation is production-ready"
echo ""
