#!/bin/bash
# Build script for Sparse Rust core

set -e

echo "Building Sparse Rust core..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Maturin not found. Installing..."
    pip install maturin
fi

# Build in release mode
echo "Building with maturin..."
maturin develop --release

echo "Build complete! Rust acceleration is now available."
echo ""
echo "Test the installation:"
echo "  python -c 'import sparse_core; print(\"Rust core loaded successfully!\")'"
