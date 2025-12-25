#!/bin/bash
# Install Rust and build sparse_core extension for HF Spaces

set -e

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Build Rust extension
cd rust
pip install maturin
maturin build --release --out ../dist

# Install the wheel
pip install ../dist/sparse_core-*.whl

echo "✅ Rust acceleration installed"
