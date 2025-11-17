#!/bin/bash
set -euo pipefail

echo "[package] Building tenpak release binary..."
cargo build --release

echo "[package] Creating package structure..."
rm -rf dist
mkdir -p dist/bin
mkdir -p dist/lib

echo "[package] Copying binary (scripts are embedded)..."
cp target/release/tenpak dist/bin/
chmod +x dist/bin/tenpak

echo "[package] Copying install script..."
cp scripts/install_release.sh dist/install.sh
chmod +x dist/install.sh

echo "[package] Copying documentation..."
cp README.md dist/
[ -f LICENSE ] && cp LICENSE dist/ || true

echo "[package] Creating tarball..."
tar -czf tenpak.tar.gz -C dist .
echo "[package] ✓ Package created: tenpak.tar.gz"
echo "[package] Package contents:"
tar -tzf tenpak.tar.gz
echo ""
echo "[package] Note: Scripts are embedded in the binary - no external scripts needed!"
