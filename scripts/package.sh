#!/bin/bash
set -euo pipefail

echo "[package] Building tenpak release binary..."
cargo build --release

echo "[package] Creating package structure..."
rm -rf dist
mkdir -p dist/tenpak/bin
mkdir -p dist/tenpak/lib

echo "[package] Copying binary (scripts are embedded)..."
cp target/release/tenpak dist/tenpak/bin/
chmod +x dist/tenpak/bin/tenpak

echo "[package] Copying documentation..."
cp README.md dist/tenpak/

echo "[package] Creating tarball..."
cd dist
tar -czf tenpak.tar.gz tenpak/
mv tenpak.tar.gz ../

cd ..
echo "[package] ✓ Package created: tenpak.tar.gz"
echo "[package] Package contents:"
tar -tzf tenpak.tar.gz
echo ""
echo "[package] Note: Scripts are embedded in the binary - no external scripts needed!"
