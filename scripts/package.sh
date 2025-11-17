#!/bin/bash
set -euo pipefail

echo "[package] Building tenpak release binary..."
cargo build --release

echo "[package] Creating package structure..."
rm -rf dist
mkdir -p dist/tenpak/bin
mkdir -p dist/tenpak/scripts
mkdir -p dist/tenpak/lib

echo "[package] Copying binary..."
cp target/release/tenpak dist/tenpak/bin/
chmod +x dist/tenpak/bin/tenpak

echo "[package] Copying required scripts for runeval..."
cp scripts/download_demo_models.sh dist/tenpak/scripts/
cp scripts/run_eval_and_update_readme.py dist/tenpak/scripts/
chmod +x dist/tenpak/scripts/download_demo_models.sh

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
