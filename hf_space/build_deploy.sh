#!/bin/bash
# Build a deployment package for HuggingFace Spaces
# Creates a standalone directory with all files (no symlinks)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_ROOT/hf_space_deploy}"

echo "Building HF Space deployment package..."
echo "  Source: $SCRIPT_DIR"
echo "  Output: $OUTPUT_DIR"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy hf_space files (excluding symlinks)
cp "$SCRIPT_DIR/app.py" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/Dockerfile" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/pyproject.toml" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/README.md" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/.dockerignore" "$OUTPUT_DIR/"
cp "$SCRIPT_DIR/.gitignore" "$OUTPUT_DIR/"

# Copy actual directories from main project
cp -r "$PROJECT_ROOT/core" "$OUTPUT_DIR/core"
cp -r "$PROJECT_ROOT/optimizer" "$OUTPUT_DIR/optimizer"
cp -r "$PROJECT_ROOT/rust" "$OUTPUT_DIR/rust"

# Clean up
rm -rf "$OUTPUT_DIR/core/__pycache__"
rm -rf "$OUTPUT_DIR/optimizer/__pycache__"
rm -rf "$OUTPUT_DIR/rust/target"

echo "✅ Deployment package ready at: $OUTPUT_DIR"
echo ""
echo "To deploy to HuggingFace Spaces:"
echo "  cd $OUTPUT_DIR"
echo "  git init && git add . && git commit -m 'Deploy'"
echo "  git remote add space https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE"
echo "  git push space main"
