#!/usr/bin/env bash
# Simple installer for pre-built tenpak release binaries
set -euo pipefail

PREFIX="/usr/local"

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Options:
  --prefix DIR          Install binaries under DIR (default: /usr/local)
  -h, --help            Show this help message

This script installs the pre-built tenpak binary for model compression.

The binary includes embedded evaluation scripts - no external files needed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      if [[ $# -lt 2 ]]; then
        echo "Error: --prefix requires an argument" >&2
        exit 1
      fi
      PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Find the tenpak binary - check common locations
if [[ -f "${SCRIPT_DIR}/bin/tenpak" ]]; then
  # Standard release structure: install.sh and bin/ in same directory
  TENPAK_BIN="${SCRIPT_DIR}/bin/tenpak"
elif [[ -f "${SCRIPT_DIR}/../bin/tenpak" ]]; then
  # Alternative: install.sh in scripts/, bin/ at parent level
  TENPAK_BIN="${SCRIPT_DIR}/../bin/tenpak"
elif [[ -f "${SCRIPT_DIR}/tenpak" ]]; then
  # Flat structure: binary next to install script
  TENPAK_BIN="${SCRIPT_DIR}/tenpak"
else
  echo "Error: tenpak binary not found" >&2
  echo "" >&2
  echo "Expected release bundle structure:" >&2
  echo "  ./bin/tenpak" >&2
  echo "  ./install.sh" >&2
  echo "" >&2
  echo "Current directory: ${SCRIPT_DIR}" >&2
  exit 1
fi

echo "Found tenpak binary at: ${TENPAK_BIN}"

TARGET_BIN_DIR="${PREFIX}/bin"
TARGET_BIN="${TARGET_BIN_DIR}/tenpak"

# Helper functions
mkdir_if_needed() {
  local dir="$1"
  if [[ -d "$dir" ]]; then
    return
  fi
  if mkdir -p "$dir" 2>/dev/null; then
    return
  fi
  echo "Creating $dir with sudo" >&2
  sudo mkdir -p "$dir"
}

copy_file() {
  local src="$1"
  local dest="$2"
  if cp "$src" "$dest" 2>/dev/null; then
    return
  fi
  echo "Copying $src to $dest with sudo" >&2
  sudo cp "$src" "$dest"
}

chmod_file() {
  local mode="$1"
  local file="$2"
  if chmod "$mode" "$file" 2>/dev/null; then
    return
  fi
  echo "Setting permissions on $file with sudo" >&2
  sudo chmod "$mode" "$file"
}

# Main installation
echo ""
echo "🚀 Installing tenpak"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📦 Pre-built binary with embedded evaluation scripts"
echo "   Binary: ${TENPAK_BIN}"
echo "   Target: ${TARGET_BIN}"
echo ""

# Install binary
mkdir_if_needed "$TARGET_BIN_DIR"
copy_file "$TENPAK_BIN" "$TARGET_BIN"
chmod_file 755 "$TARGET_BIN"

echo "✅ Binary installed: ${TARGET_BIN}"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  tenpak compress input.json output.tenpak"
echo "  tenpak decompress input.tenpak output.json"
echo "  tenpak runeval                              # Run evaluation pipeline"
echo ""
echo "For evaluation, ensure Python 3 with torch and transformers is installed."
echo ""
