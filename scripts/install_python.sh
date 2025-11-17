#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
  echo "[tenpak] Python is already installed."
  exit 0
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "[tenpak] Installing Python via apt-get..."
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip
elif command -v yum >/dev/null 2>&1; then
  echo "[tenpak] Installing Python via yum..."
  sudo yum install -y python3
elif command -v dnf >/dev/null 2>&1; then
  echo "[tenpak] Installing Python via dnf..."
  sudo dnf install -y python3
elif command -v brew >/dev/null 2>&1; then
  echo "[tenpak] Installing Python via Homebrew..."
  brew install python
else
  echo "[tenpak] Could not detect a supported package manager. Please install Python manually." >&2
  exit 1
fi
