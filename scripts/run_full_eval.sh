#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the full 10pak demo:
#  - Create venv and download demo models
#  - Build 10pak in release mode
#  - Run Python eval to compute metrics and update README

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[10pak] Root directory: ${ROOT_DIR}"

# Step 1: download models into ./models using the helper script
if [[ ! -x "${ROOT_DIR}/scripts/download_demo_models.sh" ]]; then
  chmod +x "${ROOT_DIR}/scripts/download_demo_models.sh"
fi
"${ROOT_DIR}/scripts/download_demo_models.sh"

# Step 2: build 10pak binary in release mode
echo "[10pak] Building 10pak (release)"
cargo build --release

# Step 3: activate the eval virtualenv and run the Python eval script
VENV_DIR="${ROOT_DIR}/.tenpak-eval-venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

chmod +x "${ROOT_DIR}/scripts/run_eval_and_update_readme.py"
python "${ROOT_DIR}/scripts/run_eval_and_update_readme.py"

echo "[10pak] Full eval completed. README has been updated with results."
