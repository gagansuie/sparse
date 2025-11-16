#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the full tenpak demo:
#  - Create venv and download demo models
#  - Build tenpak in release mode
#  - Run Python eval to compute metrics and update README

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[tenpak] Root directory: ${ROOT_DIR}"

# Step 1: download models into ./models using the helper script
if [[ ! -x "${ROOT_DIR}/scripts/download_demo_models.sh" ]]; then
  chmod +x "${ROOT_DIR}/scripts/download_demo_models.sh"
fi
"${ROOT_DIR}/scripts/download_demo_models.sh"

# Step 2: build 10pak binary in release mode
echo "[tenpak] Building tenpak (release)"
cargo build --release

# Step 3: activate the eval virtualenv and run the Python eval script
VENV_DIR="${ROOT_DIR}/.tenpak-eval-venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

chmod +x "${ROOT_DIR}/scripts/run_eval_and_update_readme.py"
python "${ROOT_DIR}/scripts/run_eval_and_update_readme.py"

echo "[tenpak] Full eval completed. README has been updated with results."
