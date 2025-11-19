#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the full tenpak demo:
#  - Create venv and download demo models
#  - Build tenpak in release mode
#  - Run Python eval to compute metrics and update README

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements-eval.txt"

echo "[tenpak] Root directory: ${ROOT_DIR}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "[tenpak] Missing requirements file at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

# Step 1: download models into ./models using the helper script
if [[ ! -x "${ROOT_DIR}/scripts/download_demo_models.sh" ]]; then
  chmod +x "${ROOT_DIR}/scripts/download_demo_models.sh"
fi
"${ROOT_DIR}/scripts/download_demo_models.sh"

# Step 2: ensure we have a tenpak binary (build if cargo is available, otherwise
# reuse the packaged binary from the release tarball)
TARGET_BUILD_BIN="${ROOT_DIR}/target/release/tenpak"
PACKAGED_BIN="${ROOT_DIR}/bin/tenpak"

if command -v cargo >/dev/null 2>&1; then
  echo "[tenpak] Building tenpak (ci-release)"
  cargo build --profile ci-release
  mkdir -p "${ROOT_DIR}/target/release"
  cp "${ROOT_DIR}/target/ci-release/tenpak" "${TARGET_BUILD_BIN}"
  TENPAK_BIN_PATH="${TARGET_BUILD_BIN}"
else
  echo "[tenpak] Cargo not found; reusing packaged binary"
  TENPAK_BIN_PATH="${PACKAGED_BIN}"
fi

if [[ ! -x "${TENPAK_BIN_PATH}" ]]; then
  echo "[tenpak] Error: tenpak binary not available at ${TENPAK_BIN_PATH}" >&2
  echo "         Install Rust/cargo to build from source OR ensure bin/tenpak is present." >&2
  exit 1
fi

export TENPAK_BIN="${TENPAK_BIN_PATH}"
echo "[tenpak] Using CLI binary: ${TENPAK_BIN}"

# Step 3: create/activate the eval virtualenv, install deps, and run the Python eval script
VENV_DIR="${ROOT_DIR}/.tenpak-eval-venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[tenpak] Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip >/dev/null
python -m pip install -r "${REQUIREMENTS_FILE}"

chmod +x "${ROOT_DIR}/scripts/run_eval.py"
python "${ROOT_DIR}/scripts/run_eval.py"

echo "[tenpak] Full eval completed. README has been updated with results."
