#!/usr/bin/env bash
set -euo pipefail

# Simple helper to download demo models/checkpoints for 10pak evaluation.
# Intended to be run on an EC2 instance with Python + pip available.
#
# This script will:
#   - Create a Python virtualenv (./.tenpak-eval-venv)
#   - Install minimal dependencies (transformers, datasets, accelerate, huggingface_hub)
#   - Download TinyLlama or GPT-2 style models and tokenizers via HuggingFace
#   - Materialize local checkpoints under ./models/
#
# Usage:
#   chmod +x scripts/download_demo_models.sh
#   ./scripts/download_demo_models.sh
#
# You can control which models to download via the MODELS env var, e.g.:
#   MODELS="gpt2 TinyLlama/TinyLlama-1.1B-Chat-v1.0" ./scripts/download_demo_models.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.tenpak-eval-venv"
MODELS_DEFAULT=("gpt2" "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

if [[ -n "${MODELS:-}" ]]; then
  # Split MODELS env var on whitespace into an array
  read -r -a MODELS_TO_DOWNLOAD <<< "${MODELS}"
else
  MODELS_TO_DOWNLOAD=("${MODELS_DEFAULT[@]}")
fi

MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "${MODELS_DIR}"

echo "[10pak] Root directory: ${ROOT_DIR}"
echo "[10pak] Models directory: ${MODELS_DIR}"

# Check if required packages are already available
if python3 -c "import torch, transformers" 2>/dev/null; then
  echo "[10pak] Using existing Python environment (torch & transformers found)"
  PYTHON_CMD="python3"
  # Install missing packages if needed
  python3 -m pip install --quiet datasets accelerate huggingface_hub 2>/dev/null || true
else
  echo "[10pak] Creating virtualenv at ${VENV_DIR} (if not exists)"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_CMD="python"
fi

if [[ "${PYTHON_CMD}" == "python" ]]; then
  # Only install if using venv
  pip install --upgrade pip >/dev/null
  pip install "transformers[torch]" datasets accelerate huggingface_hub >/dev/null
fi

echo "[10pak] Downloading models: ${MODELS_TO_DOWNLOAD[*]}"

${PYTHON_CMD} << 'EOF'
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = root_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

models_to_download = os.environ.get("MODELS")
if models_to_download:
    names = models_to_download.split()
else:
    names = ["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]

for name in names:
    print(f"[10pak] Downloading {name}...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)

    safe_name = name.replace("/", "_")
    out_dir = models_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[10pak] Saving model and tokenizer for {name} under {out_dir}")
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

print("[10pak] Done. Models are available under ./models/<model_name>/")
EOF
