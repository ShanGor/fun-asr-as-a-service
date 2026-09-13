#!/usr/bin/env bash
# Start the FunASR core inference server.
# Configuration is via FUNASR_* environment variables (see funasr_core/config.py).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" = "Darwin" ]; then
  # PyTorch MPS sends unsupported operators to the CPU when this fallback is
  # enabled. This keeps the service usable across model/backend combinations.
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

export FUNASR_HOST="${FUNASR_HOST:-127.0.0.1}"
export FUNASR_PORT="${FUNASR_PORT:-8000}"

# Never reach the network at runtime: models are pre-downloaded into ./models
export FUNASR_OFFLINE="${FUNASR_OFFLINE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models/.modelscope-cache}"
mkdir -p "$MODELSCOPE_CACHE"

PYTHON_BIN="${FUNASR_PYTHON:-.venv/bin/python}"
exec "$PYTHON_BIN" -m funasr_core.server
