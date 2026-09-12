#!/usr/bin/env bash
# Start the FunASR core inference server.
# Configuration is via FUNASR_* environment variables (see funasr_core/config.py).
set -euo pipefail
cd "$(dirname "$0")/.."

export FUNASR_HOST="${FUNASR_HOST:-127.0.0.1}"
export FUNASR_PORT="${FUNASR_PORT:-8000}"

# Never reach the network at runtime: models are pre-downloaded into ./models
export FUNASR_OFFLINE="${FUNASR_OFFLINE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models/.modelscope-cache}"
mkdir -p "$MODELSCOPE_CACHE"

exec .venv/bin/python -m funasr_core.server
