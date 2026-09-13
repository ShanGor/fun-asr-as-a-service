#!/usr/bin/env bash
# Install FunASR Core on an Apple Silicon Mac and download the baseline models.
# Requires macOS, arm64, Python 3, and network access for the first install.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ]; then
  echo "error: install_macos.sh must be run on macOS" >&2
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "error: this installer targets Apple Silicon (arm64) Macs" >&2
  exit 1
fi

PYTHON_BIN="${PYMACOS:-python3.12}"
VENV_DIR="${FUNASR_VENV:-.venv}"
DOWNLOAD_MODELS=1
if [ "${1:-}" = "--skip-models" ]; then
  DOWNLOAD_MODELS=0
elif [ "${1:-}" != "" ]; then
  echo "usage: $0 [--skip-models]" >&2
  exit 2
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements-macos.txt

if [ "$DOWNLOAD_MODELS" -eq 1 ]; then
  "$VENV_DIR/bin/python" scripts/download_models.py
fi

"$VENV_DIR/bin/python" - <<'PY'
import platform
import torch

if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
    raise SystemExit("Apple Silicon macOS is required")
if not torch.backends.mps.is_available():
    raise SystemExit(
        "PyTorch MPS is unavailable; use a current macOS/PyTorch install "
        "or run with FUNASR_DEVICE=cpu"
    )
print(f"Apple Silicon ready: torch={torch.__version__}, MPS available")
PY

echo "macOS install complete"
echo "start with: scripts/run_macos.sh"
echo "install ffmpeg separately for mp3/aac/m4a uploads: brew install ffmpeg"
