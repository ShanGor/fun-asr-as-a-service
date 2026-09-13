#!/usr/bin/env bash
# Start FunASR Core with the Apple Silicon/MPS backend.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ]; then
  echo "error: run_macos.sh must be run on macOS" >&2
  exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
  echo "error: this installer targets Apple Silicon (arm64) Macs" >&2
  exit 1
fi

export FUNASR_DEVICE="${FUNASR_DEVICE:-mps}"
exec scripts/run_server.sh
