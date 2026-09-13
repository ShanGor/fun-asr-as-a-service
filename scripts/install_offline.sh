#!/usr/bin/env bash
# Install the Python environment on the target server with NO network access.
# Usage: install_offline.sh [cuda|macos]
# Requires the matching bundled wheels directory and lock file.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-cuda}"
case "$TARGET" in
  cuda|nvidia|--cuda)
    TARGET="cuda"
    PYTHON_BIN="${PY312:-python3.12}"
    WHEELS_DIR="wheels"
    LOCK_FILE="requirements-lock.txt"
    ;;
  macos|mac|--macos)
    TARGET="macos"
    if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
      echo "error: macOS installation requires an Apple Silicon Mac" >&2
      exit 1
    fi
    PYTHON_BIN="${PYMACOS:-python3.12}"
    WHEELS_DIR="wheels-macos"
    LOCK_FILE="requirements-macos-lock.txt"
    ;;
  *)
    echo "usage: $0 [cuda|macos]" >&2
    exit 2
    ;;
esac

if [ ! -d "$WHEELS_DIR" ] || [ -z "$(find "$WHEELS_DIR" -maxdepth 1 -type f -print -quit)" ]; then
  echo "error: $WHEELS_DIR is empty; bundle it first with scripts/download_wheels.sh $TARGET" >&2
  exit 1
fi
if [ ! -f "$LOCK_FILE" ]; then
  echo "error: $LOCK_FILE is missing; bundle it first with scripts/download_wheels.sh $TARGET" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv

# build tooling first (offline sdists need --no-build-isolation + setuptools)
.venv/bin/pip install --no-index --find-links "$WHEELS_DIR" pip setuptools wheel

.venv/bin/pip install \
  --no-index --find-links "$WHEELS_DIR" \
  --no-build-isolation \
  -r "$LOCK_FILE"

if [ "$TARGET" = "macos" ]; then
  .venv/bin/python - <<'PY'
import platform
import torch
assert platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}, "Apple Silicon macOS required"
assert torch.backends.mps.is_available(), "MPS not available"
print("MPS available")
PY
else
  .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
fi
echo "offline $TARGET install complete"
