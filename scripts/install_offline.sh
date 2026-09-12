#!/usr/bin/env bash
# Install the Python environment on the target server with NO network access.
# Requires: python3.12 on PATH and the bundled wheels/ + requirements-lock.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

PY312="${PY312:-python3.12}"

if [ ! -d wheels ] || [ -z "$(ls wheels 2>/dev/null)" ]; then
  echo "error: wheels/ is empty; bundle it first with scripts/download_wheels.sh" >&2
  exit 1
fi

"$PY312" -m venv .venv

# build tooling first (offline sdists need --no-build-isolation + setuptools)
.venv/bin/pip install --no-index --find-links wheels pip setuptools wheel

.venv/bin/pip install \
  --no-index --find-links wheels \
  --no-build-isolation \
  -r requirements-lock.txt

.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "offline install complete"
