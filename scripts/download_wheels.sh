#!/usr/bin/env bash
# Bundle all pip wheels/sdists into ./wheels so the target server can install
# without network access. Run once on a machine with network access.
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-.venv/bin/python}"

# Freeze the exact working environment first
"$PYBIN" -m pip freeze --exclude-editable > requirements-lock.txt

mkdir -p wheels
# Wheels where available; sdists for the few packages that publish none
# (e.g. antlr4-python3-runtime). Build tooling is bundled so those sdists can
# be built offline by install_offline.sh.
"$PYBIN" -m pip download \
  --dest wheels \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -r requirements-lock.txt

"$PYBIN" -m pip download \
  --dest wheels \
  --index-url https://pypi.org/simple \
  setuptools wheel pip

echo "wheels ready: $(ls wheels | wc -l) files, $(du -sh wheels | cut -f1)"
