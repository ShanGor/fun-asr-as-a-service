#!/usr/bin/env bash
# Bundle all pip wheels/sdists into ./wheels so the target server can install
# without network access. Run once on a machine with network access.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-cuda}"
case "$TARGET" in
  cuda|nvidia|--cuda)
    TARGET="cuda"
    PYBIN="${PYBIN:-.venv/bin/python}"
    REQUIREMENTS_FILE="requirements-lock.txt"
    WHEELS_DIR="wheels"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    EXTRA_INDEX="https://pypi.org/simple"
    ;;
  macos|mac|--macos)
    TARGET="macos"
    if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
      echo "error: macOS wheels must be downloaded on an Apple Silicon Mac" >&2
      exit 1
    fi
    PYBIN="${PYBIN:-.venv/bin/python}"
    REQUIREMENTS_FILE="requirements-macos-lock.txt"
    WHEELS_DIR="wheels-macos"
    TORCH_INDEX="https://pypi.org/simple"
    EXTRA_INDEX=""
    ;;
  *)
    echo "usage: $0 [cuda|macos]" >&2
    exit 2
    ;;
esac

if [ ! -x "$PYBIN" ]; then
  echo "error: Python environment not found at $PYBIN" >&2
  echo "run the matching installer first" >&2
  exit 1
fi

# Freeze the exact working environment first
"$PYBIN" -m pip freeze --exclude-editable > "$REQUIREMENTS_FILE"

mkdir -p "$WHEELS_DIR"
# Wheels where available; sdists for the few packages that publish none
# (e.g. antlr4-python3-runtime). Build tooling is bundled so those sdists can
# be built offline by install_offline.sh.
PIP_INDEX_ARGS=(--index-url "$TORCH_INDEX")
if [ -n "$EXTRA_INDEX" ]; then
  PIP_INDEX_ARGS+=(--extra-index-url "$EXTRA_INDEX")
fi
"$PYBIN" -m pip download \
  --dest "$WHEELS_DIR" \
  "${PIP_INDEX_ARGS[@]}" \
  -r "$REQUIREMENTS_FILE"

"$PYBIN" -m pip download \
  --dest "$WHEELS_DIR" \
  --index-url https://pypi.org/simple \
  setuptools wheel pip

echo "wheels ready: $(find "$WHEELS_DIR" -maxdepth 1 -type f | wc -l) files in $WHEELS_DIR"
