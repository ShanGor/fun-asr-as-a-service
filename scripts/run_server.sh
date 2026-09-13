#!/usr/bin/env bash
# Start the FunASR core inference server.
# Configuration is via FUNASR_* environment variables (see funasr_core/config.py).
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  printf '%s\n' \
    "Usage: scripts/run_server.sh [options]" \
    "" \
    "Options:" \
    "  --https                 Enable HTTPS using deploy/tls/fullchain.pem and deploy/tls/privkey.pem" \
    "  --host HOST             Bind address (default: FUNASR_HOST or 127.0.0.1)" \
    "  --port PORT             Bind port (default: FUNASR_PORT or 8000)" \
    "  --ssl-certfile FILE    HTTPS certificate PEM file" \
    "  --ssl-keyfile FILE     HTTPS private key PEM file" \
    "  --help                  Show this help" \
    "" \
    "Environment equivalents: FUNASR_HOST, FUNASR_PORT, FUNASR_SSL_CERTFILE," \
    "FUNASR_SSL_KEYFILE, FUNASR_SSL_KEYFILE_PASSWORD"
}

host_value="${FUNASR_HOST:-127.0.0.1}"
port_value="${FUNASR_PORT:-8000}"
ssl_certfile="${FUNASR_SSL_CERTFILE:-}"
ssl_keyfile="${FUNASR_SSL_KEYFILE:-}"

while (($# > 0)); do
  case "$1" in
    --https)
      ssl_certfile="${ssl_certfile:-deploy/tls/fullchain.pem}"
      ssl_keyfile="${ssl_keyfile:-deploy/tls/privkey.pem}"
      shift
      ;;
    --host)
      [[ $# -ge 2 ]] || { echo "--host requires a value" >&2; exit 2; }
      host_value="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      port_value="$2"
      shift 2
      ;;
    --ssl-certfile)
      [[ $# -ge 2 ]] || { echo "--ssl-certfile requires a value" >&2; exit 2; }
      ssl_certfile="$2"
      shift 2
      ;;
    --ssl-keyfile)
      [[ $# -ge 2 ]] || { echo "--ssl-keyfile requires a value" >&2; exit 2; }
      ssl_keyfile="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ssl_certfile" || -n "$ssl_keyfile" ]]; then
  [[ -n "$ssl_certfile" && -n "$ssl_keyfile" ]] || {
    echo "HTTPS requires both a certificate and a private key." >&2
    exit 2
  }
  [[ -f "$ssl_certfile" ]] || { echo "Certificate not found: $ssl_certfile" >&2; exit 2; }
  [[ -f "$ssl_keyfile" ]] || { echo "Private key not found: $ssl_keyfile" >&2; exit 2; }
fi

if [ "$(uname -s)" = "Darwin" ]; then
  # PyTorch MPS sends unsupported operators to the CPU when this fallback is
  # enabled. This keeps the service usable across model/backend combinations.
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

export FUNASR_HOST="$host_value"
export FUNASR_PORT="$port_value"
export FUNASR_SSL_CERTFILE="$ssl_certfile"
export FUNASR_SSL_KEYFILE="$ssl_keyfile"

# Never reach the network at runtime: models are pre-downloaded into ./models
export FUNASR_OFFLINE="${FUNASR_OFFLINE:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models/.modelscope-cache}"
mkdir -p "$MODELSCOPE_CACHE"

PYTHON_BIN="${FUNASR_PYTHON:-.venv/bin/python}"
exec "$PYTHON_BIN" -m funasr_core.server
