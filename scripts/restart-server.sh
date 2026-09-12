#!/usr/bin/env bash
# Gracefully restart the FunASR core inference server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PORT="${FUNASR_PORT:-8000}"
SHUTDOWN_TIMEOUT="${RESTART_SHUTDOWN_TIMEOUT:-30}"
START_SERVER="$SCRIPT_DIR/run_server.sh"

is_funasr_server() {
    local pid="$1"
    local cwd
    local args

    [[ -d "/proc/$pid" ]] || return 1
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" || return 1
    [[ "$cwd" == "$PROJECT_DIR" ]] || return 1
    args="$(ps -p "$pid" -o args= 2>/dev/null)" || return 1
    [[ "$args" == *"-m funasr_core.server"* ]]
}

server_pids=()

# Prefer the process listening on the configured port, and verify its command
# and working directory before sending it a signal.
if command -v lsof >/dev/null 2>&1; then
    while IFS= read -r pid; do
        if is_funasr_server "$pid"; then
            server_pids+=("$pid")
        fi
    done < <(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u)
else
    # Fall back to process discovery on hosts without lsof.
    while IFS= read -r pid; do
        if is_funasr_server "$pid"; then
            server_pids+=("$pid")
        fi
    done < <(pgrep -f '(^|[[:space:]])-m[[:space:]]+funasr_core\.server([[:space:]]|$)' || true)
fi

if ((${#server_pids[@]} > 0)); then
    echo "Requesting graceful shutdown of FunASR server (PID ${server_pids[*]})..."
    for pid in "${server_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
    while :; do
        remaining=()
        for pid in "${server_pids[@]}"; do
            if is_funasr_server "$pid"; then
                remaining+=("$pid")
            fi
        done

        ((${#remaining[@]} == 0)) && break
        if ((SECONDS >= deadline)); then
            echo "Timed out waiting for PID ${remaining[*]} to shut down; not starting a second server." >&2
            exit 1
        fi
        sleep 1
    done
    echo "FunASR server stopped gracefully."
else
    echo "No existing FunASR server found on TCP port $PORT."
fi

exec "$START_SERVER"
