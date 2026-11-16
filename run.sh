#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
exec ./.venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT"
