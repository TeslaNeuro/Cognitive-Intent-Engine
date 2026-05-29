#!/usr/bin/env bash
# macOS / Linux dev launcher. Run with: ./scripts/run_dev.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

(python -m backend.main &) >/tmp/cse-backend.log 2>&1
(cd frontend && npm run dev &) >/tmp/cse-frontend.log 2>&1

echo "Backend on http://localhost:8000 — frontend on http://localhost:5173"
echo "Logs: /tmp/cse-backend.log  /tmp/cse-frontend.log"
