#!/usr/bin/env bash
# Serve the demo built by scripts/build_demo.sh on port 8765 (never touches app/data, never port 8000).
cd "$(dirname "$0")/.."
DEMO="${TRACEMARK_DEMO_DIR:-$HOME/tracemark_demo}"
[ -f "$DEMO/.tracemark_demo" ] || { echo "no demo at $DEMO: run bash scripts/build_demo.sh first"; exit 1; }
echo "TraceMark demo: http://127.0.0.1:8765   suspects to upload: $DEMO/suspects/"
TRACEMARK_DATA="$DEMO" exec .venv/bin/python -m uvicorn app.main:app --port 8765
