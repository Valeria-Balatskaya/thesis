#!/usr/bin/env bash
# Build the TraceMark demo in ~/tracemark_demo (NEVER app/data) and check that every suspect gives the expected
# verdict through the same /api/scan the UI uses.
#
#   bash scripts/build_demo.sh              build + verify (server on 8765 is stopped afterwards)
#   bash scripts/build_demo.sh --reset      rebuild: removes ~/tracemark_demo first (only if it has the demo marker)
#   bash scripts/run_demo.sh                serve the finished demo on http://127.0.0.1:8765
#
# Needs data/real_world (real captures) and data/products_png, data/abo_sample, data/coco_sample (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
DEMO="${TRACEMARK_DEMO_DIR:-$HOME/tracemark_demo}"
PORT=8765
PY=".venv/bin/python"

case "$(cd "$(dirname "$DEMO")" 2>/dev/null && pwd)/$(basename "$DEMO")" in
  "$(pwd)/app/data"*) echo "refusing: $DEMO is the app's own data folder"; exit 1;;
esac
if [ -e "$DEMO" ]; then
  if [ ! -f "$DEMO/.tracemark_demo" ]; then echo "$DEMO exists and is not a demo folder (no .tracemark_demo marker): not touching it"; exit 1; fi
  if [ "${1:-}" != "--reset" ]; then echo "$DEMO already built. Use --reset to rebuild, or scripts/run_demo.sh to serve it."; exit 1; fi
  rm -rf "$DEMO"
fi
mkdir -p "$DEMO"; touch "$DEMO/.tracemark_demo"

export TRACEMARK_DATA="$DEMO"
$PY -W ignore scripts/build_demo.py 2>&1 | grep -Ev "^\{|WARN|ffmpeg|attention|Working|loaded|exists"

# serve and verify through the HTTP API (never the app on port 8000)
$PY -m uvicorn app.main:app --port $PORT --log-level warning > "$DEMO/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/api/overview" >/dev/null && break; sleep 2; done
$PY scripts/verify_demo.py --base "http://127.0.0.1:$PORT" --suspects "$DEMO/suspects"
