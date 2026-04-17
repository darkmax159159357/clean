#!/usr/bin/env bash
# One-shot remote setup for the GPU server on vast.ai.
# Usage on vast.ai (after pasting TOKEN):
#   TOKEN="<your_token>"
#   BASE="https://174ca3b9-d088-4b34-aeb0-dbc9976fa8aa-00-u5olo7fa1zd.worf.replit.dev/gpu_bootstrap/$TOKEN"
#   wget -qO /tmp/setup.sh "$BASE/gpu_server/setup_remote.sh" && bash /tmp/setup.sh "$TOKEN"

set -e

TOKEN="${1:-${GPU_ADMIN_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  echo "❌ Missing token. Usage: bash setup_remote.sh <TOKEN>"
  exit 1
fi

BASE_URL="${REPLIT_BASE_URL:-https://174ca3b9-d088-4b34-aeb0-dbc9976fa8aa-00-u5olo7fa1zd.worf.replit.dev}"
BOOT="$BASE_URL/gpu_bootstrap/$TOKEN"

WORKDIR="${WORKDIR:-/workspace/gpu_server}"
CTDDIR="${CTDDIR:-/workspace/ctd}"

echo "=============================================="
echo "🚀 LALA GPU Server — Remote Setup"
echo "=============================================="
echo "Base URL : $BASE_URL"
echo "Workdir  : $WORKDIR"
echo ""

mkdir -p "$WORKDIR" "$CTDDIR" /workspace/models/sd-inpaint

fetch() {
  local rel="$1"
  local dst="$2"
  echo "⬇️  $rel → $dst"
  if ! wget -q --show-progress -O "$dst.tmp" "$BOOT/$rel"; then
    echo "❌ Failed to download $rel"
    rm -f "$dst.tmp"
    exit 1
  fi
  # Guard against auth/404 HTML/JSON errors served with 200 by reverse proxy
  if head -c 200 "$dst.tmp" | grep -q '"error"'; then
    echo "❌ Server returned error for $rel:"
    cat "$dst.tmp"
    rm -f "$dst.tmp"
    exit 1
  fi
  mv "$dst.tmp" "$dst"
}

echo "=== 1/4 Syncing code from Replit ==="
fetch "gpu_server/server.py"            "$WORKDIR/server.py"
fetch "gpu_server/requirements.txt"     "$WORKDIR/requirements.txt"
fetch "ctd/__init__.py"                 "$CTDDIR/__init__.py"
fetch "ctd/ctd_onnx_detector.py"        "$CTDDIR/ctd_onnx_detector.py"
fetch "ctd/lama_inpaint.py"             "$CTDDIR/lama_inpaint.py"
fetch "ctd/sd_inpaint.py"               "$CTDDIR/sd_inpaint.py"
fetch "ctd/easyocr_enhancer.py"         "$CTDDIR/easyocr_enhancer.py"
# Remove obsolete files from previous deployments
rm -f "$CTDDIR/detector.py" "$CTDDIR/smart_clean.py"

echo ""
echo "=== 2/4 Installing Python dependencies ==="
pip install --quiet --upgrade -r "$WORKDIR/requirements.txt"

echo ""
echo "=== 3/4 Restarting server.py ==="
pkill -f "python.*server.py" 2>/dev/null || true
sleep 2

cd "$WORKDIR"
# Start in background with nohup so SSH disconnect doesn't kill it
mkdir -p /workspace/logs
PORT="${SERVER_PORT:-7860}"
nohup python server.py --port "$PORT" > /workspace/logs/server.log 2>&1 &
SERVER_PID=$!
echo "✅ server.py started (PID=$SERVER_PID), logs → /workspace/logs/server.log"

sleep 5
echo ""
echo "=== 4/4 Health check ==="
PORT="${SERVER_PORT:-7860}"
for i in 1 2 3 4 5 6; do
  if curl -sf "http://localhost:$PORT/health" > /tmp/health.json; then
    echo "✅ Server is healthy:"
    cat /tmp/health.json
    echo ""
    break
  fi
  echo "  waiting for server... ($i/6)"
  sleep 5
done

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo "Tail logs:   tail -f /workspace/logs/server.log"
echo "Stop server: pkill -f 'python.*server.py'"
echo ""
echo "ℹ️  SD model will auto-download (~4GB) to"
echo "   /workspace/models/sd-inpaint/ on first /clean engine:sd call."
