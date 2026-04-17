#!/usr/bin/env bash
# One-shot remote setup for the GPU server on vast.ai.
# Pulls the latest code from GitHub: https://github.com/darkmax159159357/clean
#
# Usage on vast.ai:
#   wget -qO /tmp/setup.sh https://raw.githubusercontent.com/darkmax159159357/clean/main/gpu_server/setup_remote.sh
#   bash /tmp/setup.sh

set -e

REPO_URL="${REPO_URL:-https://github.com/darkmax159159357/clean.git}"
BRANCH="${BRANCH:-main}"
REPO_DIR="${REPO_DIR:-/workspace/lala-repo}"
WORKDIR="${WORKDIR:-/workspace/gpu_server}"
CTDDIR="${CTDDIR:-/workspace/ctd}"

echo "=============================================="
echo "🚀 LALA GPU Server — Remote Setup (git)"
echo "=============================================="
echo "Repo    : $REPO_URL ($BRANCH)"
echo "Repo dir: $REPO_DIR"
echo "Workdir : $WORKDIR"
echo ""

mkdir -p "$WORKDIR" "$CTDDIR" /workspace/models/sd-inpaint /workspace/models

echo "=== 1/4 Syncing code from GitHub ==="
if [ -d "$REPO_DIR/.git" ]; then
  echo "↻ Updating existing clone…"
  git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
  git -C "$REPO_DIR" clean -fd
else
  echo "⬇️  Cloning $REPO_URL…"
  rm -rf "$REPO_DIR"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

echo "📋 Copying gpu_server/ → $WORKDIR"
cp -f "$REPO_DIR/gpu_server/server.py"        "$WORKDIR/server.py"
cp -f "$REPO_DIR/gpu_server/requirements.txt" "$WORKDIR/requirements.txt"

echo "📋 Copying ctd/ → $CTDDIR"
# Wipe stale modules from previous deployments
rm -f "$CTDDIR/detector.py" "$CTDDIR/smart_clean.py"
cp -f "$REPO_DIR/ctd/__init__.py"           "$CTDDIR/__init__.py"
cp -f "$REPO_DIR/ctd/ctd_onnx_detector.py"  "$CTDDIR/ctd_onnx_detector.py"
cp -f "$REPO_DIR/ctd/lama_inpaint.py"       "$CTDDIR/lama_inpaint.py"
cp -f "$REPO_DIR/ctd/sd_inpaint.py"         "$CTDDIR/sd_inpaint.py"
cp -f "$REPO_DIR/ctd/easyocr_enhancer.py"   "$CTDDIR/easyocr_enhancer.py"

echo ""
echo "=== 2/4 Installing Python dependencies ==="
pip install --quiet --upgrade -r "$WORKDIR/requirements.txt"

echo ""
echo "=== 3/4 Restarting server.py ==="
pkill -f "python.*server.py" 2>/dev/null || true
sleep 2

cd "$WORKDIR"
mkdir -p /workspace/logs
PORT="${SERVER_PORT:-7860}"
nohup python server.py --port "$PORT" > /workspace/logs/server.log 2>&1 &
SERVER_PID=$!
echo "✅ server.py started (PID=$SERVER_PID), logs → /workspace/logs/server.log"

sleep 5
echo ""
echo "=== 4/4 Health check ==="
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
echo "ℹ️  comictextdetector.pt.onnx auto-downloads (~75MB) on first /clean call."
echo "ℹ️  SD model auto-downloads (~4GB) on first engine=sd call."
