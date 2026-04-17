#!/bin/bash
set -e

echo "========================================="
echo "  LALA Bot GPU Server Setup"
echo "  RTX 4070 (12GB) recommended"
echo "========================================="

echo ""
echo "[1/4] Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[2/4] Checking models directory..."

MODELS_DIR="../models"

if [ ! -d "$MODELS_DIR/rtdetr" ]; then
    echo "ERROR: models/rtdetr/ not found!"
    echo "Copy the rtdetr/ folder with config.json, model.safetensors, preprocessor_config.json"
    exit 1
fi

if [ ! -f "$MODELS_DIR/comic-text-segmenter-yolov8m.pt" ]; then
    echo "ERROR: models/comic-text-segmenter-yolov8m.pt not found!"
    exit 1
fi

if [ ! -f "$MODELS_DIR/anime-manga-big-lama.pt" ]; then
    echo "ERROR: models/anime-manga-big-lama.pt not found!"
    exit 1
fi

echo "All models found:"
echo "  - rtdetr/ (RT-DETR v2 text detector)"
echo "  - comic-text-segmenter-yolov8m.pt (YOLOv8 segmenter)"
echo "  - anime-manga-big-lama.pt (LaMa inpainter)"

echo ""
echo "[3/4] Checking GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  GPU: {gpu} ({mem:.1f} GB)')
else:
    print('  WARNING: No GPU detected! Server will run on CPU (slow)')
"

echo ""
echo "[4/4] Setup complete!"
echo ""
echo "========================================="
echo "  To start the server:"
echo "  python server.py --port 7860"
echo ""
echo "  Then set in your bot's environment:"
echo "  GPU_CLEAN_SERVER_URL=http://YOUR_IP:7860"
echo "========================================="
