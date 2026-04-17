# GPU Server Deployment Guide

## Quick Start (vast.ai)

### 1. Rent a GPU
- **Recommended**: RTX 4070 (12GB VRAM) — $0.080/hr
- **Budget**: RTX 3060 Ti (8GB VRAM) — $0.056/hr
- Template: PyTorch (Vast)
- Container Size: 16GB+

### 2. Upload Files to Server
```bash
# From your local machine, upload these to the vast.ai instance:
scp -r gpu_server/ vast-server:/workspace/gpu_server/
scp -r ctd/ vast-server:/workspace/ctd/
scp -r models/ vast-server:/workspace/models/
```

Required files:
```
/workspace/
├── gpu_server/
│   ├── server.py
│   ├── requirements.txt
│   └── setup.sh
├── ctd/
│   ├── __init__.py
│   ├── detector.py
│   ├── smart_clean.py
│   ├── lama_inpaint.py
│   ├── easyocr_enhancer.py
│   ├── textmask.py
│   ├── textblock.py
│   ├── common.py
│   ├── model.py
│   ├── utils.py
│   └── yolo.py
└── models/
    ├── rtdetr/
    │   ├── config.json
    │   ├── model.safetensors
    │   └── preprocessor_config.json
    ├── comic-text-segmenter-yolov8m.pt
    └── anime-manga-big-lama.pt
```

### 3. Install & Run
```bash
cd /workspace/gpu_server
bash setup.sh
python server.py --port 7860
```

### 4. Configure Bot
Set environment variable in Replit:
```
GPU_CLEAN_SERVER_URL=http://YOUR_VAST_IP:7860
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + GPU info |
| `/clean` | POST | Clean single image |
| `/clean_batch` | POST | Clean multiple images, return ZIP |
| `/clean_stream` | POST | Clean multiple images, return binary stream |

### 5. Expose via Cloudflare Tunnel (free public HTTPS)
```bash
# Install cloudflared
curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared

# In a second terminal (keep server.py running in the first):
./cloudflared tunnel --url http://localhost:7860
```
You will get a URL like `https://xxxx-xxxx.trycloudflare.com`. Put it in Replit as `GPU_CLEAN_SERVER_URL`.

## Performance (RTX 4070)

| Metric | Expected |
|--------|----------|
| Single image (1000x1500) | ~0.3-0.5s |
| Batch 20 images | ~6-10s |
| Chapter (20 pages) | ~10-15s |
| 15 chapters concurrent | ~40-60s |
