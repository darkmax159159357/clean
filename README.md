# LALA Clean Server (GPU)

  FastAPI server for manga/comic text cleaning (تبييض). Includes pre-downloaded ML models.

  ## Contents
  - `gpu_server/` — FastAPI server (port 7860) + setup scripts
  - `ctd/` — ML pipeline: ComicTextDetector + LaMa + EasyOCR + optional SD inpaint
  - `models/` — pre-downloaded weights (tracked via Git LFS, ~600 MB)

  ## Install (vast.ai, one time)

  ```bash
  cd /workspace
  apt-get update && apt-get install -y git-lfs && git lfs install
  git clone https://github.com/darkmax159159357/clean.git lala-clean
  cd lala-clean/gpu_server
  pip install -r requirements.txt
  mkdir -p /workspace/logs
  nohup python server.py --port 7860 > /workspace/logs/server.log 2>&1 &
  sleep 5 && tail -30 /workspace/logs/server.log
  ```

  ## Update

  ```bash
  cd /workspace/lala-clean && git pull
  pkill -f "python.*server.py"
  cd gpu_server && nohup python server.py --port 7860 > /workspace/logs/server.log 2>&1 &
  ```
  