#!/usr/bin/env python3
"""
GPU Cleaning Server for LALA Bot
Deploy on vast.ai with RTX 4070 (12GB) or similar GPU.

Usage:
  pip install -r requirements.txt
  Copy models/ folder (rtdetr/, comic-text-segmenter-yolov8m.pt, anime-manga-big-lama.pt)
  Copy ctd/ folder
  python server.py --port 7860

The bot connects via GPU_CLEAN_SERVER_URL env var.
"""

import os
import sys
import io
import cv2
import time
import numpy as np
import torch
import logging
import zipfile
import struct
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from typing import List, Optional
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class _Filter404Noise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "404" in msg and ("/portal-resolver" in msg or "/favicon" in msg or "/robots.txt" in msg):
            return False
        return True


logging.getLogger("uvicorn.access").addFilter(_Filter404Noise())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="LALA GPU Clean Server")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {DEVICE}")

_detector = None
_lama_model = None


def get_detector():
    global _detector
    if _detector is None:
        from ctd import ComicTextDetector
        _detector = ComicTextDetector.get_instance(device=DEVICE)
        logger.info(f"CTD loaded on {DEVICE}")
    return _detector


def get_lama():
    global _lama_model
    if _lama_model is None:
        from ctd.lama_inpaint import LAMA_MODEL_PATH
        _lama_model = torch.jit.load(LAMA_MODEL_PATH, map_location=DEVICE)
        _lama_model.eval()
        logger.info(f"LaMa loaded on {DEVICE} (fp32 — required for FFT layers)")
    return _lama_model


CHUNK_MAX_HEIGHT = 2000
INPAINT_MAX_SIZE = 1024


def _pad_to_multiple(val, multiple):
    return val if val % multiple == 0 else val + (multiple - val % multiple)


@torch.no_grad()
def _run_lama_gpu(model, img_bgr, msk):
    pad_size = 8
    h, w = img_bgr.shape[:2]
    new_h = _pad_to_multiple(h, pad_size)
    new_w = _pad_to_multiple(w, pad_size)

    img = img_bgr.copy()
    m = msk.copy()
    if new_h != h or new_w != w:
        img = cv2.resize(img, (new_w, new_h))
        m = cv2.resize(m, (new_w, new_h))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float().to(DEVICE) / 255.0
    mask_t[mask_t < 0.5] = 0
    mask_t[mask_t >= 0.5] = 1

    img_t = img_t * (1 - mask_t)
    result_t = model(img_t, mask_t)

    result = (result_t.float().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if new_h != h or new_w != w:
        result = cv2.resize(result, (w, h))
    return result


def lama_inpaint_gpu(image_bgr, mask_gray):
    model = get_lama()
    orig_h, orig_w = image_bgr.shape[:2]

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    expanded_mask = cv2.dilate(mask_gray, dilate_kernel, iterations=2)

    if max(orig_h, orig_w) <= INPAINT_MAX_SIZE:
        result = _run_lama_gpu(model, image_bgr, expanded_mask)
    else:
        scale = INPAINT_MAX_SIZE / max(orig_h, orig_w)
        small_img = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(expanded_mask, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_NEAREST)
        small_result = _run_lama_gpu(model, small_img, small_mask)
        result = cv2.resize(small_result, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    feather = cv2.dilate(expanded_mask, feather_kernel, iterations=2)
    feather = cv2.GaussianBlur(feather, (11, 11), 3.0)
    feather_f = feather.astype(np.float32) / 255.0
    mask_3 = np.stack([feather_f] * 3, axis=-1)

    final = (result * mask_3 + image_bgr * (1 - mask_3)).astype(np.uint8)
    return final


def _find_smart_split_points(image_bgr, max_height=CHUNK_MAX_HEIGHT, search_margin=400):
    h, w = image_bgr.shape[:2]
    if h <= max_height:
        return [(0, h)]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    row_density = np.mean(edges > 0, axis=1)

    chunks = []
    pos = 0

    while pos < h:
        if h - pos <= max_height:
            chunks.append((pos, h))
            break

        ideal = pos + max_height
        search_start = max(pos + max_height - search_margin, pos + int(max_height * 0.5))
        search_end = min(pos + max_height + search_margin, h)
        window = row_density[search_start:search_end]

        if len(window) == 0:
            chunks.append((pos, min(pos + max_height, h)))
            pos = min(pos + max_height, h)
            continue

        kernel_size = min(21, len(window))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size >= 3:
            smoothed = np.convolve(window, np.ones(kernel_size) / kernel_size, mode='same')
        else:
            smoothed = window

        best_offset = int(np.argmin(smoothed))
        split_y = search_start + best_offset

        remaining = h - split_y
        if remaining < max_height * 0.3:
            chunks.append((pos, h))
            break

        chunks.append((pos, split_y))
        pos = split_y

    return chunks


def _encode_debug_jpg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    out = image_bgr.copy()
    m = mask > 127
    if not np.any(m):
        return out
    out[m] = (out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def _collect_debug_stage(debug_files: list, image_bgr, raw_mask, enhanced_mask, inpaint_mask, blk_list):
    debug_files.append(("01_raw_mask.jpg", _encode_debug_jpg(raw_mask)))
    debug_files.append(("02_enhanced_mask.jpg", _encode_debug_jpg(enhanced_mask)))
    debug_files.append(("03_inpaint_mask.jpg", _encode_debug_jpg(inpaint_mask)))
    overlay_boxes = _overlay_mask(image_bgr, raw_mask, color=(0, 255, 0), alpha=0.45)
    for b in blk_list:
        x1, y1, x2, y2 = b.xyxy
        cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
    debug_files.append(("04_raw_overlay.jpg", _encode_debug_jpg(overlay_boxes)))
    debug_files.append(("05_enhanced_overlay.jpg", _encode_debug_jpg(_overlay_mask(image_bgr, enhanced_mask, color=(255, 128, 0), alpha=0.5))))
    debug_files.append(("06_inpaint_overlay.jpg", _encode_debug_jpg(_overlay_mask(image_bgr, inpaint_mask, color=(255, 0, 255), alpha=0.5))))


def _run_inpaint_engine(patch, engine, debug_files, patch_idx):
    img = patch['image']
    msk = patch['mask']

    if engine == 'lama':
        return lama_inpaint_gpu(img, msk)

    if engine == 'sd':
        from ctd.sd_inpaint import sd_inpaint
        try:
            return sd_inpaint(img, msk, device=DEVICE)
        except Exception as e:
            logger.error(f"SD inpaint failed on patch {patch_idx}: {e}", exc_info=True)
            return lama_inpaint_gpu(img, msk)

    if engine == 'compare':
        lama_out = lama_inpaint_gpu(img, msk)
        sd_out = None
        try:
            from ctd.sd_inpaint import sd_inpaint
            sd_out = sd_inpaint(img, msk, device=DEVICE)
        except Exception as e:
            logger.error(f"SD inpaint failed on patch {patch_idx}: {e}", exc_info=True)
        if debug_files is not None:
            debug_files.append((f"comparison/patch_{patch_idx:03d}_orig.jpg", _encode_debug_jpg(img)))
            debug_files.append((f"comparison/patch_{patch_idx:03d}_mask.jpg", _encode_debug_jpg(msk)))
            debug_files.append((f"comparison/patch_{patch_idx:03d}_lama.jpg", _encode_debug_jpg(lama_out)))
            if sd_out is not None:
                debug_files.append((f"comparison/patch_{patch_idx:03d}_sd.jpg", _encode_debug_jpg(sd_out)))
        return lama_out

    return lama_inpaint_gpu(img, msk)


def clean_image_gpu(image_bgr, debug_files=None, engine='lama'):
    from ctd.smart_clean import smart_clean, extract_inpaint_patches, apply_inpainted_patches
    from ctd.easyocr_enhancer import enhance_mask_with_easyocr

    detector = get_detector()
    h, w = image_bgr.shape[:2]

    if h <= CHUNK_MAX_HEIGHT:
        raw_text_mask, blk_list = detector.detect_for_cleaning(image_bgr, dilate_size=5, dilate_iter=2)
        text_mask = raw_text_mask
        try:
            text_mask = enhance_mask_with_easyocr(image_bgr, raw_text_mask)
        except Exception as e:
            logger.warning(f"EasyOCR enhance failed: {e}")
        if not blk_list and np.sum(text_mask > 127) == 0:
            if debug_files is not None:
                empty = np.zeros((h, w), dtype=np.uint8)
                _collect_debug_stage(debug_files, image_bgr, raw_text_mask, text_mask, empty, [])
            return image_bgr
        boxes = [tuple(b.xyxy) for b in blk_list]
        result, inpaint_mask, bubble_filled = smart_clean(image_bgr, text_mask, boxes)
        if debug_files is not None:
            _collect_debug_stage(debug_files, image_bgr, raw_text_mask, text_mask, inpaint_mask, blk_list)
        patches = extract_inpaint_patches(result, inpaint_mask, padding=60)
        if patches:
            inpainted = [_run_inpaint_engine(p, engine, debug_files, i) for i, p in enumerate(patches)]
            result = apply_inpainted_patches(result, patches, inpainted)
        return result

    chunks = _find_smart_split_points(image_bgr)
    logger.info(f"Split {w}x{h} → {len(chunks)} chunks")
    result = image_bgr.copy()

    full_raw_mask = np.zeros((h, w), dtype=np.uint8) if debug_files is not None else None
    full_enh_mask = np.zeros((h, w), dtype=np.uint8) if debug_files is not None else None
    full_inpaint_mask = np.zeros((h, w), dtype=np.uint8) if debug_files is not None else None
    all_blks = [] if debug_files is not None else None

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_img = result[chunk_start:chunk_end, :, :].copy()
        raw_text_mask, blk_list = detector.detect_for_cleaning(chunk_img, dilate_size=5, dilate_iter=2)
        text_mask = raw_text_mask
        try:
            text_mask = enhance_mask_with_easyocr(chunk_img, raw_text_mask)
        except Exception as e:
            logger.warning(f"EasyOCR enhance failed on chunk {chunk_idx+1}: {e}")

        if not blk_list and np.sum(text_mask > 127) == 0:
            continue

        boxes = [tuple(b.xyxy) for b in blk_list]
        cleaned, inpaint_mask, bubble_filled = smart_clean(chunk_img, text_mask, boxes)
        result[chunk_start:chunk_end, :, :] = cleaned

        if debug_files is not None:
            full_raw_mask[chunk_start:chunk_end, :] = np.maximum(full_raw_mask[chunk_start:chunk_end, :], raw_text_mask)
            full_enh_mask[chunk_start:chunk_end, :] = np.maximum(full_enh_mask[chunk_start:chunk_end, :], text_mask)
            full_inpaint_mask[chunk_start:chunk_end, :] = np.maximum(full_inpaint_mask[chunk_start:chunk_end, :], inpaint_mask)
            for b in blk_list:
                bx1, by1, bx2, by2 = b.xyxy
                shifted = type(b)([bx1, by1 + chunk_start, bx2, by2 + chunk_start], cls_name=b.cls_name)
                all_blks.append(shifted)

        patches = extract_inpaint_patches(cleaned, inpaint_mask, padding=60)
        if patches:
            for p in patches:
                px1, py1, px2, py2 = p['position']
                p['position'] = (px1, py1 + chunk_start, px2, py2 + chunk_start)
                p['image'] = result[py1 + chunk_start:py2 + chunk_start, px1:px2].copy()

            inpainted = [_run_inpaint_engine(p, engine, debug_files, chunk_idx * 100 + i) for i, p in enumerate(patches)]
            result = apply_inpainted_patches(result, patches, inpainted)
            logger.info(f"Chunk {chunk_idx+1}/{len(chunks)}: {len(boxes)} text → {len(patches)} patches inpainted ({engine})")

    if debug_files is not None and full_raw_mask is not None:
        _collect_debug_stage(debug_files, image_bgr, full_raw_mask, full_enh_mask, full_inpaint_mask, all_blks)

    return result


@app.get("/health")
async def health():
    gpu_name = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else "none"
    gpu_mem = 0
    if DEVICE == 'cuda':
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    return {
        "status": "ok",
        "device": DEVICE,
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_mem,
        "models_loaded": {
            "detector": _detector is not None,
            "lama": _lama_model is not None,
        }
    }


import asyncio
from concurrent.futures import ThreadPoolExecutor

_clean_executor = ThreadPoolExecutor(max_workers=4)


VALID_ENGINES = {"lama", "sd", "compare"}


def _normalize_engine(value: str) -> str:
    v = (value or "lama").strip().lower()
    return v if v in VALID_ENGINES else "lama"


def _sync_clean(img_bytes: bytes, img_name: str = "", debug: bool = False, engine: str = "lama"):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, img_name, []
    h, w = img_bgr.shape[:2]
    start = time.time()
    debug_files: list = [] if debug else None
    result_bgr = clean_image_gpu(img_bgr, debug_files=debug_files, engine=engine)
    _, jpg_data = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elapsed = time.time() - start
    logger.info(f"Cleaned {img_name} ({w}x{h}) engine={engine} in {elapsed:.2f}s{' [+debug]' if debug else ''}")
    return jpg_data.tobytes(), img_name, (debug_files or [])


@app.post("/clean")
async def clean_image(file: UploadFile = File(...), debug: bool = Form(False),
                      engine: str = Form("lama")):
    img_bytes = await file.read()
    loop = asyncio.get_event_loop()
    result, _, _ = await loop.run_in_executor(
        _clean_executor, _sync_clean, img_bytes, file.filename or "image",
        bool(debug), _normalize_engine(engine))
    if result is None:
        return Response(content=b"", status_code=400)
    return Response(content=result, media_type="image/jpeg")


def _sync_clean_batch(images_data: list, debug: bool = False, engine: str = "lama"):
    results = []
    batch_start = time.time()
    for img_bytes, img_name in images_data:
        try:
            result, name, dbg = _sync_clean(img_bytes, img_name, debug=debug, engine=engine)
            if result is not None:
                results.append((name, result, dbg))
            else:
                logger.warning(f"Failed to decode: {img_name}")
                results.append((img_name, None, []))
        except Exception as e:
            logger.error(f"Error cleaning {img_name}: {e}", exc_info=True)
            results.append((img_name, None, []))

    elapsed = time.time() - batch_start
    success = sum(1 for _, r, _ in results if r is not None)
    logger.info(f"Batch complete: {success}/{len(results)} images in {elapsed:.2f}s "
                f"({elapsed/max(len(results),1):.2f}s/img avg){' [+debug]' if debug else ''}")
    return results


@app.post("/clean_batch")
async def clean_batch(files: List[UploadFile] = File(...), debug: bool = Form(False),
                      engine: str = Form("lama")):
    if not files:
        return Response(content=b"", status_code=400)

    images_data = []
    for f in files:
        data = await f.read()
        images_data.append((data, f.filename or f"image_{len(images_data)}"))

    eng = _normalize_engine(engine)
    logger.info(f"Batch request: {len(images_data)} images (debug={debug}, engine={eng})")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(_clean_executor, _sync_clean_batch, images_data, bool(debug), eng)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_STORED) as zf:
        for name, data, dbg in results:
            if data is not None:
                stem = os.path.splitext(name)[0]
                zf.writestr(stem + "_cleaned.jpg", data)
                for dbg_name, dbg_bytes in dbg:
                    if dbg_bytes:
                        zf.writestr(f"_debug/{stem}/{dbg_name}", dbg_bytes)

    zip_buf.seek(0)
    return Response(content=zip_buf.getvalue(), media_type="application/zip",
                    headers={"X-Cleaned-Count": str(sum(1 for _, r, _ in results if r is not None)),
                             "X-Total-Count": str(len(results))})


@app.post("/clean_stream")
async def clean_stream(files: List[UploadFile] = File(...), debug: bool = Form(False),
                       engine: str = Form("lama")):
    if not files:
        return Response(content=b"", status_code=400)

    images_data = []
    for f in files:
        data = await f.read()
        images_data.append((data, f.filename or f"image_{len(images_data)}"))

    eng = _normalize_engine(engine)
    logger.info(f"Stream request: {len(images_data)} images (debug={debug}, engine={eng}) — streaming per image")

    async def gen():
        loop = asyncio.get_event_loop()
        t0 = time.time()
        success = 0
        for idx, (img_bytes, img_name) in enumerate(images_data):
            try:
                result, name, dbg = await loop.run_in_executor(
                    _clean_executor, _sync_clean, img_bytes, img_name, bool(debug), eng)
            except Exception as e:
                logger.error(f"Error cleaning {img_name}: {e}", exc_info=True)
                result, name, dbg = None, img_name, []

            name_bytes = name.encode('utf-8')
            if result is not None:
                chunk = (struct.pack('>I', len(name_bytes)) + name_bytes +
                         struct.pack('>I', len(result)) + result)
                yield chunk
                success += 1
                for dbg_name, dbg_bytes in dbg:
                    if not dbg_bytes:
                        continue
                    entry_name = f"{name}::_debug/{dbg_name}"
                    entry_bytes = entry_name.encode('utf-8')
                    yield (struct.pack('>I', len(entry_bytes)) + entry_bytes +
                           struct.pack('>I', len(dbg_bytes)) + dbg_bytes)
            else:
                yield (struct.pack('>I', len(name_bytes)) + name_bytes +
                       struct.pack('>I', 0))
        logger.info(f"Stream complete: {success}/{len(images_data)} in {time.time()-t0:.2f}s")

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Total-Count": str(len(images_data))})


GPU_ADMIN_TOKEN = os.environ.get("GPU_ADMIN_TOKEN", "")


@app.post("/admin/upload_file")
async def admin_upload_file(
    file: UploadFile = File(...),
    rel_path: str = Form(...),
    x_admin_token: Optional[str] = Header(None),
):
    if not GPU_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin endpoint disabled (GPU_ADMIN_TOKEN not set)")
    if x_admin_token != GPU_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    rel_path = rel_path.replace("..", "").lstrip("/\\")
    if not rel_path:
        raise HTTPException(status_code=400, detail="Empty rel_path")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.join(base, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = await file.read()
    with open(dest, "wb") as fh:
        fh.write(data)
    logger.info(f"[admin] Wrote {len(data)} bytes to {dest}")
    return {"ok": True, "path": rel_path, "bytes": len(data)}


@app.post("/admin/restart")
async def admin_restart(x_admin_token: Optional[str] = Header(None)):
    if not GPU_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin endpoint disabled")
    if x_admin_token != GPU_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    logger.warning("[admin] Restart requested — exiting process")

    async def _delayed_exit():
        await asyncio.sleep(0.5)
        os._exit(0)
    asyncio.ensure_future(_delayed_exit())
    return {"ok": True, "message": "Restarting"}


@app.on_event("startup")
async def startup():
    logger.info("Preloading models on GPU...")
    get_detector()
    get_lama()
    if DEVICE == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
    logger.info("Models ready!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
