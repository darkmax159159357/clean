"""
ComicTextDetector (zyddnys) — ONNX wrapper.

Replaces the previous custom rtdetr+yolov8 pipeline with the dual-head model
from https://github.com/dmMaze/comic-text-detector that powers
manga-image-translator and BallonsTranslator.

The model outputs a per-pixel text segmentation mask directly, so we no longer
need any of the ad-hoc bubble/mask scaffolding from smart_clean.py.

Public API:
    detect_text_mask(image_bgr, input_size=1024, mask_thresh=0.30)
        -> (mask_uint8 [H,W] {0,255}, boxes List[(x1,y1,x2,y2)])
"""
from __future__ import annotations

import logging
import os
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CTD_MODEL_URL = (
    "https://github.com/zyddnys/manga-image-translator/releases/"
    "download/beta-0.3/comictextdetector.pt.onnx"
)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTD_MODEL_PATH = os.environ.get(
    "CTD_ONNX_PATH",
    os.path.join(_PROJECT_ROOT, "models", "comictextdetector.pt.onnx"),
)

_session = None
_input_name = None
_output_names: List[str] = []


def _ensure_model_file() -> str:
    if os.path.exists(CTD_MODEL_PATH) and os.path.getsize(CTD_MODEL_PATH) > 1_000_000:
        return CTD_MODEL_PATH
    os.makedirs(os.path.dirname(CTD_MODEL_PATH), exist_ok=True)
    logger.info(f"Downloading comictextdetector.pt.onnx → {CTD_MODEL_PATH}")
    tmp = CTD_MODEL_PATH + ".part"
    urllib.request.urlretrieve(CTD_MODEL_URL, tmp)
    os.replace(tmp, CTD_MODEL_PATH)
    logger.info(f"Downloaded {os.path.getsize(CTD_MODEL_PATH) / 1e6:.1f} MB")
    return CTD_MODEL_PATH


def _get_session():
    global _session, _input_name, _output_names
    if _session is not None:
        return _session

    path = _ensure_model_file()
    try:
        import onnxruntime as ort  # type: ignore
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(path, sess_options=sess_opts, providers=providers)
        _input_name = _session.get_inputs()[0].name
        _output_names = [o.name for o in _session.get_outputs()]
        logger.info(f"CTD-ONNX loaded with providers={_session.get_providers()}")
    except ImportError:
        logger.warning("onnxruntime not installed; falling back to cv2.dnn (CPU)")
        net = cv2.dnn.readNetFromONNX(path)
        _output_names = list(net.getUnconnectedOutLayersNames())
        _session = ("cv2", net)
    return _session


def _letterbox(img: np.ndarray, new_size: int = 1024,
               color: Tuple[int, int, int] = (114, 114, 114)
               ) -> Tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio, then bottom/right-pad to (new_size, new_size).

    Matches the convention used in dmMaze/comic-text-detector (top=0, left=0).
    Returns: padded_img, scale, dw_padding_right, dh_padding_bottom.
    """
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new_size - nw, new_size - nh
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if dw > 0 or dh > 0:
        img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh


def detect_text_mask(image_bgr: np.ndarray,
                     input_size: int = 1024,
                     mask_thresh: float = 0.30,
                     ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """Run comictextdetector on a BGR image and return a binary text mask.

    The returned mask is at the original image resolution, value 0 or 255.
    Boxes are best-effort YOLO outputs (currently unused by the cleaner; the
    pixel mask is what matters for inpainting).
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8), []

    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, dw, dh = _letterbox(img_rgb, input_size)

    blob = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0  # NCHW

    sess = _get_session()
    if isinstance(sess, tuple) and sess[0] == "cv2":
        net = sess[1]
        net.setInput(blob)
        outs = net.forward(_output_names)
    else:
        outs = sess.run(_output_names, {_input_name: blob})

    # Pick mask among outputs by shape (NCHW with C=1, H=W=input_size)
    mask_arr = None
    for o in outs:
        if o.ndim == 4 and o.shape[1] == 1 and o.shape[2] == input_size:
            mask_arr = o
            break
    if mask_arr is None:
        # Fallback: ordering documented as (blks, mask, lines_map)
        mask_arr = outs[1]

    mask = mask_arr.squeeze()
    if mask.ndim != 2:
        mask = mask.reshape(input_size, input_size)
    binary = (mask > mask_thresh).astype(np.uint8) * 255

    # Un-letterbox: crop the bottom/right padding then resize to original.
    nh, nw = input_size - dh, input_size - dw
    binary = binary[:nh, :nw]
    if (nw, nh) != (w, h):
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    return binary, []


def warmup():
    """Eagerly download the model and build the session."""
    _get_session()


__all__ = ["detect_text_mask", "warmup", "CTD_MODEL_PATH", "CTD_MODEL_URL"]
