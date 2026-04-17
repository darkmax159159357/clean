import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

_readers = {}
_reader_failed = set()

LANG_GROUPS = [
    ('ja_en', ['ja', 'en']),
    ('ko_en', ['ko', 'en']),
    ('ch_en', ['ch_sim', 'en']),
]


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_easyocr_reader(group_key: str, langs: list):
    global _readers, _reader_failed
    if group_key in _reader_failed:
        return None
    if group_key in _readers:
        return _readers[group_key]
    try:
        import warnings
        import easyocr
        use_gpu = _has_cuda()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            reader = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
        _readers[group_key] = reader
        device_str = "GPU" if use_gpu else "CPU"
        logger.info(f"EasyOCR reader loaded ({group_key}: {langs}, {device_str})")
        return reader
    except Exception as e:
        logger.error(f"Failed to load EasyOCR reader {group_key}: {e}", exc_info=True)
        _reader_failed.add(group_key)
        return None


def _detect_with_reader(reader, image_rgb: np.ndarray, w: int, h: int,
                         scale: float, confidence_thresh: float) -> List[Tuple[int, int, int, int]]:
    try:
        results = reader.readtext(image_rgb, paragraph=False)
    except Exception as e:
        logger.error(f"EasyOCR readtext error: {e}")
        return []

    boxes = []
    for (bbox, text, conf) in results:
        if conf < confidence_thresh:
            continue
        if len(text.strip()) < 1:
            continue

        pts = np.array(bbox)
        x1 = int(pts[:, 0].min() / scale)
        y1 = int(pts[:, 1].min() / scale)
        x2 = int(pts[:, 0].max() / scale)
        y2 = int(pts[:, 1].max() / scale)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) > 5 and (y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _boxes_overlap(a: Tuple, b: Tuple, threshold: float = 0.5) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    min_area = min(area_a, area_b)
    return (inter / max(min_area, 1)) > threshold


def _deduplicate_boxes(all_boxes: List[Tuple]) -> List[Tuple]:
    if not all_boxes:
        return []
    unique = [all_boxes[0]]
    for box in all_boxes[1:]:
        is_dup = False
        for existing in unique:
            if _boxes_overlap(box, existing, 0.5):
                is_dup = True
                break
        if not is_dup:
            unique.append(box)
    return unique


def easyocr_detect_text_boxes(image_bgr: np.ndarray,
                                confidence_thresh: float = 0.15) -> List[Tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    max_dim = 2048
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    all_boxes = []
    loaded_count = 0

    for group_key, langs in LANG_GROUPS:
        reader = get_easyocr_reader(group_key, langs)
        if reader is None:
            continue
        loaded_count += 1
        boxes = _detect_with_reader(reader, image_rgb, w, h, scale, confidence_thresh)
        all_boxes.extend(boxes)

    if loaded_count == 0:
        logger.warning("No EasyOCR readers available")
        return []

    unique_boxes = _deduplicate_boxes(all_boxes)
    logger.info(f"EasyOCR detected {len(unique_boxes)} unique text regions "
                f"({len(all_boxes)} raw from {loaded_count} readers)")
    return unique_boxes


def enhance_mask_with_easyocr(image_bgr: np.ndarray, ctd_mask: np.ndarray,
                                padding: int = 8, dilate_size: int = 5) -> np.ndarray:
    easyocr_boxes = easyocr_detect_text_boxes(image_bgr)
    if not easyocr_boxes:
        return ctd_mask

    h, w = ctd_mask.shape[:2]
    enhanced_mask = ctd_mask.copy()
    new_regions = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

    for (x1, y1, x2, y2) in easyocr_boxes:
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)

        existing_coverage = ctd_mask[py1:py2, px1:px2]
        coverage_ratio = np.sum(existing_coverage > 127) / max(existing_coverage.size, 1)

        if coverage_ratio > 0.3:
            continue

        region = image_bgr[py1:py2, px1:px2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        bg_mean = float(np.mean(gray))
        if bg_mean > 180:
            thresh_val = max(50, int(bg_mean - 80))
            _, text_region = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        elif bg_mean < 80:
            thresh_val = min(200, int(bg_mean + 80))
            _, text_region = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            text_region = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 10
            )

        text_region = cv2.dilate(text_region, kernel, iterations=1)

        text_pixel_ratio = np.sum(text_region > 127) / max(text_region.size, 1)
        if text_pixel_ratio < 0.02 or text_pixel_ratio > 0.85:
            continue

        enhanced_mask[py1:py2, px1:px2] = np.maximum(
            enhanced_mask[py1:py2, px1:px2], text_region)
        new_regions += 1

    if new_regions > 0:
        logger.info(f"EasyOCR enhanced mask: added {new_regions} new text regions")

    return enhanced_mask
