import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

MASK_DILATE_SIZE = 7
MASK_DILATE_ITER = 3
BUBBLE_STD_THRESHOLD = 40.0
BUBBLE_BG_RATIO_MIN = 0.25


def _is_simple_background(image_bgr: np.ndarray, text_mask: np.ndarray,
                           x1: int, y1: int, x2: int, y2: int) -> Optional[dict]:
    if y2 - y1 < 10 or x2 - x1 < 10:
        return None

    inner_mask = text_mask[y1:y2, x1:x2]
    inner_region = image_bgr[y1:y2, x1:x2]

    bg_pixels = inner_region[inner_mask < 127]
    if len(bg_pixels) < 20:
        return None

    bg_ratio = len(bg_pixels) / max(inner_mask.size, 1)
    if bg_ratio < BUBBLE_BG_RATIO_MIN:
        return None

    bg_std = float(np.mean(np.std(bg_pixels.astype(np.float32), axis=0)))
    if bg_std > BUBBLE_STD_THRESHOLD:
        return None

    mean_color = np.mean(bg_pixels, axis=0).astype(np.uint8)

    return {
        'mean_color': mean_color,
        'bg_std': bg_std,
    }


def _fill_simple_region(image_bgr: np.ndarray, text_mask: np.ndarray,
                        fill_info: dict, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    result = image_bgr.copy()
    fill_color = fill_info['mean_color']

    region_mask = text_mask[y1:y2, x1:x2].copy()

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_SIZE, MASK_DILATE_SIZE))
    region_mask = cv2.dilate(region_mask, dilate_kernel, iterations=MASK_DILATE_ITER)

    fill_f = cv2.GaussianBlur(region_mask, (7, 7), 2.0).astype(np.float32) / 255.0

    region = result[y1:y2, x1:x2]
    for c in range(3):
        region[:, :, c] = (
            float(fill_color[c]) * fill_f +
            region[:, :, c].astype(np.float32) * (1 - fill_f)
        ).astype(np.uint8)
    result[y1:y2, x1:x2] = region

    return result


def smart_clean(image_bgr: np.ndarray, text_mask: np.ndarray,
                boxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    h, w = image_bgr.shape[:2]

    result = image_bgr.copy()
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    handled_mask = np.zeros((h, w), dtype=np.uint8)
    bubble_filled_indices = []

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_SIZE, MASK_DILATE_SIZE))

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        fill_info = _is_simple_background(result, text_mask, x1, y1, x2, y2)
        if fill_info is not None:
            result = _fill_simple_region(result, text_mask, fill_info, x1, y1, x2, y2)
            bubble_filled_indices.append(idx)
            handled_mask[y1:y2, x1:x2] = 255
            logger.debug(f"Box {idx}: bubble fill (std={fill_info['bg_std']:.1f}, "
                        f"color={fill_info['mean_color']})")
            continue

        pad = 10
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)

        region_mask = text_mask[ry1:ry2, rx1:rx2].copy()
        region_mask = cv2.dilate(region_mask, dilate_kernel, iterations=MASK_DILATE_ITER)

        inpaint_mask[ry1:ry2, rx1:rx2] = np.maximum(
            inpaint_mask[ry1:ry2, rx1:rx2], region_mask)
        handled_mask[ry1:ry2, rx1:rx2] = 255

    orphan_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(handled_mask))
    orphan_pixels = int(np.sum(orphan_mask > 127))
    orphan_regions = 0
    if orphan_pixels > 30:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (orphan_mask > 127).astype(np.uint8), 4, cv2.CV_32S)
        orphan_add = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:
                continue
            orphan_add[labels == i] = 255
            orphan_regions += 1
        if orphan_regions > 0:
            orphan_add = cv2.dilate(orphan_add, dilate_kernel, iterations=MASK_DILATE_ITER)
            inpaint_mask = np.maximum(inpaint_mask, orphan_add)

    if not boxes and orphan_regions == 0:
        mask_pixels = np.sum(text_mask > 127)
        if mask_pixels > 50:
            inpaint_mask = cv2.dilate(text_mask, dilate_kernel, iterations=MASK_DILATE_ITER)

    total_regions = len(boxes) + orphan_regions if boxes or orphan_regions else (1 if np.sum(inpaint_mask > 127) > 0 else 0)
    bubble_count = len(bubble_filled_indices)
    lama_count = total_regions - bubble_count
    logger.info(f"Smart clean: {total_regions} text regions "
                f"({bubble_count} bubble-filled, {lama_count} for inpaint, "
                f"{orphan_regions} orphan mask regions)")

    return result, inpaint_mask, bubble_filled_indices


def extract_inpaint_patches(image_bgr: np.ndarray, inpaint_mask: np.ndarray,
                           padding: int = 50) -> List[dict]:
    if inpaint_mask.sum() == 0:
        return []

    h, w = image_bgr.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inpaint_mask, 4, cv2.CV_16U)

    raw_patches = []
    for i in range(1, num_labels):
        x, y, pw, ph, area = stats[i]
        if area < 10:
            continue
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(w, x + pw + padding)
        py2 = min(h, y + ph + padding)
        raw_patches.append((px1, py1, px2, py2))

    merged = _merge_patch_boxes(raw_patches)

    patches = []
    for (px1, py1, px2, py2) in merged:
        patch_img = image_bgr[py1:py2, px1:px2].copy()
        patch_mask = inpaint_mask[py1:py2, px1:px2].copy()
        patches.append({
            'image': patch_img,
            'mask': patch_mask,
            'position': (px1, py1, px2, py2),
        })

    return patches


def _merge_patch_boxes(boxes: List[tuple], overlap_thresh: float = 0.3) -> List[tuple]:
    if not boxes:
        return []
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used:
                continue
            x1, y1, x2, y2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                bx1, by1, bx2, by2 = boxes[j]
                ix1 = max(x1, bx1)
                iy1 = max(y1, by1)
                ix2 = min(x2, bx2)
                iy2 = min(y2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used.add(j)
                    merged = True
            new_boxes.append((x1, y1, x2, y2))
            used.add(i)
        boxes = new_boxes
    return boxes


def apply_inpainted_patches(image_bgr: np.ndarray, patches: List[dict],
                           inpainted_patches: List[np.ndarray]) -> np.ndarray:
    result = image_bgr.copy()
    for patch_info, inpainted in zip(patches, inpainted_patches):
        px1, py1, px2, py2 = patch_info['position']
        mask = patch_info['mask']

        if inpainted.shape[:2] != (py2 - py1, px2 - px1):
            inpainted = cv2.resize(inpainted, (px2 - px1, py2 - py1))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blend_mask = cv2.dilate(mask, kernel, iterations=2)
        blend_mask = cv2.GaussianBlur(blend_mask, (11, 11), 3)
        blend_f = blend_mask.astype(np.float32) / 255.0

        for c in range(3):
            result[py1:py2, px1:px2, c] = (
                inpainted[:, :, c] * blend_f +
                result[py1:py2, px1:px2, c] * (1 - blend_f)
            ).astype(np.uint8)

    return result
