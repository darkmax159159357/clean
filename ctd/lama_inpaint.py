import os
import cv2
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

LAMA_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'models', 'anime-manga-big-lama.pt')

_lama_model = None


def get_lama_model():
    global _lama_model
    if _lama_model is None:
        if not os.path.exists(LAMA_MODEL_PATH):
            raise FileNotFoundError(f"LaMa model not found at {LAMA_MODEL_PATH}")
        logger.info(f"Loading LaMa inpainting model from {LAMA_MODEL_PATH}...")
        _lama_model = torch.jit.load(LAMA_MODEL_PATH, map_location='cpu')
        _lama_model.eval()
        logger.info("LaMa model loaded successfully")
    return _lama_model


INPAINT_MAX_SIZE = 2048


def _pad_to_multiple(val: int, multiple: int) -> int:
    return val if val % multiple == 0 else val + (multiple - val % multiple)


@torch.no_grad()
def _run_lama(model, img_bgr: np.ndarray, msk: np.ndarray) -> np.ndarray:
    pad_size = 8
    h, w = img_bgr.shape[:2]
    new_h = _pad_to_multiple(h, pad_size)
    new_w = _pad_to_multiple(w, pad_size)

    img = img_bgr.copy()
    m = msk.copy()
    if new_h != h or new_w != w:
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).float() / 255.0
    mask_t[mask_t < 0.5] = 0
    mask_t[mask_t >= 0.5] = 1

    img_t = img_t * (1 - mask_t)
    result_t = model(img_t, mask_t)

    result = (result_t.squeeze(0).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    if new_h != h or new_w != w:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def _gradient_blend(result_bgr: np.ndarray, image_bgr: np.ndarray,
                    mask_u8: np.ndarray) -> np.ndarray:
    """Per-component smooth blend that NEVER reads in-mask (text) pixels.

    For components whose surrounding ring shows a real color gradient, replace
    LaMa's flat patch with: cv2.inpaint(NS) of the original (a smooth color
    field interpolated from pixels OUTSIDE the dilated mask only) plus LaMa's
    residual high-frequency texture. For uniform/plain bubbles, keep LaMa
    output as-is. Because cv2.inpaint reads only unmasked pixels, no original
    text can leak back into the cleaned region."""
    H, W = image_bgr.shape[:2]
    final = image_bgr.copy()
    bin_mask = (mask_u8 > 127).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return final

    # Smooth color field interpolated ONLY from pixels outside the dilated mask.
    # Computed once per call; cheap relative to LaMa.
    bg_field = cv2.inpaint(image_bgr, bin_mask, 5, cv2.INPAINT_NS)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    pad = 8
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        ys_all, xs_all = np.where(labels == i)
        if area < 32:
            final[ys_all, xs_all] = result_bgr[ys_all, xs_all]
            continue
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        comp_mask = ((labels[y0:y1, x0:x1] == i).astype(np.uint8)) * 255
        if comp_mask.shape[0] < 4 or comp_mask.shape[1] < 4:
            final[ys_all, xs_all] = result_bgr[ys_all, xs_all]
            continue
        # Sample the ring just OUTSIDE the component to test for a gradient.
        ring_outer = cv2.dilate(comp_mask, np.ones((9, 9), np.uint8))
        ring_inner = cv2.dilate(comp_mask, np.ones((3, 3), np.uint8))
        ring = (ring_outer > 0) & (ring_inner == 0)
        roi_orig = image_bgr[y0:y1, x0:x1]
        if ring.sum() < 20:
            final[ys_all, xs_all] = result_bgr[ys_all, xs_all]
            continue
        ring_std = float(roi_orig[ring].astype(np.float32).std(axis=0).mean())
        if ring_std < 8.0:
            final[ys_all, xs_all] = result_bgr[ys_all, xs_all]
            continue

        # Combine smooth external gradient (bg_field) with LaMa's high-freq
        # residual to keep any subtle texture LaMa produced. Sigma scales with
        # component size so the low-frequency split matches the bubble.
        roi_lama = result_bgr[y0:y1, x0:x1].astype(np.float32)
        roi_bg = bg_field[y0:y1, x0:x1].astype(np.float32)
        size = max(w, h)
        sigma = max(3.0, size / 4.0)
        k = int(sigma * 4) | 1
        k = max(3, min(k, 51))
        lama_low = cv2.GaussianBlur(roi_lama, (k, k), sigma)
        merged = roi_bg + (roi_lama - lama_low)
        merged = np.clip(merged, 0, 255).astype(np.uint8)
        sel = comp_mask > 0
        final[y0:y1, x0:x1][sel] = merged[sel]
    return final


@torch.no_grad()
def lama_inpaint(image_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    model = get_lama_model()
    orig_h, orig_w = image_bgr.shape[:2]

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    expanded_mask = cv2.dilate(mask_gray, dilate_kernel, iterations=2)

    if max(orig_h, orig_w) <= INPAINT_MAX_SIZE:
        result = _run_lama(model, image_bgr, expanded_mask)
    else:
        scale = INPAINT_MAX_SIZE / max(orig_h, orig_w)
        small_img = cv2.resize(image_bgr, (int(orig_w * scale), int(orig_h * scale)),
                               interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(expanded_mask, (int(orig_w * scale), int(orig_h * scale)),
                                interpolation=cv2.INTER_NEAREST)
        small_result = _run_lama(model, small_img, small_mask)
        result = cv2.resize(small_result, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    blended = _gradient_blend(result, image_bgr, expanded_mask)

    feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    feather = cv2.dilate(expanded_mask, feather_kernel, iterations=1)
    feather = cv2.GaussianBlur(feather, (11, 11), 3.0)
    feather_f = feather.astype(np.float32) / 255.0
    mask_3 = np.stack([feather_f] * 3, axis=-1)

    final = (blended * mask_3 + image_bgr * (1 - mask_3)).astype(np.uint8)
    return final


def inpaint_patches(patches: list) -> list:
    results = []
    for i, p in enumerate(patches):
        try:
            out = lama_inpaint(p['image'], p['mask'])
            results.append(out)
            logger.debug(f"Inpainted patch {i} ({p['image'].shape[1]}x{p['image'].shape[0]})")
        except Exception as e:
            logger.error(f"Failed to inpaint patch {i}: {e}")
            results.append(p['image'])
    return results
