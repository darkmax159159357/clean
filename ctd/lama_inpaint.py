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

    feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    feather = cv2.dilate(expanded_mask, feather_kernel, iterations=2)
    feather = cv2.GaussianBlur(feather, (11, 11), 3.0)
    feather_f = feather.astype(np.float32) / 255.0
    mask_3 = np.stack([feather_f] * 3, axis=-1)

    final = (result * mask_3 + image_bgr * (1 - mask_3)).astype(np.uint8)
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
