"""Stable Diffusion 1.5 inpainting wrapper for comparison vs LaMa.

Lazy-loads the SD inpaint pipeline on first use. Intended as an OPTIONAL engine
alongside LaMa — much slower (~3-5s/patch vs 150ms for LaMa) and much larger
(5.2GB model). Use only when comparing or experimenting with SFX-on-art cases.
"""
import logging
import os
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"
SD_HF_REPO = "runwayml/stable-diffusion-inpainting"
SD_HF_FILENAME = "sd-v1-5-inpainting.ckpt"
SD_DEFAULT_DIR = "/workspace/models/sd-inpaint"
SD_MODEL_PATH = os.environ.get("SD_MODEL_PATH", "").strip()


def _ensure_sd_checkpoint() -> str:
    """Return a local path to the SD inpaint .ckpt, downloading it once if missing.

    Priority:
      1. SD_MODEL_PATH env var (if file exists, use as-is).
      2. Auto-download into /workspace/models/sd-inpaint/ (persistent on vast.ai).
    """
    if SD_MODEL_PATH and os.path.isfile(SD_MODEL_PATH):
        return SD_MODEL_PATH

    target_dir = os.path.dirname(SD_MODEL_PATH) if SD_MODEL_PATH else SD_DEFAULT_DIR
    target_file = os.path.basename(SD_MODEL_PATH) if SD_MODEL_PATH else SD_HF_FILENAME
    os.makedirs(target_dir, exist_ok=True)
    local_path = os.path.join(target_dir, target_file)

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1_000_000_000:
        return local_path

    logger.info(f"SD checkpoint not found — downloading from HuggingFace ({SD_HF_REPO}/{SD_HF_FILENAME}) → {local_path}")
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=SD_HF_REPO,
            filename=SD_HF_FILENAME,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"✅ SD checkpoint downloaded: {cached} ({os.path.getsize(cached)/1e9:.2f} GB)")
        return cached
    except Exception as e:
        logger.warning(f"hf_hub_download failed ({e}); falling back to direct wget-style stream")
        import urllib.request
        url = f"https://huggingface.co/{SD_HF_REPO}/resolve/main/{SD_HF_FILENAME}"
        tmp_path = local_path + ".part"
        urllib.request.urlretrieve(url, tmp_path)
        os.rename(tmp_path, local_path)
        logger.info(f"✅ SD checkpoint downloaded via urllib: {local_path}")
        return local_path


SD_TRAIN_RES = 512
SD_NUM_STEPS = int(os.environ.get("SD_NUM_STEPS", "28"))
SD_GUIDANCE = float(os.environ.get("SD_GUIDANCE", "3.5"))
SD_STRENGTH = float(os.environ.get("SD_STRENGTH", "0.55"))

# Positive prompt: a single coherent sentence describing what should be in the
# masked area. SD 1.5 is NOT an instruction-following model (it cannot obey
# "remove the text" — that requires InstructPix2Pix). We instead describe the
# scene we want: an empty, clean continuation of the surrounding manga art
# where the speech bubble used to hold dialogue. Written as natural prose so
# the text encoder reads a unified scene rather than disconnected tags.
SD_PROMPT = os.environ.get(
    "SD_PROMPT",
    "an empty manga speech bubble with a clean blank interior where the "
    "dialogue text used to be, the surrounding screentone shading, ink line "
    "art and halftone dots continuing smoothly into the area, monochrome "
    "black and white manga page, professional published manga illustration, "
    "sharp inked lines, untouched original artwork"
).strip()

# Negative prompt: aggressively forbid everything that would qualify as
# "added content". SD 1.5 ignores standalone negations like "no text" inside
# the positive prompt, but takes negative_prompt seriously.
SD_NEG_PROMPT = os.environ.get(
    "SD_NEG_PROMPT",
    # text artifacts (the whole point of the cleaner)
    "text, letters, characters, kanji, hiragana, katakana, hangul, chinese, "
    "japanese text, korean text, words, writing, calligraphy, typography, "
    "font, caption, subtitle, label, logo, watermark, signature, stamp, "
    "speech bubble, dialogue, sound effect, sfx, onomatopoeia, "
    # invented content
    "new character, additional character, extra person, person, people, "
    "face, eyes, mouth, hand, finger, hair, body, figure, silhouette, "
    "creature, animal, monster, object, item, weapon, building, vehicle, "
    "plant, flower, tree, "
    # quality
    "blurry, smudged, smear, low quality, jpeg artifacts, compression "
    "artifacts, noise, grain, distortion, deformed, ghosting, double image, "
    "color bleed, oversaturated, washed out, halo, glow, "
    # style drift
    "photo, photograph, photorealistic, 3d render, painting, oil painting, "
    "watercolor, color illustration, colored, full color"
).strip()

_sd_pipe = None


def get_sd_pipeline(device: str = 'cuda'):
    global _sd_pipe
    if _sd_pipe is None:
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError as e:
            raise RuntimeError(
                "diffusers not installed. Add `diffusers` and `accelerate` to requirements.txt"
            ) from e

        dtype = torch.float16 if device == 'cuda' else torch.float32

        ckpt_path = _ensure_sd_checkpoint()
        logger.info(f"Loading SD inpaint pipeline from local file: {ckpt_path}")
        _sd_pipe = StableDiffusionInpaintPipeline.from_single_file(
            ckpt_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
            load_safety_checker=False,
        )
        _sd_pipe.to(device)
        try:
            _sd_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        _sd_pipe.set_progress_bar_config(disable=True)

        _sd_pipe.safety_checker = None
        _sd_pipe.requires_safety_checker = False
        try:
            _sd_pipe.run_safety_checker = lambda image, device, dtype: (image, [False] * len(image))
        except Exception:
            pass

        logger.info(f"SD inpaint pipeline loaded on {device} (dtype={dtype}) — safety checker disabled")
    return _sd_pipe


def _compute_sd_size(h: int, w: int, target: int = SD_TRAIN_RES) -> tuple:
    scale = target / max(h, w)
    new_h = max(64, int(round(h * scale)) // 8 * 8)
    new_w = max(64, int(round(w * scale)) // 8 * 8)
    return new_h, new_w


@torch.no_grad()
def sd_inpaint(image_bgr: np.ndarray, mask_gray: np.ndarray,
               device: str = 'cuda') -> np.ndarray:
    """Run SD 1.5 inpainting on a single patch. Returns same shape as input."""
    pipe = get_sd_pipeline(device)
    orig_h, orig_w = image_bgr.shape[:2]

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    expanded_mask = cv2.dilate(mask_gray, dilate_kernel, iterations=3)

    new_h, new_w = _compute_sd_size(orig_h, orig_w)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_small = cv2.resize(expanded_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    from PIL import Image
    img_pil = Image.fromarray(img_small)
    mask_pil = Image.fromarray(mask_small)

    out = pipe(
        prompt=SD_PROMPT,
        negative_prompt=SD_NEG_PROMPT,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=SD_NUM_STEPS,
        guidance_scale=SD_GUIDANCE,
        strength=SD_STRENGTH,
        height=new_h,
        width=new_w,
    ).images[0]

    out_rgb = np.array(out)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    if (new_h, new_w) != (orig_h, orig_w):
        out_bgr = cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    feather_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    feather = cv2.dilate(expanded_mask, feather_kernel, iterations=1)
    feather = cv2.GaussianBlur(feather, (11, 11), 3.0)
    feather_f = feather.astype(np.float32) / 255.0
    mask_3 = np.stack([feather_f] * 3, axis=-1)

    final = (out_bgr * mask_3 + image_bgr * (1 - mask_3)).astype(np.uint8)
    return final
