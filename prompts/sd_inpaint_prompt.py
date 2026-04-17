"""
SD 1.5 INPAINTING PROMPT — MANGA TEXT REMOVAL (PROFESSIONAL)
=============================================================

PURPOSE
-------
Build the positive + negative prompt that drives Stable Diffusion 1.5
inpainting when the cleaner is asked to ERASE Japanese / Korean / Chinese
text from a manga, manhwa, or manhua page and REDRAW the original artwork
that used to sit underneath it.

CORE PHILOSOPHY (READ FIRST)
----------------------------
1. SD 1.5 is NOT instruction-following.
   - It cannot obey "remove the text", "delete this", "erase the dialogue".
   - The prompt describes WHAT SHOULD EXIST in the masked area, not what
     to do. Treat the prompt as a caption for the desired output, not a
     command to the model.

2. Token budget is hard-capped at 77 (CLIP).
   - Long flowery prose past token 77 is silently dropped.
   - Front-load the most important concepts; keep total length tight.

3. Negative prompt is where text-removal actually happens.
   - "text", "letters", "kanji", "speech bubble dialogue" in the negative
     prompt is the primary mechanism that prevents SD from re-drawing the
     original characters.
   - Without an aggressive negative prompt SD WILL hallucinate writing
     because the surrounding visual context (bubble + manga page) makes
     "text" the statistically expected content.

4. Two visual contexts → two different prompts.
   - INSIDE A BUBBLE (text on white/colored bubble interior):
     We want a clean, untouched bubble surface. Keep prompt minimal so SD
     paints flat continuation, not new artwork.
   - SFX OVER ARTWORK (text drawn on top of the scene):
     We want SD to continue the scene's line art, screentone, shading.
     Prompt must mention the surrounding art so SD blends, not invents.

5. Style guard.
   - Page may be monochrome (Japanese manga) OR full color (Korean manhwa,
     Chinese manhua). Wrong style guard = wrong palette = visible patch.
   - Detect the palette of the surrounding region and pick the matching
     style anchor.

PROMPT STRUCTURE (FOR EACH PROMPT WE BUILD)
-------------------------------------------
[ANCHOR] [SURFACE] [STYLE] [QUALITY]
- ANCHOR  = "clean continuation of the surrounding artwork"
- SURFACE = "empty bubble interior" OR "manga line art and screentone"
- STYLE   = "monochrome black and white" OR "full color manhwa"
- QUALITY = "professional published page, sharp inked lines, high detail"

NEGATIVE PROMPT STRUCTURE
-------------------------
[TEXT BAN] [CONTENT BAN] [STYLE BAN] [QUALITY BAN]
- TEXT BAN    = exhaustive list of writing systems, glyphs, sfx,
                speech indicators
- CONTENT BAN = no new characters/objects/creatures invented in the gap
- STYLE BAN   = no photo / 3d / watercolor / oil drift
- QUALITY BAN = no blur / smear / ghosting / artifacts

TUNING NOTES
------------
- guidance_scale ~3.5 : prompt heard but not dominant (1.0 = ignored, 7.5 = drift)
- strength ~0.55      : refinement, not generation (1.0 = full repaint, dangerous)
- steps 28            : enough for inpaint quality without latency tax
- All values overridable via env vars in ctd/sd_inpaint.py
"""
from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. POSITIVE PROMPT FRAGMENTS
# =============================================================================

# Inside a (mostly) plain speech bubble — keep it simple and FLAT.
POS_BUBBLE_INTERIOR = (
    "a clean empty speech bubble interior, smooth blank surface where the "
    "dialogue text used to be, untouched original bubble shape, "
    "professional published manga page, sharp clean edges, high detail"
)

# Text drawn ON TOP of the artwork (SFX, narration without a box, dialogue
# floating on a scene). We MUST mention the underlying art so SD continues
# it instead of painting a blank patch.
POS_OVER_ARTWORK_MONO = (
    "a clean continuation of the surrounding manga artwork, screentone "
    "halftone dots, ink line art, hatching and shading flowing smoothly "
    "into the area, monochrome black and white manga illustration, "
    "professional published manga page, sharp inked lines, high detail"
)

POS_OVER_ARTWORK_COLOR = (
    "a clean continuation of the surrounding manhwa artwork, soft cell "
    "shading, line art and color flat fills flowing smoothly into the "
    "area, full color digital manhwa illustration, professional published "
    "webtoon page, sharp clean lines, high detail"
)


# =============================================================================
# 2. NEGATIVE PROMPT (single canonical version, exhaustive)
# =============================================================================
#
# SD 1.5 negative prompts also share the 77-token CLIP cap. The list below is
# tuned to fit; do NOT casually extend it. Concepts appearing earlier carry
# more weight.

NEG_PROMPT = (
    # 1) TEXT BAN — the entire reason this pipeline exists
    "text, letters, characters, kanji, hiragana, katakana, hangul, "
    "chinese characters, japanese writing, korean writing, words, "
    "calligraphy, typography, font, caption, subtitle, label, logo, "
    "watermark, signature, stamp, dialogue, sfx, sound effect, onomatopoeia, "
    # 2) CONTENT BAN — do not invent people / objects in the gap
    "new character, extra person, face, eyes, mouth, hand, finger, hair, "
    "body, silhouette, creature, animal, monster, object, weapon, building, "
    "vehicle, plant, flower, "
    # 3) STYLE BAN — do not drift away from manga style
    "photo, photograph, photorealistic, 3d render, oil painting, "
    "watercolor, "
    # 4) QUALITY BAN — common SD failure modes
    "blurry, smudged, low quality, jpeg artifacts, noise, distortion, "
    "deformed, ghosting, double image, color bleed, halo, glow"
)


# =============================================================================
# 3. CONTEXT DETECTION (decide which positive prompt to use)
# =============================================================================

def _is_mostly_white(roi: np.ndarray, thresh: float = 0.55) -> bool:
    """≥ thresh fraction of pixels near white → likely inside a bubble."""
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float((gray > 220).mean()) >= thresh


def _is_color_page(roi: np.ndarray, thresh: float = 12.0) -> bool:
    """Std-dev across channels > thresh → colored page (manhwa/manhua)."""
    if roi.size == 0:
        return False
    sample = roi.reshape(-1, 3).astype(np.float32)
    chan_spread = float(sample.std(axis=0).mean())
    if chan_spread < 4.0:
        return False
    # Compare per-pixel max-min channel difference; mono pages stay near 0.
    diff = sample.max(axis=1) - sample.min(axis=1)
    return float(diff.mean()) >= thresh


def _ring_around_mask(image_bgr: np.ndarray, mask_u8: np.ndarray,
                      ring_px: int = 12) -> np.ndarray:
    """Sample a thin ring just OUTSIDE the masked area for context detection."""
    if mask_u8.sum() == 0:
        return image_bgr
    bin_mask = (mask_u8 > 127).astype(np.uint8) * 255
    outer = cv2.dilate(bin_mask, np.ones((ring_px * 2 + 1,) * 2, np.uint8))
    inner = cv2.dilate(bin_mask, np.ones((3, 3), np.uint8))
    ring = (outer > 0) & (inner == 0)
    if ring.sum() < 50:
        return image_bgr
    return image_bgr[ring]


# =============================================================================
# 4. PUBLIC API
# =============================================================================

def build_inpaint_prompt(image_bgr: np.ndarray,
                         mask_u8: np.ndarray) -> Tuple[str, str]:
    """Pick the right positive prompt for this image+mask pair.

    Returns (positive_prompt, negative_prompt).

    Decision tree:
      1. Sample a ring of pixels just outside the masked area.
      2. Mostly white ring         → text was inside a bubble  → POS_BUBBLE_INTERIOR
      3. Otherwise, colored ring   → text was on color art     → POS_OVER_ARTWORK_COLOR
      4. Otherwise (mono ring)     → text was on B/W art       → POS_OVER_ARTWORK_MONO
    """
    try:
        ring = _ring_around_mask(image_bgr, mask_u8, ring_px=14)
        if _is_mostly_white(ring.reshape(-1, 1, 3) if ring.ndim == 2 else ring):
            return POS_BUBBLE_INTERIOR, NEG_PROMPT
        if _is_color_page(ring.reshape(-1, 1, 3) if ring.ndim == 2 else ring):
            return POS_OVER_ARTWORK_COLOR, NEG_PROMPT
        return POS_OVER_ARTWORK_MONO, NEG_PROMPT
    except Exception as e:
        logger.warning(f"prompt context detection failed ({e}); using mono artwork prompt")
        return POS_OVER_ARTWORK_MONO, NEG_PROMPT


__all__ = [
    "build_inpaint_prompt",
    "POS_BUBBLE_INTERIOR",
    "POS_OVER_ARTWORK_MONO",
    "POS_OVER_ARTWORK_COLOR",
    "NEG_PROMPT",
]
