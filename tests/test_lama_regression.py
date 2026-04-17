"""Regression test for ctd.lama_inpaint against the 16-patch chapter fixture.

Purpose
-------
The bubble-cleaner has regressed twice in a way that re-prints the original
Japanese / Korean text inside the cleaned bubble. This test runs the current
`lama_inpaint` against a known-good 16-patch fixture and FAILS if any cleaned
patch still looks too much like the original inside the masked region.

Two assertions per patch:
  1. MEAN_ABS_DIFF(new_clean, orig) inside the mask must be > MIN_DIFF.
     If text reappears, the cleaned region is nearly identical to the
     original → diff stays low → test fails loudly.
  2. MEAN_ABS_DIFF(new_clean, ref_lama) inside the mask must be < MAX_DRIFT.
     The reference _lama.jpg in the fixture is a known-good clean. The new
     output should stay close to it.

How to run
----------
    pytest tests/test_lama_regression.py -v

Skip controls
-------------
    LAMA_REGRESSION_SKIP=1 pytest ...   # skip the whole module
    LAMA_REGRESSION_LIMIT=3 pytest ...  # only test first N patches (CPU-only envs)
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ZIP = ROOT / "attached_assets" / "comparison_1776387603932.zip"
FIXTURE_DIR = ROOT / "tests" / "_fixtures" / "comparison"
LAMA_MODEL = ROOT / "models" / "anime-manga-big-lama.pt"

# Thresholds (uint8 mean abs diff, 0–255 scale).
MIN_DIFF_FROM_ORIG = 12.0  # below this → cleaning didn't change much → regression
MAX_DRIFT_FROM_REF = 35.0  # above this → drifted from the known-good clean


def _extract_fixture():
    if FIXTURE_DIR.exists() and any(FIXTURE_DIR.glob("patch_*_orig.jpg")):
        return
    FIXTURE_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FIXTURE_ZIP) as zf:
        zf.extractall(FIXTURE_DIR.parent)


def _patch_ids():
    if not FIXTURE_ZIP.exists():
        return []
    _extract_fixture()
    ids = sorted({
        p.name.split("_")[1]
        for p in FIXTURE_DIR.glob("patch_*_orig.jpg")
    })
    limit = int(os.environ.get("LAMA_REGRESSION_LIMIT", "0") or 0)
    return ids if limit <= 0 else ids[:limit]


pytestmark = pytest.mark.skipif(
    os.environ.get("LAMA_REGRESSION_SKIP") == "1",
    reason="LAMA_REGRESSION_SKIP=1 set",
)


@pytest.fixture(scope="module")
def lama():
    if not LAMA_MODEL.exists():
        pytest.skip(f"LaMa model missing at {LAMA_MODEL}")
    if not FIXTURE_ZIP.exists():
        pytest.skip(f"Fixture zip missing at {FIXTURE_ZIP}")
    from ctd.lama_inpaint import lama_inpaint
    return lama_inpaint


def _load(pid: str):
    orig = cv2.imread(str(FIXTURE_DIR / f"patch_{pid}_orig.jpg"))
    mask = cv2.imread(str(FIXTURE_DIR / f"patch_{pid}_mask.jpg"), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(str(FIXTURE_DIR / f"patch_{pid}_lama.jpg"))
    assert orig is not None, f"missing orig for {pid}"
    assert mask is not None, f"missing mask for {pid}"
    assert ref is not None, f"missing ref lama for {pid}"
    return orig, mask, ref


def _masked_mean_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    sel = mask > 127
    if sel.sum() == 0:
        return 0.0
    diff = cv2.absdiff(a, b).astype(np.float32).mean(axis=2)
    return float(diff[sel].mean())


@pytest.mark.parametrize("pid", _patch_ids())
def test_clean_diverges_from_original(lama, pid):
    orig, mask, _ref = _load(pid)
    cleaned = lama(orig, mask)
    diff = _masked_mean_abs_diff(cleaned, orig, mask)
    assert diff >= MIN_DIFF_FROM_ORIG, (
        f"patch {pid}: cleaned region too similar to original "
        f"(mean abs diff {diff:.2f} < {MIN_DIFF_FROM_ORIG}). "
        f"Likely the original text is reappearing in the cleaned bubble."
    )


@pytest.mark.parametrize("pid", _patch_ids())
def test_clean_stays_close_to_reference(lama, pid):
    orig, mask, ref = _load(pid)
    cleaned = lama(orig, mask)
    drift = _masked_mean_abs_diff(cleaned, ref, mask)
    assert drift <= MAX_DRIFT_FROM_REF, (
        f"patch {pid}: cleaned region drifted from known-good reference "
        f"(mean abs diff {drift:.2f} > {MAX_DRIFT_FROM_REF})."
    )
