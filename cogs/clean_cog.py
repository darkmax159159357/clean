import discord
from discord.ext import commands
from discord import app_commands
import os
import io
import re
import cv2
import time
import numpy as np
import aiohttp
import asyncio
import logging
import zipfile
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
from functools import partial

from app import app, db
from guild import guild_registry, guild_settings_helper
import data_manager
from models import User, GuildSettings
from bot import create_stop_button_view, active_extractions
from downloaders import google_drive_direct
from downloaders.download_from_gofile import is_gofile_url, fetch_gofile_metadata, download_single_gofile_chapter, upload_bytes_to_gofile
from extraction.url_name_extractor import extract_name_from_url, sanitize_filename

logger = logging.getLogger(__name__)

DISCORD_MAX_FILE_SIZE = 8 * 1024 * 1024
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
_active_clean_ops: dict = {}
CHUNK_MAX_HEIGHT = 2000

_detector = None

from concurrent.futures import ThreadPoolExecutor
_global_clean_executor = ThreadPoolExecutor(max_workers=2)
_global_clean_sem = asyncio.Semaphore(2)

GPU_SERVER_URL = os.environ.get("GPU_CLEAN_SERVER_URL", "")
_gpu_available = None
_gpu_sem = asyncio.Semaphore(4)
_gpu_session = None


def _get_gpu_session():
    global _gpu_session
    import aiohttp
    if _gpu_session is None or _gpu_session.closed:
        timeout = aiohttp.ClientTimeout(total=180)
        _gpu_session = aiohttp.ClientSession(timeout=timeout)
    return _gpu_session


async def _clean_via_gpu(img_bytes: bytes, img_name: str = "", debug: bool = False,
                          engine: str = "lama"):
    """Returns (cleaned_bytes, debug_files) or (None, []). Debug files only populated when debug=True."""
    global _gpu_available
    if not GPU_SERVER_URL:
        return None, []
    try:
        import aiohttp
        async with _gpu_sem:
            session = _get_gpu_session()
            form = aiohttp.FormData()
            form.add_field('file', img_bytes, filename=img_name or 'image.jpg', content_type='image/jpeg')
            if debug:
                form.add_field('debug', 'true')
            if engine and engine != "lama":
                form.add_field('engine', engine)
            if debug:
                endpoint = f"{GPU_SERVER_URL}/clean_stream"
                form2 = aiohttp.FormData()
                form2.add_field('files', img_bytes, filename=img_name or 'image.jpg', content_type='image/jpeg')
                form2.add_field('debug', 'true')
                if engine and engine != "lama":
                    form2.add_field('engine', engine)
                timeout = aiohttp.ClientTimeout(total=900 if engine == "compare" else 300)
                async with session.post(endpoint, data=form2, timeout=timeout) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(f"GPU debug clean failed for {img_name}: {resp.status} - {body[:200]}")
                        return None, []
                    data = await resp.read()
                    import struct
                    cleaned_bytes = None
                    dbg_files = []
                    pos = 0
                    while pos < len(data):
                        if pos + 4 > len(data):
                            break
                        name_len = struct.unpack('>I', data[pos:pos+4])[0]
                        pos += 4
                        if pos + name_len > len(data):
                            break
                        entry_name = data[pos:pos+name_len].decode('utf-8', errors='replace')
                        pos += name_len
                        if pos + 4 > len(data):
                            break
                        data_len = struct.unpack('>I', data[pos:pos+4])[0]
                        pos += 4
                        if data_len > 0 and pos + data_len <= len(data):
                            entry_data = data[pos:pos+data_len]
                            pos += data_len
                            if "::_debug/" in entry_name:
                                rel = entry_name.split("::_debug/", 1)[1]
                                dbg_files.append((rel, entry_data))
                            else:
                                cleaned_bytes = entry_data
                        else:
                            pos += data_len
                    if cleaned_bytes:
                        _gpu_available = True
                    return cleaned_bytes, dbg_files
            async with session.post(f"{GPU_SERVER_URL}/clean", data=form) as resp:
                if resp.status == 200:
                    result = await resp.read()
                    if result and len(result) > 100:
                        _gpu_available = True
                        return result, []
                    else:
                        logger.warning(f"GPU returned empty/small response for {img_name} (status={resp.status}, size={len(result) if result else 0})")
                else:
                    body = await resp.text()
                    logger.warning(f"GPU server returned {resp.status} for {img_name}: {body[:200]}")
        return None, []
    except Exception as e:
        logger.warning(f"GPU server error for {img_name}: {type(e).__name__}: {e}")
        return None, []


async def _clean_batch_via_gpu(images: List[Tuple[str, bytes]], batch_size: int = 8,
                                debug: bool = False, on_image_done=None,
                                engine: str = "lama"):
    """Returns (cleaned_dict, debug_dict) or None.
    on_image_done: optional callback(int) called after each image is received (for progress)."""
    global _gpu_available
    if not GPU_SERVER_URL:
        return None
    try:
        import aiohttp
        import struct

        indexed_names = []
        for i, (fname, _) in enumerate(images):
            safe_name = f"{i:04d}_{fname}"
            indexed_names.append(safe_name)

        all_results = {}
        all_debug: Dict[str, list] = {}
        for batch_start in range(0, len(images), batch_size):
            batch_indices = list(range(batch_start, min(batch_start + batch_size, len(images))))
            batch = [(indexed_names[i], images[i][1]) for i in batch_indices]

            session = _get_gpu_session()
            form = aiohttp.FormData()
            for safe_name, img_bytes in batch:
                form.add_field('files', img_bytes, filename=safe_name, content_type='image/jpeg')
            if debug:
                form.add_field('debug', 'true')
            if engine and engine != "lama":
                form.add_field('engine', engine)

            if engine == "compare":
                base_timeout = 1800
            elif engine == "sd":
                base_timeout = 1200
            else:
                base_timeout = 600 if debug else 300
            timeout = aiohttp.ClientTimeout(total=base_timeout)
            async with session.post(f"{GPU_SERVER_URL}/clean_stream", data=form, timeout=timeout) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"GPU batch failed: {resp.status} - {body[:200]}")
                    return None

                total_count = int(resp.headers.get("X-Total-Count", str(len(batch))))
                buf = b""
                parsed_count = 0
                success_count = 0
                while True:
                    chunk = await resp.content.read(65536)
                    if not chunk:
                        break
                    buf += chunk

                    while True:
                        if len(buf) < 4:
                            break
                        name_len = struct.unpack('>I', buf[:4])[0]
                        if name_len > 10000:
                            logger.warning(f"GPU stream: invalid name length {name_len}")
                            buf = b""
                            break
                        if len(buf) < 4 + name_len + 4:
                            break
                        data_len = struct.unpack('>I', buf[4+name_len:4+name_len+4])[0]
                        total_entry = 4 + name_len + 4 + data_len
                        if len(buf) < total_entry:
                            break

                        name = buf[4:4+name_len].decode('utf-8', errors='replace')
                        entry_data = buf[4+name_len+4:total_entry] if data_len > 0 else b""
                        buf = buf[total_entry:]

                        if data_len == 0:
                            all_results[name] = None
                            parsed_count += 1
                            if on_image_done:
                                try:
                                    on_image_done(1)
                                except Exception:
                                    pass
                        elif "::_debug/" in name:
                            base_name, rel = name.split("::_debug/", 1)
                            all_debug.setdefault(base_name, []).append((rel, entry_data))
                        else:
                            all_results[name] = entry_data
                            parsed_count += 1
                            success_count += 1
                            if on_image_done:
                                try:
                                    on_image_done(1)
                                except Exception:
                                    pass

                if parsed_count != total_count:
                    logger.warning(f"GPU batch parse mismatch: parsed {parsed_count}, expected {total_count}")

                logger.info(f"GPU batch {batch_start//batch_size + 1}: {success_count}/{total_count} cleaned{' [+debug]' if debug else ''}")

        _gpu_available = True
        return (all_results, all_debug)

    except Exception as e:
        logger.warning(f"GPU batch error: {type(e).__name__}: {e}")
        return None


async def check_gpu_health() -> bool:
    if not GPU_SERVER_URL:
        return False
    import aiohttp
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{GPU_SERVER_URL}/health") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "ok":
                            logger.info(f"GPU health check OK (attempt {attempt+1}): {data}")
                            return True
        except Exception as e:
            logger.warning(f"GPU health check attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                await asyncio.sleep(3)
    return False


def get_detector():
    global _detector
    if _detector is None:
        try:
            from ctd.ctd_onnx_detector import warmup
            warmup()
            _detector = True
            logger.info("CTD-ONNX detector loaded (CPU) for local cleaning fallback")
        except Exception as e:
            logger.error(f"Failed to load CTD-ONNX detector: {e}", exc_info=True)
    return _detector


def _is_macos_junk(filename: str) -> bool:
    if '__MACOSX' in filename:
        return True
    basename = filename.split('/')[-1]
    return basename == '.DS_Store' or basename.startswith('._')


def _natsort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _find_smart_split_points(image_bgr: np.ndarray, max_height: int = CHUNK_MAX_HEIGHT,
                              search_range: int = 400) -> list:
    h, w = image_bgr.shape[:2]
    if h <= max_height:
        return [(0, h)]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    row_edge_density = np.sum(edges > 0, axis=1).astype(np.float32) / w

    chunks = []
    pos = 0

    while pos < h:
        if h - pos <= max_height:
            chunks.append((pos, h))
            break

        ideal_end = pos + max_height
        search_start = max(pos + max_height - search_range, pos + int(max_height * 0.5))
        search_end = min(ideal_end + search_range // 2, h - 1)

        if search_start >= search_end:
            chunks.append((pos, min(pos + max_height, h)))
            pos = min(pos + max_height, h)
            continue

        window = row_edge_density[search_start:search_end]

        kernel_size = 15
        if len(window) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(window, kernel, mode='same')
        else:
            smoothed = window

        best_offset = int(np.argmin(smoothed))
        split_y = search_start + best_offset

        min_density = float(smoothed[best_offset])
        if min_density < 0.02:
            stripe = gray[max(0, split_y - 3):min(h, split_y + 4), :]
            mean_val = float(np.mean(stripe))
            std_val = float(np.std(stripe))
            if mean_val > 200 and std_val < 30:
                split_y = search_start + best_offset

        remaining = h - split_y
        if remaining < max_height * 0.3:
            chunks.append((pos, h))
            break

        chunks.append((pos, split_y))
        pos = split_y

    return chunks


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray,
                   color=(0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    out = image_bgr.copy()
    m = mask > 127
    if not np.any(m):
        return out
    out[m] = (out[m] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def _encode_debug_jpg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return data.tobytes() if ok else b""


def _collect_debug_stage(debug_files: list, prefix: str,
                          image_bgr: np.ndarray, raw_mask: np.ndarray,
                          enhanced_mask: np.ndarray, inpaint_mask: np.ndarray) -> None:
    debug_files.append((f"{prefix}01_raw_mask.jpg", _encode_debug_jpg(raw_mask)))
    debug_files.append((f"{prefix}02_enhanced_mask.jpg", _encode_debug_jpg(enhanced_mask)))
    debug_files.append((f"{prefix}03_inpaint_mask.jpg", _encode_debug_jpg(inpaint_mask)))
    debug_files.append((f"{prefix}04_raw_overlay.jpg", _encode_debug_jpg(_overlay_mask(image_bgr, raw_mask, color=(0, 255, 0), alpha=0.45))))
    debug_files.append((f"{prefix}05_enhanced_overlay.jpg", _encode_debug_jpg(_overlay_mask(image_bgr, enhanced_mask, color=(255, 128, 0), alpha=0.5))))
    debug_files.append((f"{prefix}06_inpaint_overlay.jpg", _encode_debug_jpg(_overlay_mask(image_bgr, inpaint_mask, color=(255, 0, 255), alpha=0.5))))


class _CleanCancelled(Exception):
    """Raised from inside the CPU worker when the user pressed Cancel."""
    pass


def _clean_region_local(region_bgr: np.ndarray, tag: str = ""):
    """Detect text mask via comictextdetector → enhance with EasyOCR → LaMa inpaint."""
    from ctd.ctd_onnx_detector import detect_text_mask
    from ctd.easyocr_enhancer import enhance_mask_with_easyocr
    from ctd.lama_inpaint import lama_inpaint

    raw_mask, _ = detect_text_mask(region_bgr)
    enh_mask = raw_mask
    try:
        enh_mask = enhance_mask_with_easyocr(region_bgr, raw_mask)
    except Exception as e:
        logger.warning(f"{tag}EasyOCR enhance failed: {e}")

    if np.sum(enh_mask > 127) == 0:
        return region_bgr, raw_mask, enh_mask

    cleaned = lama_inpaint(region_bgr, enh_mask)
    return cleaned, raw_mask, enh_mask


def _split_and_clean_long_image(image_bgr: np.ndarray, img_name: str = "",
                                  debug_files: Optional[list] = None,
                                  stop_event=None) -> np.ndarray:
    tag = f"[{img_name}] " if img_name else ""
    h, w = image_bgr.shape[:2]
    chunks = _find_smart_split_points(image_bgr, max_height=CHUNK_MAX_HEIGHT)
    logger.info(f"{tag}Split {w}x{h} → {len(chunks)} chunks")

    result = image_bgr.copy()
    full_raw = np.zeros((h, w), dtype=np.uint8) if debug_files is not None else None
    full_enh = np.zeros((h, w), dtype=np.uint8) if debug_files is not None else None

    for idx, (y0, y1) in enumerate(chunks):
        if stop_event is not None and stop_event.is_set():
            logger.info(f"{tag}Cancelled at chunk {idx+1}/{len(chunks)}")
            raise _CleanCancelled()
        chunk_img = result[y0:y1, :, :].copy()
        cleaned, raw_mask, enh_mask = _clean_region_local(chunk_img, tag=tag)
        result[y0:y1, :, :] = cleaned
        if debug_files is not None:
            full_raw[y0:y1, :] = np.maximum(full_raw[y0:y1, :], raw_mask)
            full_enh[y0:y1, :] = np.maximum(full_enh[y0:y1, :], enh_mask)
        logger.info(f"{tag}Chunk {idx+1}/{len(chunks)}: {int((enh_mask > 127).sum())} text px")

    if debug_files is not None and full_raw is not None:
        _collect_debug_stage(debug_files, "", image_bgr, full_raw, full_enh, full_enh)
    return result


def smart_clean_image(image_bgr: np.ndarray, img_name: str = "",
                       debug_files: Optional[list] = None,
                       stop_event=None):
    if stop_event is not None and stop_event.is_set():
        raise _CleanCancelled()

    if get_detector() is None:
        raise RuntimeError("CTD-ONNX detector not available")

    h, w = image_bgr.shape[:2]
    if h > CHUNK_MAX_HEIGHT:
        return _split_and_clean_long_image(image_bgr, img_name, debug_files=debug_files, stop_event=stop_event)

    tag = f"[{img_name}] " if img_name else ""
    result, raw_mask, enh_mask = _clean_region_local(image_bgr, tag=tag)
    if debug_files is not None:
        _collect_debug_stage(debug_files, "", image_bgr, raw_mask, enh_mask, enh_mask)
    return result


def encode_result(result_bgr: np.ndarray, for_zip: bool = False) -> Tuple[bytes, str]:
    if for_zip:
        _, jpg_data = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jpg_data.tobytes(), ".jpg"

    _, png_data = cv2.imencode('.png', result_bgr)
    png_bytes = png_data.tobytes()
    if len(png_bytes) <= DISCORD_MAX_FILE_SIZE:
        return png_bytes, ".png"

    for quality in (95, 85, 75):
        _, jpg_data = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg_bytes = jpg_data.tobytes()
        if len(jpg_bytes) <= DISCORD_MAX_FILE_SIZE:
            return jpg_bytes, ".jpg"

    h, w = result_bgr.shape[:2]
    scale = 0.75
    while scale > 0.25:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(result_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, jpg_data = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_bytes = jpg_data.tobytes()
        if len(jpg_bytes) <= DISCORD_MAX_FILE_SIZE:
            return jpg_bytes, ".jpg"
        scale -= 0.1

    _, jpg_data = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return jpg_data.tobytes(), ".jpg"


def process_single_image(img_bytes: bytes, for_zip: bool = False, img_name: str = "",
                          debug: bool = False, stop_event=None) -> Optional[Tuple[bytes, str, list]]:
    try:
        if stop_event is not None and stop_event.is_set():
            return None
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        debug_files: list = [] if debug else None
        result_bgr = smart_clean_image(img_bgr, img_name=img_name,
                                        debug_files=debug_files, stop_event=stop_event)
        encoded, ext = encode_result(result_bgr, for_zip=for_zip)
        return encoded, ext, (debug_files or [])
    except _CleanCancelled:
        logger.info(f"Cleaning cancelled mid-image: {img_name}")
        return None
    except Exception as e:
        logger.error(f"Error processing image {img_name}: {e}", exc_info=True)
        return None


def extract_images_from_zip(data: bytes) -> List[Tuple[str, bytes]]:
    images = []
    with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
        image_files = sorted([
            name for name in zf.namelist()
            if not name.endswith('/')
            and not _is_macos_junk(name)
            and _is_image_file(name)
        ], key=_natsort_key)
        for filename in image_files:
            images.append((os.path.basename(filename), zf.read(filename)))
    return images


async def download_from_url(url: str) -> Tuple[Optional[bytes], Optional[str]]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=300)
            async with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if not data:
                        return None, "Downloaded file is empty"
                    return data, None
                else:
                    return None, f"Download failed: status {resp.status}"
    except asyncio.TimeoutError:
        return None, "Download timeout (5 min)"
    except Exception as e:
        return None, f"Download error: {str(e)}"


async def update_clean_progress(message: discord.Message,
                                 interaction: discord.Interaction,
                                 total_images: int,
                                 processed: int,
                                 phase: str,
                                 start_time: float,
                                 chapter_name: str = ""):
    if not message:
        return
    try:
        percentage = int((processed / total_images) * 100) if total_images > 0 else 0
        progress_bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))
        elapsed = int(time.time() - start_time)

        embed = discord.Embed(
            title="🧹 Text Cleaning in Progress...",
            color=discord.Color.gold()
        )
        embed.set_author(
            name=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url
        )
        if chapter_name:
            embed.add_field(name="📁 Source", value=f"`{chapter_name}`", inline=False)
        embed.add_field(name="Phase", value=f"**{phase}**", inline=False)
        embed.add_field(
            name="Progress",
            value=f"`[{progress_bar}]` **{percentage}%** ({processed}/{total_images})\nElapsed: {elapsed}s",
            inline=False
        )

        view = active_extractions.get(message.id, {}).get('view')
        await message.edit(embed=embed, view=view)
    except discord.NotFound:
        stop_event = active_extractions.get(message.id, {}).get('stop_event')
        if stop_event:
            stop_event.set()
    except Exception as e:
        logger.warning(f"Error updating clean progress: {e}")


async def collect_images_from_source(source_type: str, source_data, progress_msg=None, interaction=None) -> Tuple[List[Tuple[str, bytes]], Optional[str], Optional[str]]:
    images = []
    auto_name = None

    if source_type == 'attachment':
        data = await source_data.read()
        if data[:4] == b'PK\x03\x04':
            images = extract_images_from_zip(data)
            auto_name = os.path.splitext(source_data.filename)[0]
        elif _is_image_file(source_data.filename):
            images = [(source_data.filename, data)]
            auto_name = os.path.splitext(source_data.filename)[0]
        else:
            return [], f"Unsupported file: `{source_data.filename}`. Upload images (PNG/JPG) or a ZIP.", None

    elif source_type == 'url':
        url_str = str(source_data).strip()
        _drive_hosts = ('drive.google.com', 'docs.google.com')
        is_gofile = is_gofile_url(url_str)
        is_drive = any(h in url_str for h in _drive_hosts)

        if is_gofile:
            if progress_msg and interaction:
                await update_clean_progress(progress_msg, interaction, 1, 0, "📦 Analyzing GoFile link...", time.time())

            gofile_metadata, gofile_error = await fetch_gofile_metadata(url_str, use_playwright_first=True)
            if gofile_error:
                return [], f"GoFile error: {gofile_error}", None
            if not gofile_metadata:
                return [], "No files found in GoFile link", None

            chapter_names = []
            for idx, chapter_meta in enumerate(gofile_metadata):
                ch_name = chapter_meta.get("name", f"Chapter {idx+1}")
                chapter_names.append(ch_name)
                if progress_msg and interaction:
                    await update_clean_progress(
                        progress_msg, interaction, len(gofile_metadata), idx,
                        f"📥 Downloading: {ch_name} ({idx+1}/{len(gofile_metadata)})", time.time())

                downloaded_chapters, dl_error = await download_single_gofile_chapter(chapter_meta)
                if dl_error or not downloaded_chapters:
                    logger.warning(f"GoFile chapter download failed: {dl_error}")
                    continue
                for ch in downloaded_chapters:
                    for fname, fdata in ch.get("images", []):
                        if _is_image_file(fname):
                            images.append((fname, fdata))

            if len(chapter_names) == 1:
                auto_name = sanitize_filename(chapter_names[0])
            elif chapter_names:
                auto_name = sanitize_filename(chapter_names[0]) + f"..{sanitize_filename(chapter_names[-1])}"

        elif is_drive:
            if progress_msg and interaction:
                await update_clean_progress(progress_msg, interaction, 1, 0, "📂 Analyzing Google Drive link...", time.time())

            drive_id = google_drive_direct.extract_id_from_url(url_str)
            if not drive_id:
                return [], "Invalid Google Drive link - could not extract ID", None

            chapters, folder_error = await google_drive_direct.process_drive_folder(drive_id)
            if chapters:
                folder_names = []
                for ch in chapters:
                    ch_name = ch.get("name", "")
                    if ch_name:
                        folder_names.append(ch_name)
                    for fname, fdata in ch.get("images", []):
                        if _is_image_file(fname):
                            images.append((fname, fdata))
                if len(folder_names) == 1:
                    auto_name = sanitize_filename(folder_names[0])
                elif folder_names:
                    auto_name = sanitize_filename(folder_names[0]) + f"..{sanitize_filename(folder_names[-1])}"
            elif folder_error and ('غير متاح للعامة' in folder_error or 'not publicly shared' in folder_error.lower()):
                return [], f"Google Drive error: {folder_error}", None
            else:
                if progress_msg and interaction:
                    await update_clean_progress(progress_msg, interaction, 1, 0, "📄 Downloading single file...", time.time())
                data, error_msg = await google_drive_direct.download_file(file_id=drive_id)
                if error_msg:
                    return [], f"Google Drive error: {error_msg}", None
                if data:
                    if data[:4] == b'PK\x03\x04':
                        images = extract_images_from_zip(data)
                    elif len(data) > 100:
                        images = [("image.jpg", data)]

        else:
            extracted_url_name = await extract_name_from_url(url_str)
            if extracted_url_name:
                auto_name = sanitize_filename(extracted_url_name)

            if progress_msg and interaction:
                await update_clean_progress(progress_msg, interaction, 1, 0, "🔗 Downloading from URL...", time.time())
            data, error_msg = await download_from_url(url_str)
            if error_msg:
                return [], f"Download error: {error_msg}", None
            if data:
                if data[:4] == b'PK\x03\x04':
                    images = extract_images_from_zip(data)
                else:
                    images = [("image.jpg", data)]

    if not images:
        return [], "No images found to process", None

    return images, None, auto_name


def build_zip_chunks(results: List[Tuple[str, bytes]]) -> List[List[Tuple[str, bytes]]]:
    max_chunk = DISCORD_MAX_FILE_SIZE - 4096
    chunks = []
    current_chunk = []
    current_size = 0

    for name, data in results:
        entry_size = len(data) + 200
        if current_chunk and current_size + entry_size > max_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append((name, data))
        current_size += entry_size

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


class CleanCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._preload_task = asyncio.ensure_future(self._preload_models())

    async def _preload_models(self):
        global _gpu_available
        if GPU_SERVER_URL:
            gpu_ok = await check_gpu_health()
            _gpu_available = gpu_ok
            if gpu_ok:
                logger.info(f"GPU server connected: {GPU_SERVER_URL} — local CPU models will load lazily only if GPU fails")
            else:
                logger.warning(f"GPU server configured but not reachable: {GPU_SERVER_URL} — will fallback to CPU (models load on first use)")
        else:
            logger.info("No GPU server configured — CPU models will load on first use")

    async def _handle_clean_request(
        self,
        interaction: discord.Interaction,
        source_type: str,
        source_data,
        file_name: Optional[str] = None,
        debug: bool = False,
        engine: str = "lama"
    ):
        try:
            await interaction.response.defer(ephemeral=False)
        except discord.NotFound:
            return
        except discord.InteractionResponded:
            return
        except discord.HTTPException as e:
            if e.code != 40060:
                return

        url_dedup_key = None
        if source_type == 'url' and source_data:
            url_dedup_key = f"clean:{interaction.user.id}:{str(source_data).strip()}"
            now = time.time()
            last_time = _active_clean_ops.get(url_dedup_key)
            if last_time and (now - last_time) < 20:
                await interaction.followup.send(
                    embed=discord.Embed(
                        title="⚠️ Duplicate Request",
                        description="This URL is already being cleaned. Please wait.",
                        color=discord.Color.orange()
                    ), ephemeral=True)
                return
            _active_clean_ops[url_dedup_key] = now

        total_start_time = time.time()

        initial_embed = discord.Embed(
            title="🧹 Cleaning Request Received!",
            description="Your text cleaning task has been queued and will begin shortly.",
            color=discord.Color.blue()
        )
        initial_embed.set_author(
            name=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url
        )
        try:
            await interaction.followup.send(embed=initial_embed)
            loading_msg = await interaction.original_response()
        except discord.NotFound:
            return

        user_id = str(interaction.user.id)
        guild_id = str(interaction.guild_id)

        with app.app_context():
            user_profile = db.session.get(User, user_id)
            if user_profile and user_profile.is_banned:
                await interaction.followup.send("❌ You are banned.", ephemeral=True)
                return

            if guild_registry.is_bot_disabled_in_guild(guild_id):
                await loading_msg.edit(embed=discord.Embed(
                    title="🔒 Bot Disabled",
                    description="Bot is temporarily disabled in this server.",
                    color=discord.Color.orange()), view=None)
                return

            if not guild_settings_helper.is_command_enabled(guild_id, 'clean'):
                await loading_msg.edit(embed=discord.Embed(
                    title="🚫 Command Disabled",
                    description=guild_settings_helper.format_command_disabled_message('clean'),
                    color=discord.Color.red()), view=None)
                return

            cost_per_image = float(data_manager.get_setting('clean_cost_per_image', '0.5'))
            max_images = int(data_manager.get_setting('clean_max_images', '50'))

            guild_settings = guild_settings_helper.get_guild_settings(guild_id)
            if guild_settings and guild_settings.cost_clean is not None:
                cost_per_image = guild_settings.cost_clean

            use_server_balance = False
            use_independent_balance = False
            is_dependent_server = False
            dependent_settings = guild_registry.get_guild_dependent_settings(guild_id)

            if dependent_settings and dependent_settings.get('is_dependent_server', False):
                is_dependent_server = True
                if guild_registry.is_user_authorized_for_dependent_server(guild_id, user_id):
                    use_server_balance = True
                else:
                    await loading_msg.edit(embed=discord.Embed(
                        title="🔒 Authorization Required",
                        description="This server uses **Dependent Server Mode**.\n"
                                    "You are not authorized to use this server's balance.\n\n"
                                    "Contact the server admin to be added to the authorized users list.",
                        color=discord.Color.red()), view=None)
                    return

            if not is_dependent_server and dependent_settings and dependent_settings.get('is_independent_server', False):
                use_independent_balance = True

            if use_server_balance:
                current_balance = guild_registry.get_user_allocated_balance(guild_id, user_id)
            elif use_independent_balance:
                current_balance = guild_registry.get_independent_balance(guild_id)
            else:
                current_balance = guild_registry.get_guild_balance(guild_id, user_id)

            if current_balance < cost_per_image:
                await loading_msg.edit(embed=discord.Embed(
                    title="❌ Insufficient Balance",
                    description=f"You need at least **{cost_per_image}** points.\nYour balance: **{current_balance:.2f}** points.",
                    color=discord.Color.red()), view=None)
                return

        await update_clean_progress(loading_msg, interaction, 1, 0,
                                     "📥 Downloading images...", total_start_time)

        download_start = time.time()
        images, error, auto_name = await collect_images_from_source(
            source_type, source_data, progress_msg=loading_msg, interaction=interaction)

        if error:
            await loading_msg.edit(embed=discord.Embed(
                title="❌ Error",
                description=error,
                color=discord.Color.red()), view=None)
            return

        source_name = file_name or auto_name or ""
        if not source_name:
            source_name = "images"

        download_time = time.time() - download_start
        total_images = len(images)

        if total_images > max_images:
            images = images[:max_images]
            total_images = max_images
            logger.info(f"Capped images to max {max_images}")

        with app.app_context():
            total_cost = cost_per_image * total_images
            if use_server_balance:
                current_balance = guild_registry.get_user_allocated_balance(guild_id, user_id)
            elif use_independent_balance:
                current_balance = guild_registry.get_independent_balance(guild_id)
            else:
                current_balance = guild_registry.get_guild_balance(guild_id, user_id)

            max_affordable = int(current_balance / cost_per_image) if cost_per_image > 0 else total_images
            if max_affordable < total_images:
                images = images[:max_affordable]
                total_images = max_affordable
                if total_images <= 0:
                    await loading_msg.edit(embed=discord.Embed(
                        title="❌ Insufficient Balance",
                        description=f"Cannot clean any images. Balance: **{current_balance:.2f}** points.",
                        color=discord.Color.red()), view=None)
                    return

        view, stop_event, extraction_key = await create_stop_button_view(interaction, loading_msg)

        await update_clean_progress(loading_msg, interaction, total_images, 0,
                                     f"🧹 Cleaning {total_images} image(s)...", total_start_time, source_name)

        loop = asyncio.get_event_loop()
        results_dict = {}
        debug_dict: Dict[int, list] = {}
        failed = 0
        clean_start = time.time()

        progress_state = {'done': 0, 'phase': 'starting', 'running': True}

        async def _progress_ticker():
            while progress_state['running']:
                try:
                    await update_clean_progress(
                        loading_msg, interaction, total_images, progress_state['done'],
                        progress_state['phase'],
                        total_start_time, source_name)
                except Exception:
                    pass
                await asyncio.sleep(3)

        ticker_task = asyncio.ensure_future(_progress_ticker())

        use_gpu = GPU_SERVER_URL and _gpu_available is not False

        async def _clean_batch_gpu():
            nonlocal failed
            progress_state['phase'] = f"🚀 GPU Batch Cleaning {total_images} images{' [+debug]' if debug else ''}..."

            def _on_done(n: int):
                progress_state['done'] = min(progress_state['done'] + n, total_images)
                ext_data = active_extractions.get(loading_msg.id, {})
                if ext_data.get('view'):
                    try:
                        ext_data['view'].update_chapter_progress(progress_state['done'], total_images)
                    except Exception:
                        pass

            batch_res = await _clean_batch_via_gpu(
                images, batch_size=8, debug=debug, on_image_done=_on_done, engine=engine)
            if batch_res is None:
                return False

            cleaned_results, debug_results = batch_res
            indexed_names = [f"{i:04d}_{fname}" for i, (fname, _) in enumerate(images)]

            for i, safe_name in enumerate(indexed_names):
                data = cleaned_results.get(safe_name)
                if data is not None:
                    out_name = os.path.splitext(images[i][0])[0] + "_cleaned.jpg"
                    results_dict[i] = (out_name, data)
                    if debug:
                        dbg_entries = debug_results.get(safe_name, [])
                        if dbg_entries:
                            debug_dict[i] = dbg_entries

            return True

        async def _clean_one_cpu(idx, fname, img_bytes):
            nonlocal failed
            if stop_event.is_set():
                return
            progress_state['phase'] = f"Cleaning ({progress_state['done']+1}/{total_images})"

            async with _global_clean_sem:
                if stop_event.is_set():
                    return
                result = await loop.run_in_executor(
                    _global_clean_executor,
                    partial(process_single_image, img_bytes, total_images > 1, fname, debug, stop_event))
                if result:
                    result_bytes, ext, dbg_files = result
                    out_name = os.path.splitext(fname)[0] + f"_cleaned{ext}"
                    results_dict[idx] = (out_name, result_bytes)
                    if debug and dbg_files:
                        debug_dict[idx] = dbg_files
                else:
                    failed += 1
                    logger.warning(f"Failed to clean: {fname}")
                progress_state['done'] += 1
                ext_data = active_extractions.get(loading_msg.id, {})
                if ext_data.get('view'):
                    ext_data['view'].update_chapter_progress(progress_state['done'], total_images)

        async def _clean_one_gpu_fallback(idx, fname, img_bytes):
            nonlocal failed, use_gpu
            if stop_event.is_set():
                return
            progress_state['phase'] = f"Cleaning ({progress_state['done']+1}/{total_images})"

            if use_gpu:
                gpu_result, gpu_debug = await _clean_via_gpu(img_bytes, fname, debug=debug, engine=engine)
                if gpu_result:
                    out_name = os.path.splitext(fname)[0] + "_cleaned.jpg"
                    results_dict[idx] = (out_name, gpu_result)
                    if debug and gpu_debug:
                        debug_dict[idx] = gpu_debug
                    progress_state['done'] += 1
                    ext_data = active_extractions.get(loading_msg.id, {})
                    if ext_data.get('view'):
                        ext_data['view'].update_chapter_progress(progress_state['done'], total_images)
                    return

            await _clean_one_cpu(idx, fname, img_bytes)

        try:
            gpu_batch_ok = False
            if use_gpu and total_images >= 2:
                try:
                    gpu_batch_ok = await _clean_batch_gpu()
                except Exception as e:
                    logger.warning(f"GPU batch failed, falling back to individual: {e}")

            remaining = [(i, fname, img_bytes) for i, (fname, img_bytes) in enumerate(images) if i not in results_dict]
            if remaining:
                if remaining:
                    logger.info(f"Processing {len(remaining)} remaining images via {'GPU+CPU fallback' if use_gpu else 'CPU'}")
                    progress_state['done'] = total_images - len(remaining)
                    tasks = [_clean_one_gpu_fallback(i, fname, img_bytes) for i, fname, img_bytes in remaining]
                    await asyncio.gather(*tasks)
        finally:
            progress_state['running'] = False
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass

        results = [results_dict[i] for i in sorted(results_dict.keys())]

        clean_time = time.time() - clean_start
        total_time = time.time() - total_start_time
        success_count = len(results)

        if stop_event.is_set() and not results:
            cancel_embed = discord.Embed(
                title="⛔ Cleaning Cancelled",
                description="Process was cancelled. No images were cleaned.",
                color=discord.Color.red())
            await loading_msg.edit(embed=cancel_embed, view=None)
            if loading_msg.id in active_extractions:
                del active_extractions[loading_msg.id]
            return

        actual_cost = 0.0
        new_balance = 0.0
        with app.app_context():
            if success_count > 0:
                actual_cost = cost_per_image * success_count
                if use_server_balance:
                    guild_registry.deduct_server_balance_only(guild_id, user_id, actual_cost, "Text cleaning")
                    new_balance = guild_registry.get_user_allocated_balance(guild_id, user_id)
                elif use_independent_balance:
                    guild_registry.deduct_independent_balance_only(guild_id, user_id, actual_cost)
                    new_balance = guild_registry.get_independent_balance(guild_id)
                else:
                    guild_registry.deduct_balance_only(guild_id, user_id, actual_cost)
                    new_balance = guild_registry.get_guild_balance(guild_id, user_id)
            else:
                if use_server_balance:
                    new_balance = guild_registry.get_user_allocated_balance(guild_id, user_id)
                elif use_independent_balance:
                    new_balance = guild_registry.get_independent_balance(guild_id)
                else:
                    new_balance = guild_registry.get_guild_balance(guild_id, user_id)

        cancelled = stop_event.is_set()
        status_title = "⛔ Cleaning Stopped" if cancelled else "✅ Cleaning Complete!"
        status_color = discord.Color.orange() if cancelled else discord.Color.green()

        if actual_cost > 0:
            cost_str = f"💰 **{actual_cost:.2f} points deducted. Balance: {new_balance:.2f} points.**"
        else:
            cost_str = f"💰 **No points deducted.** Balance: {new_balance:.2f} points."

        stats_str = (
            f"📊 **Stats:**\n"
            f"• **Total images:** `{total_images}`\n"
            f"• **Successfully cleaned:** `{success_count}`\n"
            f"• **Failed:** `{failed}`"
        )
        if cancelled:
            stats_str += f"\n• **Skipped (cancelled):** `{total_images - success_count - failed}`"

        time_str = (
            f"⏱️ **Processing Time:**\n"
            f"• **Download:** {download_time:.2f}s\n"
            f"• **Cleaning:** {clean_time:.2f}s\n"
            f"• **Total:** {total_time:.2f}s"
        )

        final_embed = discord.Embed(
            title=status_title,
            description=f"{cost_str}\n\n{stats_str}\n\n{time_str}",
            color=status_color
        )
        final_embed.set_author(
            name=f"Requested by {interaction.user.display_name}",
            icon_url=interaction.user.display_avatar.url
        )
        if source_name:
            final_embed.add_field(name="📁 Source", value=f"`{source_name}`", inline=False)

        if not results:
            await loading_msg.edit(embed=final_embed, view=None)
            if loading_msg.id in active_extractions:
                del active_extractions[loading_msg.id]
            return

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, data in results:
                zf.writestr(name, data)
            if debug and debug_dict:
                for idx, (fname, _) in enumerate(images):
                    dbg_files = debug_dict.get(idx)
                    if not dbg_files:
                        continue
                    stem = os.path.splitext(os.path.basename(fname))[0]
                    for dbg_name, dbg_bytes in dbg_files:
                        if dbg_bytes:
                            zf.writestr(f"_debug/{stem}/{dbg_name}", dbg_bytes)
        zip_buf.seek(0)
        zip_bytes = zip_buf.getvalue()

        safe_name = sanitize_filename(source_name) if source_name else "cleaned"
        zip_name = f"{safe_name}_cleaned.zip"

        single_image_mode = (len(results) == 1 and not debug)
        if single_image_mode:
            img_name, img_bytes = results[0]
            if len(img_bytes) <= DISCORD_MAX_FILE_SIZE:
                await loading_msg.edit(embed=final_embed, view=None)
                if loading_msg.id in active_extractions:
                    del active_extractions[loading_msg.id]
                file = discord.File(io.BytesIO(img_bytes), filename=img_name)
                await interaction.channel.send(
                    content=f"🧹 Cleaned for {interaction.user.mention}",
                    file=file, reference=loading_msg)
                if url_dedup_key and url_dedup_key in _active_clean_ops:
                    del _active_clean_ops[url_dedup_key]
                return

        gofile_link = None
        try:
            gofile_result = await upload_bytes_to_gofile(zip_bytes, zip_name)
            if gofile_result and gofile_result.get("status") == "ok":
                gofile_link = gofile_result["data"].get("downloadPage")
                logger.info(f"Cleaned ZIP uploaded to GoFile: {gofile_link}")
            else:
                err_msg = gofile_result.get("data", {}).get("message", "Unknown") if gofile_result else "No response"
                logger.warning(f"GoFile upload failed: {err_msg}")
        except Exception as e:
            logger.warning(f"GoFile upload error: {e}")

        if gofile_link:
            final_embed.add_field(
                name="\u200b",
                value=f"📥 **[⬇️ Click Here to Download]({gofile_link})**",
                inline=False
            )

        await loading_msg.edit(embed=final_embed, view=None)

        if loading_msg.id in active_extractions:
            del active_extractions[loading_msg.id]

        if not gofile_link:
            if len(zip_bytes) > DISCORD_MAX_FILE_SIZE:
                chunks = build_zip_chunks(results)
                for idx, chunk_files in enumerate(chunks, 1):
                    chunk_zip = io.BytesIO()
                    with zipfile.ZipFile(chunk_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for name, data in chunk_files:
                            zf.writestr(name, data)
                    chunk_zip.seek(0)
                    suffix = f"_part{idx}" if len(chunks) > 1 else ""
                    file = discord.File(chunk_zip, filename=f"{safe_name}_cleaned{suffix}.zip")
                    await interaction.channel.send(file=file)
            else:
                file = discord.File(io.BytesIO(zip_bytes), filename=zip_name)
                await interaction.channel.send(
                    content=f"🧹 Cleaned for {interaction.user.mention}",
                    file=file, reference=loading_msg)

        if url_dedup_key and url_dedup_key in _active_clean_ops:
            del _active_clean_ops[url_dedup_key]

    ENGINE_CHOICES = [
        app_commands.Choice(name="LaMa (fast, default)", value="lama"),
        app_commands.Choice(name="Stable Diffusion (slow, experimental)", value="sd"),
        app_commands.Choice(name="Compare LaMa vs SD (debug only)", value="compare"),
    ]

    @app_commands.command(
        name="clean",
        description="Clean/remove text from manga images - upload image or ZIP file"
    )
    @app_commands.describe(
        attachment="Upload an image (PNG/JPG/WEBP) or ZIP file containing images",
        file_name="[Optional] Custom name for the output file",
        debug="[Optional] Include mask debug folder (_debug/) in the output ZIP",
        engine="[Optional] Inpainting engine (default: LaMa)"
    )
    @app_commands.choices(engine=ENGINE_CHOICES)
    async def clean_command(
        self,
        interaction: discord.Interaction,
        attachment: discord.Attachment,
        file_name: Optional[str] = None,
        debug: Optional[bool] = False,
        engine: Optional[app_commands.Choice[str]] = None
    ):
        engine_val = engine.value if engine else "lama"
        debug_val = bool(debug) or engine_val == "compare"
        await self._handle_clean_request(
            interaction=interaction,
            source_type='attachment',
            source_data=attachment,
            file_name=file_name,
            debug=debug_val,
            engine=engine_val
        )

    @app_commands.command(
        name="clean_url",
        description="Clean/remove text from manga via URL - GoFile, Google Drive, or direct link"
    )
    @app_commands.describe(
        url="Any URL: GoFile, Google Drive, or direct image/ZIP URL",
        file_name="[Optional] Custom name for the output file",
        debug="[Optional] Include mask debug folder (_debug/) in the output ZIP",
        engine="[Optional] Inpainting engine (default: LaMa)"
    )
    @app_commands.choices(engine=ENGINE_CHOICES)
    async def clean_url_command(
        self,
        interaction: discord.Interaction,
        url: str,
        file_name: Optional[str] = None,
        debug: Optional[bool] = False,
        engine: Optional[app_commands.Choice[str]] = None
    ):
        url = url.strip()
        if not url.startswith('http'):
            await interaction.response.send_message(
                "❌ Please provide a valid URL starting with `http`.",
                ephemeral=True)
            return

        engine_val = engine.value if engine else "lama"
        debug_val = bool(debug) or engine_val == "compare"
        await self._handle_clean_request(
            interaction=interaction,
            source_type='url',
            source_data=url,
            file_name=file_name,
            debug=debug_val,
            engine=engine_val
        )


async def setup(bot: commands.Bot):
    await bot.add_cog(CleanCog(bot))
    logger.info("CleanCog has been loaded.")
