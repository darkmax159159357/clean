import discord
from discord import app_commands
from discord.ext import commands
import os
import logging
import asyncio
import time
from datetime import datetime
from openai import OpenAI
from core.security import is_admin
from prompts.translation_prompt import build_translation_prompt, parse_translation_response

logger = logging.getLogger(__name__)

def _split_text_chunks(text, max_len=1016):
    bubbles = text.split('\n\n')
    chunks = []
    current = ""
    for bubble in bubbles:
        if not bubble.strip():
            continue
        candidate = f"{current}\n\n{bubble}" if current else bubble
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(bubble) <= max_len:
                current = bubble
            else:
                while bubble:
                    if len(bubble) <= max_len:
                        current = bubble
                        break
                    cut = bubble.rfind('\n', 0, max_len)
                    if cut <= 0:
                        cut = max_len
                    chunks.append(bubble[:cut])
                    bubble = bubble[cut:].lstrip('\n')
                    current = ""
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_len]]

def _embed_size(embed: discord.Embed) -> int:
    size = 0
    if embed.title:
        size += len(embed.title)
    if embed.description:
        size += len(embed.description)
    if embed.footer and embed.footer.text:
        size += len(embed.footer.text)
    if embed.author and embed.author.name:
        size += len(embed.author.name)
    for field in embed.fields:
        size += len(field.name or '') + len(field.value or '')
    return size

async def _reply_or_send(target, **kwargs):
    """Reply to a message; if the original was deleted, fall back to channel.send."""
    try:
        return await target.reply(**kwargs)
    except discord.HTTPException as e:
        # 50035 Invalid Form Body / Unknown message in message_reference
        # 10008 Unknown Message — original deleted before we replied
        unknown_ref = (
            e.code in (10008,)
            or (e.code == 50035 and "message_reference" in str(e))
        )
        if not unknown_ref:
            raise
        channel = getattr(target, "channel", None)
        if channel is None:
            raise
        kwargs.pop("mention_author", None)
        return await channel.send(**kwargs)


async def _send_embeds_safe(target, embeds, **kwargs):
    MAX_TOTAL = 5900
    MAX_PER_MSG = 10
    batch = []
    batch_size = 0
    for embed in embeds:
        es = _embed_size(embed)
        if batch and (batch_size + es > MAX_TOTAL or len(batch) >= MAX_PER_MSG):
            await _reply_or_send(target, embeds=batch, **kwargs)
            batch = []
            batch_size = 0
        batch.append(embed)
        batch_size += es
    if batch:
        await _reply_or_send(target, embeds=batch, **kwargs)

def _build_split_embeds(original, translation=None, title="✅ Text Extracted Successfully", color=0x2ECC71):
    embeds = []
    if translation:
        orig_lines = [l.strip() for l in original.split('\n\n') if l.strip()]
        trans_lines = [l.strip() for l in translation.split('\n\n') if l.strip()]
        max_lines = max(len(orig_lines), len(trans_lines))
        while len(orig_lines) < max_lines:
            orig_lines.append("")
        while len(trans_lines) < max_lines:
            trans_lines.append("")

        FIELD_MAX = 1016
        paired_chunks_orig = []
        paired_chunks_trans = []
        cur_orig = ""
        cur_trans = ""
        for o_line, t_line in zip(orig_lines, trans_lines):
            next_orig = f"{cur_orig}\n\n{o_line}" if cur_orig else o_line
            next_trans = f"{cur_trans}\n\n{t_line}" if cur_trans else t_line
            if (len(next_orig) > FIELD_MAX or len(next_trans) > FIELD_MAX) and cur_orig:
                paired_chunks_orig.append(cur_orig)
                paired_chunks_trans.append(cur_trans)
                cur_orig = o_line[:FIELD_MAX]
                cur_trans = t_line[:FIELD_MAX]
            else:
                cur_orig = next_orig[:FIELD_MAX]
                cur_trans = next_trans[:FIELD_MAX]
        if cur_orig or cur_trans:
            paired_chunks_orig.append(cur_orig)
            paired_chunks_trans.append(cur_trans)

        total = len(paired_chunks_orig)
        for idx in range(total):
            part_label = f" (Part {idx+1}/{total})" if total > 1 else ""
            embed = discord.Embed(title=f"{title}{part_label}", color=color)
            o_text = paired_chunks_orig[idx]
            t_text = paired_chunks_trans[idx]
            if o_text:
                embed.add_field(name="📝 Original", value=f"```\n{o_text}\n```", inline=True)
            if t_text:
                embed.add_field(name="🌐 Translation", value=f"```\n{t_text}\n```", inline=True)
            embeds.append(embed)
    else:
        chunks = _split_text_chunks(original, max_len=4080)
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            part_label = f" (Part {idx+1}/{total})" if total > 1 else ""
            embed = discord.Embed(
                title=f"{title}{part_label}",
                description=f"```\n{chunk}\n```",
                color=color,
            )
            embeds.append(embed)
    return embeds

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')

_ocr_prompt_cache = {'text': None, 'mtime': 0}

def _get_ocr_prompt():
    prompt_file = os.path.join(PROMPTS_DIR, 'prompt_general_no_connected.py')
    try:
        current_mtime = os.path.getmtime(prompt_file)
        if _ocr_prompt_cache['text'] and _ocr_prompt_cache['mtime'] == current_mtime:
            return _ocr_prompt_cache['text']
        import importlib, sys
        module_name = 'prompts.prompt_general_no_connected'
        if module_name in sys.modules:
            del sys.modules[module_name]
        mod = importlib.import_module(module_name)
        _ocr_prompt_cache['text'] = mod.PROMPT_GENERAL_NO_CONNECTED
        _ocr_prompt_cache['mtime'] = current_mtime
        return mod.PROMPT_GENERAL_NO_CONNECTED
    except Exception as e:
        logger.error(f"Error loading OCR prompt: {e}")
        return "Extract all text from this manga/comic image. One bubble per line, separated by blank lines."


COOLDOWN_SECONDS = 300
CACHE_TTL = 120
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
MAX_IMAGES_NORMAL = 1
MAX_IMAGES_VIP = 3


TARGET_GUILD_ID = 1389003237893341225


def _is_no_text(text: str) -> bool:
    if not text:
        return True
    s = text.strip()
    return s in ('NO_TEXT_FOUND', '[NO TEXT]', 'NO TEXT')

UNREADABLE_PATTERNS = [
    '읽을 수 없음', '읽을수없음', '판독불가', '判読不能', '読めない',
    '(unreadable)', 'unreadable', '(illegible)', 'illegible',
    '(can\'t read)', '(cannot read)', '(unclear)', '(blurry)',
]

def _has_unreadable_problem(text: str) -> bool:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    unreadable_count = 0
    for line in lines:
        clean = line.lower()
        for pattern in UNREADABLE_PATTERNS:
            if pattern.lower() in clean:
                unreadable_count += 1
                break
    return len(lines) > 0 and unreadable_count / len(lines) > 0.3

def _format_bubbles(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return '\n\n'.join(lines)





class FreeOcrCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.client = OpenAI(
            base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL"),
            api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
        )
        self._cooldowns = {}
        self._config_cache = None
        self._config_cache_time = 0
        self._processing = set()
        self._setup_done = False

    @commands.Cog.listener()
    async def on_ready(self):
        if self._setup_done:
            return
        self._setup_done = True
        await self._auto_setup_ocr()

    def _load_config(self):
        now = time.time()
        if self._config_cache and now - self._config_cache_time < CACHE_TTL:
            return self._config_cache

        from app import app, db
        from models import AdminConfig
        config = {}
        try:
            with app.app_context():
                for key in ('free_ocr_enabled', 'free_ocr_channel_id', 'free_ocr_history_channel_id',
                            'free_ocr_cooldown', 'free_ocr_vip_role_id', 'free_ocr_model'):
                    cfg = db.session.query(AdminConfig).filter_by(key=key).first()
                    config[key] = cfg.value if cfg else None
        except Exception as e:
            logger.error(f"Error loading free OCR config: {e}")

        self._config_cache = config
        self._config_cache_time = now
        return config

    def _is_enabled(self):
        config = self._load_config()
        return config.get('free_ocr_enabled') == 'true'

    def _get_ocr_channel_id(self):
        config = self._load_config()
        val = config.get('free_ocr_channel_id')
        return int(val) if val else None

    def _get_history_channel_id(self):
        config = self._load_config()
        val = config.get('free_ocr_history_channel_id')
        return int(val) if val else None

    def _get_cooldown_seconds(self):
        config = self._load_config()
        val = config.get('free_ocr_cooldown')
        try:
            return int(val) if val else COOLDOWN_SECONDS
        except (ValueError, TypeError):
            return COOLDOWN_SECONDS

    def _get_vip_role_id(self):
        config = self._load_config()
        val = config.get('free_ocr_vip_role_id')
        return int(val) if val else None

    def _get_model(self):
        config = self._load_config()
        return config.get('free_ocr_model') or 'gpt-4o'

    def _get_translation_model(self):
        config = self._load_config()
        return config.get('free_ocr_translation_model') or self._get_model()

    def _is_vip(self, member: discord.Member):
        try:
            from cogs.vip_cog import _get_all_vip_role_ids
            vip_role_ids = _get_all_vip_role_ids()
            if vip_role_ids:
                return any(role.id in vip_role_ids for role in member.roles)
        except Exception:
            pass
        vip_role_id = self._get_vip_role_id()
        if not vip_role_id:
            return False
        return any(role.id == vip_role_id for role in member.roles)

    def _check_cooldown(self, user_id: int) -> int:
        cooldown = self._get_cooldown_seconds()
        if user_id in self._cooldowns:
            elapsed = time.time() - self._cooldowns[user_id]
            if elapsed < cooldown:
                return int(cooldown - elapsed)
        return 0

    def _set_cooldown(self, user_id: int, timestamp: float = None):
        self._cooldowns[user_id] = timestamp or time.time()

    async def _load_cooldown_from_history(self, user_id: int):
        if user_id in self._cooldowns:
            return
        history_channel_id = self._get_history_channel_id()
        if not history_channel_id:
            return
        try:
            channel = self.bot.get_channel(history_channel_id)
            if not channel:
                channel = await self.bot.fetch_channel(history_channel_id)
            if not channel:
                return
            async for msg in channel.history(limit=200):
                if msg.author == self.bot.user and msg.embeds:
                    embed = msg.embeds[0]
                    if embed.footer and embed.footer.text and f'uid:{user_id}' in embed.footer.text:
                        self._set_cooldown(user_id, msg.created_at.timestamp())
                        return
        except Exception as e:
            logger.error(f"Error loading cooldown from history: {e}")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        if not self._is_enabled():
            return

        ocr_channel_id = self._get_ocr_channel_id()
        if not ocr_channel_id or message.channel.id != ocr_channel_id:
            return

        image_urls = []
        for att in message.attachments:
            if any(att.filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_urls.append(att.url)

        if not image_urls:
            return

        user_id = message.author.id
        processing_key = f"{user_id}_{message.id}"

        if processing_key in self._processing:
            return
        self._processing.add(processing_key)

        is_vip = self._is_vip(message.author)
        max_images = MAX_IMAGES_VIP if is_vip else MAX_IMAGES_NORMAL

        try:
            if not is_vip:
                await self._load_cooldown_from_history(user_id)
                remaining = self._check_cooldown(user_id)
                if remaining > 0:
                    minutes = remaining // 60
                    seconds = remaining % 60
                    cooldown_embed = discord.Embed(
                        title="Cooldown",
                        description=f"Please wait **{minutes}m {seconds}s** before sending another image.\nGet VIP for no cooldown!",
                        color=0xFF6B6B,
                    )
                    cooldown_embed.set_footer(text=f"⏱️ Cooldown: {self._get_cooldown_seconds() // 60}min | 🌟 VIP = No cooldown + Auto-translation + 3 images")
                    reply = await message.reply(embed=cooldown_embed, mention_author=False)
                    await asyncio.sleep(10)
                    try:
                        await reply.delete()
                    except Exception:
                        pass
                    return

            if not is_vip and len(image_urls) > MAX_IMAGES_NORMAL:
                limit_embed = discord.Embed(
                    description=f"⚠️ Only **{MAX_IMAGES_NORMAL}** image per message for free users. Get **VIP** to process up to **{MAX_IMAGES_VIP}** images at once!",
                    color=0xFFA500,
                )
                await message.reply(embed=limit_embed, mention_author=False, delete_after=10)

            images_to_process = image_urls[:max_images]
            skipped = len(image_urls) - len(images_to_process)

            async with message.channel.typing():
                if len(images_to_process) == 1:
                    extracted_text = await self._extract_text(images_to_process[0])

                    if extracted_text and not _is_no_text(extracted_text):
                        formatted_text = _format_bubbles(extracted_text)
                        char_count = len(formatted_text)

                        translation = None
                        if is_vip:
                            translation = await self._translate_text(extracted_text)

                        if is_vip and translation:
                            trans_formatted = _format_bubbles(translation)
                            result_embeds = _build_split_embeds(formatted_text, trans_formatted)
                        else:
                            result_embeds = _build_split_embeds(formatted_text)

                        if not is_vip:
                            cooldown_mins = self._get_cooldown_seconds() // 60
                            result_embeds[-1].set_footer(text=f"⏱️ Cooldown: {cooldown_mins}min | 🌟 VIP = No cooldown + Auto-translation + 3 images")

                        await _send_embeds_safe(message, result_embeds, mention_author=False)
                        await self._log_to_history(message, extracted_text, char_count, success=True)
                        try:
                            await message.delete()
                        except (discord.Forbidden, discord.NotFound):
                            pass
                    else:
                        fail_embed = discord.Embed(
                            title="No Text Found",
                            description="No readable text was detected in your image.\nTry sending a clearer image with visible text.",
                            color=0xE74C3C,
                        )
                        await message.reply(embed=fail_embed, mention_author=False)
                        await self._log_to_history(message, None, 0, success=False)
                        try:
                            await message.delete()
                        except (discord.Forbidden, discord.NotFound):
                            pass
                else:
                    ocr_tasks = [self._extract_text(url) for url in images_to_process]
                    ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

                    extracted_texts = []
                    for result in ocr_results:
                        if isinstance(result, Exception) or _is_no_text(result):
                            extracted_texts.append(None)
                        else:
                            extracted_texts.append(result)

                    if is_vip:
                        async def _noop():
                            return None

                        trans_tasks = []
                        for text in extracted_texts:
                            if text:
                                trans_tasks.append(self._translate_text(text))
                            else:
                                trans_tasks.append(_noop())
                        trans_results = await asyncio.gather(*trans_tasks, return_exceptions=True)
                    else:
                        trans_results = [None] * len(extracted_texts)

                    embeds = []
                    total_chars = 0
                    any_success = False

                    for i, (ocr_result, trans_result) in enumerate(zip(ocr_results, trans_results)):
                        if isinstance(ocr_result, Exception):
                            embed = discord.Embed(
                                title=f"❌ Image {i+1}",
                                description="Error processing this image.",
                                color=0xE74C3C,
                            )
                        elif extracted_texts[i]:
                            formatted = _format_bubbles(extracted_texts[i])
                            total_chars += len(formatted)
                            any_success = True

                            if is_vip and not isinstance(trans_result, Exception) and trans_result:
                                trans_formatted = _format_bubbles(trans_result)
                                img_embeds = _build_split_embeds(formatted, trans_formatted, title=f"✅ Image {i+1}")
                            else:
                                img_embeds = _build_split_embeds(formatted, title=f"✅ Image {i+1}")
                            embeds.extend(img_embeds)
                            continue
                        else:
                            embed = discord.Embed(
                                title=f"⚠️ Image {i+1}",
                                description="No text found.",
                                color=0xFFA500,
                            )
                        embeds.append(embed)

                    if skipped > 0:
                        skip_embed = discord.Embed(
                            description=f"⚠️ {skipped} image(s) skipped — VIP limit is {max_images} per message.",
                            color=0xFFA500,
                        )
                        embeds.append(skip_embed)

                    await _send_embeds_safe(message, embeds, mention_author=False)
                    await self._log_to_history(
                        message, f"[Multi-image: {len(images_to_process)} images]",
                        total_chars, success=any_success
                    )
                    try:
                        await message.delete()
                    except (discord.Forbidden, discord.NotFound):
                        pass

            if not is_vip:
                self._set_cooldown(user_id)

        except Exception as e:
            logger.error(f"Free OCR error: {e}", exc_info=True)
            try:
                error_embed = discord.Embed(
                    title="Error",
                    description="An error occurred while processing your image. Please try again.",
                    color=0xE74C3C,
                )
                await message.reply(embed=error_embed, mention_author=False)
            except Exception:
                pass
        finally:
            self._processing.discard(processing_key)

    async def _download_image(self, url: str) -> bytes:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.read()
        return None

    def _split_tall_image(self, img_bytes: bytes, max_segments: int = 10, overlap_pct: float = 0.90) -> list:
        from io import BytesIO
        from PIL import Image
        import base64
        import time as _time

        t0 = _time.monotonic()
        img = Image.open(BytesIO(img_bytes))
        original_format = img.format or 'JPEG'
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        width, height = img.size
        img.load()
        logger.info(f"Original image: {width}x{height}, format={original_format}, size={len(img_bytes)} bytes")

        min_segment_h = 800
        if height <= min_segment_h * 1.5:
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=95, subsampling=0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            logger.info(f"Single segment (no split): {len(buf.getvalue())//1024}KB in {_time.monotonic()-t0:.2f}s")
            return [f"data:image/jpeg;base64,{b64}"]

        num_segments = min(max_segments, max(2, height // min_segment_h))
        segment_h = int(height / (1.0 + (num_segments - 1) * (1.0 - overlap_pct)))
        segment_h = max(segment_h, min_segment_h)
        segment_h = min(segment_h, height)
        step = max(1, int(segment_h * (1.0 - overlap_pct)))

        actual_needed = 1 + max(0, (height - segment_h + step - 1) // step)
        num_segments = min(int(actual_needed), max_segments)

        crop_coords = []
        for i in range(num_segments):
            y_start = i * step
            y_end = min(y_start + segment_h, height)
            if i == num_segments - 1:
                y_end = height
                y_start = max(0, y_end - segment_h)
            crop_coords.append((y_start, y_end))

        def _process_segment(idx, y_start, y_end):
            crop = img.crop((0, y_start, width, y_end))
            buf = BytesIO()
            crop.save(buf, format='JPEG', quality=95, subsampling=0)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return idx, f"data:image/jpeg;base64,{b64}", len(buf.getvalue()), crop.size

        from concurrent.futures import ThreadPoolExecutor
        segments = [None] * num_segments
        with ThreadPoolExecutor(max_workers=min(4, num_segments)) as pool:
            futures = [
                pool.submit(_process_segment, i, ys, ye)
                for i, (ys, ye) in enumerate(crop_coords)
            ]
            for future in futures:
                idx, data_url, raw_size, (cw, ch) = future.result()
                segments[idx] = data_url
                ys, ye = crop_coords[idx]
                if idx > 0:
                    prev_end = min(crop_coords[idx-1][0] + segment_h, height)
                    actual_overlap = max(0, prev_end - ys)
                    overlap_ratio = actual_overlap / (ye - ys) if (ye - ys) > 0 else 0
                else:
                    actual_overlap = 0
                    overlap_ratio = 0
                logger.info(
                    f"Segment {idx+1}/{num_segments}: y={ys}-{ye} "
                    f"(h={ye-ys}px, {cw}x{ch}, overlap={overlap_ratio:.0%}, "
                    f"size={raw_size//1024}KB)"
                )

        elapsed = _time.monotonic() - t0
        logger.info(
            f"Smart split: {width}x{height} -> {num_segments} segments in {elapsed:.1f}s, "
            f"segment_h={segment_h}px, step={step}px, overlap={overlap_pct:.0%}"
        )
        return segments

    def _deduplicate_overlapping_text(self, texts: list) -> str:
        if not texts:
            return ""
        if len(texts) == 1:
            return self._dedup_within_segment(texts[0])

        from difflib import SequenceMatcher

        def _normalize(line: str) -> str:
            return line.strip().lower()

        def _is_duplicate(norm: str, seen: list, threshold: float = 0.60) -> bool:
            if not norm:
                return True
            for ex in seen:
                if SequenceMatcher(None, norm, ex).ratio() >= threshold:
                    return True
            return False

        seen_lines = []
        result_lines = []

        for idx, segment_text in enumerate(texts):
            deduped_segment = self._dedup_within_segment(segment_text)
            current_lines = deduped_segment.strip().split('\n')
            added = 0
            skipped = 0
            for line in current_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if _is_duplicate(_normalize(stripped), seen_lines):
                    skipped += 1
                else:
                    result_lines.append(stripped)
                    added += 1
            seen_lines.extend(
                _normalize(l.strip()) for l in current_lines if l.strip()
            )
            logger.info(f"Dedup: segment {idx+1} kept={added}, skipped={skipped}")

        return '\n'.join(result_lines)

    def _dedup_within_segment(self, text: str) -> str:
        from difflib import SequenceMatcher
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return text

        result = [lines[0]]
        seen_norms = [lines[0].strip().lower()]

        for line in lines[1:]:
            norm = line.strip().lower()
            is_dup = False
            for s in seen_norms:
                if SequenceMatcher(None, norm, s).ratio() >= 0.85:
                    is_dup = True
                    break
            if not is_dup:
                result.append(line)
                seen_norms.append(norm)

        removed = len(lines) - len(result)
        if removed > 0:
            logger.info(f"Intra-segment dedup: removed {removed} duplicate lines from {len(lines)}")
        return '\n'.join(result)

    async def _extract_text(self, image_url: str) -> str:
        prompt = _get_ocr_prompt()
        model = self._get_model()

        try:
            img_bytes = await self._download_image(image_url)
            if not img_bytes:
                logger.warning(f"Failed to download image, trying direct URL")
                return await self._extract_single(image_url, prompt, model)

            logger.info(f"Downloaded image: {len(img_bytes)} bytes")
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(None, self._split_tall_image, img_bytes)
            logger.info(f"Image split into {len(segments)} segment(s)")

            if len(segments) == 1:
                result = await self._extract_single(segments[0], prompt, model)

                if result and not _is_no_text(result) and _has_unreadable_problem(result):
                    logger.warning(f"OCR returned unreadable placeholders, retrying with stronger prompt")
                    retry_msg = (
                        "The image may be low quality but you MUST try to read the actual text. "
                        "NEVER write 'unreadable', '읽을 수 없음', or any placeholder. "
                        "Read each character as best you can, even if blurry. Output your best guess for every bubble."
                    )
                    result = await self._extract_single(segments[0], prompt, model, user_msg=retry_msg)

                    if result and _has_unreadable_problem(result):
                        logger.warning(f"OCR retry still returned unreadable, returning result as-is")

                logger.info(f"Single segment result: {'found text' if result and not _is_no_text(result) else 'no text'}")
                return result

            tasks = [
                self._extract_single(
                    seg_url, prompt, model,
                    user_msg=f'Extract all text from this image segment ({i+1}/{len(segments)}). Read each character individually and precisely. IMPORTANT: Every output line MUST start with a valid tag ("": or :: or (): or []: or <>: or SFX: or ST: or OT:). No untagged lines! One bubble = one line (join multi-row text). Do NOT hallucinate text you cannot clearly see.'
                )
                for i, seg_url in enumerate(segments)
            ]
            results = await asyncio.gather(*tasks)

            all_text = []
            needs_retry = []
            for i, part_text in enumerate(results):
                if part_text is None:
                    logger.warning(f"Segment {i+1}/{len(segments)} returned None (API error)")
                elif _is_no_text(part_text):
                    logger.info(f"Segment {i+1}/{len(segments)} had no text")
                elif _has_unreadable_problem(part_text):
                    logger.warning(f"Segment {i+1}/{len(segments)} has unreadable placeholders, will retry")
                    needs_retry.append(i)
                else:
                    logger.info(f"Segment {i+1}/{len(segments)} found {len(part_text)} chars")
                    all_text.append(part_text.strip())

            if needs_retry:
                retry_tasks = [
                    self._extract_single(
                        segments[i], prompt, model,
                        user_msg=(
                            f"Extract text from segment {i+1}/{len(segments)}. "
                            "The image may be low quality but you MUST read the actual text. "
                            "NEVER write 'unreadable' or '읽을 수 없음'. Output your best guess for every character."
                        )
                    )
                    for i in needs_retry
                ]
                retry_results = await asyncio.gather(*retry_tasks)
                for idx, retry_text in zip(needs_retry, retry_results):
                    if retry_text and not _is_no_text(retry_text):
                        logger.info(f"Segment {idx+1} retry: {len(retry_text)} chars")
                        all_text.append(retry_text.strip())

            if not all_text:
                logger.warning(f"All {len(segments)} segments returned no text")
                return "[NO TEXT]"

            combined = self._deduplicate_overlapping_text(all_text)
            logger.info(f"Combined {len(all_text)} segments -> {len(combined)} chars (deduped)")
            return combined

        except Exception as e:
            logger.error(f"Free OCR extraction error: {e}", exc_info=True)
            return None

    def _save_ocr_cost(self, model: str, input_tokens: int, output_tokens: int):
        MODEL_PRICES = {
            'gpt-4o':       (2.50, 10.00),
            'gpt-4o-mini':  (0.15,  0.60),
            'gpt-4.1':      (2.00,  8.00),
            'gpt-4.1-mini': (0.40,  1.60),
            'gpt-4.1-nano': (0.10,  0.40),
            'gpt-5.2':      (3.00, 12.00),
            'gpt-5.1':      (3.00, 12.00),
            'gpt-5':        (3.00, 12.00),
            'gpt-5-mini':   (1.50,  6.00),
            'gpt-5-nano':   (0.50,  2.00),
            'o4-mini':      (1.10,  4.40),
            'o3':           (2.00, 8.00),
            'o3-mini':      (1.10,  4.40),
        }
        inp_price, out_price = MODEL_PRICES.get(model, (2.50, 10.00))
        cost = (input_tokens * inp_price + output_tokens * out_price) / 1_000_000
        try:
            from app import app, db
            from models import AdminConfig
            with app.app_context():
                for key, delta in [('ocr_total_cost', cost), ('ocr_total_extractions', 1)]:
                    cfg = db.session.query(AdminConfig).filter_by(key=key).first()
                    if cfg:
                        cfg.value = str(round(float(cfg.value) + delta, 6))
                    else:
                        db.session.add(AdminConfig(key=key, value=str(round(delta, 6))))
                db.session.commit()
        except Exception as e:
            logger.error(f"Error saving OCR cost: {e}")
        return cost

    async def _extract_single(self, image_url: str, prompt: str, model: str, user_msg: str = None) -> str:
        try:
            return await self._ocr_api_call(image_url, prompt, model, user_msg)
        except self._ContentFilterError:
            logger.warning("Content filter hit — splitting segment into 2 halves and retrying")
            sub_results = await self._retry_with_splits(image_url, prompt, model, user_msg, depth=1)
            if sub_results:
                return "\n".join(sub_results)
            logger.warning("All content-filter retries failed for this segment")
            return None
        except Exception as e:
            logger.error(f"Free OCR single extraction error: {e}", exc_info=True)
            return None

    class _ContentFilterError(Exception):
        pass

    async def _ocr_api_call(self, image_url: str, prompt: str, model: str, user_msg: str = None) -> str:
        url_preview = image_url[:80] + "..." if len(image_url) > 80 else image_url
        logger.info(f"OCR API call: model={model}, url={url_preview}")
        chat_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_msg or "Extract all text from this image. Read each character individually and precisely."},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                ]}
            ],
            "temperature": 0.0,
        }
        if model.startswith(('gpt-5', 'o3', 'o4')):
            chat_kwargs["max_completion_tokens"] = 4000
        else:
            chat_kwargs["max_tokens"] = 4000
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **chat_kwargs,
            )
        except Exception as e:
            err_str = str(e).lower()
            if 'content_filter' in err_str or 'responsibleaipolicyviolation' in err_str:
                raise self._ContentFilterError(str(e))
            raise
        text = response.choices[0].message.content.strip()
        logger.info(f"OCR API response: {len(text)} chars, finish_reason={response.choices[0].finish_reason}")
        if response.usage:
            await asyncio.to_thread(
                self._save_ocr_cost, model,
                response.usage.prompt_tokens, response.usage.completion_tokens
            )
        return text

    async def _retry_with_splits(self, image_url: str, prompt: str, model: str, user_msg: str, depth: int) -> list:
        MAX_DEPTH = 3
        if depth > MAX_DEPTH:
            return []

        halves = await asyncio.to_thread(self._split_base64_image_in_half, image_url)
        if not halves:
            return []

        results = []
        for i, half_url in enumerate(halves):
            label = f"half-{i+1} depth-{depth}"
            try:
                text = await self._ocr_api_call(half_url, prompt, model, user_msg)
                if text:
                    logger.info(f"Content-filter retry {label}: got {len(text)} chars")
                    results.append(text)
            except self._ContentFilterError:
                logger.warning(f"Content-filter retry {label}: still blocked, splitting deeper")
                sub = await self._retry_with_splits(half_url, prompt, model, user_msg, depth + 1)
                results.extend(sub)
            except Exception as e:
                logger.error(f"Content-filter retry {label} error: {e}")
        return results

    def _split_base64_image_in_half(self, data_url: str) -> list:
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            header, b64_data = data_url.split(',', 1)
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(BytesIO(img_bytes))
            w, h = img.size
            if h < 200:
                return []
            mid = h // 2
            halves = []
            for (y_start, y_end) in [(0, mid), (mid, h)]:
                crop = img.crop((0, y_start, w, y_end))
                buf = BytesIO()
                crop.save(buf, format='JPEG', quality=95, subsampling=0)
                b64 = base64.b64encode(buf.getvalue()).decode()
                halves.append(f"data:image/jpeg;base64,{b64}")
            logger.info(f"Split image {w}x{h} into 2 halves: top={w}x{mid}, bottom={w}x{h-mid}")
            return halves
        except Exception as e:
            logger.error(f"Error splitting image: {e}")
            return []

    async def _translate_text(self, text: str) -> str:
        try:
            system_prompt, user_prompt = build_translation_prompt(
                source_text=text,
                glossary="",
                source_language="auto",
            )
            model = self._get_translation_model()
            chat_kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
            }
            if model.startswith(('gpt-5', 'o3', 'o4')):
                chat_kwargs["max_completion_tokens"] = 4000
            else:
                chat_kwargs["max_tokens"] = 4000

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **chat_kwargs,
            )
            raw = response.choices[0].message.content.strip()

            if response.usage:
                await asyncio.to_thread(
                    self._save_ocr_cost, model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

            parsed = parse_translation_response(raw)
            if parsed.get('translation'):
                return parsed['translation']
            return raw
        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            return None

    async def _log_to_history(self, message: discord.Message, text: str, char_count: int, success: bool):
        history_channel_id = self._get_history_channel_id()
        if not history_channel_id:
            return

        try:
            channel = self.bot.get_channel(history_channel_id)
            if not channel:
                channel = await self.bot.fetch_channel(history_channel_id)

            if not channel:
                return

            if success:
                embed = discord.Embed(
                    title="OCR Extraction Log",
                    color=0x2ECC71,
                )
                embed.add_field(name="User", value=f"{message.author.display_name}", inline=True)
                embed.add_field(name="Characters", value=str(char_count), inline=True)
                text_preview = f"```\n{text[:990]}\n```" if text else "N/A"
                embed.add_field(name="Extracted Text", value=text_preview, inline=False)
                embed.set_thumbnail(url=message.author.display_avatar.url if message.author.display_avatar else None)
                embed.set_footer(text=f"uid:{message.author.id}")
                embed.timestamp = datetime.utcnow()
            else:
                embed = discord.Embed(
                    title="OCR Extraction Log — No Text Found",
                    color=0xE74C3C,
                )
                embed.add_field(name="User", value=f"{message.author.display_name}", inline=True)
                embed.set_thumbnail(url=message.author.display_avatar.url if message.author.display_avatar else None)
                embed.set_footer(text=f"uid:{message.author.id}")
                embed.timestamp = datetime.utcnow()

            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Error logging to history channel: {e}")

    def _save_config_batch(self, updates: dict) -> bool:
        from app import app, db
        from models import AdminConfig
        try:
            with app.app_context():
                for key, value in updates.items():
                    cfg = db.session.query(AdminConfig).filter_by(key=key).first()
                    if cfg:
                        cfg.value = value
                    else:
                        db.session.add(AdminConfig(key=key, value=value))
                db.session.commit()
            self._config_cache = None
            self._config_cache_time = 0
            return True
        except Exception as e:
            logger.error(f"Error saving config batch: {e}")
            try:
                db.session.rollback()
            except Exception:
                pass
            return False

    async def _auto_setup_ocr(self):
        try:
            guild = self.bot.get_guild(TARGET_GUILD_ID)
            if not guild:
                logger.warning(f"OCR auto-setup: Guild {TARGET_GUILD_ID} not found")
                return

            config = self._load_config()
            ocr_ch_id = config.get('free_ocr_channel_id')
            if ocr_ch_id:
                existing = guild.get_channel(int(ocr_ch_id))
                if existing:
                    logger.info(f"OCR auto-setup: Already configured (channel {ocr_ch_id}), skipping")
                    return

            existing_category = None
            for cat in guild.categories:
                if cat.name.upper() == "OCR SYSTEM":
                    existing_category = cat
                    break

            if existing_category:
                category = existing_category
            else:
                category = await guild.create_category(
                    "OCR SYSTEM",
                    reason="Free OCR System auto-setup"
                )
                logger.info(f"OCR auto-setup: Created category '{category.name}' in {guild.name}")

            ocr_channel = None
            history_channel = None
            for ch in category.text_channels:
                if ch.name == "free-ocr":
                    ocr_channel = ch
                elif ch.name == "history":
                    history_channel = ch

            if not ocr_channel:
                ocr_channel = await category.create_text_channel(
                    "free-ocr",
                    topic="Send images here to extract text automatically | ابعت صور هنا لاستخراج النصوص",
                    reason="Free OCR System auto-setup"
                )
                logger.info(f"OCR auto-setup: Created #free-ocr ({ocr_channel.id})")

            if not history_channel:
                history_overwrites = {
                    guild.default_role: discord.PermissionOverwrite(view_channel=False),
                    guild.me: discord.PermissionOverwrite(
                        view_channel=True, send_messages=True, embed_links=True
                    ),
                }
                for role in guild.roles:
                    if role.permissions.administrator and not role.is_default():
                        history_overwrites[role] = discord.PermissionOverwrite(view_channel=True)

                history_channel = await category.create_text_channel(
                    "history",
                    topic="OCR extraction logs",
                    overwrites=history_overwrites,
                    reason="Free OCR System auto-setup"
                )
                logger.info(f"OCR auto-setup: Created #history ({history_channel.id}) [private]")

            saved = self._save_config_batch({
                'free_ocr_channel_id': str(ocr_channel.id),
                'free_ocr_history_channel_id': str(history_channel.id),
                'free_ocr_enabled': 'true',
            })

            if saved:
                logger.info(f"OCR auto-setup: Complete! free-ocr={ocr_channel.id}, history={history_channel.id}, enabled=true")
            else:
                logger.error("OCR auto-setup: Channels created but failed to save config to DB")

        except discord.Forbidden:
            logger.error("OCR auto-setup: Missing 'Manage Channels' permission")
        except Exception as e:
            logger.error(f"OCR auto-setup error: {e}", exc_info=True)

    @app_commands.command(
        name="setup-ocr",
        description="[Admin] Setup OCR System category and channels"
    )
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_channels=True)
    async def setup_ocr(self, interaction: discord.Interaction):
        if not is_admin(str(interaction.user.id)):
            await interaction.response.send_message(
                "This command is for admins only.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True)

        guild = interaction.guild
        if not guild:
            await interaction.followup.send("This command must be used in a server.", ephemeral=True)
            return

        try:
            existing_category = None
            for cat in guild.categories:
                if cat.name.upper() == "OCR SYSTEM":
                    existing_category = cat
                    break

            if existing_category:
                category = existing_category
            else:
                category = await guild.create_category(
                    "OCR SYSTEM",
                    reason="Free OCR System setup"
                )

            ocr_channel = None
            history_channel = None
            for ch in category.text_channels:
                if ch.name == "free-ocr":
                    ocr_channel = ch
                elif ch.name == "history":
                    history_channel = ch

            if not ocr_channel:
                ocr_channel = await category.create_text_channel(
                    "free-ocr",
                    topic="Send images here to extract text automatically",
                    reason="Free OCR System setup"
                )

            if not history_channel:
                history_overwrites = {
                    guild.default_role: discord.PermissionOverwrite(view_channel=False),
                    guild.me: discord.PermissionOverwrite(
                        view_channel=True, send_messages=True, embed_links=True
                    ),
                }
                for role in guild.roles:
                    if role.permissions.administrator and not role.is_default():
                        history_overwrites[role] = discord.PermissionOverwrite(view_channel=True)

                history_channel = await category.create_text_channel(
                    "history",
                    topic="OCR extraction logs",
                    overwrites=history_overwrites,
                    reason="Free OCR System setup"
                )

            saved = self._save_config_batch({
                'free_ocr_channel_id': str(ocr_channel.id),
                'free_ocr_history_channel_id': str(history_channel.id),
                'free_ocr_enabled': 'true',
            })

            if not saved:
                await interaction.followup.send(
                    "Channels created but failed to save settings to database. "
                    "Please set the channel IDs manually from the dashboard.",
                    ephemeral=True
                )
                return

            result_embed = discord.Embed(
                title="OCR System Setup Complete",
                color=0x2ECC71,
            )
            result_embed.add_field(
                name="Category",
                value=f"**{category.name}**",
                inline=False
            )
            result_embed.add_field(
                name="Free OCR Channel",
                value=f"{ocr_channel.mention} (`{ocr_channel.id}`)",
                inline=True
            )
            result_embed.add_field(
                name="History Channel",
                value=f"{history_channel.mention} (`{history_channel.id}`)",
                inline=True
            )
            result_embed.add_field(
                name="Status",
                value="Enabled | Cooldown: 5 min | Model: gpt-4o",
                inline=False
            )
            result_embed.set_footer(text="Channel IDs saved automatically. Use the dashboard to adjust settings.")

            await interaction.followup.send(embed=result_embed, ephemeral=True)
            logger.info(f"OCR System setup in guild {guild.id} by {interaction.user.id}: "
                        f"ocr={ocr_channel.id}, history={history_channel.id}")

        except discord.Forbidden:
            await interaction.followup.send(
                "I don't have permission to create channels/categories. Please give me **Manage Channels** permission.",
                ephemeral=True
            )
        except Exception as e:
            logger.error(f"Error in setup-ocr: {e}", exc_info=True)
            await interaction.followup.send(
                "An error occurred during setup. Please check bot logs for details.",
                ephemeral=True
            )


async def setup(bot):
    await bot.add_cog(FreeOcrCog(bot))
