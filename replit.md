# LALA Bot — Discord Bot with Web Dashboard

## Overview
The LALA Bot project integrates a Discord bot with a Flask web dashboard to automate manga/comic extraction using Google Gemini AI and other AI providers. Its core function is to acquire content from various sources, extract text using AI, and manage user interactions and subscriptions. Content is stored and displayed in its original language. The web dashboard provides comprehensive administration capabilities, including user management, API key configuration, usage tracking, and "Follow The New" (FTN) chapter tracking. The project aims to become a robust, AI-powered platform for manga and comic enthusiasts, offering advanced content acquisition and management.

## User Preferences
- **Main Server ID**: 1389003237893341225 (السيرفر الرئيسي للبوت)
- Pricing Display: Hardcoded prices removed from command descriptions to allow flexible pricing via dashboard settings
- Cost Display: All costs in the dashboard are now displayed in USD ($) with proper conversion from points (1 point = $0.10, configurable via settings)
- Date Format: Gregorian dates used in dashboard tables (en-GB locale)
- Timezone: Egypt (Africa/Cairo, UTC+2) - TZ env var set, all display dates converted from UTC via `to_egypt()` utility, date filters convert Egypt boundaries to UTC for DB queries. All "today" calculations (dashboard stats, chart, operations) consistently use Egypt midnight as day boundary.
- Stats Metric: All dashboard/chart/operations statistics count chapters (`sum(Extraction.chapters_count)`) not extraction records, ensuring consistency across all views. `Guild.total_extractions` and `GuildUser.total_extractions` are denormalized counters kept in sync with Extraction records. `/api/stats` and `get_realtime_stats()` both use `SUM(Extraction.chapters_count)` as the single source of truth.
- API Cost Filtering: `/api/extractions/stats` splits API cost calculation around `period_start` (OpenRouter billing reset). Pre-period costs use per-record token calculation, post-period uses OpenRouter live data (when no non-date filters). Date filters are properly bounded with `filter_to_utc` in both segments to avoid overcounting.

## System Architecture
The project utilizes a Flask web dashboard for administration and an optional Discord bot for user interaction. The system architecture emphasizes a multi-provider AI framework for text extraction, offering dynamic mode/provider management, automatic fallback, and smart failover mechanisms. Security features like CSRF protection and rate limiting are integrated into the dashboard.

**Key Architectural Decisions & Features:**

*   **Multi-Provider AI Architecture**: Supports Google Gemini, OpenRouter, NeuroRouters, and OpenAI with dynamic management, automatic fallback, smart failover, and an API key auto-ban system. Includes cross-provider image-level fallback and project-aware quota management.
*   **Quality Tiers**: "Normal" and "Advanced" modes for extraction with configurable costs and separate pricing for Manga B&W.
*   **Intelligent Image Processing**: Features smart image splitting, Japanese vertical text extraction, optimized splitting, and a junk image filter.
*   **Unified `/extract` Command**: Handles various extraction sources (GoFile, Google Drive, Discord, Direct URLs) with automatic source detection.
*   **Automated Account Management**: Integration with Tapas.io, Lezhin Comics, and Toptoon Korea using Playwright for session management.
*   **Per-Guild Settings**: Customizable extraction costs, initial balances, command toggles, daily limits, and allowed channels per Discord server.
*   **Triple Server Mode**: Supports global, independent, or dependent user balances.
*   **Auto-Resume Downloads**: Automatic retry and cleanup for failed downloads.
*   **Real-time Statistics**: Dashboard stats update every 5 seconds, including alerts and bot status monitoring.
*   **Advanced FTN Tools**: Bulk management, filtering, notification history, and statistics for "Follow The New" subscriptions.
*   **Data Export System**: CSV, JSON, and Excel export with date range filtering.
*   **Modern Dashboard Design**: Dark theme with cyan/teal accents, modern stat cards, activity charts, and enhanced tables.
*   **Accurate Cost & Profit Calculations**: Monthly OpenRouter profit tracking and real API cost tracking.
*   **Multi-Image Single Request Optimization**: Tall image chunks are sent in a single API request with explicit part labeling.
*   **User Profiles System**: Users can save and switch between named extraction settings profiles.
*   **Google Drive Enhancements**: Reliable file listing and robust download retry mechanism.
*   **Shared HTTP Session**: Centralized `aiohttp.ClientSession` for efficient resource management.
*   **Environment Variable Configuration**: Sensitive IDs loaded from environment variables with fallback defaults.
*   **GoFile Cloud Backup**: Automatic database backup to GoFile every 6 hours, with automatic deletion of backups older than 3 days. Local backup every 24 hours (keeps last 7 copies).
*   **Dashboard Member Balance Adjustment**: Provides an interface for adjusting member balances in dependent servers.
*   **Security Hardening**: Atomic balance updates, database constraints, API input validation, CSRF token auto-refresh, timing-attack resistant password comparison, session fingerprinting, and generic error messages.
*   **Hallucination Detection & Prevention**: Multi-layer defense: (1) Anti-looping rules in all extraction prompts and API system instructions prevent AI from generating repetitive output at the source, (2) API-level hallucination detection triggers retries, (3) post-processing cleanup catches any remaining repetitive content. Compression-ratio based detection with phrase deduplication for dense handwritten/CJK text pages. Rules carefully allow legitimate repetition (same text in different bubbles/panels, emphasis) while blocking looping hallucinations.
*   **AI Reasoning Filter**: Automatically removes AI model's "thinking" output.
*   **Japanese Mode Optimization**: Forces image splitting off for Japanese extraction.
*   **B&W Manga Auto-Detection**: Analyzes image saturation (HSV) to detect B&W manga when users select Normal/General mode. Samples up to 5 images evenly across the chapter — requires ALL sampled images to be B&W (mean saturation < 12, >97% pixels with saturation < 25). When detected, auto-switches to Japanese mode (prompt + pricing) and notifies the user. Works across all 3 extraction paths (GoFile, Google Drive, direct/ZIP). Minimum 3 images required for detection to avoid false positives on small batches.
*   **Translation System Enhancements**: Dual-Glossary Architecture, enhanced prompts, and retry/backoff for AI API calls.
*   **Interactive Menu Systems**: `/menu` and `/server` commands use discord.py Components V2.
*   **Watermark Text Filter**: Post-extraction filter to remove watermark-related lines.
*   **Purchase Request Fixes**: Enhanced proof image handling, duplicate request prevention, race condition prevention, and automatic expiration.
*   **Interaction Safety Patterns**: Deferral of button/view callbacks, `db.session.expunge()` for SQLAlchemy objects outside `app_context`, and `rollback()` on DB commit failures.
*   **AI Chat Feature (Lala)**: Global unified chat system using GPT-5.2 (configurable) via OpenAI. Supports custom models/prompts, user memory, image generation, cost tracking, and bot support knowledge base.
*   **Free OCR System**: Cog for image attachment text extraction using OpenAI vision in a configured channel, with cooldowns and optional logging.
*   **Manga Typesetting System**: AI-powered manga typesetting via OpenRouter. Supports multi-chapter processing from GoFile and Google Drive, with Normal and Advanced quality modes. Utilizes content masking for NSFW handling.
*   **AI Text Cleaning (Whitening/تبييض)**: GPU pipeline for manga/comic/manhwa text removal, powered by zyddnys's `comictextdetector.pt.onnx` (the same dual-head DBNet+YOLO model used by manga-image-translator and BallonsTranslator). Models: models/comictextdetector.pt.onnx (auto-downloads from GitHub on first call) and models/anime-manga-big-lama.pt (LaMa inpainting, FP32 — FFT layers require fp32). Pipeline stages: (1) Detection (ctd/ctd_onnx_detector.py): a single ONNX session produces a per-pixel text-segmentation mask at the image's original resolution — no custom bubble heuristics. (2) Mask enhancement (ctd/easyocr_enhancer.py): EasyOCR singleton reader (ja+ko+en, CPU) fills gaps — catches text the neural segmenter missed. (3) Inpainting (ctd/lama_inpaint.py / ctd/sd_inpaint.py): the enhanced mask is fed straight to LaMa (default) or Stable Diffusion ("sd"/"compare" engine), which redraws the background. LaMa output is then merged back via a per-component gradient-aware blend: for bubbles whose surrounding ring shows a real color gradient, the cleaned region is replaced with cv2.inpaint(NS) of the original (a smooth color field interpolated from pixels OUTSIDE the dilated mask only — never reads in-mask text pixels) plus LaMa's high-frequency residual; for plain/uniform bubbles LaMa's output is kept as-is. Followed by a feathered blend for soft edges. SD runs in repaint mode (empty prompt, guidance_scale=1.0, strength=0.7, expanded mask context, exhaustive negative prompt covering people/anime/objects) so it only fills the background without generating new content. Server splits very tall images (>CHUNK_MAX_HEIGHT) at safe whitespace boundaries so each chunk runs independently. Files: ctd/__init__.py, ctd/ctd_onnx_detector.py, ctd/easyocr_enhancer.py, ctd/lama_inpaint.py, ctd/sd_inpaint.py. Commands: /clean (attachment: image or ZIP) and /clean_url (GoFile, Google Drive, direct URL). Progress embeds with progress bar, stop button, cost deduction, balance checks, ZIP chunking for large outputs, duplicate prevention.
*   **V2 Pipeline (Extract → Translate → Edit)**: Primary pipeline uses a 3-step flow: text extraction, text-only translation, and AI-powered text replacement on the image. Enforces horizontal English text.
*   **VIP Package System (v2)**: 5-tier subscription system with role degradation, purchasable with points, and dashboard management.

## External Dependencies
*   **Database**: PostgreSQL, SQLite
*   **AI/ML**: Google Gemini AI (gemini-3-flash, gemini-3-pro), NeuroRouters, OpenRouter, OpenAI (via Replit Integration), PyTorch (CPU, comictextdetector.pt for text segmentation)
*   **Web Framework**: Flask, Gunicorn
*   **Discord API**: `discord.py` library
*   **Scraping**: Playwright, Chromium
*   **Database ORM**: SQLAlchemy
*   **Flask Extensions**: Flask-WTF, Flask-Limiter, ProxyFix, Flask-Dance, Flask-Login
*   **Authentication**: PyJWT

## GPU Server Integration (Optional)
*   **Purpose**: Offload CTD text detection + LaMa inpainting to a remote GPU server for ~5-10x faster cleaning
*   **Recommended GPU**: RTX 4070 (12GB VRAM) on vast.ai — $0.080/hr. Handles 15 chapters (20 imgs each) in ~40-60s
*   **Setup**: Deploy `gpu_server/server.py` on vast.ai (or any GPU machine). Copy `models/` and `ctd/` folders to the server. Run `bash gpu_server/setup.sh` then `python server.py --port 7860`
*   **Env Var**: Set `GPU_CLEAN_SERVER_URL` (e.g. `http://your-vast-ip:7860`) — bot auto-detects at startup
*   **Fallback**: If GPU server is down or unreachable, automatically falls back to local CPU processing
*   **Batch Processing**: `/clean_stream` endpoint accepts multiple images in one request, processes sequentially on GPU, returns binary stream. Bot sends images in batches of 8 to reduce network overhead by ~30-50%
*   **FP16 Optimization**: LaMa runs in half-precision (FP16) on GPU for ~40% faster inference with minimal quality loss
*   **Endpoints**: `GET /health` (status + GPU info), `POST /clean` (single image), `POST /clean_batch` (batch → ZIP), `POST /clean_stream` (batch → binary stream)
*   **Files**: `gpu_server/server.py` (FastAPI server), `gpu_server/requirements.txt`, `gpu_server/setup.sh`, `gpu_server/DEPLOY.md`, `cogs/clean_cog.py` (client integration)