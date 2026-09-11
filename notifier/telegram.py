"""
notifier/telegram.py — Telegram delivery for NeuroSync Top-10.

Completes the pipeline: pipeline.recommender → scheduler.jobs.rec_push_job
→ this push_recommendations() → Telegram Bot API → AgainOwner.

Stdlib only (urllib). No new pip deps.

Setup:
    export TELEGRAM_BOT_TOKEN="123456:ABC-..."
    # optional:
    export TELEGRAM_CHAT_ID="6802929470"   # default: OWNER id below
    export PARLAY_PUSH_CALLBACK="notifier.telegram:push_recommendations"

Then:
    python main.py --once --user-ids 6802929470
    python main.py   # hourly push to owner

Callback shape matches scheduler/jobs.py:
    async def push_recommendations(user_id: int, recs: list[dict]) -> None
Each rec: session_id, song_id, rank, title, channel, yt_url,
thumbnail, final_score, why.
Inline + / - buttons carry callback_data "fb:+:<song_id>" /
"fb:-:<song_id>" (64-byte limit). Wire them in your Parlay bot back to
NeuroSyncPipeline.record_feedback(user_id, song_id, +1/-1, session_id).
"""

import asyncio
import html
import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger("parlay.notifier.telegram")

API_BASE = "https://api.telegram.org"
MAX_MSG = 4000  # under Telegram's 4096 limit

OWNER_TELEGRAM_ID = int(os.environ.get("TELEGRAM_OWNER_ID", "6802929470"))
OWNER_USERNAME = os.environ.get("TELEGRAM_OWNER_USERNAME", "AgainOwner")


def _get_token() -> str:
    from config import TELEGRAM_BOT_TOKEN
    return (TELEGRAM_BOT_TOKEN or "").strip()


def _resolve_chat_id(user_id: int):
    """Map a pipeline user_id to a Telegram chat_id, or None to skip."""
    from config import TELEGRAM_CHAT_ID
    forced = (TELEGRAM_CHAT_ID or "").strip()
    if forced:
        return forced
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return None
    if uid == OWNER_TELEGRAM_ID:
        return str(OWNER_TELEGRAM_ID)
    if uid > 1_000_000:  # looks like a real Telegram chat id
        return str(uid)
    # Small internal / synthetic ids (0..N): skip unless broadcast enabled.
    if os.environ.get("TELEGRAM_BROADCAST", "").strip() == "1":
        return str(OWNER_TELEGRAM_ID)
    return None


def _api_post(token: str, method: str, payload: dict) -> dict:
    url = f"{API_BASE}/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _mark_session_sent(session_id: str) -> None:
    if not session_id:
        return
    try:
        from config import DB_RECS
        from core.database import get_conn
        with get_conn(DB_RECS) as conn:
            conn.execute(
                "UPDATE recommendation_sessions SET status='sent' WHERE session_id=?",
                (session_id,),
            )
    except Exception as e:
        logger.warning("Could not mark session %s sent: %s", session_id, e)


def _format_item(rec: dict) -> str:
    rank = rec.get("rank", 0)
    title = html.escape(str(rec.get("title") or "Unknown"))
    channel = html.escape(str(rec.get("channel") or ""))
    url = rec.get("yt_url") or ""
    score = rec.get("final_score", 0.0)
    why = rec.get("why") or {}
    drivers = ", ".join((why.get("drivers") or [])[:3]) or "general popularity"
    line = f"<b>{rank}.</b> {title}"
    if channel:
        line += f" — {channel}"
    line += f"\n   └ score={score:.3f} · {html.escape(drivers)}"
    if url:
        line += f'\n   └ <a href="{html.escape(url, quote=True)}">▶ YouTube</a>'
    return line


def build_messages(user_id: int, recs: list[dict]) -> tuple[list[str], dict]:
    """Render recs into 1+ sendMessage texts + shared inline keyboard."""
    session_id = recs[0].get("session_id", "") if recs else ""
    name = OWNER_USERNAME if int(user_id) == OWNER_TELEGRAM_ID else f"user {user_id}"
    header = f"🎧 <b>NeuroSync Top-{len(recs)} for {html.escape(name)}</b>\n"
    header += f"<i>Tap + / − under each pick to teach me your taste.</i>\n"
    chunks = [header]
    for rec in recs:
        item = _format_item(rec) + "\n"
        if len(chunks[-1]) + len(item) > MAX_MSG:
            chunks.append("")
        chunks[-1] += ("\n" if chunks[-1] and not chunks[-1].endswith("\n") else "") + item
    if session_id:
        footer = f"\n\n<code>session={html.escape(session_id)}</code>"
        if len(chunks[-1]) + len(footer) > MAX_MSG:
            chunks.append(footer)
        else:
            chunks[-1] += footer
    keyboard = {"inline_keyboard": []}
    for rec in recs:
        sid = str(rec.get("song_id", ""))[:48]
        title = str(rec.get("title") or sid)[:28]
        keyboard["inline_keyboard"].append([
            {"text": f"👍 {rec.get('rank', 0)} {title}", "callback_data": f"fb:+:{sid}"[:64]},
            {"text": f"👎 {rec.get('rank', 0)}", "callback_data": f"fb:-:{sid}"[:64]},
        ])
    return chunks, keyboard


async def _send_message(token: str, chat_id: str, text: str, reply_markup: dict | None = None) -> None:
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _api_post(token, "sendMessage", payload))


async def push_recommendations(user_id: int, recs: list[dict]) -> None:
    """Bot hook for PARLAY_PUSH_CALLBACK and scheduler/jobs.py."""
    if not recs:
        logger.info("📲 → user %s: no recs, nothing to send.", user_id)
        return
    token = _get_token()
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — logging Top-%d for user %s instead.", len(recs), user_id)
        for r in recs:
            logger.info("   [%2d] %-50s score=%.4f  [ + ] [ − ]",
                        r.get("rank", 0), (r.get("title") or "")[:50], r.get("final_score", 0.0))
        return
    chat_id = _resolve_chat_id(user_id)
    if chat_id is None:
        logger.info("Skipping Telegram for internal user %s (set TELEGRAM_BROADCAST=1 to force).", user_id)
        return
    try:
        texts, keyboard = build_messages(int(user_id), recs)
        for i, text in enumerate(texts):
            # Attach buttons only to the last chunk so they appear once.
            markup = keyboard if i == len(texts) - 1 else None
            await _send_message(token, str(chat_id), text, markup)
        _mark_session_sent(recs[0].get("session_id", ""))
        logger.info("📲 → chat %s: sent %d recs in %d msg(s).", chat_id, len(recs), len(texts))
    except Exception as e:
        logger.error("Telegram push to %s failed: %s", chat_id, e, exc_info=True)


def push_recommendations_sync(user_id: int, recs: list[dict]) -> None:
    """Sync wrapper for scripts / non-async callers."""
    asyncio.run(push_recommendations(user_id, recs))
