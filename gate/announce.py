"""announce.py — "Parlay says": hearings rendered as messages.

When the machine hears something, it should say so — the listening feed
is how Parlay tells you (and your agents) that it listened. Log-first;
Telegram delivery reuses notifier.telegram when configured.
"""

import logging

logger = logging.getLogger("parlay.gate.announce")


def announce_listen(song_id: str, title: str, summary: dict, caption: str = "") -> str:
    """One hearing → one message. Returns the text (also logged)."""
    if not caption:
        try:
            from audio.caption import caption as _cap
            caption = _cap({**summary, "musical_key": summary.get("key", "")})
        except Exception:
            caption = ""
    bpm = summary.get("bpm", 0)
    key = f"{summary.get('key', '?')} {summary.get('mode', '')}".strip()
    text = f"🎧 Parlay heard: {title or song_id}"
    if bpm:
        text += f" (~{float(bpm):.0f} BPM, {key})"
    if caption:
        text += f"\n{caption}"
    logger.info("📢 %s", text.replace("\n", " · "))
    return text


def announce_batch(heard: list[tuple[str, str, dict]]) -> list[str]:
    """A night of listening → a morning digest of messages."""
    msgs = []
    for sid, title, summary in heard:
        try:
            from audio.caption import caption_for
            cap = caption_for(sid, title)
        except Exception:
            cap = ""
        msgs.append(announce_listen(sid, title, summary, cap))
    if msgs:
        msgs.append(f"🌙 Night over: {len(msgs)} new hearing(s). The graph rewired itself.")
    return msgs
