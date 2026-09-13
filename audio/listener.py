"""listener.py — the nightly listening session.

Priority queue (owner taste + precompute pools + fresh feed first, then
popular unheard tracks), N tracks per night, one at a time, lock-aware,
RSS-logged. Runs 02:00, before deep-train at 03:00 — never overlapping.

Only ~24 numbers per track survive. Audio is deleted the same night.
"""

import logging
import time

from config import DB_CATALOG, DB_FEED, PRECOMPUTE_DIR, TRAIN_LOCK_FILE
from core.database import get_conn

from .decode import read_wav_mono
from .features import analyze, summarize
from .fetch import cleanup_wav, fetch_clip
from .store import analyzed_ids, blocked_ids, ensure_schema, note_failure, save

logger = logging.getLogger("parlay.audio.listener")

TRACKS_PER_NIGHT = int(__import__("os").environ.get("AUDIO_TRACKS_PER_NIGHT", "150"))


def _locked() -> bool:
    try:
        if TRAIN_LOCK_FILE.exists():
            return (time.time() - TRAIN_LOCK_FILE.stat().st_mtime) < 4 * 3600
    except OSError:
        pass
    return False


def priority_queue(limit: int = TRACKS_PER_NIGHT) -> list[tuple[str, str]]:
    """(song_id, title) ordered by listening priority, unheard only."""
    ensure_schema()
    heard = analyzed_ids()
    blocked = blocked_ids()
    skip = heard | blocked
    ordered: list[str] = []

    def _add(ids: list[str]) -> None:
        for s in ids:
            if s not in skip and s not in ordered:
                ordered.append(s)
            if len(ordered) >= limit * 3:  # overfetch; title lookup may miss
                break

    # 1. Owner taste first (the ears that matter most).
    try:
        with get_conn(DB_CATALOG) as conn:
            rows = conn.execute("SELECT song_id FROM songs WHERE song_id LIKE 'owner\\_%' ESCAPE '\\'").fetchall()
            _add([r["song_id"] for r in rows])
    except Exception:
        pass
    # 2. Precomputed Top-200 pools (what we actually serve).
    try:
        import json as _json
        for p in sorted(PRECOMPUTE_DIR.glob("top200_*.json")):
            try:
                payload = _json.loads(p.read_text())
                _add([d["song_id"] for d in payload.get("items", [])])
            except Exception:
                continue
    except Exception:
        pass
    # 3. Latest fresh feed.
    try:
        with get_conn(DB_FEED) as conn:
            snap = conn.execute("SELECT snapshot_id FROM feed_snapshots ORDER BY fetched_at DESC LIMIT 1").fetchone()
            if snap:
                rows = conn.execute("SELECT song_id FROM feed_songs WHERE snapshot_id=? ORDER BY rank_in_snapshot LIMIT 500",
                                    (snap["snapshot_id"],)).fetchall()
                _add([r["song_id"] for r in rows])
    except Exception:
        pass
    # 4. Popular unheard filler.
    try:
        with get_conn(DB_CATALOG) as conn:
            rows = conn.execute("SELECT song_id FROM songs ORDER BY view_count DESC LIMIT 1000").fetchall()
            _add([r["song_id"] for r in rows])
    except Exception:
        pass
    # Resolve titles, keep order, cut to limit.
    out: list[tuple[str, str]] = []
    if ordered:
        with get_conn(DB_CATALOG) as conn:
            for s in ordered:
                if len(out) >= limit:
                    break
                try:
                    row = conn.execute("SELECT title FROM songs WHERE song_id=?", (s,)).fetchone()
                except Exception:
                    row = None
                if row:
                    out.append((s, row["title"]))
    return out


def listen_one(song_id: str) -> dict | None:
    """Fetch → hear → store → delete. Returns summary or None."""
    wav = None
    try:
        wav = fetch_clip(str(song_id))
        if not wav:
            note_failure(str(song_id), "no audio (blocked?)")
            return None
        y, sr = read_wav_mono(wav)
        feat = analyze(y, sr)
        summary = summarize(feat)
        save(str(song_id), summary)
        return summary
    except Exception as e:
        logger.warning("listen %s failed: %s", song_id, e)
        try:
            note_failure(str(song_id), str(e))
        except Exception:
            pass
        return None
    finally:
        cleanup_wav(wav)


def listen_night(limit: int = TRACKS_PER_NIGHT) -> dict:
    """One night of listening. Returns stats."""
    import resource

    if _locked():
        logger.warning("audio night skipped: train lock held.")
        return {"skipped": True}
    queue = priority_queue(limit)
    heard, failed = 0, 0
    t0 = time.time()
    for i, (sid, title) in enumerate(queue, 1):
        if _locked():  # trainer woke up early — yield the box
            logger.info("audio yielding to trainer after %d tracks.", heard)
            break
        s = listen_one(sid)
        if s:
            heard += 1
        else:
            failed += 1
        if i % 10 == 0:
            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
            logger.info("  listening %d/%d heard=%d failed=%d RSS=%.0fMB", i, len(queue), heard, failed, rss)
    dt = time.time() - t0
    stats = {"heard": heard, "failed": failed, "secs": round(dt, 1),
             "per_track_s": round(dt / max(heard + failed, 1), 1)}
    logger.info("🌙 Night over: %s", stats)
    return stats
