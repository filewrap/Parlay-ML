"""store.py — audio_features table: what the machine heard, per song."""

import json
import logging
import time

from config import DB_CATALOG
from core.database import get_conn

logger = logging.getLogger("parlay.audio.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_features (
    song_id           TEXT PRIMARY KEY,
    bpm               REAL DEFAULT 0,
    musical_key       TEXT DEFAULT '',
    mode              TEXT DEFAULT '',
    key_alt           TEXT DEFAULT '',
    danceability      REAL DEFAULT 0.5,
    valence           REAL DEFAULT 0.5,
    energy            REAL DEFAULT 0.5,
    brightness        REAL DEFAULT 0.5,
    harmonic_clarity  REAL DEFAULT 0.5,
    tempo_strength    REAL DEFAULT 0,
    key_strength      REAL DEFAULT 0,
    chroma            TEXT DEFAULT '',
    analyzed_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_audio_bpm ON audio_features(bpm);
-- Emotional arcs: the journey, not just the average.
CREATE TABLE IF NOT EXISTS audio_arcs (
    song_id     TEXT PRIMARY KEY,
    n_windows   INTEGER DEFAULT 0,
    arousal     TEXT DEFAULT '',
    valence     TEXT DEFAULT '',
    climax_frac REAL DEFAULT 0.5,
    lift        REAL DEFAULT 0,
    melody_10s  TEXT DEFAULT '',
    presence_10s TEXT DEFAULT '',
    analyzed_at REAL
);
-- Words: synced lyrics cache (lrclib, free, no key).
CREATE TABLE IF NOT EXISTS track_lyrics (
    song_id     TEXT PRIMARY KEY,
    plain       TEXT DEFAULT '',
    synced_json TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    fetched_at  REAL
);
-- Fetch backoff: don't hammer blocked videos every night.
CREATE TABLE IF NOT EXISTS audio_fetch_state (
    song_id     TEXT PRIMARY KEY,
    attempts    INTEGER DEFAULT 0,
    last_error  TEXT DEFAULT '',
    updated_at  REAL
);
"""

MAX_ATTEMPTS = 5


def ensure_schema() -> None:
    with get_conn(DB_CATALOG) as conn:
        conn.executescript(SCHEMA)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(audio_features)").fetchall()]
        for col, ddl in (("key_alt", "TEXT DEFAULT ''"),
                         ("climax_frac", "REAL DEFAULT 0.5"),
                         ("lift", "REAL DEFAULT 0"),
                         ("source", "TEXT DEFAULT ''"),
                         ("clip_secs", "REAL DEFAULT 0"),
                         ("voice_pct", "REAL DEFAULT 0"),
                         ("vibrato_pct", "REAL DEFAULT 0"),
                         ("median_f0", "REAL DEFAULT 0"),
                         ("f0_lo", "REAL DEFAULT 0"),
                         ("f0_hi", "REAL DEFAULT 0"),
                         ("peak_f0", "REAL DEFAULT 0"),
                         ("register", "TEXT DEFAULT ''"),
                         ("voice_enter_s", "REAL DEFAULT -1")):
            if col not in cols:
                conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {ddl}")
        acols = [r[1] for r in conn.execute("PRAGMA table_info(audio_arcs)").fetchall()]
        for col, ddl in (("melody_10s", "TEXT DEFAULT ''"), ("presence_10s", "TEXT DEFAULT ''")):
            if col not in acols:
                conn.execute(f"ALTER TABLE audio_arcs ADD COLUMN {col} {ddl}")


def _enter(summary: dict) -> float:
    v = summary.get("voice_enter_s", -1)
    return float(v) if v is not None else -1.0


def save(song_id: str, summary: dict) -> None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO audio_features
            (song_id, bpm, musical_key, mode, key_alt, danceability, valence, energy,
             brightness, harmonic_clarity, tempo_strength, key_strength,
             climax_frac, lift, chroma, source, clip_secs,
             voice_pct, vibrato_pct, median_f0, f0_lo, f0_hi, peak_f0, register,
             voice_enter_s, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            bpm=excluded.bpm, musical_key=excluded.musical_key, mode=excluded.mode,
            key_alt=excluded.key_alt,
            danceability=excluded.danceability, valence=excluded.valence,
            energy=excluded.energy, brightness=excluded.brightness,
            harmonic_clarity=excluded.harmonic_clarity,
            tempo_strength=excluded.tempo_strength, key_strength=excluded.key_strength,
            climax_frac=excluded.climax_frac, lift=excluded.lift,
            chroma=excluded.chroma, source=excluded.source,
            clip_secs=excluded.clip_secs,
            voice_pct=excluded.voice_pct, vibrato_pct=excluded.vibrato_pct,
            median_f0=excluded.median_f0, f0_lo=excluded.f0_lo, f0_hi=excluded.f0_hi,
            peak_f0=excluded.peak_f0, register=excluded.register,
            voice_enter_s=excluded.voice_enter_s, analyzed_at=excluded.analyzed_at
        """, (str(song_id), float(summary.get("bpm", 0)), str(summary.get("key", "")),
              str(summary.get("mode", "")), str(summary.get("key_alt", "")),
              float(summary.get("danceability", 0.5)),
              float(summary.get("valence", 0.5)), float(summary.get("energy", 0.5)),
              float(summary.get("brightness", 0.5)), float(summary.get("harmonic_clarity", 0.5)),
              float(summary.get("tempo_strength", 0)), float(summary.get("key_strength", 0)),
              float(summary.get("climax_frac", 0.5)), float(summary.get("lift", 0)),
              str(summary.get("chroma", "")), str(summary.get("source", "")),
              float(summary.get("clip_secs", 0)),
              float(summary.get("voice_pct", 0)), float(summary.get("vibrato_pct", 0)),
              float(summary.get("median_f0", 0)), float(summary.get("f0_lo", 0)),
              float(summary.get("f0_hi", 0)), float(summary.get("peak_f0", 0)),
              str(summary.get("register", "")), _enter(summary),
              time.time()))
    # Full arc + melody series live beside it.
    try:
        save_arc(str(song_id), {"arousal": str(summary.get("arc_arousal", "")),
                                "valence": str(summary.get("arc_valence", "")),
                                "climax_frac": float(summary.get("climax_frac", 0.5)),
                                "lift": float(summary.get("lift", 0)),
                                "melody_10s": str(summary.get("melody_10s", "")),
                                "presence_10s": str(summary.get("presence_10s", ""))})
    except Exception:
        pass
    # The Gate hears about every hearing: append to the listen ledger.
    try:
        from gate.state import record_listen
        title = ""
        try:
            with get_conn(DB_CATALOG) as conn:
                row = conn.execute("SELECT title FROM songs WHERE song_id=?",
                                   (str(song_id),)).fetchone()
            title = row["title"] if row else ""
        except Exception:
            pass
        cap = ""
        try:
            from audio.caption import caption as _cap
            cap = _cap({**summary, "musical_key": summary.get("key", "")})
        except Exception:
            pass
        record_listen(str(song_id), title, summary, cap)
    except Exception:
        pass


def get(song_id: str) -> dict | None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT * FROM audio_features WHERE song_id=?", (str(song_id),)).fetchone()
    return dict(row) if row else None

def save_arc(song_id: str, arc: dict) -> None:
    ensure_schema()
    arousal = arc.get("arousal", "")
    n = len([v for v in str(arousal).split(",") if v.strip()])
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO audio_arcs (song_id, n_windows, arousal, valence, climax_frac, lift,
                                melody_10s, presence_10s, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            n_windows=excluded.n_windows, arousal=excluded.arousal,
            valence=excluded.valence, climax_frac=excluded.climax_frac,
            lift=excluded.lift, melody_10s=excluded.melody_10s,
            presence_10s=excluded.presence_10s, analyzed_at=excluded.analyzed_at
        """, (str(song_id), n, str(arousal), str(arc.get("valence", "")),
              float(arc.get("climax_frac", 0.5)), float(arc.get("lift", 0)),
              str(arc.get("melody_10s", "")), str(arc.get("presence_10s", "")),
              time.time()))


def get_arc(song_id: str) -> dict | None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT * FROM audio_arcs WHERE song_id=?", (str(song_id),)).fetchone()
    return dict(row) if row else None


def save_lyrics(song_id: str, lyrics: dict) -> None:
    import json as _json
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO track_lyrics (song_id, plain, synced_json, source, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET plain=excluded.plain,
            synced_json=excluded.synced_json, source=excluded.source,
            fetched_at=excluded.fetched_at
        """, (str(song_id), str(lyrics.get("plain", "")),
              _json.dumps(lyrics.get("synced", []))[:20000],
              str(lyrics.get("source", "")), time.time()))


def get_lyrics(song_id: str) -> dict | None:
    import json as _json
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT * FROM track_lyrics WHERE song_id=?", (str(song_id),)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["synced"] = _json.loads(d.get("synced_json") or "[]")
    except Exception:
        d["synced"] = []
    return d


def recent_heard(since_days: float = 7.0, limit: int = 20) -> list[dict]:
    """Recently-analyzed tracks, newest first — the fresh-ears pool."""
    ensure_schema()
    cutoff = time.time() - since_days * 86400
    with get_conn(DB_CATALOG) as conn:
        try:
            rows = conn.execute(
                "SELECT * FROM audio_features WHERE analyzed_at > ? ORDER BY analyzed_at DESC LIMIT ?",
                (cutoff, limit)).fetchall()
        except Exception:
            rows = []
    return [dict(r) for r in rows]


def analyzed_ids() -> set[str]:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute("SELECT song_id FROM audio_features").fetchall()
    return {r["song_id"] for r in rows}


def blocked_ids() -> set[str]:
    """Songs that failed fetching too often — leave them alone."""
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        try:
            rows = conn.execute("SELECT song_id FROM audio_fetch_state WHERE attempts >= ?",
                                (MAX_ATTEMPTS,)).fetchall()
        except Exception:
            rows = []
    return {r["song_id"] for r in rows}


def note_failure(song_id: str, error: str = "") -> int:
    """Record a fetch failure. Returns total attempts."""
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO audio_fetch_state (song_id, attempts, last_error, updated_at)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            attempts = attempts + 1, last_error = excluded.last_error,
            updated_at = excluded.updated_at
        """, (str(song_id), str(error or "")[-200:], time.time()))
        row = conn.execute("SELECT attempts FROM audio_fetch_state WHERE song_id=?",
                           (str(song_id),)).fetchone()
    return int(row["attempts"]) if row else 1


def count() -> int:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT COUNT(*) c FROM audio_features").fetchone()
    return int(row["c"]) if row else 0
