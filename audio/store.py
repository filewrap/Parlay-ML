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
    analyzed_at REAL
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
                         ("lift", "REAL DEFAULT 0")):
            if col not in cols:
                conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {ddl}")


def save(song_id: str, summary: dict) -> None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO audio_features
            (song_id, bpm, musical_key, mode, key_alt, danceability, valence, energy,
             brightness, harmonic_clarity, tempo_strength, key_strength,
             climax_frac, lift, chroma, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            bpm=excluded.bpm, musical_key=excluded.musical_key, mode=excluded.mode,
            key_alt=excluded.key_alt,
            danceability=excluded.danceability, valence=excluded.valence,
            energy=excluded.energy, brightness=excluded.brightness,
            harmonic_clarity=excluded.harmonic_clarity,
            tempo_strength=excluded.tempo_strength, key_strength=excluded.key_strength,
            climax_frac=excluded.climax_frac, lift=excluded.lift,
            chroma=excluded.chroma, analyzed_at=excluded.analyzed_at
        """, (str(song_id), float(summary.get("bpm", 0)), str(summary.get("key", "")),
              str(summary.get("mode", "")), str(summary.get("key_alt", "")),
              float(summary.get("danceability", 0.5)),
              float(summary.get("valence", 0.5)), float(summary.get("energy", 0.5)),
              float(summary.get("brightness", 0.5)), float(summary.get("harmonic_clarity", 0.5)),
              float(summary.get("tempo_strength", 0)), float(summary.get("key_strength", 0)),
              float(summary.get("climax_frac", 0.5)), float(summary.get("lift", 0)),
              str(summary.get("chroma", "")), time.time()))
    # Full arc series lives beside it.
    try:
        save_arc(str(song_id), {"arousal": str(summary.get("arc_arousal", "")),
                                "valence": str(summary.get("arc_valence", "")),
                                "climax_frac": float(summary.get("climax_frac", 0.5)),
                                "lift": float(summary.get("lift", 0))})
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
        INSERT INTO audio_arcs (song_id, n_windows, arousal, valence, climax_frac, lift, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            n_windows=excluded.n_windows, arousal=excluded.arousal,
            valence=excluded.valence, climax_frac=excluded.climax_frac,
            lift=excluded.lift, analyzed_at=excluded.analyzed_at
        """, (str(song_id), n, str(arousal), str(arc.get("valence", "")),
              float(arc.get("climax_frac", 0.5)), float(arc.get("lift", 0)), time.time()))


def get_arc(song_id: str) -> dict | None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT * FROM audio_arcs WHERE song_id=?", (str(song_id),)).fetchone()
    return dict(row) if row else None


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
