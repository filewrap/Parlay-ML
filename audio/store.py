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
        if "key_alt" not in cols:
            conn.execute("ALTER TABLE audio_features ADD COLUMN key_alt TEXT DEFAULT ''")


def save(song_id: str, summary: dict) -> None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO audio_features
            (song_id, bpm, musical_key, mode, key_alt, danceability, valence, energy,
             brightness, harmonic_clarity, tempo_strength, key_strength, chroma, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(song_id) DO UPDATE SET
            bpm=excluded.bpm, musical_key=excluded.musical_key, mode=excluded.mode,
            key_alt=excluded.key_alt,
            danceability=excluded.danceability, valence=excluded.valence,
            energy=excluded.energy, brightness=excluded.brightness,
            harmonic_clarity=excluded.harmonic_clarity,
            tempo_strength=excluded.tempo_strength, key_strength=excluded.key_strength,
            chroma=excluded.chroma, analyzed_at=excluded.analyzed_at
        """, (str(song_id), float(summary.get("bpm", 0)), str(summary.get("key", "")),
              str(summary.get("mode", "")), str(summary.get("key_alt", "")),
              float(summary.get("danceability", 0.5)),
              float(summary.get("valence", 0.5)), float(summary.get("energy", 0.5)),
              float(summary.get("brightness", 0.5)), float(summary.get("harmonic_clarity", 0.5)),
              float(summary.get("tempo_strength", 0)), float(summary.get("key_strength", 0)),
              str(summary.get("chroma", "")), time.time()))


def get(song_id: str) -> dict | None:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT * FROM audio_features WHERE song_id=?", (str(song_id),)).fetchone()
    return dict(row) if row else None


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
