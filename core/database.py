"""
core/database.py — SQLite connection pool + schema bootstrap

All tables are created here once. Every module imports get_conn()
instead of creating their own sqlite3 connections.
"""

import sqlite3
import threading
import time
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from config import (
    DB_FEED, DB_HISTORY, DB_FEEDBACK, DB_CATALOG,
    DB_RECS, DB_BANDIT, DB_EMBEDDINGS, DB_TRAINING,
)

# Thread-local connection cache (one conn per thread per db)
_local = threading.local()


def _make_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")   # 64MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


@contextmanager
def get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Thread-safe context-managed connection."""
    key = str(db_path)
    if not hasattr(_local, "conns"):
        _local.conns = {}
    if key not in _local.conns:
        _local.conns[key] = _make_conn(db_path)
    conn = _local.conns[key]
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ─── Schema definitions ───────────────────────────────────

def init_catalog():
    with get_conn(DB_CATALOG) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS songs (
            song_id         TEXT PRIMARY KEY,   -- yt video_id
            title           TEXT NOT NULL,
            channel         TEXT,
            channel_id      TEXT,
            duration        INTEGER,
            view_count      INTEGER DEFAULT 0,
            like_count      INTEGER DEFAULT 0,
            upload_date     TEXT,
            thumbnail_url   TEXT,
            yt_url          TEXT,
            genre_code      INTEGER DEFAULT 20,
            language_code   INTEGER DEFAULT 0,
            has_official    INTEGER DEFAULT 0,
            has_lyric       INTEGER DEFAULT 0,
            energy_score    REAL DEFAULT 0.5,
            description     TEXT DEFAULT '',
            mood_tags       TEXT DEFAULT '',   -- comma-separated free keywords
            first_seen      REAL,   -- unix ts
            last_seen       REAL,
            times_fetched   INTEGER DEFAULT 1,
            bloom_hash      TEXT    -- comma-separated bloom signatures
        );
        CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre_code);
        CREATE INDEX IF NOT EXISTS idx_songs_views ON songs(view_count DESC);
        CREATE INDEX IF NOT EXISTS idx_songs_last_seen ON songs(last_seen DESC);
        -- MAX: artist co-count graph (artist → song co-counts from descriptions)
        CREATE TABLE IF NOT EXISTS artist_graph (
            artist_a        TEXT NOT NULL,
            artist_b        TEXT NOT NULL,
            co_count        INTEGER DEFAULT 1,
            updated_at      REAL,
            PRIMARY KEY (artist_a, artist_b)
        );
        -- MAX: companion prefs (/never blacklist, /anchor, /adventurous)
        CREATE TABLE IF NOT EXISTS companion_prefs (
            user_id         INTEGER PRIMARY KEY,
            blacklist_json  TEXT DEFAULT '[]',
            anchor_song_id  TEXT DEFAULT '',
            adventurous     REAL DEFAULT 0.3,
            updated_at      REAL
        );
        -- MAX: listening journal + streaks
        CREATE TABLE IF NOT EXISTS listening_journal (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            day             TEXT NOT NULL,   -- YYYY-MM-DD
            plays           INTEGER DEFAULT 0,
            likes           INTEGER DEFAULT 0,
            top_genre       INTEGER DEFAULT 20,
            UNIQUE (user_id, day)
        );
        -- MAX: learned blender weights per (user x mood)
        CREATE TABLE IF NOT EXISTS blender_weights (
            user_id         INTEGER NOT NULL,
            mood            TEXT NOT NULL DEFAULT '',
            weights_json    TEXT NOT NULL,
            ndcg10          REAL DEFAULT 0,
            updated_at      REAL,
            PRIMARY KEY (user_id, mood)
        );
        """)


def init_feed():
    with get_conn(DB_FEED) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feed_snapshots (
            snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at      REAL NOT NULL,
            query_used      TEXT
        );
        CREATE TABLE IF NOT EXISTS feed_songs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id     INTEGER REFERENCES feed_snapshots(snapshot_id),
            song_id         TEXT NOT NULL,
            rank_in_snapshot INTEGER,
            raw_score       REAL DEFAULT 0.0
        );
        CREATE INDEX IF NOT EXISTS idx_feed_songs_sid ON feed_songs(song_id);
        CREATE INDEX IF NOT EXISTS idx_feed_snap ON feed_songs(snapshot_id);
        """)


def init_history():
    with get_conn(DB_HISTORY) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS listens (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            song_id         TEXT NOT NULL,
            started_at      REAL NOT NULL,
            duration_played REAL DEFAULT 0,   -- seconds actually played
            completion_pct  REAL DEFAULT 0.0, -- 0.0–1.0
            source          TEXT DEFAULT 'manual',  -- manual/rec/auto
            context_genre   INTEGER DEFAULT 20,
            context_hour    INTEGER DEFAULT 0,  -- 0–23
            context_dow     INTEGER DEFAULT 0   -- 0=Mon..6=Sun
        );
        CREATE INDEX IF NOT EXISTS idx_listens_user ON listens(user_id, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_listens_song ON listens(song_id);
        CREATE INDEX IF NOT EXISTS idx_listens_user_song ON listens(user_id, song_id);
        """)


def init_feedback():
    with get_conn(DB_FEEDBACK) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            song_id         TEXT NOT NULL,
            rec_session_id  TEXT,
            signal          INTEGER NOT NULL,   -- +1 like, -1 dislike
            created_at      REAL NOT NULL,
            model_version   TEXT DEFAULT 'v0'
        );
        CREATE TABLE IF NOT EXISTS dislike_counts (
            user_id         INTEGER PRIMARY KEY,
            count           INTEGER DEFAULT 0,
            last_reset      REAL
        );
        CREATE INDEX IF NOT EXISTS idx_fb_user ON feedback(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_fb_song ON feedback(song_id);
        """)


def init_recs():
    with get_conn(DB_RECS) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS recommendation_sessions (
            session_id      TEXT PRIMARY KEY,
            user_id         INTEGER NOT NULL,
            created_at      REAL NOT NULL,
            model_version   TEXT,
            algorithm       TEXT,   -- 'ensemble'
            status          TEXT DEFAULT 'pending'  -- pending/sent/expired
        );
        CREATE TABLE IF NOT EXISTS rec_items (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT REFERENCES recommendation_sessions(session_id),
            song_id         TEXT NOT NULL,
            rank            INTEGER NOT NULL,
            final_score     REAL,
            svd_score       REAL,
            ncf_score       REAL,
            content_score   REAL,
            bandit_score    REAL,
            recency_score   REAL,
            why_top10       TEXT    -- JSON explanation string
        );
        CREATE INDEX IF NOT EXISTS idx_rec_items_sess ON rec_items(session_id);
        CREATE INDEX IF NOT EXISTS idx_rec_sess_user ON recommendation_sessions(user_id, created_at DESC);
        """)


def init_bandit():
    with get_conn(DB_BANDIT) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS bandit_arms (
            user_id         INTEGER NOT NULL,
            song_id         TEXT NOT NULL,
            alpha           REAL DEFAULT 1.0,   -- successes + 1 (Beta prior)
            beta_param      REAL DEFAULT 1.0,   -- failures + 1
            n_pulls         INTEGER DEFAULT 0,
            last_pulled     REAL,
            PRIMARY KEY (user_id, song_id)
        );
        """)


def init_embeddings():
    with get_conn(DB_EMBEDDINGS) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS song_embeddings (
            song_id         TEXT PRIMARY KEY,
            vector          BLOB NOT NULL,      -- numpy float32 array
            dim             INTEGER NOT NULL,
            model_version   TEXT DEFAULT 'v0',
            updated_at      REAL
        );
        CREATE TABLE IF NOT EXISTS user_embeddings (
            user_id         INTEGER PRIMARY KEY,
            vector          BLOB NOT NULL,
            dim             INTEGER NOT NULL,
            model_version   TEXT DEFAULT 'v0',
            updated_at      REAL
        );
        """)


def init_training_log():
    with get_conn(DB_TRAINING) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id          TEXT PRIMARY KEY,
            started_at      REAL,
            finished_at     REAL,
            model_type      TEXT,   -- 'svd' | 'ncf' | 'als' | 'feature_mf' | 'all'
            n_samples       INTEGER,
            n_users         INTEGER,
            n_songs         INTEGER,
            train_loss      REAL,
            val_loss        REAL,
            rmse            REAL,
            model_path      TEXT,
            notes           TEXT
        );
        -- MAX: split fingerprint + full metric bundle + registry pointers
        CREATE TABLE IF NOT EXISTS split_fingerprints (
            run_id          TEXT PRIMARY KEY,
            cutoff_ts       REAL,
            leave_last_n    INTEGER,
            frame_hash      TEXT,
            n_train         INTEGER,
            n_valid         INTEGER,
            created_at      REAL
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            key             TEXT PRIMARY KEY,  -- e.g. 'prod' | 'staging'
            run_id          TEXT,
            model_version   TEXT,
            ndcg10          REAL,
            updated_at      REAL
        );
        CREATE TABLE IF NOT EXISTS model_artifacts (
            run_id          TEXT,
            model_name      TEXT,   -- als | feature_mf | retrievers | sequence | svd | ncf | two_tower | blender
            model_path      TEXT,
            metrics_json    TEXT,
            PRIMARY KEY (run_id, model_name)
        );
        -- OPS: every promotion remembered (rollback needs history, not just pointers)
        CREATE TABLE IF NOT EXISTS registry_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            key             TEXT NOT NULL,   -- 'prod' | 'staging'
            run_id          TEXT,
            model_version   TEXT,
            ndcg10          REAL,
            updated_at      REAL
        );
        CREATE INDEX IF NOT EXISTS idx_reg_hist ON registry_history(key, updated_at DESC);
        """)


def _ensure_column(db_path, table: str, col: str, ddl: str) -> None:
    """Idempotent ADD COLUMN for existing SQLite files (migration)."""
    with get_conn(db_path) as conn:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def bootstrap_all():
    """Call once at startup to ensure all tables exist."""
    init_catalog()
    init_feed()
    init_history()
    init_feedback()
    init_recs()
    init_bandit()
    init_embeddings()
    init_training_log()
    # ── MAX migrations for pre-existing DB files ──
    try:
        from config import DB_CATALOG as _DBC
        for _col, _ddl in [
            ("description", "TEXT DEFAULT ''"),
            ("mood_tags", "TEXT DEFAULT ''"),
            ("has_remix", "INTEGER DEFAULT 0"),
        ]:
            _ensure_column(_DBC, "songs", _col, _ddl)
    except Exception:
        pass
    print("✅  Database schema bootstrapped.")


# ─── Numpy blob helpers ───────────────────────────────────

def ndarray_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def blob_to_ndarray(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)
