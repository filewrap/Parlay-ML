"""joblog.py — every job leaves a trace. No more grepping logs blind."""

import logging
import time

from config import DB_TRAINING
from core.database import get_conn

logger = logging.getLogger("parlay.ops.joblog")

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job         TEXT NOT NULL,
    started_at  REAL NOT NULL,
    secs        REAL DEFAULT 0,
    ok          INTEGER DEFAULT 1,
    note        TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_job_runs_job ON job_runs(job, started_at DESC);
"""


def ensure_schema() -> None:
    with get_conn(DB_TRAINING) as conn:
        conn.executescript(SCHEMA)


def log_job(job: str, secs: float, ok: bool = True, note: str = "") -> None:
    try:
        ensure_schema()
        with get_conn(DB_TRAINING) as conn:
            conn.execute("INSERT INTO job_runs (job, started_at, secs, ok, note)"
                         " VALUES (?, ?, ?, ?, ?)",
                         (job, time.time() - secs, round(secs, 1), 1 if ok else 0,
                          str(note)[:500]))
    except Exception as e:
        logger.warning("joblog failed: %s", e)


def recent_jobs(limit: int = 20) -> list[dict]:
    ensure_schema()
    with get_conn(DB_TRAINING) as conn:
        rows = conn.execute("SELECT * FROM job_runs ORDER BY started_at DESC LIMIT ?",
                            (limit,)).fetchall()
    return [dict(r) for r in rows]


async def logged(job: str, coro):
    """Wrap a job coroutine: time it, log ok/fail, re-raise nothing."""
    t0 = time.time()
    try:
        out = await coro
        log_job(job, time.time() - t0, True)
        return out
    except Exception as e:
        log_job(job, time.time() - t0, False, str(e)[:200])
        raise
