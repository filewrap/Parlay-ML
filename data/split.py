"""
data/split.py — MAX Phase 0 temporal split.

No random 80/20 splits (they leak the future).
Leave-last-N-per-user + global time cutoff, RecBole-style.

Saves split fingerprint (cutoff ts, hash) into training_log.db
so runs are comparable.
"""

import hashlib
import logging
import time

from core.database import get_conn
from config import DB_TRAINING

logger = logging.getLogger("parlay.data.split")


def temporal_split(frame: list[dict], leave_last_n: int = 5,
                   cutoff_quantile: float = 0.85) -> tuple[list[dict], list[dict], dict]:
    """Split a time-sorted interaction frame.

    - Global cutoff = quantile(cutoff_quantile) of timestamps.
    - Per user: last `leave_last_n` positives always go to valid
      (even if before cutoff) → leave-last-N-per-user.
    - Everything else before cutoff → train; at/after cutoff → valid.

    Returns (train, valid, fingerprint).
    """
    if not frame:
        return [], [], {"cutoff_ts": time.time(), "leave_last_n": leave_last_n,
                        "frame_hash": "empty", "n_train": 0, "n_valid": 0}
    ordered = sorted(frame, key=lambda d: float(d["timestamp"]))
    stamps = sorted(float(d["timestamp"]) for d in ordered)
    k = min(max(int(len(stamps) * cutoff_quantile), 1), len(stamps) - 1)
    cutoff = float(stamps[k]) if len(stamps) > 1 else float(stamps[0])

    # Per-user last-N positives → forced valid.
    positives_by_user: dict[int, list[dict]] = {}
    for d in ordered:
        if float(d["plan_reward"]) > 0:
            positives_by_user.setdefault(int(d["user_id"]), []).append(d)
    forced_valid_ids: set[int] = set()
    for _uid, plist in positives_by_user.items():
        tail = sorted(plist, key=lambda d: float(d["timestamp"]))[-leave_last_n:]
        forced_valid_ids.update(id(d) for d in tail)

    train, valid = [], []
    for d in ordered:
        if id(d) in forced_valid_ids or float(d["timestamp"]) >= cutoff:
            valid.append(d)
        else:
            train.append(d)
    # Guard: never return empty train when data exists.
    if not train and valid:
        train = valid[: max(1, len(valid) // 2)]
    if not valid and train:
        valid = train[-max(1, len(train) // 5):]

    h = hashlib.sha256()
    for d in ordered:
        h.update(f"{d['user_id']}|{d['song_id']}|{d['timestamp']:.0f}".encode())
    fp = {
        "cutoff_ts": cutoff,
        "leave_last_n": leave_last_n,
        "frame_hash": h.hexdigest()[:16],
        "n_train": len(train),
        "n_valid": len(valid),
    }
    logger.info("Temporal split: train=%d valid=%d cutoff=%.0f leave_last=%d hash=%s",
                len(train), len(valid), cutoff, leave_last_n, fp["frame_hash"])
    return train, valid, fp


def save_fingerprint(run_id: str, fp: dict) -> None:
    """Persist split fingerprint so runs are comparable."""
    with get_conn(DB_TRAINING) as conn:
        conn.execute("""
        INSERT OR REPLACE INTO split_fingerprints
            (run_id, cutoff_ts, leave_last_n, frame_hash, n_train, n_valid, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (run_id, float(fp.get("cutoff_ts", 0.0)), int(fp.get("leave_last_n", 5)),
              str(fp.get("frame_hash", "")), int(fp.get("n_train", 0)),
              int(fp.get("n_valid", 0)), time.time()))
