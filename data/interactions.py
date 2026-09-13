"""
data/interactions.py — MAX Phase 0 unified interaction frame.

One data pipeline: listens (history.db) + feedback (feedback.db)
→ (user_id, song_id, confidence, timestamp).

Confidence worldview (the `implicit` library, natively reimplemented):
    c = 1 + alpha * reward
    preference p = 1 if reward > 0 else 0

Plan-native rewards:
    full-play=3, like=5, partial=1, skip-fast=-2, dislike=-5
Legacy REWARD_* values from config are mapped onto the same scale so
old SVD/NCF code keeps working.

Time decay is applied AT TRAIN TIME (30-day half-life on confidence),
not only at score time.

Negative sampler: 4 negatives per positive
(50% uniform, 50% popularity-biased). Required for pairwise models.
"""

import hashlib
import logging
import math
import random
import time

import numpy as np

from config import (
    DB_CATALOG,
    DB_FEEDBACK,
    DB_HISTORY,
    IMPLICIT_ALPHA,
    INTERACTION_HALFLIFE_DAYS,
    NEGATIVES_PER_POSITIVE,
    PLAN_REWARD_DISLIKE,
    PLAN_REWARD_FULL_PLAY,
    PLAN_REWARD_LIKE,
    PLAN_REWARD_PARTIAL_PLAY,
    PLAN_REWARD_SKIP_FAST,
    REWARD_DISLIKE,
    REWARD_FULL_PLAY,
    REWARD_LIKE,
    REWARD_PARTIAL_PLAY,
)
from core.database import get_conn

logger = logging.getLogger("parlay.data.interactions")

# Map legacy reward values → plan-native rewards (for confidence math).
_LEGACY_TO_PLAN = {
    REWARD_LIKE: PLAN_REWARD_LIKE,
    REWARD_DISLIKE: PLAN_REWARD_DISLIKE,
    REWARD_FULL_PLAY: PLAN_REWARD_FULL_PLAY,
    REWARD_PARTIAL_PLAY: PLAN_REWARD_PARTIAL_PLAY,
}
try:
    from config import REWARD_SKIP_FAST as _SKIP

    _LEGACY_TO_PLAN[_SKIP] = PLAN_REWARD_SKIP_FAST
except Exception:
    _LEGACY_TO_PLAN[-0.5] = PLAN_REWARD_SKIP_FAST


def legacy_to_plan_reward(r: float) -> float:
    """Map a legacy REWARD_* value onto the plan-native scale."""
    for k, v in _LEGACY_TO_PLAN.items():
        if abs(float(r) - float(k)) < 1e-9:
            return float(v)
    # Already plan-native or unknown → pass through, clamped.
    return max(-5.0, min(5.0, float(r)))


def reward_to_confidence(reward: float, alpha: float = IMPLICIT_ALPHA) -> float:
    """c = 1 + alpha * max(reward,0); negatives keep c=1 but p=0."""
    r = legacy_to_plan_reward(reward)
    if r > 0:
        return 1.0 + alpha * r
    return 1.0


def apply_time_decay(confidence: float, age_days: float,
                     halflife_days: float = INTERACTION_HALFLIFE_DAYS) -> float:
    """Exponential half-life decay on confidence at train time."""
    if age_days <= 0:
        return confidence
    return confidence * (0.5 ** (age_days / halflife_days))


def build_interaction_frame(alpha: float = IMPLICIT_ALPHA,
                            halflife_days: float = INTERACTION_HALFLIFE_DAYS,
                            now: float | None = None) -> list[dict]:
    """Unified builder: listens + feedback → confidence frame.

    Returns list of dicts:
      {user_id, song_id, reward, plan_reward, confidence, timestamp, source}
    Explicit feedback wins over implicit listens for the same (user, song).
    Confidence already includes time decay.
    """
    now = now if now is not None else time.time()
    frame: dict[tuple[int, str], dict] = {}

    # --- Explicit feedback (highest priority) ---
    with get_conn(DB_FEEDBACK) as conn:
        try:
            rows = conn.execute(
                "SELECT user_id, song_id, signal, created_at FROM feedback"
            ).fetchall()
        except Exception:
            rows = []
    for row in rows:
        try:
            sig = float(row["signal"])
        except Exception:
            continue
        plan_r = PLAN_REWARD_LIKE if sig > 0 else PLAN_REWARD_DISLIKE
        ts = float(row["created_at"] or now)
        age_days = max(0.0, (now - ts) / 86400.0)
        c = apply_time_decay(reward_to_confidence(plan_r, alpha), age_days, halflife_days)
        frame[(int(row["user_id"]), str(row["song_id"]))] = {
            "user_id": int(row["user_id"]),
            "song_id": str(row["song_id"]),
            "reward": sig,
            "plan_reward": plan_r,
            "confidence": c,
            "timestamp": ts,
            "source": "feedback",
        }

    # --- Implicit from listen history ---
    with get_conn(DB_HISTORY) as conn:
        try:
            rows = conn.execute(
                "SELECT user_id, song_id, completion_pct, started_at FROM listens"
            ).fetchall()
        except Exception:
            rows = []
    for row in rows:
        key = (int(row["user_id"]), str(row["song_id"]))
        if key in frame:
            continue  # explicit signal already set
        pct = float(row["completion_pct"] or 0.0)
        ts = float(row["started_at"] or now)
        if pct >= 0.80:
            plan_r = PLAN_REWARD_FULL_PLAY
        elif pct >= 0.30:
            plan_r = PLAN_REWARD_PARTIAL_PLAY
        else:
            plan_r = PLAN_REWARD_SKIP_FAST
        age_days = max(0.0, (now - ts) / 86400.0)
        c = apply_time_decay(reward_to_confidence(plan_r, alpha), age_days, halflife_days)
        frame[key] = {
            "user_id": key[0],
            "song_id": key[1],
            "reward": plan_r,
            "plan_reward": plan_r,
            "confidence": c,
            "timestamp": ts,
            "source": "listen",
        }

    out = sorted(frame.values(), key=lambda d: d["timestamp"])
    logger.info("Built interaction frame: %d pairs (alpha=%.1f, HL=%dd)",
                len(out), alpha, halflife_days)
    return out


def frame_to_triples(frame: list[dict]) -> list[tuple[int, str, float]]:
    """Back-compat: frame → [(user_id, song_id, reward)] for SVD/NCF."""
    return [(d["user_id"], d["song_id"], float(d["plan_reward"])) for d in frame]


def frame_hash(frame: list[dict]) -> str:
    h = hashlib.sha256()
    for d in frame:
        h.update(f"{d['user_id']}|{d['song_id']}|{d['plan_reward']:.2f}".encode())
    return h.hexdigest()[:16]


def sample_negatives(user_id: int, pos_ids: set[str], n: int,
                     popularity: dict[str, int] | None = None,
                     all_song_ids: list[str] | None = None,
                     seed: int | None = None) -> list[str]:
    """4 negatives per positive: 50% uniform, 50% popularity-biased."""
    rng = random.Random(seed if seed is not None else user_id)
    if not all_song_ids:
        with get_conn(DB_CATALOG) as conn:
            try:
                rows = conn.execute("SELECT song_id, view_count FROM songs").fetchall()
            except Exception:
                rows = []
        all_song_ids = [r["song_id"] for r in rows]
        popularity = {r["song_id"]: int(r["view_count"] or 0) for r in rows}
    if not all_song_ids:
        return []
    popularity = popularity or {}
    pos = set(pos_ids)
    out: list[str] = []
    # Popularity weights (log-compressed, avoids megahit domination).
    pop_ids = list(all_song_ids)
    pop_w = np.array([math.log10(max(popularity.get(s, 1), 1) + 1) for s in pop_ids],
                     dtype=np.float64)
    if pop_w.sum() <= 0:
        pop_w = np.ones_like(pop_w)
    pop_w = pop_w / pop_w.sum()

    n_uni = n // 2
    n_pop = n - n_uni
    tries = 0
    while len([x for x in out[:n_uni]]) < n_uni and tries < n_uni * 20:
        tries += 1
        s = rng.choice(all_song_ids)
        if s not in pos and s not in out:
            out.append(s)
    tries = 0
    got_pop = 0
    while got_pop < n_pop and tries < n_pop * 20:
        tries += 1
        s = str(np.random.choice(pop_ids, p=pop_w))
        if s not in pos and s not in out:
            out.append(s)
            got_pop += 1
    return out[:n]


def pairwise_training_pairs(frame: list[dict],
                            n_neg: int = NEGATIVES_PER_POSITIVE,
                            seed: int = 42) -> list[tuple[int, str, str]]:
    """(user, pos_song, neg_song) triples for BPR/WARP models."""
    rng = random.Random(seed)
    by_user: dict[int, list[str]] = {}
    for d in frame:
        if float(d["plan_reward"]) > 0:
            by_user.setdefault(int(d["user_id"]), []).append(str(d["song_id"]))
    with get_conn(DB_CATALOG) as conn:
        try:
            rows = conn.execute("SELECT song_id, view_count FROM songs").fetchall()
        except Exception:
            rows = []
    all_ids = [r["song_id"] for r in rows]
    pop = {r["song_id"]: int(r["view_count"] or 0) for r in rows}
    pairs: list[tuple[int, str, str]] = []
    for uid, positives in by_user.items():
        pos_set = set(positives)
        for pos in positives:
            negs = sample_negatives(uid, pos_set, n_neg, pop, all_ids,
                                    seed=rng.randint(0, 10 ** 9))
            for neg in negs:
                pairs.append((uid, pos, neg))
    return pairs
