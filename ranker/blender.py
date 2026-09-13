"""
ranker/blender.py — MAX Phase 3 learned ensemble v2.

Replaces fixed W_SVD=0.30… with logistic regression per (user × mood)
on validation NDCG. Inputs: every sub-score + freshness + artist-repeat
penalty. Falls back to plan defaults when data is thin.

Serve path: nightly precompute Top-200/user to disk; hourly job reranks
with the fresh feed only.

Hard rules (always applied post-blend):
  - user blacklist (/never)
  - max-3-per-genre
  - one anchor slot (Mehrama-energy for owner)
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from config import (ANCHOR_SONG_ID, ANCHOR_USER_ID, BLENDER_EPOCHS, BLENDER_L2,
                    BLENDER_LR, BLENDER_MAX_PER_GENRE, DB_CATALOG, PRECOMPUTE_DIR,
                    W_BANDIT, W_CONTENT, W_NCF, W_RECENCY, W_SVD)
from core.database import get_conn

logger = logging.getLogger("parlay.blender")

FEATURES = ["als", "feature_mf", "svd", "ncf", "two_tower", "content",
            "bandit", "recency", "retriever", "sequence", "freshness", "repeat_pen"]
DEFAULT_W = np.array([0.22, 0.18, 0.12, 0.08, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, -0.06],
                     dtype=np.float64)


def feature_vector(scores: dict) -> np.ndarray:
    return np.array([float(scores.get(f, 0.0)) for f in FEATURES], dtype=np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Blender:
    """Logistic-regression blender. One weight vector per (user, mood)."""

    def __init__(self, version: str = "v0"):
        self.version = version
        self.global_w: np.ndarray = DEFAULT_W.copy()
        self.per_user_mood: dict[tuple[int, str], np.ndarray] = {}
        self._load_all()

    # ── persistence (SQLite blender_weights) ──
    def _load_all(self) -> None:
        try:
            with get_conn(DB_CATALOG) as conn:
                rows = conn.execute("SELECT user_id, mood, weights_json FROM blender_weights").fetchall()
            for r in rows:
                try:
                    w = np.array(json.loads(r["weights_json"]), dtype=np.float64)
                    if w.shape == DEFAULT_W.shape:
                        self.per_user_mood[(int(r["user_id"]), str(r["mood"] or ""))] = w
                except Exception:
                    pass
        except Exception:
            pass

    def weights_for(self, user_id: int, mood: str = "") -> np.ndarray:
        return self.per_user_mood.get((int(user_id), mood or ""),
                                      self.per_user_mood.get((int(user_id), ""), self.global_w))

    def _save(self, user_id: int, mood: str, w: np.ndarray, ndcg: float = 0.0) -> None:
        with get_conn(DB_CATALOG) as conn:
            conn.execute("""
            INSERT INTO blender_weights (user_id, mood, weights_json, ndcg10, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, mood) DO UPDATE SET
                weights_json=excluded.weights_json, ndcg10=excluded.ndcg10,
                updated_at=excluded.updated_at
            """, (int(user_id), mood or "", json.dumps([float(x) for x in w]), float(ndcg), time.time()))

    # ── training: pairwise logistic (BPR-style) on validation judged pairs ──
    def fit(self, judged: list[dict], mood: str = "",
            lr: float = BLENDER_LR, epochs: int = BLENDER_EPOCHS, l2: float = BLENDER_L2) -> dict:
        """judged: [{user_id, pos_scores: dict, neg_scores: dict}].

        pos_scores/neg_scores map FEATURES → float in [-1,1].
        Learns per-user weights; tiny-user fallback = global.
        """
        by_user: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for j in judged:
            try:
                by_user.setdefault(int(j["user_id"]), []).append(
                    (feature_vector(j["pos_scores"]), feature_vector(j["neg_scores"])))
            except Exception:
                continue
        report: dict = {}
        # Global pass first.
        w_g = self.global_w.copy()
        all_pairs = [p for v in by_user.values() for p in v]
        if all_pairs:
            w_g = self._train_vec(w_g, all_pairs, lr, epochs, l2)
            self.global_w = w_g
            self._save(-1, mood, w_g)
        for uid, pairs in by_user.items():
            if len(pairs) < 5:
                continue  # too thin → fallback to global at serve time
            w = self._train_vec(w_g.copy(), pairs, lr, min(epochs, 80), l2 * 5)
            self.per_user_mood[(uid, mood or "")] = w
            self._save(uid, mood, w)
            report[str(uid)] = round(float(self._pairwise_acc(w, pairs)), 4)
        logger.info("✅ Blender fitted (mood=%r): %d users, global_acc=%.3f",
                    mood, len(report), float(self._pairwise_acc(self.global_w, all_pairs)) if all_pairs else 0.0)
        return {"users": len(report), "pairs": len(all_pairs)}

    @staticmethod
    def _train_vec(w: np.ndarray, pairs: list[tuple[np.ndarray, np.ndarray]],
                   lr: float, epochs: int, l2: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            order = rng.permutation(len(pairs))
            for t in order:
                d = pairs[t][0] - pairs[t][1]
                p = float(sigmoid(d @ w))
                grad = -(1 - p) * d + l2 * w
                w -= lr * grad
        return w

    @staticmethod
    def _pairwise_acc(w: np.ndarray, pairs: list) -> float:
        if not pairs:
            return 0.0
        return float(np.mean([1.0 if (p[0] - p[1]) @ w > 0 else 0.0 for p in pairs]))

    # ── serving ──
    def score(self, user_id: int, scores: dict, mood: str = "") -> float:
        w = self.weights_for(int(user_id), mood or "")
        return float(sigmoid(feature_vector(scores) @ w) * 2 - 1)

    def score_batch(self, user_id: int, rows: list[dict], mood: str = "") -> list[tuple[str, float]]:
        w = self.weights_for(int(user_id), mood or "")
        out = []
        for r in rows:
            v = float(sigmoid(feature_vector(r["scores"]) @ w) * 2 - 1)
            out.append((str(r["song_id"]), v))
        out.sort(key=lambda x: x[1], reverse=True)
        return out


# ─── Hard rules + precompute ────────────────────────────

def _prefs(user_id: int) -> dict:
    with get_conn(DB_CATALOG) as conn:
        try:
            row = conn.execute("SELECT blacklist_json, anchor_song_id, adventurous FROM companion_prefs WHERE user_id=?",
                               (int(user_id),)).fetchone()
        except Exception:
            row = None
    if not row:
        return {"blacklist": set(), "anchor": "", "adventurous": 0.3}
    try:
        bl = set(json.loads(row["blacklist_json"] or "[]"))
    except Exception:
        bl = set()
    return {"blacklist": bl, "anchor": row["anchor_song_id"] or "",
            "adventurous": float(row["adventurous"] or 0.3)}


def apply_hard_rules(user_id: int, ranked: list[tuple[str, float]],
                     meta: dict[str, dict], top_k: int = 10,
                     max_per_genre: int = BLENDER_MAX_PER_GENRE) -> list[tuple[str, float]]:
    """Blacklist + max-3-per-genre + one anchor slot."""
    prefs = _prefs(int(user_id))
    bl = prefs["blacklist"]
    anchor = prefs["anchor"] or (ANCHOR_SONG_ID if int(user_id) == ANCHOR_USER_ID else "")
    genre_count: dict[int, int] = {}
    out: list[tuple[str, float]] = []
    for sid, sc in ranked:
        if sid in bl:
            continue
        g = int((meta.get(sid) or {}).get("genre_code", 20))
        if genre_count.get(g, 0) >= max_per_genre:
            continue
        genre_count[g] = genre_count.get(g, 0) + 1
        out.append((sid, sc))
        if len(out) >= top_k:
            break
    # Anchor slot: ensure anchor present (swap last slot, never drop rank-1).
    if anchor and anchor not in [s for s, _ in out]:
        a_score = next((sc for s, sc in ranked if s == anchor), None)
        if a_score is not None and len(out) >= 2:
            out[-1] = (anchor, a_score)
        elif a_score is not None:
            out.append((anchor, a_score))
    return out[:top_k]


def precompute_path(user_id: int) -> Path:
    return PRECOMPUTE_DIR / f"top200_{int(user_id)}.json"


def save_precompute(user_id: int, ranked: list[tuple[str, float]], meta: dict[str, dict]) -> Path:
    p = precompute_path(int(user_id))
    payload = {"user_id": int(user_id), "ts": time.time(),
               "items": [{"song_id": s, "score": sc,
                          "title": (meta.get(s) or {}).get("title", ""),
                          "genre_code": (meta.get(s) or {}).get("genre_code", 20)} for s, sc in ranked[:200]]}
    p.write_text(json.dumps(payload))
    return p


def load_precompute(user_id: int, max_age_h: float = 26.0) -> list[tuple[str, float]] | None:
    p = precompute_path(int(user_id))
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
        if time.time() - float(payload.get("ts", 0)) > max_age_h * 3600:
            return None
        return [(d["song_id"], float(d["score"])) for d in payload.get("items", [])]
    except Exception:
        return None
