"""
models/retrievers.py — Cornac-style zoo (cheap, strong).

Five retrievers → ~2000 candidate pool (up from 500/3):
  - ItemKNN (cosine on co-listen vectors)
  - Co-occurrence (direct pair counts — beat pure collab 0.647 vs
    0.338 R-precision on the MPD lineage)
  - Item2Vec (skip-gram over listen sequences, numpy)
  - Popular (view-count floor)
  - Fresh (recent feed)

All CPU, minutes on 8 GB. Batched predict paths only.
"""

import logging
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from config import DB_CATALOG, DB_FEED, DB_FEEDBACK, DB_HISTORY, MODELS_DIR, REC_CANDIDATE_POOL
from core.database import get_conn

logger = logging.getLogger("parlay.retrievers")


class RetrieverZoo:
    """Fitted cheap retrievers. All state is plain numpy/dicts."""

    def __init__(self, version: str = "v0"):
        self.version = version
        self.song_idx: dict[str, int] = {}
        self.idx_song: dict[int, str] = {}
        self.item_knn: dict[str, list[tuple[str, float]]] = {}
        self.cooccur: dict[str, Counter] = {}
        self.item2vec: dict[str, np.ndarray] = {}
        self.popular: list[str] = []
        self.fresh: list[str] = []
        self.user_history: dict[int, list[str]] = {}
        self.trained_at: Optional[float] = None

    # ── fit ──
    def fit(self, frame: list[dict]) -> dict:
        t0 = time.time()
        songs = sorted({str(d["song_id"]) for d in frame})
        self.song_idx = {s: i for i, s in enumerate(songs)}
        self.idx_song = {i: s for s, i in self.song_idx.items()}
        by_user: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for d in frame:
            if float(d.get("plan_reward", 0)) > 0:
                by_user[int(d["user_id"])].append((float(d["timestamp"]), str(d["song_id"])))
        self.user_history = {u: [s for _, s in sorted(v)] for u, v in by_user.items()}

        n = len(songs)
        # Co-occurrence pair counts (session = user's full history).
        co: dict[str, Counter] = defaultdict(Counter)
        for _u, seq in self.user_history.items():
            uniq = list(dict.fromkeys(seq))[:60]
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:i + 11]:  # window 10
                    co[a][b] += 1
                    co[b][a] += 1
        self.cooccur = dict(co)

        # ItemKNN: cosine on binary user vectors (sparse → dense per-item loop, batched).
        # Build item→users incidence via frame.
        item_users: dict[str, set[int]] = defaultdict(set)
        for d in frame:
            if float(d.get("plan_reward", 0)) > 0:
                item_users[str(d["song_id"])].add(int(d["user_id"]))
        norms = {s: math.sqrt(len(u)) for s, u in item_users.items()}
        knn: dict[str, list[tuple[str, float]]] = {}
        s_list = list(item_users.keys())
        for a_i, a in enumerate(s_list):
            ua = item_users[a]
            sims = []
            for b in s_list:
                if b == a:
                    continue
                inter = len(ua & item_users[b])
                if inter == 0:
                    continue
                sims.append((b, inter / max(norms[a] * norms[b], 1e-9)))
            sims.sort(key=lambda x: x[1], reverse=True)
            knn[a] = sims[:50]
        self.item_knn = knn

        # Item2Vec: skip-gram with negative sampling, numpy, tiny.
        dim = 32
        rng = np.random.default_rng(3)
        W_in = {s: rng.normal(0, 0.1, dim).astype(np.float32) for s in songs}
        W_out = {s: rng.normal(0, 0.1, dim).astype(np.float32) for s in songs}
        lr = 0.05
        for _ep in range(3):
            for _u, seq in self.user_history.items():
                seq = seq[:60]
                for pos, center in enumerate(seq):
                    if center not in W_in:
                        continue
                    ctx = [seq[j] for j in range(max(0, pos - 3), min(len(seq), pos + 4))
                           if j != pos and seq[j] in W_in]
                    for c in ctx:
                        # Positive update.
                        x = float(W_in[center] @ W_out[c])
                        g = 1.0 - 1.0 / (1.0 + np.exp(-x))
                        W_in[center] += lr * g * W_out[c]
                        W_out[c] += lr * g * W_in[center]
                        # 2 negatives.
                        for _n in range(2):
                            neg = songs[rng.integers(0, len(songs))]
                            xn = float(W_in[center] @ W_out[neg])
                            gn = -1.0 / (1.0 + np.exp(-xn))
                            W_in[center] += lr * gn * W_out[neg]
                            W_out[neg] += lr * gn * W_in[center]
        # L2-normalise.
        for s in W_in:
            v = W_in[s]
            nv = float(np.linalg.norm(v))
            if nv > 0:
                W_in[s] = (v / nv).astype(np.float32)
        self.item2vec = W_in

        # Popular + fresh from catalog/feed.
        with get_conn(DB_CATALOG) as conn:
            try:
                rows = conn.execute("SELECT song_id FROM songs ORDER BY view_count DESC LIMIT 500").fetchall()
                self.popular = [r["song_id"] for r in rows]
            except Exception:
                self.popular = songs[:500]
        with get_conn(DB_FEED) as conn:
            try:
                snap = conn.execute("SELECT snapshot_id FROM feed_snapshots ORDER BY fetched_at DESC LIMIT 1").fetchone()
                if snap:
                    rows = conn.execute("SELECT song_id FROM feed_songs WHERE snapshot_id=? ORDER BY rank_in_snapshot LIMIT 500",
                                        (snap["snapshot_id"],)).fetchall()
                    self.fresh = [r["song_id"] for r in rows]
            except Exception:
                self.fresh = []
        if not self.fresh:
            self.fresh = self.popular[:200]
        self.trained_at = time.time()
        stats = {"songs": n, "knn_items": len(self.item_knn),
                 "co_items": len(self.cooccur), "secs": round(time.time() - t0, 1)}
        logger.info("✅ Retrievers fitted: %s", stats)
        return stats

    # ── serve ──
    def candidates_for(self, user_id: int, limit: int = REC_CANDIDATE_POOL,
                       per_source: int = 500) -> list[str]:
        """Merge 5 sources → deduped pool (default ~2000)."""
        hist = self.user_history.get(int(user_id), [])
        hist_set = set(hist[-60:])
        pool: list[str] = []
        seen: set[str] = set()

        def _add(items: list[str]) -> None:
            for s in items:
                if s not in seen and s not in hist_set:
                    seen.add(s)
                    pool.append(s)
                if len(pool) >= limit:
                    break

        # 1. Co-occurrence expansion from recent history.
        co_rank: Counter = Counter()
        for s in hist[-20:]:
            for other, c in self.cooccur.get(s, {}).most_common(30):
                co_rank[other] += c
        _add([s for s, _ in co_rank.most_common(per_source)])
        # 2. ItemKNN expansion.
        knn_rank: Counter = Counter()
        for s in hist[-20:]:
            for other, sim in self.item_knn.get(s, [])[:30]:
                knn_rank[other] += sim
        _add([s for s, _ in knn_rank.most_common(per_source)])
        # 3. Item2Vec nearest to mean history vector.
        if hist and self.item2vec:
            vecs = [self.item2vec[s] for s in hist[-20:] if s in self.item2vec]
            if vecs:
                mean = np.mean(vecs, axis=0)
                sims = [(s, float(mean @ v)) for s, v in self.item2vec.items()
                        if s not in hist_set]
                sims.sort(key=lambda x: x[1], reverse=True)
                _add([s for s, _ in sims[:per_source]])
        # 4+5. Popular + fresh floors.
        _add([s for s in self.fresh if s not in hist_set][:per_source])
        _add([s for s in self.popular if s not in hist_set][:per_source])
        return pool[:limit]

    def score_candidates(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        """Cheap retriever score per candidate (for blender input)."""
        hist = set(self.user_history.get(int(user_id), [])[-20:])
        out: dict[str, float] = {}
        for s in song_ids:
            co = sum(self.cooccur.get(h, {}).get(s, 0) for h in hist)
            knn = sum(sim for h in hist for o, sim in self.item_knn.get(h, []) if o == s)
            out[s] = min(1.0, math.log10(1 + co) * 0.5 + min(knn, 2.0) * 0.25
                         + (0.2 if s in self.fresh[:200] else 0.0)
                         + (0.1 if s in self.popular[:200] else 0.0))
        # Normalise to [-1, 1] like other scorers.
        return {s: v * 2 - 1 for s, v in out.items()}

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"retrievers_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 Retrievers saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RetrieverZoo":
        with open(path, "rb") as f:
            m = pickle.load(f)
        logger.info("📂 Retrievers loaded from %s", path)
        return m
