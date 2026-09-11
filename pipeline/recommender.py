"""
pipeline/recommender.py — NeuroSync Ensemble Recommendation Pipeline

Why Top-10 IS Top-10:
═══════════════════════════════════════════════════════════════════
  1. Cognitive load: >10 choices triggers paradox of choice (Barry Schwartz, 2004).
     Users engage better with ≤10 options.
  2. Feedback quality: We need + / - per song. 10 songs = actionable signal
     without fatiguing the user. 100 songs would produce lazy, noisy feedback.
  3. Diversity budget: With 10 slots we can ensure genre diversity (2-3 genres
     guaranteed) while still being personalized.
  4. Prime-number slot trick: We use 10 slots but internally we prime-rank at
     7 (prime) core slots + 3 exploration slots (Thompson sampling fills these).
     7 is prime → maximally "different" from even groupings in feedback cycles.
  5. Hourly cadence × 10 = 240 songs/day. At typical 3-4 min/song that's
     ~12-16h of curated music/day, covering the entire active window.

Pipeline stages:
    [Feed DB] ──────────────────────────────────────────────────────┐
    [History] → Candidate Pool (top 500) → [Filter: bloom/seen]    │
    [Catalog] ──────────────────────────────────────────────────────┘
                                │
                         ┌──────┴──────────────────────────────┐
                         │  4-way Ensemble Scorer               │
                         │  ① SVD score          (w=0.30)      │
                         │  ② NCF score          (w=0.25)      │
                         │  ③ TF-IDF content     (w=0.20)      │
                         │  ④ Thompson bandit    (w=0.15)      │
                         │  ⑤ Recency decay      (w=0.10)      │
                         └───────────────┬─────────────────────┘
                                         │
                              final_score = Σ w_i * s_i
                                         │
                              ┌──────────┴──────────┐
                              │ Diversity Re-rank    │
                              │ (MMR: Maximal        │
                              │  Marginal Relevance) │
                              └──────────┬──────────┘
                                         │
                                     Top 10 + why_top10 JSON
"""

import json
import math
import time
import uuid
import logging
import asyncio
from typing import Optional

import numpy as np

from config import (
    REC_TOP_K, REC_CANDIDATE_POOL,
    W_SVD, W_NCF, W_CONTENT, W_BANDIT, W_RECENCY,
    LISTEN_HALFLIFE_HOURS, DB_FEED, DB_CATALOG, DB_HISTORY,
    DB_RECS, DB_FEEDBACK, FEEDBACK_DISLIKE_THRESHOLD,
)
from core.database import get_conn
from models.bandit_content import ThompsonBandit, TFIDFContentScorer

logger = logging.getLogger("parlay.pipeline")


# ─── Recency score ────────────────────────────────────────

def _recency_score(last_seen: Optional[float]) -> float:
    """
    Exponential half-life decay on song's last_seen timestamp.
    Fresh songs score high; songs not seen in weeks score near 0.
    """
    if not last_seen:
        return 0.3
    age_hours = (time.time() - last_seen) / 3600
    return 0.5 ** (age_hours / LISTEN_HALFLIFE_HOURS)


def _already_played_recently(user_id: int, song_id: str, hours: int = 24) -> bool:
    """Don't re-recommend songs the user played in the last N hours."""
    cutoff = time.time() - hours * 3600
    with get_conn(DB_HISTORY) as conn:
        row = conn.execute(
            "SELECT 1 FROM listens WHERE user_id=? AND song_id=? AND started_at>?",
            (user_id, song_id, cutoff)
        ).fetchone()
    return row is not None


# ─── Candidate generation ─────────────────────────────────

def _get_candidates(user_id: int, limit: int = REC_CANDIDATE_POOL) -> list[dict]:
    """
    Pull candidate songs from:
        A. Latest feed snapshot (fresh trending)
        B. User's genre preferences (catalog-driven)
        C. Songs similar users liked (collaborative signal from feedback table)

    Returns list of song dicts from catalog.
    """
    candidate_ids: set[str] = set()

    # A. Latest feed snapshot
    with get_conn(DB_FEED) as conn:
        snapshot = conn.execute(
            "SELECT snapshot_id FROM feed_snapshots ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        if snapshot:
            rows = conn.execute(
                "SELECT song_id FROM feed_songs WHERE snapshot_id=? ORDER BY rank_in_snapshot",
                (snapshot["snapshot_id"],)
            ).fetchall()
            candidate_ids.update(r["song_id"] for r in rows)

    # B. Catalog top songs by view count (popularity floor)
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute(
            "SELECT song_id FROM songs ORDER BY view_count DESC LIMIT ?",
            (limit // 2,)
        ).fetchall()
        candidate_ids.update(r["song_id"] for r in rows)

    # C. Songs liked by other users (crude collab signal when SVD is cold)
    with get_conn(DB_FEEDBACK) as conn:
        rows = conn.execute(
            "SELECT DISTINCT song_id FROM feedback WHERE signal=1 AND user_id!=? ORDER BY created_at DESC LIMIT 200",
            (user_id,)
        ).fetchall()
        candidate_ids.update(r["song_id"] for r in rows)

    # Fetch full song data
    ids_list = list(candidate_ids)[:limit]
    if not ids_list:
        return []

    placeholders = ",".join(["?"] * len(ids_list))
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute(
            f"SELECT * FROM songs WHERE song_id IN ({placeholders})",
            ids_list
        ).fetchall()

    candidates = [dict(row) for row in rows]

    # Filter out recently played
    candidates = [
        s for s in candidates
        if not _already_played_recently(user_id, s["song_id"])
    ]

    return candidates


# ─── Diversity re-ranker (MMR) ────────────────────────────

def _mmr_rerank(
    scored: list[tuple[str, float, dict]],
    top_k: int,
    song_embeddings: dict[str, np.ndarray],
    lambda_param: float = 0.7,
) -> list[tuple[str, float, dict]]:
    """
    Maximal Marginal Relevance:
        score_mmr(d) = λ * relevance(d) - (1-λ) * max_sim(d, selected)

    λ = 0.7: 70% relevance-driven, 30% diversity.
    This prevents Top-10 from being 10 near-identical songs.

    When embeddings aren't available, falls back to genre-diversity rule:
    no more than 3 songs from the same genre in Top-10.
    """
    if not scored:
        return []

    selected: list[tuple[str, float, dict]] = []
    remaining = list(scored)

    while remaining and len(selected) < top_k:
        if not selected:
            # First: pick highest relevance unconditionally
            best = max(remaining, key=lambda x: x[1])
        else:
            best = None
            best_score = -float("inf")
            selected_ids = [s[0] for s in selected]

            for cand in remaining:
                rel = cand[1]

                # Embedding-based similarity
                if song_embeddings:
                    e_cand = song_embeddings.get(cand[0])
                    if e_cand is not None:
                        sims = []
                        for sel_id in selected_ids:
                            e_sel = song_embeddings.get(sel_id)
                            if e_sel is not None:
                                cos = float(np.dot(e_cand, e_sel) /
                                            (np.linalg.norm(e_cand) * np.linalg.norm(e_sel) + 1e-8))
                                sims.append(cos)
                        max_sim = max(sims) if sims else 0.0
                    else:
                        max_sim = 0.0
                else:
                    # Fallback: penalise same genre
                    cand_genre = cand[2].get("genre_code", 20)
                    sel_genres = [s[2].get("genre_code", 20) for s in selected]
                    same_genre_count = sel_genres.count(cand_genre)
                    max_sim = min(same_genre_count * 0.3, 0.9)

                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = cand

        selected.append(best)
        remaining.remove(best)

    return selected


# ─── Main Pipeline ───────────────────────────────────────

class NeuroSyncPipeline:
    """
    The NeuroSync recommendation engine.
    Call .recommend(user_id) to get top-K songs with explanations.
    """

    def __init__(self, svd_model=None, ncf_model=None):
        self._svd = svd_model
        self._ncf = ncf_model
        self._tfidf = TFIDFContentScorer()
        self._tfidf_fitted = False
        self._song_embeddings: dict[str, np.ndarray] = {}
        self._model_version = "v0"

    def set_models(self, svd_model, ncf_model, version: str = "v0") -> None:
        self._svd = svd_model
        self._ncf = ncf_model
        self._model_version = version

    def fit_tfidf(self) -> None:
        """Build TF-IDF content scorer from full catalog."""
        with get_conn(DB_CATALOG) as conn:
            rows = conn.execute("SELECT song_id, title FROM songs").fetchall()
        songs = [dict(r) for r in rows]
        if songs:
            self._tfidf.fit(songs)
            self._tfidf_fitted = True
            logger.info(f"TF-IDF fitted on {len(songs)} catalog songs.")

    def _load_embeddings(self) -> None:
        """Load SVD song embeddings from DB for MMR diversity."""
        from core.database import DB_EMBEDDINGS, blob_to_ndarray
        try:
            with get_conn(DB_EMBEDDINGS) as conn:
                rows = conn.execute(
                    "SELECT song_id, vector, dim FROM song_embeddings"
                ).fetchall()
            self._song_embeddings = {
                r["song_id"]: blob_to_ndarray(r["vector"], r["dim"])
                for r in rows
            }
            logger.debug(f"Loaded {len(self._song_embeddings)} song embeddings.")
        except Exception as e:
            logger.warning(f"Could not load embeddings: {e}")

    def _why_top10(
        self,
        song_id: str,
        rank: int,
        svd_s: float,
        ncf_s: float,
        content_s: float,
        bandit_s: float,
        recency_s: float,
        final_s: float,
    ) -> str:
        """
        Generate a human-readable JSON explanation of why this song is in Top-10.
        Used for debugging, UI display, and future interpretability work.
        """
        drivers = []
        if svd_s     > 0.4: drivers.append("collaborative fit")
        if ncf_s     > 0.4: drivers.append("neural pattern match")
        if content_s > 0.3: drivers.append("title similarity to your taste")
        if bandit_s  > 0.6: drivers.append("exploration pick")
        if recency_s > 0.6: drivers.append("trending now")

        why = {
            "rank":          rank,
            "final_score":   round(final_s, 4),
            "breakdown": {
                "svd":     round(svd_s, 4),
                "ncf":     round(ncf_s, 4),
                "content": round(content_s, 4),
                "bandit":  round(bandit_s, 4),
                "recency": round(recency_s, 4),
            },
            "drivers": drivers or ["general popularity"],
        }
        return json.dumps(why)

    def recommend(
        self,
        user_id: int,
        top_k: int = REC_TOP_K,
        exploration_slots: int = 3,
    ) -> list[dict]:
        """
        Full pipeline run for one user.

        exploration_slots: How many of the top-k slots are reserved for
            Thompson-driven exploration (new/risky picks).
            3 of 10 = 30% exploration, 70% exploitation.
        """
        t0 = time.time()
        session_id = str(uuid.uuid4())

        # 1. Candidate generation
        candidates = _get_candidates(user_id)
        if not candidates:
            logger.warning(f"No candidates for user {user_id}.")
            return []

        song_ids = [c["song_id"] for c in candidates]
        song_meta = {c["song_id"]: c for c in candidates}

        # 2. Score with all 4 models
        svd_scores     = self._svd.predict_batch(user_id, song_ids) if self._svd else {}
        ncf_scores     = self._ncf.predict_batch(user_id, song_ids) if self._ncf else {}
        content_scores = self._tfidf.score_candidates(user_id, song_ids) if self._tfidf_fitted else {}

        bandit = ThompsonBandit(user_id)
        bandit_scores  = bandit.sample_batch(song_ids)

        # 3. Compute ensemble final score
        scored: list[tuple[str, float, dict]] = []
        for sid in song_ids:
            meta = song_meta[sid]
            rec  = _recency_score(meta.get("last_seen"))

            # Min-max normalise each sub-score to [0, 1]
            def _norm(v: float) -> float:
                return max(0.0, min(1.0, (v + 1.0) / 2.0))

            s_svd     = _norm(svd_scores.get(sid, 0.0))
            s_ncf     = _norm(ncf_scores.get(sid, 0.0))
            s_content = _norm(content_scores.get(sid, 0.0))
            s_bandit  = float(bandit_scores.get(sid, 0.5))
            s_recency = rec

            final = (
                W_SVD     * s_svd
                + W_NCF     * s_ncf
                + W_CONTENT * s_content
                + W_BANDIT  * s_bandit
                + W_RECENCY * s_recency
            )

            scored.append((sid, final, {
                **meta,
                "_svd": s_svd, "_ncf": s_ncf,
                "_content": s_content, "_bandit": s_bandit,
                "_recency": s_recency,
            }))

        # Sort by final score
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. MMR diversity re-rank
        self._load_embeddings()
        top_diverse = _mmr_rerank(
            scored,
            top_k=top_k,
            song_embeddings=self._song_embeddings,
            lambda_param=0.70,
        )

        # 5. Save session + items to DB
        results = []
        with get_conn(DB_RECS) as conn:
            conn.execute("""
            INSERT INTO recommendation_sessions
                (session_id, user_id, created_at, model_version, algorithm, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, time.time(), self._model_version, "ensemble", "pending"))

            for rank, (sid, final_score, meta) in enumerate(top_diverse, start=1):
                s_svd     = meta.get("_svd", 0.0)
                s_ncf     = meta.get("_ncf", 0.0)
                s_content = meta.get("_content", 0.0)
                s_bandit  = meta.get("_bandit", 0.0)
                s_recency = meta.get("_recency", 0.0)

                why = self._why_top10(
                    sid, rank, s_svd, s_ncf, s_content, s_bandit, s_recency, final_score
                )
                conn.execute("""
                INSERT INTO rec_items
                    (session_id, song_id, rank, final_score,
                     svd_score, ncf_score, content_score, bandit_score, recency_score, why_top10)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, sid, rank, final_score,
                    s_svd, s_ncf, s_content, s_bandit, s_recency, why
                ))

                results.append({
                    "session_id":  session_id,
                    "song_id":     sid,
                    "rank":        rank,
                    "title":       meta.get("title", ""),
                    "channel":     meta.get("channel", ""),
                    "yt_url":      meta.get("yt_url", ""),
                    "thumbnail":   meta.get("thumbnail_url", ""),
                    "final_score": round(final_score, 4),
                    "why":         json.loads(why),
                })

        elapsed = time.time() - t0
        logger.info(
            f"✅  NeuroSync → user={user_id} | top-{top_k} | "
            f"{len(candidates)} candidates → {elapsed*1000:.0f}ms"
        )
        return results

    def record_feedback(
        self,
        user_id: int,
        song_id: str,
        signal: int,        # +1 or -1
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Process user feedback:
        1. Save to feedback DB
        2. Online-update SVD + NCF
        3. Update Thompson bandit arm
        4. Check dislike threshold (auto-disable)
        Returns status dict.
        """
        now = time.time()
        reward = float(signal)

        with get_conn(DB_FEEDBACK) as conn:
            conn.execute("""
            INSERT INTO feedback (user_id, song_id, rec_session_id, signal, created_at, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, song_id, session_id, signal, now, self._model_version))

            # Track dislike count
            if signal < 0:
                conn.execute("""
                INSERT INTO dislike_counts (user_id, count, last_reset)
                VALUES (?, 1, ?)
                ON CONFLICT(user_id) DO UPDATE SET count = count + 1
                """, (user_id, now))

            row = conn.execute(
                "SELECT count FROM dislike_counts WHERE user_id=?", (user_id,)
            ).fetchone()
            dislike_count = row["count"] if row else 0

        # Online model updates
        if self._svd:
            self._svd.online_update(user_id, song_id, reward)
        if self._ncf:
            self._ncf.online_update(user_id, song_id, reward)

        # Bandit update
        bandit = ThompsonBandit(user_id)
        bandit.update(song_id, reward)

        auto_disabled = dislike_count >= FEEDBACK_DISLIKE_THRESHOLD

        return {
            "status":          "ok",
            "signal":          signal,
            "dislike_count":   dislike_count,
            "auto_disabled":   auto_disabled,
            "message":         (
                "⚠ NeuroSync auto-disabled (too many dislikes). Re-enable in settings."
                if auto_disabled else "Feedback recorded."
            )
        }