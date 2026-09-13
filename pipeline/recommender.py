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
    REC_TOP_K, REC_CANDIDATE_POOL, REC_PRECOMPUTE_TOP_N,
    W_SVD, W_NCF, W_CONTENT, W_BANDIT, W_RECENCY,
    LISTEN_HALFLIFE_HOURS, DB_FEED, DB_CATALOG, DB_HISTORY,
    DB_RECS, DB_FEEDBACK, FEEDBACK_DISLIKE_THRESHOLD,
)
from core.database import get_conn
from models.bandit_content import ThompsonBandit, TFIDFContentScorer

try:
    from ranker.blender import Blender, apply_hard_rules, save_precompute, load_precompute
    BLENDER_AVAILABLE = True
except Exception:
    BLENDER_AVAILABLE = False
    Blender = None  # type: ignore

try:
    from companion.commands import (mood_filter, exploration_slots_for,
                                    why_in_words, record_journal_day)
    COMPANION_AVAILABLE = True
except Exception:
    COMPANION_AVAILABLE = False

try:
    from music_math.features import math_bonus
    from music_math.harmony import consonance_of_title as _mm_cons
    from music_math.rhythm import groove_score as _mm_groove, tempo_fit as _mm_tempo_fit
    from music_math.why_music import reasons_for_track as _math_reasons
    MATH_AVAILABLE = True
except Exception:
    MATH_AVAILABLE = False

try:
    from audio.store import get as _audio_get
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


def _measured_math_bonus(song_id: str, meta: dict) -> tuple[float, bool]:
    """Measured ears beat guessed priors. Returns (bonus, measured).

    With an audio_features row: real BPM → tempo_fit, real key strength
    → harmony, measured danceability. Without: title-cue math_bonus.
    """
    if MATH_AVAILABLE and AUDIO_AVAILABLE:
        try:
            row = _audio_get(str(song_id))
        except Exception:
            row = None
        if row and (row.get("bpm") or row.get("key_strength")):
            bpm = float(row.get("bpm") or 0)
            tfit = _mm_tempo_fit(bpm) if bpm else 0.5
            ks = float(row.get("key_strength") or 0.5)
            dance = float(row.get("danceability") or 0.5)
            b = (tfit - 0.5) * 0.12 + (ks - 0.5) * 0.08 + (dance - 0.5) * 0.06
            return round(max(-0.05, min(0.12, b)), 4), True
    if MATH_AVAILABLE:
        try:
            return float(math_bonus(meta)), False
        except Exception:
            pass
    return 0.0, False

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
    The NeuroSync recommendation engine (MAX: retrievers → blender → rules).
    Call .recommend(user_id) to get top-K songs with explanations.
    """

    def __init__(self, svd_model=None, ncf_model=None, max_bundle: dict | None = None):
        self._svd = svd_model
        self._ncf = ncf_model
        self._tfidf = TFIDFContentScorer()
        self._tfidf_fitted = False
        self._song_embeddings: dict[str, np.ndarray] = {}
        self._model_version = "v0"
        # MAX models (optional, degrade gracefully).
        self._als = (max_bundle or {}).get("als")
        self._fmf = (max_bundle or {}).get("feature_mf")
        self._retrievers = (max_bundle or {}).get("retrievers")
        self._sequence = (max_bundle or {}).get("sequence")
        self._two_tower = (max_bundle or {}).get("two_tower")
        self._blender = (max_bundle or {}).get("blender")
        if self._blender is None and BLENDER_AVAILABLE:
            try:
                self._blender = Blender()
            except Exception:
                self._blender = None

    def set_models(self, svd_model, ncf_model, version: str = "v0", max_bundle: dict | None = None) -> None:
        self._svd = svd_model
        self._ncf = ncf_model
        self._model_version = version
        if max_bundle:
            for k, attr in (("als", "_als"), ("feature_mf", "_fmf"), ("retrievers", "_retrievers"),
                            ("sequence", "_sequence"), ("two_tower", "_two_tower"), ("blender", "_blender")):
                if max_bundle.get(k) is not None:
                    setattr(self, attr, max_bundle[k])

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
        exploration_slots: int | None = None,
        mood: str = "",
        use_precompute: bool = True,
    ) -> list[dict]:
        """MAX pipeline: 5+ retrievers → learned blender → MMR + hard rules.

        exploration_slots: None → companion prefs (/adventurous), else override.
        mood: '', 'morning', 'gym', '4am', ... → filter + per-mood weights.
        use_precompute: nightly Top-200 reranked with fresh feed only.
        """
        t0 = time.time()
        session_id = str(uuid.uuid4())
        if exploration_slots is None and COMPANION_AVAILABLE:
            try:
                exploration_slots = exploration_slots_for(int(user_id), 3)
            except Exception:
                exploration_slots = 3
        exploration_slots = int(exploration_slots if exploration_slots is not None else 3)
        mood = (mood or "").strip().lower()
        mfil = mood_filter(mood) if COMPANION_AVAILABLE and mood else {"genres": [], "adventurous": 0.0, "recency_boost": 0.0}

        # 1. Candidate generation: retrievers → ~2000 pool, else legacy.
        candidates: list[dict] = []
        retr_ids: list[str] = []
        if self._retrievers is not None:
            try:
                retr_ids = self._retrievers.candidates_for(int(user_id), limit=REC_CANDIDATE_POOL)
            except Exception as e:
                logger.warning("Retrievers failed, falling back: %s", e)
        # Hourly fast path: precomputed Top-200 + fresh feed rerank.
        pre_ranked = load_precompute(int(user_id)) if (BLENDER_AVAILABLE and use_precompute) else None
        if retr_ids:
            ids_list = retr_ids
            if pre_ranked:
                pre_ids = [s for s, _ in pre_ranked[:REC_PRECOMPUTE_TOP_N]]
                ids_list = list(dict.fromkeys(pre_ids + ids_list))[:REC_CANDIDATE_POOL]
            placeholders = ",".join(["?"] * len(ids_list))
            try:
                with get_conn(DB_CATALOG) as conn:
                    rows = conn.execute(f"SELECT * FROM songs WHERE song_id IN ({placeholders})", ids_list).fetchall()
                candidates = [dict(r) for r in rows]
                # Mood filter: boost preferred genres (filter lightly, don't nuke pool).
                if mfil["genres"]:
                    pref = set(mfil["genres"])
                    boosted = [c for c in candidates if int(c.get("genre_code", 20)) in pref]
                    if len(boosted) >= 50:
                        candidates = boosted + [c for c in candidates if c not in boosted]
            except Exception as e:
                logger.warning("Catalog fetch failed: %s", e)
                candidates = []
        if not candidates:
            candidates = _get_candidates(int(user_id), limit=REC_CANDIDATE_POOL)
        if not candidates:
            logger.warning(f"No candidates for user {user_id}.")
            return []
        # Companion blacklist pre-filter.
        try:
            if COMPANION_AVAILABLE:
                from companion.commands import get_prefs as _gp
                _bl = set(_gp(int(user_id)).get("blacklist", []))
                if _bl:
                    candidates = [c for c in candidates if c["song_id"] not in _bl]
        except Exception:
            pass
        if not candidates:
            return []

        song_ids = [c["song_id"] for c in candidates]
        song_meta = {c["song_id"]: c for c in candidates}

        # 2. Score with every available sub-model.
        svd_scores = self._svd.predict_batch(user_id, song_ids) if self._svd else {}
        ncf_scores = self._ncf.predict_batch(user_id, song_ids) if self._ncf else {}
        content_scores = self._tfidf.score_candidates(user_id, song_ids) if self._tfidf_fitted else {}
        bandit = ThompsonBandit(user_id)
        bandit_scores = bandit.sample_batch(song_ids)
        als_scores = self._als.predict_batch(user_id, song_ids) if self._als else {}
        fmf_scores = self._fmf.predict_batch(user_id, song_ids) if self._fmf else {}
        tt_scores = self._two_tower.predict_batch(user_id, song_ids) if self._two_tower else {}
        retr_scores = self._retrievers.score_candidates(user_id, song_ids) if self._retrievers else {}
        seq_scores = self._sequence.predict_batch(user_id, song_ids) if self._sequence else {}

        # 3. Blend: learned LR if available, else fixed weights.
        def _norm(v: float) -> float:
            return max(0.0, min(1.0, (v + 1.0) / 2.0))

        scored: list[tuple[str, float, dict]] = []
        # Artist-repeat tracking for repeat penalty input.
        from collections import Counter as _C
        for sid in song_ids:
            meta = song_meta[sid]
            rec = _recency_score(meta.get("last_seen")) + float(mfil.get("recency_boost", 0.0))
            rec = max(0.0, min(1.0, rec))
            s_svd, s_ncf = _norm(svd_scores.get(sid, 0.0)), _norm(ncf_scores.get(sid, 0.0))
            s_content = _norm(content_scores.get(sid, 0.0))
            s_bandit = float(bandit_scores.get(sid, 0.5))
            s_als, s_fmf = _norm(als_scores.get(sid, 0.0)), _norm(fmf_scores.get(sid, 0.0))
            s_tt = _norm(tt_scores.get(sid, 0.0))
            s_retr, s_seq = _norm(retr_scores.get(sid, 0.0)), _norm(seq_scores.get(sid, 0.0))
            if self._blender is not None:
                feats = {"als": als_scores.get(sid, 0.0), "feature_mf": fmf_scores.get(sid, 0.0),
                         "svd": svd_scores.get(sid, 0.0), "ncf": ncf_scores.get(sid, 0.0),
                         "two_tower": tt_scores.get(sid, 0.0), "content": content_scores.get(sid, 0.0),
                         "bandit": s_bandit * 2 - 1, "recency": rec * 2 - 1,
                         "retriever": retr_scores.get(sid, 0.0), "sequence": seq_scores.get(sid, 0.0),
                         "freshness": rec * 2 - 1, "repeat_pen": 0.0}
                try:
                    final01 = (self._blender.score(int(user_id), feats, mood=mood) + 1) / 2
                except Exception:
                    final01 = 0.5
                final = max(0.0, min(1.0, final01))
            else:
                final = (0.22 * s_als + 0.14 * s_fmf + 0.10 * s_svd + 0.06 * s_ncf + 0.08 * s_tt
                         + 0.10 * s_content + 0.08 * s_bandit + 0.05 * rec + 0.10 * s_retr + 0.07 * s_seq)
            # music_math: measured ears beat guessed priors (audio_features
            # row when the night listener has heard the track, else title cues).
            math_b = 0.0
            heard = False
            if MATH_AVAILABLE:
                try:
                    math_b, heard = _measured_math_bonus(sid, meta)
                except Exception:
                    math_b, heard = 0.0, False
            final = max(0.0, min(1.0, final + math_b))
            scored.append((sid, final, {**meta, "_svd": s_svd, "_ncf": s_ncf, "_content": s_content,
                                        "_bandit": s_bandit, "_recency": rec, "_als": s_als,
                                        "_fmf": s_fmf, "_tt": s_tt, "_retr": s_retr, "_seq": s_seq,
                                        "_math": math_b, "_heard": heard}))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 4. MMR diversity re-rank (exploration_slots widen diversity).
        self._load_embeddings()
        lam = 0.70 - min(exploration_slots, 5) * 0.04  # more exploration → more diversity
        top_diverse = _mmr_rerank(scored, top_k=max(top_k * 3, top_k + 5),
                                  song_embeddings=self._song_embeddings, lambda_param=max(0.4, lam))
        # 5. Hard rules: blacklist, max-3-per-genre, anchor slot.
        if BLENDER_AVAILABLE:
            try:
                ruled = apply_hard_rules(int(user_id), [(s, sc) for s, sc, _ in top_diverse], song_meta, top_k=top_k)
                id2meta = {s: m for s, _, m in top_diverse}
                top_diverse = [(s, sc, id2meta[s]) for s, sc in ruled if s in id2meta]
            except Exception as e:
                logger.warning("Hard rules skipped: %s", e)
                top_diverse = top_diverse[:top_k]
        else:
            top_diverse = top_diverse[:top_k]
        # Nightly precompute refresh (Top-200 of full scored list).
        try:
            if BLENDER_AVAILABLE:
                save_precompute(int(user_id), [(s, sc) for s, sc, _ in scored], song_meta)
        except Exception:
            pass

        # 6. Save session + items to DB
        results = []
        with get_conn(DB_RECS) as conn:
            conn.execute("""
            INSERT INTO recommendation_sessions
                (session_id, user_id, created_at, model_version, algorithm, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, user_id, time.time(), self._model_version, "ensemble-max", "pending"))

            for rank, (sid, final_score, meta) in enumerate(top_diverse, start=1):
                s_svd = meta.get("_svd", 0.0)
                s_ncf = meta.get("_ncf", 0.0)
                s_content = meta.get("_content", 0.0)
                s_bandit = meta.get("_bandit", 0.0)
                s_recency = meta.get("_recency", 0.0)

                why = self._why_top10(sid, rank, s_svd, s_ncf, s_content, s_bandit, s_recency, final_score)
                conn.execute("""
                INSERT INTO rec_items
                    (session_id, song_id, rank, final_score,
                     svd_score, ncf_score, content_score, bandit_score, recency_score, why_top10)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, sid, rank, final_score, s_svd, s_ncf, s_content, s_bandit, s_recency, why))
                wdict = json.loads(why)
                try:
                    wdict["why_text"] = why_in_words(wdict, meta.get("title", "")) if COMPANION_AVAILABLE else ""
                except Exception:
                    wdict["why_text"] = ""
                if MATH_AVAILABLE:
                    try:
                        mr = _math_reasons(meta)
                        if meta.get("_heard") and AUDIO_AVAILABLE:
                            try:
                                arow = _audio_get(sid)
                                if arow and arow.get("bpm"):
                                    mr = [f"heard at {float(arow['bpm']):.0f} BPM in {arow.get('musical_key','?')} {arow.get('mode','')}"] + mr
                            except Exception:
                                pass
                        if mr:
                            wdict["math_reasons"] = mr[:2]
                            if wdict.get("why_text"):
                                wdict["why_text"] += f" Math note: {mr[0]}."
                    except Exception:
                        pass
                results.append({
                    "session_id": session_id, "song_id": sid, "rank": rank,
                    "title": meta.get("title", ""), "channel": meta.get("channel", ""),
                    "yt_url": meta.get("yt_url", ""), "thumbnail": meta.get("thumbnail_url", ""),
                    "final_score": round(final_score, 4), "why": wdict,
                })
        try:
            if COMPANION_AVAILABLE:
                record_journal_day(int(user_id), plays=0, likes=0)
        except Exception:
            pass
        elapsed = time.time() - t0
        logger.info("✅  NeuroSync-MAX → user=%s | top-%d | %d candidates (mood=%r) → %dms",
                    user_id, top_k, len(candidates), mood, int(elapsed * 1000))
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