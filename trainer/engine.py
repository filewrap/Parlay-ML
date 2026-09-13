"""
trainer/engine.py — NeuroSync Training Engine

Full training pipeline:
  1. Load all interactions from history + feedback DBs
  2. Train FunkSVD (matrix factorization)
  3. Train NCF (neural CF)
  4. Save models + log training run
  5. Return new pipeline with fresh models

Also contains:
  - SyntheticDataGenerator: generates 10,000 fake song/user interactions
    for bootstrapping when real data is sparse (cold-start)
  - ModelEvaluator: RMSE, precision@k, recall@k, NDCG@k metrics
"""

import math
import time
import uuid
import random
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    MODELS_DIR, SVD_FACTORS, SVD_EPOCHS, SVD_LR, SVD_REG,
    NCF_EPOCHS, GENRE_MAP, LANGUAGE_MAP, FIB_SEQ,
    REWARD_LIKE, REWARD_DISLIKE, REWARD_FULL_PLAY, REWARD_PARTIAL_PLAY,
    TRAIN_LOCK_FILE,
)
from core.database import get_conn, DB_CATALOG, DB_HISTORY, DB_FEEDBACK, DB_TRAINING
from models.svd_model import FunkSVD, load_interactions
from models.ncf_model import NCFModel

logger = logging.getLogger("parlay.trainer")


# ─── Synthetic Data Generator ────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates realistic synthetic music listen data for cold-start training.

    Based on a power-law song popularity distribution:
        P(song i is listened to) ∝ 1/rank^alpha   (Zipf's law)

    Zipf's law governs real music consumption:
    the top 1% of songs get ~80% of listens.
    We mirror this so our model trains on realistic distributions
    rather than uniform random data.

    Also generates genre-coherent user profiles:
    each synthetic user has a preferred genre cluster,
    and listens to songs in those clusters more often.
    """

    def __init__(self, n_users=50, n_songs=10000):
        self.n_users = n_users
        self.n_songs = n_songs

    def _zipf_weights(self, n: int, alpha: float = 1.2) -> np.ndarray:
        """
        Zipf distribution: w_i = 1 / i^alpha
        alpha=1.2 matches observed music streaming distributions.
        """
        ranks = np.arange(1, n + 1, dtype=np.float64)
        weights = 1.0 / (ranks ** alpha)
        return weights / weights.sum()

    def generate_songs(self) -> list[dict]:
        genres = list(GENRE_MAP.keys())
        songs = []
        for i in range(self.n_songs):
            genre = random.choice(genres)
            views = int(random.lognormvariate(12, 2))   # log-normal view dist
            energy = FIB_SEQ[random.randint(0, len(FIB_SEQ)-1)] / max(FIB_SEQ)
            songs.append({
                "song_id":       f"syn_{i:06d}",
                "title":         f"Synthetic Track {i} [{genre}]",
                "channel":       f"Artist_{random.randint(0, 500)}",
                "channel_id":    f"ch_{random.randint(0, 500)}",
                "duration":      random.randint(90, 480),
                "view_count":    max(views, 1000),
                "like_count":    int(max(views, 1000) * random.uniform(0.01, 0.15)),
                "upload_date":   "20240101",
                "thumbnail_url": "",
                "yt_url":        f"https://youtube.com/watch?v=syn{i}",
                "genre_code":    GENRE_MAP.get(genre, 20),
                "language_code": LANGUAGE_MAP.get("en", 0),
                "has_official":  random.randint(0, 1),
                "has_lyric":     random.randint(0, 1),
                "energy_score":  energy,
                "first_seen":    time.time() - random.uniform(0, 86400 * 30),
                "last_seen":     time.time() - random.uniform(0, 86400 * 7),
                "times_fetched": random.randint(1, 10),
            })
        return songs

    def generate_interactions(
        self,
        songs: list[dict],
        interactions_per_user: int = 200,
    ) -> list[tuple[int, str, float]]:
        """
        Generate (user_id, song_id, reward) triples.
        Each user has:
          - 1-3 preferred genres (70% of listens come from preferred genres)
          - Zipf-weighted song selection within genre
        """
        genres = list(GENRE_MAP.values())
        song_by_genre: dict[int, list[str]] = {}
        for s in songs:
            g = s["genre_code"]
            song_by_genre.setdefault(g, []).append(s["song_id"])

        all_song_ids = [s["song_id"] for s in songs]
        weights = self._zipf_weights(len(all_song_ids))

        interactions = []
        for uid in range(self.n_users):
            preferred_genres = random.sample(genres, k=min(3, len(genres)))
            seen = set()

            for _ in range(interactions_per_user):
                # 70% chance: pick from preferred genre
                if random.random() < 0.70:
                    pref_genre = random.choice(preferred_genres)
                    genre_songs = song_by_genre.get(pref_genre, all_song_ids)
                    if genre_songs:
                        sid = random.choice(genre_songs)
                    else:
                        sid = np.random.choice(all_song_ids, p=weights)
                else:
                    sid = np.random.choice(all_song_ids, p=weights)

                if sid in seen:
                    continue
                seen.add(sid)

                # Generate realistic reward
                in_preferred = (
                    any(s["song_id"] == sid and s["genre_code"] in preferred_genres
                        for s in songs[:100])
                )
                if in_preferred:
                    reward = random.choices(
                        [REWARD_LIKE, REWARD_FULL_PLAY, REWARD_PARTIAL_PLAY, REWARD_DISLIKE],
                        weights=[0.35, 0.40, 0.20, 0.05]
                    )[0]
                else:
                    reward = random.choices(
                        [REWARD_LIKE, REWARD_FULL_PLAY, REWARD_PARTIAL_PLAY, REWARD_DISLIKE],
                        weights=[0.10, 0.20, 0.35, 0.35]
                    )[0]

                interactions.append((uid, sid, reward))

        logger.info(f"SyntheticDataGenerator: {len(interactions)} interactions for {self.n_users} users")
        return interactions

    def seed_db(self, songs: list[dict], interactions: list[tuple]) -> None:
        """Write synthetic data to DBs for model training."""
        with get_conn(DB_CATALOG) as conn:
            conn.executemany("""
            INSERT OR IGNORE INTO songs (
                song_id, title, channel, channel_id, duration, view_count,
                like_count, upload_date, thumbnail_url, yt_url, genre_code,
                language_code, has_official, has_lyric, energy_score,
                first_seen, last_seen, times_fetched
            ) VALUES (
                :song_id, :title, :channel, :channel_id, :duration, :view_count,
                :like_count, :upload_date, :thumbnail_url, :yt_url, :genre_code,
                :language_code, :has_official, :has_lyric, :energy_score,
                :first_seen, :last_seen, :times_fetched
            )
            """, songs)

        with get_conn(DB_HISTORY) as conn:
            conn.executemany("""
            INSERT INTO listens (user_id, song_id, started_at, completion_pct, source)
            VALUES (?, ?, ?, ?, ?)
            """, [
                (u, s, time.time() - random.uniform(0, 86400*30),
                 max(0, min(1, (r + 1) / 2.0)), 'synthetic')
                for u, s, r in interactions
            ])

        logger.info(f"Synthetic data seeded: {len(songs)} songs, {len(interactions)} listens.")


# ─── Model Evaluator ─────────────────────────────────────────

class ModelEvaluator:
    """
    Offline evaluation metrics for recommendation quality.

    Metrics:
        RMSE:       Root Mean Squared Error on predicted vs actual reward
        Precision@k: Fraction of top-k predictions that are relevant (reward>0)
        Recall@k:   Fraction of relevant items in top-k
        NDCG@k:     Normalised Discounted Cumulative Gain — rewards relevant
                    items appearing earlier in the top-k list
    """

    def __init__(self, model: FunkSVD):
        self.model = model

    def evaluate(
        self,
        test_interactions: list[tuple[int, str, float]],
        k: int = 10,
    ) -> dict:
        if not test_interactions:
            return {"error": "no test data"}

        # RMSE
        sq_err = 0.0
        n = 0
        for uid, sid, r in test_interactions:
            pred = self.model.predict(uid, sid)
            sq_err += (r - pred) ** 2
            n += 1
        rmse = math.sqrt(sq_err / max(n, 1))

        # Group by user
        user_items: dict[int, list[tuple[str, float]]] = {}
        for uid, sid, r in test_interactions:
            user_items.setdefault(uid, []).append((sid, r))

        precision_k_sum = 0.0
        recall_k_sum    = 0.0
        ndcg_k_sum      = 0.0
        user_count      = 0

        for uid, items in user_items.items():
            # Sort items by true reward (ground truth)
            relevant = {sid for sid, r in items if r > 0}
            if not relevant:
                continue

            # Predict scores for all test items of this user
            song_ids = [sid for sid, _ in items]
            preds    = self.model.predict_batch(uid, song_ids)
            ranked   = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:k]
            ranked_ids = [sid for sid, _ in ranked]

            hits = [1 if sid in relevant else 0 for sid in ranked_ids]
            precision_k = sum(hits) / k
            recall_k    = sum(hits) / len(relevant)

            # NDCG
            dcg  = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
            ideal = sorted(hits, reverse=True)
            idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
            ndcg = dcg / max(idcg, 1e-8)

            precision_k_sum += precision_k
            recall_k_sum    += recall_k
            ndcg_k_sum      += ndcg
            user_count      += 1

        n_u = max(user_count, 1)
        return {
            "rmse":          round(rmse, 4),
            f"precision@{k}": round(precision_k_sum / n_u, 4),
            f"recall@{k}":    round(recall_k_sum / n_u, 4),
            f"ndcg@{k}":      round(ndcg_k_sum / n_u, 4),
            "n_users_eval":   user_count,
            "n_interactions": n,
        }


# ─── Training Engine (MAX) ─────────────────────────────────

def _peak_rss_mb() -> float:
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def _acquire_lock() -> bool:
    try:
        if TRAIN_LOCK_FILE.exists():
            age = time.time() - TRAIN_LOCK_FILE.stat().st_mtime
            if age < 4 * 3600:  # overlapping run guard
                return False
        TRAIN_LOCK_FILE.write_text(str(time.time()))
        return True
    except Exception:
        return True


def _release_lock() -> None:
    try:
        TRAIN_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def get_registry(key: str = "prod") -> dict | None:
    with get_conn(DB_TRAINING) as conn:
        try:
            row = conn.execute("SELECT run_id, model_version, ndcg10 FROM model_registry WHERE key=?",
                               (key,)).fetchone()
        except Exception:
            return None
    return dict(row) if row else None


def set_registry(key: str, run_id: str, version: str, ndcg10: float) -> None:
    with get_conn(DB_TRAINING) as conn:
        conn.execute("""
        INSERT INTO model_registry (key, run_id, model_version, ndcg10, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET run_id=excluded.run_id, model_version=excluded.model_version,
            ndcg10=excluded.ndcg10, updated_at=excluded.updated_at
        """, (key, run_id, version, float(ndcg10), time.time()))
        try:
            conn.execute("INSERT INTO registry_history (key, run_id, model_version, ndcg10, updated_at)"
                         " VALUES (?, ?, ?, ?, ?)", (key, run_id, version, float(ndcg10), time.time()))
        except Exception:
            pass


def rollback_prod() -> dict:
    """Point prod at the previous registry entry. Returns what happened."""
    import json as _json
    with get_conn(DB_TRAINING) as conn:
        cur = conn.execute("SELECT run_id, model_version, ndcg10 FROM model_registry WHERE key='prod'").fetchone()
        hist = conn.execute("SELECT run_id, model_version, ndcg10, updated_at FROM registry_history"
                            " WHERE key='prod' ORDER BY updated_at DESC LIMIT 10").fetchall()
    if not cur:
        return {"status": "noop", "reason": "no prod pointer set"}
    prev = next((dict(h) for h in hist if h["run_id"] != cur["run_id"]), None)
    if not prev:
        return {"status": "noop", "reason": "no earlier prod entry in history"}
    # Verify the old models still exist before pointing at them.
    missing = []
    try:
        with get_conn(DB_TRAINING) as conn:
            arts = conn.execute("SELECT model_path FROM model_artifacts WHERE run_id=?",
                                (prev["run_id"],)).fetchall()
        import os as _os
        missing = [a["model_path"] for a in arts if not _os.path.exists(a["model_path"])]
    except Exception:
        pass
    set_registry("prod", prev["run_id"], prev["model_version"], float(prev["ndcg10"] or 0))
    return {"status": "rolled-back", "from_run": cur["run_id"], "to_run": prev["run_id"],
            "to_version": prev["model_version"], "missing_files": missing}


def git_sha() -> str:
    try:
        import subprocess as _sp
        r = _sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                    timeout=10, cwd=str(MODELS_DIR.parent))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_snapshot() -> str:
    """Key constants that change model behavior. Recorded per run."""
    import json as _json
    import config as _c
    keys = ["REC_CANDIDATE_POOL", "ALS_FACTORS", "ALS_ITERATIONS", "ALS_REG", "ALS_ALPHA",
            "FEATURE_MF_FACTORS", "FEATURE_MF_EPOCHS", "FEATURE_MF_LR", "TWO_TOWER_DIM",
            "SEQ_MAX_LEN", "NEGATIVES_PER_POSITIVE", "IMPLICIT_ALPHA", "INTERACTION_HALFLIFE_DAYS",
            "HEARD_BOOST", "HEARD_WINDOW_DAYS", "BLENDER_LR", "BLENDER_EPOCHS"]
    return _json.dumps({k: getattr(_c, k, None) for k in keys}, default=str)


def _ensure_manifest_cols() -> None:
    with get_conn(DB_TRAINING) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(training_runs)").fetchall()]
        for col, ddl in (("git_sha", "TEXT DEFAULT ''"), ("config_json", "TEXT DEFAULT ''"),
                         ("frame_hash", "TEXT DEFAULT ''")):
            if col not in cols:
                conn.execute(f"ALTER TABLE training_runs ADD COLUMN {col} {ddl}")


def log_artifact(run_id: str, model_name: str, model_path: str, metrics_json: str = "") -> None:
    with get_conn(DB_TRAINING) as conn:
        conn.execute("""
        INSERT OR REPLACE INTO model_artifacts (run_id, model_name, model_path, metrics_json)
        VALUES (?, ?, ?, ?)
        """, (run_id, model_name, str(model_path), metrics_json[:4000]))


class TrainingEngine:
    """
    MAX orchestrator: every model trains on the SAME interaction frame,
    evals temporally, promotes only on NDCG gain (staging vs prod).

    run() stays back-compat: returns (svd, ncf, metrics).
    New models are available as attributes after run().
    """

    def __init__(self):
        self._svd: Optional[FunkSVD] = None
        self._ncf: Optional[NCFModel] = None
        self.als = None
        self.feature_mf = None
        self.retrievers = None
        self.sequence = None
        self.two_tower = None
        self.blender = None

    def run(
        self,
        use_synthetic: bool = False,
        n_synthetic_songs: int = 10000,
        n_synthetic_users: int = 100,
        synthetic_interactions_per_user: int = 200,
        version: Optional[str] = None,
        light: bool = False,
        force_ncf: bool = False,
        smoke: bool = False,
    ) -> tuple:
        """Full MAX training run.

        light=True  → hourly light-train (ALS + retrievers + two-tower + sequence).
        light=False → deep-train (all + FeatureMF + SVD + weekly NCF + blender).
        smoke=True  → 60-second wiring check: sampled frame, toy factors,
                       cheap models only, no registry writes, no checkpoints
                       clobbered (version forced to smoke-*).
        """
        import json as _json
        from data.interactions import build_interaction_frame, frame_to_triples
        from data.split import temporal_split, save_fingerprint
        from eval.metrics import evaluate_ranker, should_promote

        run_id = str(uuid.uuid4())[:8]
        version = version or f"v{int(time.time())}"
        if smoke:
            version = f"smoke-{version}"
        started_at = time.time()
        if not _acquire_lock():
            logger.warning("Training lock held — refusing overlap (stagger the VPS).")
            return self._svd, self._ncf, {"error": "locked", "version": version}

        logger.info("══════════════════════════════════════════")
        logger.info("  NeuroSync MAX Training Run [%s] light=%s smoke=%s", run_id, light, smoke)
        logger.info("  Version: %s", version)
        logger.info("══════════════════════════════════════════")

        try:
            # ── 1. Seed synthetic if needed ──
            if use_synthetic:
                logger.info("Generating synthetic training data...")
                gen = SyntheticDataGenerator(n_synthetic_users, n_synthetic_songs)
                songs = gen.generate_songs()
                interactions = gen.generate_interactions(songs, synthetic_interactions_per_user)
                gen.seed_db(songs, interactions)

            # ── 2. ONE interaction frame for every model ──
            frame = build_interaction_frame()
            if len(frame) < 20:
                logger.warning("Very few real interactions — adding synthetic boost.")
                gen = SyntheticDataGenerator(20, 2000)
                songs = gen.generate_songs()
                synt = gen.generate_interactions(songs, 100)
                gen.seed_db(songs, synt)
                frame = build_interaction_frame()

            train_frame, valid_frame, fp = temporal_split(frame, leave_last_n=5)
            if smoke:
                # Wiring check, not learning: 1500-pair sample, toy models.
                import random as _rnd
                _rnd.Random(0).shuffle(train_frame)
                train_frame = train_frame[:1500]
                valid_frame = valid_frame[:300]
                fp = {**fp, "smoke": True}
            save_fingerprint(run_id, fp)
            triples = frame_to_triples(train_frame)
            n_users = len({d["user_id"] for d in frame})
            n_songs = len({d["song_id"] for d in frame})
            logger.info("Frame: %d | train=%d valid=%d | users=%d songs=%d",
                        len(frame), len(train_frame), len(valid_frame), n_users, n_songs)

            # ── 3. Train cheap CPU models (always) ──
            from models.als import ImplicitALS
            from models.retrievers import RetrieverZoo
            from models.sequence import SequenceModel
            from models.two_tower import TwoTower

            if smoke:
                als = ImplicitALS(n_factors=8, n_iters=2, version=version)
            else:
                als = ImplicitALS(version=version)
            als_loss = als.fit(train_frame)
            als_path = als.save() if not smoke else "smoke:skipped-save"

            ret = RetrieverZoo(version=version)
            ret_stats = ret.fit(train_frame)
            ret_path = ret.save() if not smoke else "smoke:skipped-save"

            seq = SequenceModel(version=version)
            seq_loss = seq.fit(train_frame)
            seq_path = seq.save() if not smoke else "smoke:skipped-save"

            if smoke:
                tt = TwoTower(dim=8, version=version)
                tt_loss = tt.fit(train_frame, epochs=1)
            else:
                tt = TwoTower(version=version)
                tt_loss = tt.fit(train_frame)
            tt_path = tt.save() if not smoke else "smoke:skipped-save"

            # ── 4. Deep-only: FeatureMF + SVD (+ blender later) ──
            fmf_loss, fmf_path, svd_rmse, svd_path = 999.0, "", 999.0, ""
            fmf, svd = None, None
            if not light and not smoke:
                from models.feature_mf import FeatureMF
                fmf = FeatureMF(version=version)
                fmf_loss = fmf.fit(train_frame)
                fmf_path = str(fmf.save())
                svd = FunkSVD(version=version)
                svd_rmse = svd.fit(triples)
                svd_path = str(svd.save())
                self._svd = svd
            else:
                # Light path reuses latest SVD if present.
                try:
                    svd, _ = self.load_latest()
                    self._svd = svd
                except Exception:
                    pass

            # ── 5. NCF: weekly bonus only ──
            ncf_loss, ncf_path = 999.0, ""
            ncf = None
            prod = get_registry("prod")
            last_weekly = 0.0
            try:
                ncf_files = sorted(MODELS_DIR.glob("ncf_*.pkl"), key=lambda p: p.stat().st_mtime)
                if ncf_files:
                    last_weekly = ncf_files[-1].stat().st_mtime
            except Exception:
                pass
            weekly_due = (time.time() - last_weekly) > 7 * 86400
            if not smoke and (force_ncf or (not light and weekly_due)):
                logger.info("NCF weekly training (due=%s)...", weekly_due)
                ncf = NCFModel(version=version)
                ncf_loss = ncf.fit(triples)
                ncf_path = str(ncf.save())
                self._ncf = ncf
            else:
                logger.info("NCF skipped (weekly-only; last=%.1fd ago).", (time.time() - last_weekly) / 86400)
                try:
                    _, ncf = self.load_latest()
                    self._ncf = ncf
                except Exception:
                    pass

            # ── 6. Temporal eval on ONE harness ──
            als_for_eval = als
            fmf_for_eval = fmf
            tt_for_eval = tt

            def _predict(uid: int, cands: list[str]) -> list[str]:
                s1 = als_for_eval.predict_batch(uid, cands)
                s2 = tt_for_eval.predict_batch(uid, cands)
                s3 = ret.score_candidates(uid, cands)
                s4 = seq.predict_batch(uid, cands)
                agg = {s: 0.35 * s1.get(s, 0) + 0.25 * s2.get(s, 0) + 0.25 * s3.get(s, 0) + 0.15 * s4.get(s, 0)
                       for s in cands}
                return [s for s, _ in sorted(agg.items(), key=lambda x: x[1], reverse=True)]

            with get_conn(DB_CATALOG) as conn:
                try:
                    rows = conn.execute("SELECT song_id, genre_code FROM songs").fetchall()
                    cat_meta = {r["song_id"]: {"genre_code": r["genre_code"]} for r in rows}
                except Exception:
                    cat_meta = {}
            metrics = evaluate_ranker(_predict, valid_frame, k=10, catalog_meta=cat_meta)
            ndcg10 = float(metrics.get("ndcg@10", 0.0))

            # ── 7. Blender fit on valid judged pairs (deep only) ──
            if not light and not smoke:
                try:
                    from ranker.blender import Blender
                    judged = []
                    for d in valid_frame[:2000]:
                        if float(d.get("plan_reward", 0)) <= 0:
                            continue
                        uid = int(d["user_id"])
                        pos, neg = str(d["song_id"]), None
                        # Sample a negative from train.
                        for t in train_frame:
                            if int(t["user_id"]) == uid and str(t["song_id"]) != pos:
                                neg = str(t["song_id"])
                                break
                        if not neg:
                            continue
                        cands = [pos, neg]
                        feats_pos = {"als": als.predict_batch(uid, cands).get(pos, 0),
                                     "two_tower": tt.predict_batch(uid, cands).get(pos, 0),
                                     "retriever": ret.score_candidates(uid, cands).get(pos, 0),
                                     "sequence": seq.predict_batch(uid, cands).get(pos, 0),
                                     "svd": 0.0, "ncf": 0.0, "feature_mf": 0.0,
                                     "content": 0.0, "bandit": 0.0, "recency": 0.0,
                                     "freshness": 0.0, "repeat_pen": 0.0}
                        feats_neg = {"als": als.predict_batch(uid, cands).get(neg, 0),
                                     "two_tower": tt.predict_batch(uid, cands).get(neg, 0),
                                     "retriever": ret.score_candidates(uid, cands).get(neg, 0),
                                     "sequence": seq.predict_batch(uid, cands).get(neg, 0),
                                     "svd": 0.0, "ncf": 0.0, "feature_mf": 0.0,
                                     "content": 0.0, "bandit": 0.0, "recency": 0.0,
                                     "freshness": 0.0, "repeat_pen": 0.0}
                        judged.append({"user_id": uid, "pos_scores": feats_pos, "neg_scores": feats_neg})
                    blender = Blender(version=version)
                    blender.fit(judged)
                    self.blender = blender
                except Exception as e:
                    logger.warning("Blender fit skipped: %s", e)

            # ── 8. Log + registry (staging vs prod) + run manifest ──
            finished_at = time.time()
            _ensure_manifest_cols()
            _sha, _cfg = git_sha(), config_snapshot()
            with get_conn(DB_TRAINING) as conn:
                conn.execute("""
                INSERT INTO training_runs
                    (run_id, started_at, finished_at, model_type, n_samples,
                     n_users, n_songs, train_loss, val_loss, rmse, model_path, notes,
                     git_sha, config_json, frame_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (run_id, started_at, finished_at,
                      "smoke" if smoke else ("max-all" if not light else "max-light"),
                      len(train_frame), n_users, n_songs, als_loss, tt_loss, svd_rmse,
                      str(als_path), f"metrics={metrics} rss_mb={_peak_rss_mb():.0f}",
                      _sha, _cfg, fp.get("frame_hash", "")))
            for name, p in (("als", str(als_path)), ("retrievers", str(ret_path)),
                            ("sequence", str(seq_path)), ("two_tower", str(tt_path)),
                            ("feature_mf", fmf_path), ("svd", svd_path), ("ncf", ncf_path)):
                if p and not p.startswith("smoke:"):
                    log_artifact(run_id, name, p, _json.dumps(metrics)[:2000])

            promote, reason = False, "smoke: no promotion"
            if not smoke:
                prod_metrics = None
                if prod:
                    try:
                        with get_conn(DB_TRAINING) as conn:
                            row = conn.execute("SELECT metrics_json FROM model_artifacts WHERE run_id=? AND model_name='als'",
                                               (prod["run_id"],)).fetchone()
                        prod_metrics = _json.loads(row["metrics_json"]) if row and row["metrics_json"] else None
                    except Exception:
                        prod_metrics = None
                from eval.metrics import should_promote as _sp
                promote, reason = _sp(metrics, prod_metrics)
                set_registry("staging", run_id, version, ndcg10)
                if promote:
                    set_registry("prod", run_id, version, ndcg10)
            logger.info("Eval: %s | promote=%s (%s) | peak RSS=%.0f MB | sha=%s",
                        metrics, promote, reason, _peak_rss_mb(), _sha)

            self.als, self.retrievers, self.sequence, self.two_tower = als, ret, seq, tt
            self.feature_mf = fmf
            elapsed = finished_at - started_at
            full_metrics = {**metrics, "version": version, "elapsed_s": round(elapsed, 2),
                            "als_loss": round(float(als_loss), 4), "promoted": promote,
                            "smoke": smoke, "git_sha": _sha}
            return self._svd, self._ncf, full_metrics
        finally:
            _release_lock()

    def load_latest(self) -> tuple:
        """Load the most recently saved model pair (plus MAX models as attrs)."""
        from models.als import ImplicitALS
        from models.retrievers import RetrieverZoo
        from models.sequence import SequenceModel
        from models.two_tower import TwoTower
        from models.feature_mf import FeatureMF

        def _latest(pat: str):
            files = sorted(MODELS_DIR.glob(pat), key=lambda p: p.stat().st_mtime)
            return files[-1] if files else None

        svd = FunkSVD.load(_latest("svd_*.pkl")) if _latest("svd_*.pkl") else None
        ncf = None
        try:
            ncf = NCFModel.load(_latest("ncf_*.pkl")) if _latest("ncf_*.pkl") else None
        except Exception as e:
            logger.warning("NCF load skipped: %s", e)
        for attr, pat, cls in (("als", "als_*.pkl", ImplicitALS),
                               ("retrievers", "retrievers_*.pkl", RetrieverZoo),
                               ("sequence", "sequence_*.pkl", SequenceModel),
                               ("two_tower", "two_tower_*.pkl", TwoTower),
                               ("feature_mf", "feature_mf_*.pkl", FeatureMF)):
            try:
                p = _latest(pat)
                setattr(self, attr, cls.load(p) if p else None)
                if p:
                    logger.info("Loaded %s: %s", attr, p.name)
            except Exception as e:
                logger.warning("Load %s skipped: %s", attr, e)
                setattr(self, attr, None)
        self._svd, self._ncf = svd, ncf
        try:
            from ranker.blender import Blender
            self.blender = Blender()
        except Exception:
            self.blender = None
        return svd, ncf

    def load_max_bundle(self) -> dict:
        """Convenience: load everything, return name → model dict."""
        self.load_latest()
        return {"svd": self._svd, "ncf": self._ncf, "als": self.als,
                "feature_mf": self.feature_mf, "retrievers": self.retrievers,
                "sequence": self.sequence, "two_tower": self.two_tower,
                "blender": self.blender}