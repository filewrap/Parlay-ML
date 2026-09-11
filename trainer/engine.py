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
            VALUES (?, ?, ?, ?, 'synthetic')
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


# ─── Training Engine ─────────────────────────────────────────

class TrainingEngine:
    """
    Orchestrates the full train-eval-save cycle for both models.
    """

    def __init__(self):
        self._svd: Optional[FunkSVD] = None
        self._ncf: Optional[NCFModel] = None

    def run(
        self,
        use_synthetic: bool = False,
        n_synthetic_songs: int = 10000,
        n_synthetic_users: int = 100,
        synthetic_interactions_per_user: int = 200,
        version: Optional[str] = None,
    ) -> tuple:
        """
        Full training run. Returns (FunkSVD, NCFModel, metrics).
        """
        run_id = str(uuid.uuid4())[:8]
        version = version or f"v{int(time.time())}"
        started_at = time.time()

        logger.info(f"══════════════════════════════════════════")
        logger.info(f"  NeuroSync Training Run [{run_id}]")
        logger.info(f"  Version: {version}")
        logger.info(f"══════════════════════════════════════════")

        # ── 1. Seed synthetic data if needed ──────────────────
        if use_synthetic:
            logger.info("Generating synthetic training data...")
            gen = SyntheticDataGenerator(n_synthetic_users, n_synthetic_songs)
            songs = gen.generate_songs()
            interactions = gen.generate_interactions(songs, synthetic_interactions_per_user)
            gen.seed_db(songs, interactions)

        # ── 2. Load real interactions ──────────────────────────
        all_interactions = load_interactions()
        if len(all_interactions) < 20:
            logger.warning("Very few real interactions — adding synthetic boost.")
            gen = SyntheticDataGenerator(20, 2000)
            songs = gen.generate_songs()
            synt = gen.generate_interactions(songs, 100)
            gen.seed_db(songs, synt)
            all_interactions = load_interactions()

        # ── 3. Train/Test split (80/20) ───────────────────────
        np.random.seed(42)
        np.random.shuffle(all_interactions)
        split = int(len(all_interactions) * 0.8)
        train_data = all_interactions[:split]
        test_data  = all_interactions[split:]

        n_users = len(set(u for u, _, _ in all_interactions))
        n_songs = len(set(s for _, s, _ in all_interactions))
        logger.info(f"Data: {len(all_interactions)} interactions | {n_users} users | {n_songs} songs")
        logger.info(f"Train: {len(train_data)} | Test: {len(test_data)}")

        # ── 4. Train SVD ──────────────────────────────────────
        logger.info("\n[1/2] Training FunkSVD...")
        svd = FunkSVD(version=version)
        svd_rmse = svd.fit(train_data)
        svd_path = svd.save()

        # ── 5. Evaluate SVD ───────────────────────────────────
        evaluator = ModelEvaluator(svd)
        metrics = evaluator.evaluate(test_data, k=10)
        logger.info(f"SVD Metrics: {metrics}")

        # ── 6. Train NCF ──────────────────────────────────────
        logger.info("\n[2/2] Training NCF...")
        ncf = NCFModel(version=version)
        ncf_loss = ncf.fit(train_data)
        ncf_path = ncf.save()

        # ── 7. Log to DB ──────────────────────────────────────
        finished_at = time.time()
        with get_conn(DB_TRAINING) as conn:
            conn.execute("""
            INSERT INTO training_runs
                (run_id, started_at, finished_at, model_type, n_samples,
                 n_users, n_songs, train_loss, val_loss, rmse, model_path, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, started_at, finished_at, "svd+ncf",
                len(train_data), n_users, n_songs,
                svd_rmse, ncf_loss, metrics.get("rmse", 999.0),
                str(svd_path),
                f"metrics={metrics}",
            ))

        elapsed = finished_at - started_at
        logger.info(f"\n✅  Training done in {elapsed:.1f}s")
        logger.info(f"    SVD RMSE:   {svd_rmse:.4f}")
        logger.info(f"    NCF Loss:   {ncf_loss:.4f}")
        logger.info(f"    Eval Metrics: {metrics}")

        self._svd = svd
        self._ncf = ncf

        return svd, ncf, {**metrics, "version": version, "elapsed_s": round(elapsed, 2)}

    def load_latest(self) -> tuple:
        """Load the most recently saved model pair."""
        svd_files = sorted(MODELS_DIR.glob("svd_*.pkl"), key=lambda p: p.stat().st_mtime)
        ncf_files = sorted(MODELS_DIR.glob("ncf_*.pkl"), key=lambda p: p.stat().st_mtime)

        svd = FunkSVD.load(svd_files[-1]) if svd_files else None
        ncf = NCFModel.load(ncf_files[-1]) if ncf_files else None

        if svd:
            logger.info(f"Loaded SVD: {svd_files[-1].name}")
        if ncf:
            logger.info(f"Loaded NCF: {ncf_files[-1].name}")

        return svd, ncf