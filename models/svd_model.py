"""
models/svd_model.py — Funk SVD (Stochastic Gradient Descent Matrix Factorization)

Why SVD for music recommendation?
  The user-song interaction matrix R (users × songs) is enormous and sparse.
  SVD factorises R ≈ P × Q^T where:
    P ∈ R^(n_users × k)   — user latent factors (taste embeddings)
    Q ∈ R^(n_songs × k)   — song latent factors (style embeddings)

  Predicted rating: r̂_ui = μ + b_u + b_i + p_u · q_i
    μ    = global mean rating
    b_u  = user bias (how much this user rates above/below average)
    b_i  = song bias (how popular this song is globally)
    p_u  = user embedding
    q_i  = song embedding

  Update rule (SGD):
    e_ui = r_ui - r̂_ui                       (error)
    b_u  += lr * (e_ui - reg * b_u)
    b_i  += lr * (e_ui - reg * b_i)
    p_u  += lr * (e_ui * q_i - reg * p_u)
    q_i  += lr * (e_ui * p_u - reg * q_i)

  Converges to optimal k-rank approximation that minimises MSE on observed entries.
  k (SVD_FACTORS=128) controls expressiveness vs. overfitting.

This is the same core mechanism used by the Netflix Prize winning solution.
"""

import numpy as np
import pickle
import time
import math
import logging
from pathlib import Path
from typing import Optional

from config import (
    SVD_FACTORS, SVD_EPOCHS, SVD_LR, SVD_REG, MODELS_DIR,
    REWARD_LIKE, REWARD_DISLIKE, REWARD_FULL_PLAY, REWARD_PARTIAL_PLAY,
)
from core.database import get_conn, DB_HISTORY, DB_FEEDBACK, DB_CATALOG, ndarray_to_blob, blob_to_ndarray, DB_EMBEDDINGS
import time

logger = logging.getLogger("parlay.svd")


class FunkSVD:
    """
    Funk SVD with biases, L2 regularisation, and online update capability.
    Can be incrementally updated from new feedback without full retraining.
    """

    def __init__(
        self,
        n_factors: int = SVD_FACTORS,
        n_epochs: int = SVD_EPOCHS,
        lr: float = SVD_LR,
        reg: float = SVD_REG,
        version: str = "v0",
    ):
        self.k = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.version = version

        # Mappings
        self.user_idx:  dict[int, int]  = {}
        self.song_idx:  dict[str, int]  = {}
        self.idx_song:  dict[int, str]  = {}

        # Parameters (initialised after first fit)
        self.P:    Optional[np.ndarray] = None   # (n_users, k)
        self.Q:    Optional[np.ndarray] = None   # (n_songs, k)
        self.b_u:  Optional[np.ndarray] = None   # (n_users,)
        self.b_i:  Optional[np.ndarray] = None   # (n_songs,)
        self.mu:   float = 0.0                   # global mean

        self.trained_at: Optional[float] = None
        self.train_rmse: float = 999.0

    def _build_indices(self, interactions: list[tuple]) -> None:
        users = sorted(set(u for u, _, _ in interactions))
        songs = sorted(set(s for _, s, _ in interactions))
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.song_idx = {s: i for i, s in enumerate(songs)}
        self.idx_song = {i: s for s, i in self.song_idx.items()}

    def _init_params(self, n_users: int, n_songs: int) -> None:
        rng = np.random.default_rng(42)
        # Small random init: values ~ N(0, 0.1)
        self.P   = rng.normal(0, 0.1, (n_users, self.k)).astype(np.float32)
        self.Q   = rng.normal(0, 0.1, (n_songs, self.k)).astype(np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_songs, dtype=np.float32)

    def fit(self, interactions: list[tuple[int, str, float]]) -> float:
        """
        Train on interactions: [(user_id, song_id, rating), ...]
        Rating is in [−1, +1] (our reward space).

        Returns final RMSE.
        """
        if len(interactions) < 10:
            logger.warning("SVD: too few interactions to train meaningfully.")
            return 999.0

        self._build_indices(interactions)
        n_u = len(self.user_idx)
        n_s = len(self.song_idx)
        self._init_params(n_u, n_s)

        ratings = np.array([r for _, _, r in interactions], dtype=np.float32)
        self.mu = float(ratings.mean())

        # Convert to index-triples
        data = [
            (self.user_idx[u], self.song_idx[s], r)
            for u, s, r in interactions
            if u in self.user_idx and s in self.song_idx
        ]
        rng = np.random.default_rng(7)

        logger.info(f"SVD training: {n_u} users × {n_s} songs, {len(data)} interactions")

        for epoch in range(self.n_epochs):
            rng.shuffle(data := list(data))   # in-place shuffle each epoch
            sq_err = 0.0
            for uid, iid, r in data:
                # Predicted rating
                r_hat = (
                    self.mu
                    + self.b_u[uid]
                    + self.b_i[iid]
                    + float(self.P[uid] @ self.Q[iid])
                )
                e = r - r_hat
                sq_err += e * e

                # SGD updates
                self.b_u[uid] += self.lr * (e - self.reg * self.b_u[uid])
                self.b_i[iid] += self.lr * (e - self.reg * self.b_i[iid])

                p_old = self.P[uid].copy()
                self.P[uid] += self.lr * (e * self.Q[iid] - self.reg * self.P[uid])
                self.Q[iid] += self.lr * (e * p_old      - self.reg * self.Q[iid])

            rmse = math.sqrt(sq_err / max(len(data), 1))
            if epoch % 5 == 0 or epoch == self.n_epochs - 1:
                logger.info(f"  Epoch {epoch+1:3d}/{self.n_epochs} | RMSE: {rmse:.4f}")

        self.train_rmse = rmse
        self.trained_at = time.time()
        logger.info(f"✅  SVD training done. Final RMSE: {rmse:.4f}")
        self._save_embeddings()
        return rmse

    def predict(self, user_id: int, song_id: str) -> float:
        """Predict affinity score for (user, song). Returns float in ~[-1.5, 1.5]."""
        if self.P is None:
            return 0.0
        uid = self.user_idx.get(user_id)
        iid = self.song_idx.get(song_id)
        if uid is None or iid is None:
            # Cold-start: return mean + item bias if known
            bias = self.b_i[iid] if iid is not None else 0.0
            return float(self.mu + bias)
        return float(
            self.mu
            + self.b_u[uid]
            + self.b_i[iid]
            + self.P[uid] @ self.Q[iid]
        )

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        """Vectorised batch prediction for a list of songs."""
        if self.P is None:
            return {s: 0.0 for s in song_ids}

        uid = self.user_idx.get(user_id)
        scores = {}
        for sid in song_ids:
            iid = self.song_idx.get(sid)
            if uid is None or iid is None:
                bias = (self.b_i[iid] if iid is not None else 0.0)
                scores[sid] = float(self.mu + bias)
            else:
                scores[sid] = float(
                    self.mu + self.b_u[uid] + self.b_i[iid]
                    + self.P[uid] @ self.Q[iid]
                )
        return scores

    def online_update(self, user_id: int, song_id: str, reward: float) -> None:
        """
        Single-sample SGD update when new feedback arrives.
        Lets the model adapt in real-time without full retraining.
        Adds new user/song dimensions if unseen (cold-start expansion).
        """
        if self.P is None:
            return

        # Expand matrices if new user/song
        if user_id not in self.user_idx:
            new_uid = len(self.user_idx)
            self.user_idx[user_id] = new_uid
            self.P   = np.vstack([self.P,   np.zeros((1, self.k), dtype=np.float32)])
            self.b_u = np.append(self.b_u, 0.0)

        if song_id not in self.song_idx:
            new_iid = len(self.song_idx)
            self.song_idx[song_id] = new_iid
            self.idx_song[new_iid] = song_id
            self.Q   = np.vstack([self.Q,   np.zeros((1, self.k), dtype=np.float32)])
            self.b_i = np.append(self.b_i, 0.0)

        uid = self.user_idx[user_id]
        iid = self.song_idx[song_id]

        r_hat = (
            self.mu + self.b_u[uid] + self.b_i[iid]
            + float(self.P[uid] @ self.Q[iid])
        )
        e = reward - r_hat

        # Faster lr for online updates (momentum-style)
        lr_online = self.lr * 2.0
        self.b_u[uid] += lr_online * (e - self.reg * self.b_u[uid])
        self.b_i[iid] += lr_online * (e - self.reg * self.b_i[iid])

        p_old = self.P[uid].copy()
        self.P[uid] += lr_online * (e * self.Q[iid] - self.reg * self.P[uid])
        self.Q[iid] += lr_online * (e * p_old       - self.reg * self.Q[iid])

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        uid = self.user_idx.get(user_id)
        if uid is None or self.P is None:
            return None
        return self.P[uid]

    def get_song_embedding(self, song_id: str) -> Optional[np.ndarray]:
        iid = self.song_idx.get(song_id)
        if iid is None or self.Q is None:
            return None
        return self.Q[iid]

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"svd_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"💾  SVD model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "FunkSVD":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"📂  SVD model loaded from {path}")
        return model

    def _save_embeddings(self) -> None:
        """Persist user/song embeddings to DB for retrieval by other modules."""
        if self.Q is None:
            return
        now = time.time()
        with get_conn(DB_EMBEDDINGS) as conn:
            rows = [
                (sid, ndarray_to_blob(self.Q[iid]), self.k, self.version, now)
                for sid, iid in self.song_idx.items()
            ]
            conn.executemany("""
            INSERT OR REPLACE INTO song_embeddings
                (song_id, vector, dim, model_version, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """, rows)
        logger.info(f"  Embeddings: {len(rows)} songs persisted.")


# ─── Data loader ──────────────────────────────────────────

def load_interactions() -> list[tuple[int, str, float]]:
    """
    Build interaction matrix from listen history + feedback.

    Rating mapping:
        explicit like (+1.0)    → REWARD_LIKE
        explicit dislike (−1.0) → REWARD_DISLIKE
        full play (>80%)        → REWARD_FULL_PLAY
        partial (30–80%)        → REWARD_PARTIAL_PLAY
        skip (<30%)             → REWARD_SKIP_FAST (implicit negative)
    """
    from config import REWARD_SKIP_FAST

    interactions: dict[tuple[int, str], float] = {}

    # --- Explicit feedback (highest priority) ---
    with get_conn(DB_FEEDBACK) as conn:
        rows = conn.execute(
            "SELECT user_id, song_id, signal FROM feedback"
        ).fetchall()
    for row in rows:
        key = (row["user_id"], row["song_id"])
        # Explicit signals dominate; average if multiple
        if key in interactions:
            interactions[key] = (interactions[key] + row["signal"]) / 2
        else:
            interactions[key] = float(row["signal"])

    # --- Implicit from listen history ---
    with get_conn(DB_HISTORY) as conn:
        rows = conn.execute("""
            SELECT user_id, song_id, completion_pct
            FROM listens
        """).fetchall()
    for row in rows:
        key = (row["user_id"], row["song_id"])
        if key in interactions:
            continue  # explicit signal already set, skip implicit
        pct = row["completion_pct"] or 0.0
        if pct >= 0.80:
            interactions[key] = REWARD_FULL_PLAY
        elif pct >= 0.30:
            interactions[key] = REWARD_PARTIAL_PLAY
        else:
            interactions[key] = REWARD_SKIP_FAST

    result = [(u, s, r) for (u, s), r in interactions.items()]
    logger.info(f"  Loaded {len(result)} interactions for SVD training.")
    return result
