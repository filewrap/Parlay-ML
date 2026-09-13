"""
models/als.py — Implicit ALS (native numpy, CPU-first).

The `implicit` library's core idea, reimplemented natively
(never a thin wrapper):

    confidence C = 1 + alpha * R   (R = plan_reward, >= 0 part)
    preference P = 1 if R > 0 else 0

Alternate least squares:
    Xu = (Yᵀ C_u Y + λI)⁻¹ Yᵀ C_u p_u   via Cholesky solves
    Yi = (Xᵀ C_i X + λI)⁻¹ Xᵀ C_i p_i

15–20 iterations. Sparse, vectorized per-user/item with
precomputed YᵀY / XᵀX gram trick (Hu–Koren–Volinsky 2008).

128 factors; 50k songs × 128 float32 ≈ 25 MB.
500k interactions ≈ 2–4 min CPU.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import ALS_ALPHA, ALS_FACTORS, ALS_ITERATIONS, ALS_REG, MODELS_DIR

logger = logging.getLogger("parlay.als")


class ImplicitALS:
    """Native implicit-feedback ALS. CPU-only, Cholesky solves."""

    def __init__(self, n_factors: int = ALS_FACTORS, n_iters: int = ALS_ITERATIONS,
                 reg: float = ALS_REG, alpha: float = ALS_ALPHA,
                 version: str = "v0", use_cg: bool = False):
        self.k = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.alpha = alpha
        self.version = version
        self.user_idx: dict[int, int] = {}
        self.song_idx: dict[str, int] = {}
        self.idx_song: dict[int, str] = {}
        self.X: Optional[np.ndarray] = None  # users × k
        self.Y: Optional[np.ndarray] = None  # songs × k
        self.trained_at: Optional[float] = None
        self.train_loss: float = 999.0

    # ── data (numpy + stdlib only, no scipy) ──
    def _build_matrix(self, frame: list[dict]) -> tuple[dict[int, tuple[list[int], list[float]]], dict[int, tuple[list[int], list[float]]], int, int]:
        users = sorted({int(d["user_id"]) for d in frame})
        songs = sorted({str(d["song_id"]) for d in frame})
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.song_idx = {s: i for i, s in enumerate(songs)}
        self.idx_song = {i: s for s, i in self.song_idx.items()}
        # Collapse duplicates: keep max confidence.
        best: dict[tuple[int, int], float] = {}
        for d in frame:
            r = float(d.get("plan_reward", d.get("reward", 0.0)))
            if r <= 0:
                continue
            c = float(d.get("confidence", 1.0 + self.alpha * r))
            u = self.user_idx[int(d["user_id"])]
            i = self.song_idx[str(d["song_id"])]
            if (u, i) not in best or c > best[(u, i)]:
                best[(u, i)] = c
        user_items: dict[int, tuple[list[int], list[float]]] = {u: ([], []) for u in range(len(users))}
        item_users: dict[int, tuple[list[int], list[float]]] = {i: ([], []) for i in range(len(songs))}
        triples: list[tuple[int, int, float]] = []
        for (u, i), c in best.items():
            user_items[u][0].append(i)
            user_items[u][1].append(c)
            item_users[i][0].append(u)
            item_users[i][1].append(c)
            triples.append((u, i, c))
        self._triples = triples
        return user_items, item_users, len(users), len(songs)

    # ── training ──
    def fit(self, frame: list[dict]) -> float:
        if len(frame) < 10:
            logger.warning("ALS: too few interactions.")
            return 999.0
        user_items, item_users, n_u, n_s = self._build_matrix(frame)
        nnz = len(getattr(self, "_triples", []))
        if n_u == 0 or n_s == 0 or nnz == 0:
            return 999.0
        rng = np.random.default_rng(42)
        self.X = (rng.normal(0, 0.01, (n_u, self.k))).astype(np.float32)
        self.Y = (rng.normal(0, 0.01, (n_s, self.k))).astype(np.float32)
        eye = np.eye(self.k, dtype=np.float64) * self.reg

        logger.info("ALS training: %d users × %d songs, %d positives, k=%d iters=%d",
                    n_u, n_s, nnz, self.k, self.n_iters)
        for it in range(self.n_iters):
            # Update users: Xu = (YᵀCuY + λI)⁻¹ YᵀCu pu
            YtY = (self.Y.astype(np.float64).T @ self.Y.astype(np.float64))
            for u in range(n_u):
                idx, conf_list = user_items[u]
                if not idx:
                    self.X[u] = 0
                    continue
                conf = np.array(conf_list, dtype=np.float64)
                Y_i = self.Y[np.array(idx)].astype(np.float64)
                # YᵀCuY = YᵀY + Y_iᵀ (Cu - I) Y_i
                A = YtY + (Y_i * (conf - 1.0)[:, None]).T @ Y_i + eye
                b = (Y_i * conf[:, None]).sum(axis=0)
                try:
                    L = np.linalg.cholesky(A)
                    y = np.linalg.solve(L, b)
                    self.X[u] = np.linalg.solve(L.T, y).astype(np.float32)
                except np.linalg.LinAlgError:
                    self.X[u] = (np.linalg.solve(A, b)).astype(np.float32)
            # Update songs symmetrically.
            XtX = (self.X.astype(np.float64).T @ self.X.astype(np.float64))
            for i in range(n_s):
                idx, conf_list = item_users[i]
                if not idx:
                    self.Y[i] = 0
                    continue
                conf = np.array(conf_list, dtype=np.float64)
                X_u = self.X[np.array(idx)].astype(np.float64)
                A = XtX + (X_u * (conf - 1.0)[:, None]).T @ X_u + eye
                b = (X_u * conf[:, None]).sum(axis=0)
                try:
                    L = np.linalg.cholesky(A)
                    y = np.linalg.solve(L, b)
                    self.Y[i] = np.linalg.solve(L.T, y).astype(np.float32)
                except np.linalg.LinAlgError:
                    self.Y[i] = (np.linalg.solve(A, b)).astype(np.float32)
            if (it + 1) % 5 == 0 or it == 0:
                loss = self._loss()
                logger.info("  ALS iter %2d/%d loss=%.4f", it + 1, self.n_iters, loss)
        self.train_loss = self._loss()
        self.trained_at = time.time()
        logger.info("✅ ALS done. loss=%.4f", self.train_loss)
        return float(self.train_loss)

    def _loss(self) -> float:
        triples = getattr(self, "_triples", [])
        if not triples or self.X is None or self.Y is None:
            return 999.0
        err = 0.0
        for u, i, c in triples:
            err += float(c) * (1.0 - float(self.X[u] @ self.Y[i])) ** 2
        err += self.reg * (float((self.X ** 2).sum()) + float((self.Y ** 2).sum()))
        return err / max(len(triples), 1)

    # ── serving ──
    def predict(self, user_id: int, song_id: str) -> float:
        if self.X is None or self.Y is None:
            return 0.0
        u = self.user_idx.get(int(user_id))
        i = self.song_idx.get(str(song_id))
        if u is None or i is None:
            return 0.0
        return float(self.X[u] @ self.Y[i])

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        if self.X is None or self.Y is None:
            return {s: 0.0 for s in song_ids}
        u = self.user_idx.get(int(user_id))
        if u is None:
            return {s: 0.0 for s in song_ids}
        xu = self.X[u]
        out = {}
        for s in song_ids:
            i = self.song_idx.get(str(s))
            out[s] = float(xu @ self.Y[i]) if i is not None else 0.0
        return out

    def recommend(self, user_id: int, candidates: list[str], top_k: int = 10) -> list[str]:
        scores = self.predict_batch(user_id, candidates)
        return [s for s, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # ── persistence ──
    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"als_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 ALS saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ImplicitALS":
        with open(path, "rb") as f:
            m = pickle.load(f)
        logger.info("📂 ALS loaded from %s", path)
        return m
