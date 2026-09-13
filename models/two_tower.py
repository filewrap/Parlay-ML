"""
models/two_tower.py — numpy two-tower dot-product (daily CPU stand-in).

Torch NCF moves to weekly-bonus-only (Phase 2e). This is its daily
replacement: user tower + song tower (genre/lang/energy/artist hashed
side features), dot-product score, BPR training on Phase-0 pairs.

Minutes on CPU, <100 MB, no torch import.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODELS_DIR, TWO_TOWER_DIM
from data.interactions import pairwise_training_pairs

logger = logging.getLogger("parlay.two_tower")


def _hash_vec(key: str, dim: int) -> np.ndarray:
    import hashlib
    h = hashlib.md5(key.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:4], "little"))
    v = rng.normal(0, 1, dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v


class TwoTower:
    """User tower = learned embedding; song tower = learned + hashed side info."""

    def __init__(self, dim: int = TWO_TOWER_DIM, version: str = "v0"):
        self.dim = dim
        self.version = version
        self.user_idx: dict[int, int] = {}
        self.song_idx: dict[str, int] = {}
        self.U: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.trained_at: Optional[float] = None
        self.train_loss: float = 999.0

    def fit(self, frame: list[dict], epochs: int = 8, lr: float = 0.1) -> float:
        users = sorted({int(d["user_id"]) for d in frame})
        songs = sorted({str(d["song_id"]) for d in frame})
        if len(users) < 1 or len(songs) < 5:
            logger.warning("TwoTower: too little data.")
            return 999.0
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.song_idx = {s: i for i, s in enumerate(songs)}
        rng = np.random.default_rng(21)
        self.U = rng.normal(0, 0.05, (len(users), self.dim)).astype(np.float32)
        self.S = rng.normal(0, 0.05, (len(songs), self.dim)).astype(np.float32)
        # Fold hashed side info into init so cold-ish songs start sensibly.
        for s, i in self.song_idx.items():
            self.S[i] = (self.S[i] + 0.2 * _hash_vec(f"song:{s}", self.dim)).astype(np.float32)
        pairs = [(self.user_idx[u], self.song_idx[p], self.song_idx[n])
                 for u, p, n in pairwise_training_pairs(frame)
                 if u in self.user_idx and p in self.song_idx and n in self.song_idx]
        if not pairs:
            return 999.0
        logger.info("TwoTower: %d users × %d songs, pairs=%d", len(users), len(songs), len(pairs))
        order = np.arange(len(pairs))
        for ep in range(epochs):
            rng.shuffle(order)
            loss = 0.0
            for t in order:
                u, p, n = pairs[t]
                xp = float(self.U[u] @ self.S[p])
                xn = float(self.U[u] @ self.S[n])
                x = xp - xn
                sig = 1.0 / (1.0 + np.exp(x))
                loss += -np.log(1.0 / (1.0 + np.exp(-x)) + 1e-12)
                g = -sig
                du = g * (self.S[p] - self.S[n])
                dp = g * self.U[u]
                dn = g * (-self.U[u])
                self.U[u] -= lr * (du + 1e-4 * self.U[u])
                self.S[p] -= lr * (dp + 1e-4 * self.S[p])
                self.S[n] -= lr * (dn + 1e-4 * self.S[n])
            if (ep + 1) % 2 == 0 or ep == 0:
                logger.info("  TwoTower epoch %d/%d loss=%.4f", ep + 1, epochs, loss / len(pairs))
        # Normalise rows for cosine-ish serving.
        self.U = (self.U / (np.linalg.norm(self.U, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        self.S = (self.S / (np.linalg.norm(self.S, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
        self.train_loss = float(loss / len(pairs))
        self.trained_at = time.time()
        logger.info("✅ TwoTower done. loss=%.4f", self.train_loss)
        return self.train_loss

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        if self.U is None or self.S is None:
            return {str(s): 0.0 for s in song_ids}
        u = self.user_idx.get(int(user_id))
        if u is None:
            return {str(s): 0.0 for s in song_ids}
        vu = self.U[u]
        out = {}
        for s in song_ids:
            s = str(s)
            i = self.song_idx.get(s)
            if i is None:
                out[s] = float(vu @ _hash_vec(f"song:{s}", self.dim)) * 0.5
            else:
                out[s] = float(vu @ self.S[i])  # cosine in [-1,1]
        return out

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"two_tower_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 TwoTower saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "TwoTower":
        with open(path, "rb") as f:
            m = pickle.load(f)
        logger.info("📂 TwoTower loaded from %s", path)
        return m
