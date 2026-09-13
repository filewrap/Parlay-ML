"""
models/sequence.py — next-song continuity (Markov + SASRec-lite).

The "4AM in Karachi at 4am" pattern is sequential, not a taste vector.
Companion continuity lives here.

- Markov: P(next | current) from adjacent-pair counts in listen sequences.
- SASRec-lite: 1-layer causal self-attention over last-20 listens (numpy).
  Tiny (d=32, 1 head) — CPU seconds, not minutes.

API: fit(frame) → predict_batch(user_id, song_ids) in [-1, 1].
"""

import logging
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODELS_DIR, SEQ_MAX_LEN

logger = logging.getLogger("parlay.sequence")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


class SequenceModel:
    """Markov + 1-layer causal attention, numpy-only."""

    def __init__(self, dim: int = 32, version: str = "v0"):
        self.dim = dim
        self.version = version
        self.song_idx: dict[str, int] = {}
        self.idx_song: dict[int, str] = {}
        self.markov: dict[str, Counter] = {}
        self.user_seqs: dict[int, list[str]] = {}
        self.E: Optional[np.ndarray] = None  # item embeddings (n_songs, dim)
        self.Wq: Optional[np.ndarray] = None
        self.Wk: Optional[np.ndarray] = None
        self.Wv: Optional[np.ndarray] = None
        self.trained_at: Optional[float] = None
        self.train_loss: float = 999.0

    # ── fit ──
    def fit(self, frame: list[dict]) -> float:
        t0 = time.time()
        by_user: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for d in frame:
            if float(d.get("plan_reward", 0)) > 0:
                by_user[int(d["user_id"])].append((float(d["timestamp"]), str(d["song_id"])))
        seqs = {u: [s for _, s in sorted(v)][-60:] for u, v in by_user.items()}
        self.user_seqs = seqs
        songs = sorted({s for seq in seqs.values() for s in seq})
        if len(songs) < 5:
            logger.warning("Sequence: too few songs (%d).", len(songs))
            return 999.0
        self.song_idx = {s: i for i, s in enumerate(songs)}
        self.idx_song = {i: s for s, i in self.song_idx.items()}

        # Markov pair counts.
        mk: dict[str, Counter] = defaultdict(Counter)
        for seq in seqs.values():
            for a, b in zip(seq, seq[1:]):
                mk[a][b] += 1
        self.markov = dict(mk)

        # SASRec-lite: learn item embeddings to predict next item.
        n = len(songs)
        rng = np.random.default_rng(9)
        E = rng.normal(0, 0.1, (n, self.dim)).astype(np.float32)
        Wq = rng.normal(0, 0.1, (self.dim, self.dim)).astype(np.float32)
        Wk = rng.normal(0, 0.1, (self.dim, self.dim)).astype(np.float32)
        Wv = rng.normal(0, 0.1, (self.dim, self.dim)).astype(np.float32)
        lr = 0.05
        loss = 0.0
        for _ep in range(5):
            loss = 0.0
            nb = 0
            for seq in seqs.values():
                if len(seq) < 3:
                    continue
                ids = [self.song_idx[s] for s in seq[-SEQ_MAX_LEN:] if s in self.song_idx]
                if len(ids) < 3:
                    continue
                # Train on each prefix → next prediction (causal).
                for t in range(1, len(ids) - 1):
                    ctx_ids = ids[max(0, t - 8):t + 1]
                    pos = ids[t + 1]
                    neg = ids[rng.integers(0, len(ids))]
                    if neg == pos:
                        continue
                    H = E[ctx_ids]  # (L, d)
                    Q = H @ Wq
                    K = H @ Wk
                    V = H @ Wv
                    scores = (Q @ K.T) / np.sqrt(self.dim)
                    # Causal mask.
                    L = len(ctx_ids)
                    mask = np.triu(np.ones((L, L)), k=1) * -1e9
                    A = _softmax(scores + mask)
                    O = A @ V
                    h = O[-1]  # state for next prediction
                    xp = float(h @ E[pos])
                    xn = float(h @ E[neg])
                    # BPR loss.
                    x = xp - xn
                    sig = 1.0 / (1.0 + np.exp(x))
                    loss += -np.log(1.0 / (1.0 + np.exp(-x)) + 1e-12)
                    nb += 1
                    g = -sig
                    # Gradients (simplified: through dot only, embeddings + out).
                    dh = g * (E[pos] - E[neg])
                    # Backprop through attention output (last row only).
                    dO_last = dh
                    dA_last = dO_last @ V.T  # (L,)
                    # Softmax jacobian approx: dA = A*(dA - sum).
                    a_last = A[-1]
                    dS_last = a_last * (dA_last - float(a_last @ dA_last))
                    dQ_last = dS_last @ K / np.sqrt(self.dim)
                    dK = (dS_last[:, None] * Q[-1][None, :]) / np.sqrt(self.dim)
                    dV = np.outer(a_last, dO_last)
                    H_last = H[-1]
                    # Update.
                    Wq -= lr * np.outer(H_last, dQ_last) * 0.1
                    Wk -= lr * (H.T @ dK) * 0.1
                    Wv -= lr * (H.T @ dV) * 0.1
                    E[pos] -= lr * (g * h + 0.001 * E[pos])
                    E[neg] -= lr * (-g * h + 0.001 * E[neg])
            if nb == 0:
                break
        self.E, self.Wq, self.Wk, self.Wv = E, Wq, Wk, Wv
        self.train_loss = float(loss / max(nb, 1))
        self.trained_at = time.time()
        logger.info("✅ Sequence fitted: songs=%d users=%d loss=%.4f secs=%.1f",
                    n, len(seqs), self.train_loss, time.time() - t0)
        return self.train_loss

    # ── serve ──
    def _session_vector(self, user_id: int) -> Optional[np.ndarray]:
        seq = self.user_seqs.get(int(user_id), [])
        if not seq or self.E is None:
            return None
        ids = [self.song_idx[s] for s in seq[-SEQ_MAX_LEN:] if s in self.song_idx]
        if not ids:
            return None
        H = self.E[ids]
        Q = H @ self.Wq
        K = H @ self.Wk
        V = H @ self.Wv
        scores = (Q @ K.T) / np.sqrt(self.dim)
        L = len(ids)
        mask = np.triu(np.ones((L, L)), k=1) * -1e9
        A = _softmax(scores + mask)
        return (A @ V)[-1]

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        seq = self.user_seqs.get(int(user_id), [])
        last = seq[-1] if seq else None
        h = self._session_vector(int(user_id))
        for s in song_ids:
            s = str(s)
            # Markov term.
            mk = 0.0
            if last and last in self.markov:
                tot = sum(self.markov[last].values())
                mk = self.markov[last].get(s, 0) / max(tot, 1)
            # Attention term.
            att = 0.0
            if h is not None and self.E is not None and s in self.song_idx:
                att = float(h @ self.E[self.song_idx[s]])
                att = 1.0 / (1.0 + np.exp(-att))  # → [0,1]
            v = 0.6 * att + 0.4 * min(mk * 5.0, 1.0)
            out[s] = v * 2 - 1
        return out

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"sequence_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 Sequence saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "SequenceModel":
        with open(path, "rb") as f:
            m = pickle.load(f)
        logger.info("📂 Sequence loaded from %s", path)
        return m
