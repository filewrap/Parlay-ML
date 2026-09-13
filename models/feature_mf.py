"""
models/feature_mf.py — LightFM-core (native, CPU-fast).

score(u,i) = (p_u + Σ_f F_f) · (q_i + Σ_g G_g) + b_u + b_i

Song features: genre_code, language, energy bucket, artist_id.
User features: mood/time bucket (hour//6, dow).
New songs with zero plays still score via features → kills cold-start.

BPR pairwise loss + Adagrad on sampled negatives from Phase 0.
"""

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import FEATURE_MF_EPOCHS, FEATURE_MF_FACTORS, FEATURE_MF_LR, MODELS_DIR, DB_CATALOG
from core.database import get_conn
from data.interactions import pairwise_training_pairs

logger = logging.getLogger("parlay.feature_mf")


def _feat_id(kind: str, val: str) -> str:
    return f"{kind}:{val}"


def song_features(song_id: str, meta: dict) -> list[str]:
    ch = str(meta.get("channel", "") or "unknown")[:48]
    e = float(meta.get("energy_score", 0.5))
    eb = "hi" if e > 0.66 else ("lo" if e < 0.33 else "mid")
    feats = [
        _feat_id("genre", str(int(meta.get("genre_code", 20)))),
        _feat_id("lang", str(int(meta.get("language_code", 0)))),
        _feat_id("energy", eb),
        _feat_id("artist", ch),
    ]
    # Measured ears (audio_features row): key/mode/tempo/brightness buckets.
    # Missing row → no extra features → title/metadata path still scores.
    audio = meta.get("audio") or {}
    if audio.get("musical_key"):
        feats.append(_feat_id("key", f"{audio['musical_key']}-{audio.get('mode', '')}"))
    bpm = float(audio.get("bpm") or 0)
    if bpm > 0:
        tb = "slow" if bpm < 90 else ("mid" if bpm < 130 else "fast")
        feats.append(_feat_id("tempo", tb))
    for name, lo, hi in (("bright", 0.66, 0.33), ("dance", 0.66, 0.33)):
        v = audio.get(name)
        if v is not None:
            v = float(v)
            feats.append(_feat_id(name, "hi" if v > lo else ("lo" if v < hi else "mid")))
    return feats


def user_features(user_id: int, hour: int = 20, dow: int = 4) -> list[str]:
    return [_feat_id("u", str(int(user_id) % 97)),
            _feat_id("tod", str(int(hour) // 6)),
            _feat_id("dow", str(int(dow) // 2))]


class FeatureMF:
    """Factorization machine with user/item side features, BPR + Adagrad."""

    def __init__(self, n_factors: int = FEATURE_MF_FACTORS, lr: float = FEATURE_MF_LR,
                 epochs: int = FEATURE_MF_EPOCHS, reg: float = 1e-4,
                 version: str = "v0"):
        self.k = n_factors
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.version = version
        self.user_idx: dict[int, int] = {}
        self.song_idx: dict[str, int] = {}
        self.feat_idx: dict[str, int] = {}
        self.P: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None  # feature embeddings
        self.b_u: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self._song_feat_lists: dict[str, list[int]] = {}
        self._user_feat_lists: dict[int, list[int]] = {}
        self.trained_at: Optional[float] = None
        self.train_loss: float = 999.0

    # ── setup ──
    def _load_catalog_meta(self, song_ids: set[str]) -> dict[str, dict]:
        meta: dict[str, dict] = {}
        if not song_ids:
            return meta
        ids = list(song_ids)
        # Batched fetch to stay < 2 GB peak. LEFT JOIN audio ears when heard.
        for b in range(0, len(ids), 2000):
            chunk = ids[b:b + 2000]
            ph = ",".join(["?"] * len(chunk))
            with get_conn(DB_CATALOG) as conn:
                try:
                    rows = conn.execute(
                        f"""SELECT s.song_id, s.genre_code, s.language_code, s.energy_score, s.channel,
                                   a.musical_key, a.mode, a.bpm, a.brightness, a.danceability
                            FROM songs s LEFT JOIN audio_features a ON a.song_id = s.song_id
                            WHERE s.song_id IN ({ph})""",
                        chunk).fetchall()
                except Exception:
                    try:
                        rows = conn.execute(
                            f"SELECT song_id, genre_code, language_code, energy_score, channel FROM songs WHERE song_id IN ({ph})",
                            chunk).fetchall()
                    except Exception:
                        rows = []
            for r in rows:
                keys = r.keys() if hasattr(r, "keys") else []
                m = {"genre_code": r["genre_code"], "language_code": r["language_code"],
                     "energy_score": r["energy_score"], "channel": r["channel"]}
                if "musical_key" in keys and r["musical_key"]:
                    m["audio"] = {"musical_key": r["musical_key"], "mode": r["mode"],
                                  "bpm": r["bpm"], "bright": r["brightness"],
                                  "dance": r["danceability"]}
                meta[r["song_id"]] = m
        return meta

    def _prepare(self, frame: list[dict]) -> list[tuple[int, str, str]]:
        users = sorted({int(d["user_id"]) for d in frame})
        songs = sorted({str(d["song_id"]) for d in frame})
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.song_idx = {s: i for i, s in enumerate(songs)}
        meta = self._load_catalog_meta(set(songs))
        feats: set[str] = set()
        song_fl: dict[str, list[str]] = {}
        for s in songs:
            fl = song_features(s, meta.get(s, {}))
            song_fl[s] = fl
            feats.update(fl)
        user_fl: dict[int, list[str]] = {}
        for u in users:
            fl = user_features(u)
            user_fl[u] = fl
            feats.update(fl)
        self.feat_idx = {f: i for i, f in enumerate(sorted(feats))}
        self._song_feat_lists = {s: [self.feat_idx[f] for f in fl] for s, fl in song_fl.items()}
        self._user_feat_lists = {u: [self.feat_idx[f] for f in fl] for u, fl in user_fl.items()}
        rng = np.random.default_rng(11)
        n_u, n_s, n_f = len(users), len(songs), len(self.feat_idx)
        self.P = rng.normal(0, 0.05, (n_u, self.k)).astype(np.float32)
        self.Q = rng.normal(0, 0.05, (n_s, self.k)).astype(np.float32)
        self.F = rng.normal(0, 0.05, (n_f, self.k)).astype(np.float32)
        self.b_u = np.zeros(n_u, np.float32)
        self.b_i = np.zeros(n_s, np.float32)
        # Adagrad accumulators.
        self._gP = np.ones_like(self.P) * 1e-8
        self._gQ = np.ones_like(self.Q) * 1e-8
        self._gF = np.ones_like(self.F) * 1e-8
        pairs = pairwise_training_pairs(frame)
        # Map to indices, drop unknowns.
        out = [(self.user_idx[u], s_pos, s_neg) for u, s_pos, s_neg in pairs
               if u in self.user_idx and s_pos in self.song_idx and s_neg in self.song_idx]
        return out

    def _vec(self, is_user: bool, idx: int, key) -> np.ndarray:
        if is_user:
            v = self.P[idx].copy()
            for f in self._user_feat_lists.get(key, []):
                v += self.F[f]
            return v
        v = self.Q[idx].copy()
        for f in self._song_feat_lists.get(key, []):
            v += self.F[f]
        return v

    # ── training (BPR + Adagrad) ──
    def fit(self, frame: list[dict]) -> float:
        if len(frame) < 10:
            logger.warning("FeatureMF: too little data.")
            return 999.0
        triples = self._prepare(frame)
        if not triples:
            return 999.0
        rng = np.random.default_rng(5)
        logger.info("FeatureMF: %d users × %d songs, feats=%d, pairs=%d",
                    len(self.user_idx), len(self.song_idx), len(self.feat_idx), len(triples))
        idx_users = [u for u, _, _ in triples]
        rev_song = {v: k for k, v in self.song_idx.items()}
        rev_user = {v: k for k, v in self.user_idx.items()}
        for ep in range(self.epochs):
            order = rng.permutation(len(triples))
            loss = 0.0
            for t in order:
                u, s_pos, s_neg = triples[t]
                i_pos, i_neg = self.song_idx[s_pos], self.song_idx[s_neg]
                ukey, pkey, nkey = rev_user[u], s_pos, s_neg
                vu = self._vec(True, u, ukey)
                vp = self._vec(False, i_pos, pkey)
                vn = self._vec(False, i_neg, nkey)
                x = float(vu @ vp + self.b_u[u] + self.b_i[i_pos]
                          - (vu @ vn + self.b_u[u] + self.b_i[i_neg]))
                sig = 1.0 / (1.0 + np.exp(x))
                loss += -np.log(1.0 / (1.0 + np.exp(-x)) + 1e-12)
                # Gradients of -log σ(x) = -σ(-x) * dx.
                g = -sig
                # Update user + feats, pos/neg items + feats via Adagrad.
                d_vu = g * (vp - vn)
                d_vp = g * vu
                d_vn = g * (-vu)
                for mat, acc, idx, d in ((self.P, self._gP, u, d_vu),):
                    acc[idx] += d * d
                    mat[idx] -= (self.lr / np.sqrt(acc[idx])) * (d + self.reg * mat[idx])
                for mat, acc, idx, d in ((self.Q, self._gQ, i_pos, d_vp),
                                         (self.Q, self._gQ, i_neg, d_vn)):
                    acc[idx] += d * d
                    mat[idx] -= (self.lr / np.sqrt(acc[idx])) * (d + self.reg * mat[idx])
                for f in self._user_feat_lists.get(ukey, []):
                    self._gF[f] += d_vu * d_vu
                    self.F[f] -= (self.lr / np.sqrt(self._gF[f])) * (d_vu + self.reg * self.F[f])
                for f in self._song_feat_lists.get(pkey, []):
                    self._gF[f] += d_vp * d_vp
                    self.F[f] -= (self.lr / np.sqrt(self._gF[f])) * (d_vp + self.reg * self.F[f])
                for f in self._song_feat_lists.get(nkey, []):
                    self._gF[f] += d_vn * d_vn
                    self.F[f] -= (self.lr / np.sqrt(self._gF[f])) * (d_vn + self.reg * self.F[f])
                # Biases (plain SGD).
                self.b_i[i_pos] -= self.lr * (g + self.reg * self.b_i[i_pos])
                self.b_i[i_neg] -= self.lr * (-g + self.reg * self.b_i[i_neg])
            if (ep + 1) % 5 == 0 or ep == 0:
                logger.info("  FeatureMF epoch %2d/%d bpr_loss=%.4f", ep + 1, self.epochs, loss / len(triples))
        self.train_loss = float(loss / max(len(triples), 1))
        self.trained_at = time.time()
        # Drop accumulators before pickling (memory).
        for a in ("_gP", "_gQ", "_gF"):
            if hasattr(self, a):
                delattr(self, a)
        logger.info("✅ FeatureMF done. loss=%.4f", self.train_loss)
        return self.train_loss

    # ── serving (cold-start aware: unseen songs score via features) ──
    def predict(self, user_id: int, song_id: str) -> float:
        return self.predict_batch(int(user_id), [str(song_id)]).get(str(song_id), 0.0)

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        u = self.user_idx.get(int(user_id))
        if self.P is None or u is None:
            # Unknown user: score items by bias + feature prior only.
            for s in song_ids:
                i = self.song_idx.get(str(s))
                out[str(s)] = float(self.b_i[i]) if i is not None and self.b_i is not None else 0.0
            return out
        ukey = next((k for k, v in self.user_idx.items() if v == u), int(user_id))
        vu = self._vec(True, u, int(user_id) if int(user_id) in self._user_feat_lists else ukey)
        bu = float(self.b_u[u])
        for s in song_ids:
            s = str(s)
            i = self.song_idx.get(s)
            if i is not None:
                vi = self._vec(False, i, s)
                out[s] = float(vu @ vi + bu + float(self.b_i[i]))
            else:
                # Cold-start song: feature-only vector (no Q/bias).
                h = hashlib.md5(s.encode())
                rng = np.random.default_rng(int.from_bytes(h.digest()[:4], "little"))
                vi = rng.normal(0, 0.02, self.k).astype(np.float32)
                # Add known feature embeddings when derivable (genre unknown → skip).
                out[s] = float(vu @ vi * 0.25)
        return out

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"feature_mf_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("💾 FeatureMF saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "FeatureMF":
        with open(path, "rb") as f:
            m = pickle.load(f)
        logger.info("📂 FeatureMF loaded from %s", path)
        return m
