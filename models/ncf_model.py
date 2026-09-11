"""
models/ncf_model.py — Neural Collaborative Filtering (NCF)

Architecture: GMF + MLP fusion (He et al., 2017 — "Neural Collaborative Filtering")

                User_ID ──→ Embedding_u (k)
                Song_ID ──→ Embedding_i (k)
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  GMF branch          MLP branch
                 (u ⊙ i)           [u∥i → FC layers]
                    │                   │
                    └────── concat ─────┘
                                │
                           Output FC → sigmoid → affinity score

GMF (Generalised Matrix Factorization) captures linear user-item interactions.
MLP captures non-linear, higher-order feature interactions.
Fusion gives the best of both worlds.

Input also includes song metadata features (genre, energy, freshness, etc.),
concatenated to the MLP branch to give the model content awareness alongside
collaborative signals.
"""

import numpy as np
import time
import math
import logging
import pickle
from pathlib import Path
from typing import Optional

from config import (
    NCF_EMBEDDING_DIM, NCF_HIDDEN_LAYERS, NCF_DROPOUT,
    NCF_LR, NCF_BATCH_SIZE, NCF_EPOCHS, MODELS_DIR,
    METADATA_FEATURES,
)
from core.database import get_conn, DB_HISTORY, DB_FEEDBACK, DB_CATALOG, DB_EMBEDDINGS, ndarray_to_blob

logger = logging.getLogger("parlay.ncf")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. NCF model will be disabled. `pip install torch`")


# ─── NCF Architecture ────────────────────────────────────

class NCFNet(nn.Module if TORCH_AVAILABLE else object):
    """
    GMF + MLP hybrid network.
    Input:
        user_idx (long)
        song_idx (long)
        meta     (float tensor of METADATA_FEATURES)
    Output:
        affinity score (float, [0, 1] via sigmoid)
    """

    def __init__(
        self,
        n_users: int,
        n_songs: int,
        emb_dim: int = NCF_EMBEDDING_DIM,
        hidden: list[int] = NCF_HIDDEN_LAYERS,
        dropout: float = NCF_DROPOUT,
        n_meta: int = len(METADATA_FEATURES),
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # ── Embeddings ──────────────────────────────────────
        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.song_emb_gmf = nn.Embedding(n_songs, emb_dim)
        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.song_emb_mlp = nn.Embedding(n_songs, emb_dim)

        # ── MLP branch ──────────────────────────────────────
        # Input = concat(user_mlp, song_mlp, meta_features)
        mlp_input_dim = emb_dim * 2 + n_meta
        layers = []
        in_dim = mlp_input_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

        # ── Output fusion ───────────────────────────────────
        # GMF output = emb_dim, MLP output = hidden[-1]
        self.output = nn.Sequential(
            nn.Linear(emb_dim + hidden[-1], 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.song_emb_gmf,
                    self.user_emb_mlp, self.song_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_idx, song_idx, meta):
        # GMF path: element-wise product of embeddings
        u_gmf = self.user_emb_gmf(user_idx)
        i_gmf = self.song_emb_gmf(song_idx)
        gmf_out = u_gmf * i_gmf                          # (B, emb_dim)

        # MLP path: concatenate + feed-forward
        u_mlp = self.user_emb_mlp(user_idx)
        i_mlp = self.song_emb_mlp(song_idx)
        mlp_in = torch.cat([u_mlp, i_mlp, meta], dim=1) # (B, 2*emb + n_meta)
        mlp_out = self.mlp(mlp_in)                       # (B, hidden[-1])

        # Fuse
        fused = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output(fused).squeeze(1)              # (B,)


# ─── Trainer Wrapper ─────────────────────────────────────

class NCFModel:
    def __init__(self, version: str = "v0"):
        self.version = version
        self.net: Optional["NCFNet"] = None
        self.user_idx: dict[int, int] = {}
        self.song_idx: dict[str, int] = {}
        self.idx_song: dict[int, str] = {}
        self.song_meta: dict[str, np.ndarray] = {}   # precomputed meta vecs
        self.n_meta = len(METADATA_FEATURES)
        self.trained_at: Optional[float] = None
        self.train_loss: float = 999.0
        self._device = "cpu"
        if TORCH_AVAILABLE:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"NCF will train on: {self._device}")

    def _load_song_meta(self, song_ids: list[str]) -> dict[str, np.ndarray]:
        """Build metadata feature vectors for each song."""
        meta = {}
        with get_conn(DB_CATALOG) as conn:
            for sid in song_ids:
                row = conn.execute(
                    "SELECT * FROM songs WHERE song_id=?", (sid,)
                ).fetchone()
                if row:
                    v = np.array([
                        math.log10(max(row["view_count"], 1)),       # view_count_log
                        row["like_count"] / max(row["view_count"], 1), # like_ratio
                        min(row["duration"] / 600.0, 1.0),           # duration_norm
                        0.0,                                          # channel_sub_log (N/A)
                        0.5,                                          # days_since_upload_norm
                        min(len(row["title"]) / 100.0, 1.0),         # title_len_norm
                        float(row["has_official"]),
                        float(row["has_lyric"]),
                        float(row["has_remix"] if "has_remix" in row.keys() else 0),
                        float(row["language_code"]) / 9.0,           # normalised
                        float(row["genre_code"]) / 20.0,
                        float(row["energy_score"]),
                    ], dtype=np.float32)
                else:
                    v = np.zeros(self.n_meta, dtype=np.float32)
                meta[sid] = v
        return meta

    def fit(self, interactions: list[tuple[int, str, float]]) -> float:
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — NCF skipped.")
            return 999.0

        if len(interactions) < 20:
            logger.warning("NCF: not enough data to train.")
            return 999.0

        # Build index maps
        users = sorted(set(u for u, _, _ in interactions))
        songs = sorted(set(s for _, s, _ in interactions))
        self.user_idx = {u: i for i, u in enumerate(users)}
        self.song_idx = {s: i for i, s in enumerate(songs)}
        self.idx_song = {i: s for s, i in self.song_idx.items()}

        self.song_meta = self._load_song_meta(songs)

        n_u, n_s = len(users), len(songs)
        logger.info(f"NCF training: {n_u} users × {n_s} songs")

        # Prepare tensors
        U, S, R, M = [], [], [], []
        for uid, sid, rating in interactions:
            if uid not in self.user_idx or sid not in self.song_idx:
                continue
            U.append(self.user_idx[uid])
            S.append(self.song_idx[sid])
            # Normalise reward from [-1,1] to [0,1] for sigmoid output
            R.append((rating + 1.0) / 2.0)
            M.append(self.song_meta.get(sid, np.zeros(self.n_meta, np.float32)))

        U_t = torch.tensor(U, dtype=torch.long)
        S_t = torch.tensor(S, dtype=torch.long)
        R_t = torch.tensor(R, dtype=torch.float32)
        M_t = torch.tensor(np.array(M), dtype=torch.float32)

        dataset = TensorDataset(U_t, S_t, M_t, R_t)
        loader  = DataLoader(dataset, batch_size=NCF_BATCH_SIZE, shuffle=True)

        self.net = NCFNet(n_u, n_s).to(self._device)
        opt = optim.Adam(self.net.parameters(), lr=NCF_LR, weight_decay=1e-5)
        criterion = nn.BCELoss()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NCF_EPOCHS)

        logger.info(f"  Device: {self._device} | Batch: {NCF_BATCH_SIZE} | Epochs: {NCF_EPOCHS}")

        for epoch in range(NCF_EPOCHS):
            self.net.train()
            total_loss = 0.0
            batches = 0
            for u_b, s_b, m_b, r_b in loader:
                u_b = u_b.to(self._device)
                s_b = s_b.to(self._device)
                m_b = m_b.to(self._device)
                r_b = r_b.to(self._device)

                opt.zero_grad()
                pred = self.net(u_b, s_b, m_b)
                loss = criterion(pred, r_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item()
                batches += 1

            scheduler.step()
            avg_loss = total_loss / max(batches, 1)
            if epoch % 4 == 0 or epoch == NCF_EPOCHS - 1:
                logger.info(f"  NCF Epoch {epoch+1:3d}/{NCF_EPOCHS} | Loss: {avg_loss:.4f}")

        self.train_loss = avg_loss
        self.trained_at = time.time()
        self.net.eval()
        logger.info(f"✅  NCF training done. Final loss: {avg_loss:.4f}")
        return avg_loss

    def predict_batch(self, user_id: int, song_ids: list[str]) -> dict[str, float]:
        if not TORCH_AVAILABLE or self.net is None:
            return {s: 0.0 for s in song_ids}

        uid = self.user_idx.get(user_id, 0)
        scores = {}
        batch_songs, batch_iids, batch_metas = [], [], []

        for sid in song_ids:
            iid = self.song_idx.get(sid)
            if iid is None:
                scores[sid] = 0.5  # cold-start: neutral
                continue
            batch_songs.append(sid)
            batch_iids.append(iid)
            batch_metas.append(self.song_meta.get(sid, np.zeros(self.n_meta, np.float32)))

        if not batch_songs:
            return scores

        self.net.eval()
        with torch.no_grad():
            u_t = torch.tensor([uid] * len(batch_iids), dtype=torch.long).to(self._device)
            s_t = torch.tensor(batch_iids, dtype=torch.long).to(self._device)
            m_t = torch.tensor(np.array(batch_metas), dtype=torch.float32).to(self._device)
            preds = self.net(u_t, s_t, m_t).cpu().numpy()

        for sid, pred in zip(batch_songs, preds):
            # Map back from [0,1] to [-1,1]
            scores[sid] = float(pred) * 2 - 1

        return scores

    def online_update(self, user_id: int, song_id: str, reward: float) -> None:
        """Single-step gradient update on new feedback."""
        if not TORCH_AVAILABLE or self.net is None:
            return
        uid = self.user_idx.get(user_id)
        iid = self.song_idx.get(song_id)
        if uid is None or iid is None:
            return

        meta = self.song_meta.get(song_id, np.zeros(self.n_meta, np.float32))
        u_t = torch.tensor([uid], dtype=torch.long).to(self._device)
        s_t = torch.tensor([iid], dtype=torch.long).to(self._device)
        m_t = torch.tensor([meta], dtype=torch.float32).to(self._device)
        r_t = torch.tensor([(reward + 1) / 2], dtype=torch.float32).to(self._device)

        self.net.train()
        opt = optim.Adam(self.net.parameters(), lr=NCF_LR * 0.1)  # smaller lr for online
        pred = self.net(u_t, s_t, m_t)
        loss = nn.BCELoss()(pred, r_t)
        loss.backward()
        opt.step()
        self.net.eval()

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"ncf_{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"💾  NCF model saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "NCFModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"📂  NCF model loaded from {path}")
        return model
