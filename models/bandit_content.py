"""
models/bandit_content.py — Thompson Sampling Bandit + TF-IDF Content Scorer

┌─────────────────────────────────────────────────────────┐
│  THOMPSON SAMPLING (Bayesian Multi-Armed Bandit)         │
│                                                          │
│  Each (user, song) pair is treated as an "arm".          │
│  We maintain Beta(α, β) distributions per arm:           │
│      α = 1 + n_likes                                    │
│      β = 1 + n_dislikes                                 │
│                                                          │
│  At recommendation time:                                 │
│      θ_i ~ Beta(α_i, β_i)   for each candidate          │
│      Recommend top-k by sampled θ                        │
│                                                          │
│  Why Beta distribution?                                  │
│      Beta is the conjugate prior of the Bernoulli       │
│      distribution. After each like/dislike, the          │
│      posterior update is exact and O(1):                 │
│          like   → α += 1                                │
│          dislike→ β += 1                                │
│                                                          │
│  This balances:                                          │
│      EXPLOITATION: arms with many likes (high α)        │
│      EXPLORATION:  arms with few observations (wide β)  │
└─────────────────────────────────────────────────────────┘

Content scorer uses TF-IDF on song titles + tags.
Cosine similarity between user's history title corpus
and candidate titles gives a content-based affinity score.
"""

import numpy as np
import math
import time
import re
import logging
from collections import defaultdict
from typing import Optional

from config import DB_BANDIT, DB_HISTORY, DB_CATALOG, DB_FEEDBACK
from core.database import get_conn

logger = logging.getLogger("parlay.bandit")


# ─── Thompson Sampling ────────────────────────────────────

class ThompsonBandit:
    """
    Per-user Beta bandit. State persisted in DB_BANDIT.
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self._alpha: dict[str, float] = {}   # song_id → α
        self._beta:  dict[str, float] = {}   # song_id → β
        self._load()

    def _load(self) -> None:
        with get_conn(DB_BANDIT) as conn:
            rows = conn.execute(
                "SELECT song_id, alpha, beta_param FROM bandit_arms WHERE user_id=?",
                (self.user_id,)
            ).fetchall()
        for row in rows:
            self._alpha[row["song_id"]] = row["alpha"]
            self._beta[row["song_id"]]  = row["beta_param"]

    def _ensure(self, song_id: str) -> None:
        if song_id not in self._alpha:
            self._alpha[song_id] = 1.0   # Beta(1,1) = Uniform prior
            self._beta[song_id]  = 1.0

    def sample(self, song_id: str) -> float:
        """
        Draw θ ~ Beta(α, β).
        High α → mostly likes → high sample.
        Low α, low β → unexplored → wide distribution → random chance of being high.
        """
        self._ensure(song_id)
        return float(np.random.beta(self._alpha[song_id], self._beta[song_id]))

    def sample_batch(self, song_ids: list[str]) -> dict[str, float]:
        """Vectorised batch sample — much faster than looping."""
        alphas = np.array([self._alpha.get(s, 1.0) for s in song_ids])
        betas  = np.array([self._beta.get(s,  1.0) for s in song_ids])
        samples = np.random.beta(alphas, betas)
        return dict(zip(song_ids, samples.tolist()))

    def update(self, song_id: str, reward: float) -> None:
        """
        Bayesian update:
            reward > 0 → like  → α += |reward|
            reward < 0 → dislike → β += |reward|

        Using |reward| instead of binary 0/1 allows partial rewards
        (e.g. partial play = +0.2 contributes α += 0.2).
        """
        self._ensure(song_id)
        if reward > 0:
            self._alpha[song_id] += abs(reward)
        else:
            self._beta[song_id]  += abs(reward)
        self._persist(song_id)

    def _persist(self, song_id: str) -> None:
        with get_conn(DB_BANDIT) as conn:
            conn.execute("""
            INSERT INTO bandit_arms (user_id, song_id, alpha, beta_param, n_pulls, last_pulled)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(user_id, song_id) DO UPDATE SET
                alpha       = excluded.alpha,
                beta_param  = excluded.beta_param,
                n_pulls     = n_pulls + 1,
                last_pulled = excluded.last_pulled
            """, (
                self.user_id,
                song_id,
                self._alpha[song_id],
                self._beta[song_id],
                time.time(),
            ))

    def persist_all(self) -> None:
        """Batch-persist entire state. Call after bulk updates."""
        with get_conn(DB_BANDIT) as conn:
            conn.executemany("""
            INSERT INTO bandit_arms (user_id, song_id, alpha, beta_param, n_pulls, last_pulled)
            VALUES (?, ?, ?, ?, 0, ?)
            ON CONFLICT(user_id, song_id) DO UPDATE SET
                alpha       = excluded.alpha,
                beta_param  = excluded.beta_param
            """, [
                (self.user_id, sid, self._alpha[sid], self._beta[sid], time.time())
                for sid in self._alpha
            ])

    def confidence(self, song_id: str) -> float:
        """
        Confidence = how much we know about this arm.
        1 - entropy of Beta(α, β) normalised to [0, 1].
        Wide distributions (entropy high) = low confidence = explore more.
        """
        a = self._alpha.get(song_id, 1.0)
        b = self._beta.get(song_id, 1.0)
        # Variance of Beta: α*β / ((α+β)²*(α+β+1))
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return 1.0 - min(var * 10, 1.0)  # invert and scale


# ─── TF-IDF Content Scorer ───────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split."""
    return re.sub(r'[^\w\s]', '', text.lower()).split()


class TFIDFContentScorer:
    """
    Build a TF-IDF corpus from the song catalog titles.
    Score candidates by cosine similarity to user's taste profile
    (built from their listen history titles).

    Why TF-IDF for music?
        Genre/mood keywords ("lofi", "trap", "sad", "chill") are meaningful.
        TF (term freq) rewards songs whose titles align with the user's vocab.
        IDF (inverse doc freq) downweights ultra-common words ("song", "official").
        Cosine similarity is scale-invariant — title length doesn't bias it.
    """

    def __init__(self):
        self._vocab:    dict[str, int]  = {}
        self._idf:      np.ndarray      = np.array([])
        self._song_vecs: dict[str, np.ndarray] = {}
        self._fitted    = False

    def fit(self, songs: list[dict]) -> None:
        """
        Build IDF from song titles.
        songs: list of {"song_id": ..., "title": ...}
        """
        # Build vocabulary
        df: dict[str, int] = defaultdict(int)   # document frequency
        token_lists: dict[str, list[str]] = {}

        for song in songs:
            tokens = set(_tokenise(song.get("title", "")))
            for t in tokens:
                df[t] += 1
            token_lists[song["song_id"]] = list(_tokenise(song.get("title", "")))

        N = max(len(songs), 1)
        # Keep vocab words that appear in 2+ songs but not ALL songs (stop-word-like)
        vocab_words = [w for w, cnt in df.items() if 2 <= cnt <= N * 0.8]
        self._vocab = {w: i for i, w in enumerate(vocab_words)}
        V = len(self._vocab)

        # IDF = log((N + 1) / (df + 1)) + 1  [sklearn-style smooth IDF]
        self._idf = np.array([
            math.log((N + 1) / (df[w] + 1)) + 1.0
            for w in vocab_words
        ], dtype=np.float32)

        # Compute TF-IDF vectors for each song
        for song in songs:
            sid = song["song_id"]
            tokens = token_lists.get(sid, [])
            vec = np.zeros(V, dtype=np.float32)
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for w, cnt in tf.items():
                if w in self._vocab:
                    idx = self._vocab[w]
                    vec[idx] = (cnt / max(len(tokens), 1)) * self._idf[idx]
            # L2 normalise
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self._song_vecs[sid] = vec

        self._fitted = True
        logger.info(f"TF-IDF fitted: vocab={V}, songs={len(songs)}")

    def user_profile_vector(self, user_id: int) -> Optional[np.ndarray]:
        """
        Build user taste vector from their listen history titles.
        Average of TF-IDF vectors of all listened songs, weighted
        by completion_pct (so songs they finished count more).
        """
        if not self._fitted:
            return None

        with get_conn(DB_HISTORY) as conn:
            rows = conn.execute("""
                SELECT l.song_id, l.completion_pct, s.title
                FROM listens l
                LEFT JOIN songs s ON s.song_id = l.song_id
                WHERE l.user_id = ?
                ORDER BY l.started_at DESC
                LIMIT 200
            """, (user_id,)).fetchall()

        if not rows:
            return None

        V = len(self._vocab)
        profile = np.zeros(V, dtype=np.float32)
        total_w = 0.0

        for row in rows:
            sid = row["song_id"]
            w = max(float(row["completion_pct"] or 0.3), 0.1)
            vec = self._song_vecs.get(sid)
            if vec is not None:
                profile += vec * w
                total_w += w

        if total_w == 0:
            return None

        profile /= total_w
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
        return profile

    def score_candidates(
        self,
        user_id: int,
        song_ids: list[str],
    ) -> dict[str, float]:
        """
        Cosine similarity between user profile and each candidate.
        Returns scores in [-1, 1] (mapped from [0, 1] dot product).
        """
        if not self._fitted:
            return {s: 0.0 for s in song_ids}

        profile = self.user_profile_vector(user_id)
        if profile is None:
            # Cold-start: use song popularity signals only (genre diversity bonus)
            return {s: 0.0 for s in song_ids}

        scores = {}
        for sid in song_ids:
            vec = self._song_vecs.get(sid)
            if vec is None:
                scores[sid] = 0.0
            else:
                # dot product of normalised vectors = cosine similarity
                cos_sim = float(np.dot(profile, vec))
                # Map from [0, 1] → [-1, 1]
                scores[sid] = cos_sim * 2 - 1
        return scores

    def update_song(self, song: dict) -> None:
        """Add/update a single song's vector (online catalog expansion)."""
        if not self._fitted:
            return
        tokens = _tokenise(song.get("title", ""))
        V = len(self._vocab)
        vec = np.zeros(V, dtype=np.float32)
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        for w, cnt in tf.items():
            if w in self._vocab:
                idx = self._vocab[w]
                vec[idx] = (cnt / max(len(tokens), 1)) * self._idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._song_vecs[song["song_id"]] = vec
