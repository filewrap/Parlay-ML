"""features.py — numpy bridge: math priors as model side-information.

song_math_vector(meta) → float32[8] in [0,1]:
  [consonance, groove, golden_climax_prior, duration_shape,
   title_len_norm, energy, zipf_rank_prior, harmonic_density]

numpy is imported lazily so the stdlib core stays dependency-free.
"""

from typing import Optional

try:
    import numpy as np

    _NP = True
except ImportError:
    _NP = False

from .harmony import consonance_of_title
from .rhythm import groove_score
from .structure import golden_climax


def song_math_vector(meta: dict, rank: int = 500, total: int = 10000):
    """Fixed 8-dim math prior vector for one track. Requires numpy."""
    if not _NP:
        raise ImportError("song_math_vector needs numpy")
    title = str(meta.get("title", ""))
    dur = int(meta.get("duration", 0) or 0)
    cons = consonance_of_title(title)
    gro = groove_score(title, dur)
    gold = golden_climax(None)  # positional prior without audio analysis
    shape = min(dur / 600.0, 1.0) if dur else 0.5
    tlen = min(len(title) / 100.0, 1.0)
    energy = float(meta.get("energy_score", 0.5) or 0.5)
    zipf = max(0.0, min(1.0, 1.0 - rank / max(total, 1)))
    density = min((title.lower().count(" ") + 1) / 12.0, 1.0)
    return np.array([cons, gro, gold, shape, tlen, energy, zipf, density], dtype=np.float32)


def math_bonus(meta: dict) -> float:
    """Tiny stdlib-only bonus in [-0.05, +0.10] for the final blend."""
    title = str(meta.get("title", ""))
    c = consonance_of_title(title)
    g = groove_score(title, int(meta.get("duration", 0) or 0))
    return round(max(-0.05, min(0.10, (c - 0.5) * 0.08 + (g - 0.5) * 0.12)), 4)
