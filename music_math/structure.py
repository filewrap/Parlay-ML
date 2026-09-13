"""structure.py — why songs are memorable: proportion math + Zipf.

- Golden ratio (φ ≈ 1.618): climaxes landing ~61.8% through a track feel
  "right"; verse/chorus duration ratios near φ feel balanced.
- Fibonacci: phrase lengths in Fibonacci counts (3, 5, 8 bars) recur
  across traditions; our energy bonus already uses Fibonacci weights.
- Zipf's law: the top 1% of songs take ~80% of plays (rank^-alpha).
  zipf_prior() turns a catalog rank into a replay-expectation prior.

Stdlib only.
"""

import math

PHI = (1.0 + math.sqrt(5.0)) / 2.0  # 1.618...
GOLDEN_POINT = 1.0 / PHI            # 0.618...


def golden_climax(climax_fraction: float | None) -> float:
    """Scores a climax position in [0,1]. Unknown → 0.5."""
    if climax_fraction is None:
        return 0.5
    d = abs(float(climax_fraction) - GOLDEN_POINT)
    return max(0.0, 1.0 - d / GOLDEN_POINT)


def fibonacci_balance(section_a: float, section_b: float) -> float:
    """Scores a verse/chorus duration ratio against φ in [0,1]."""
    if section_b <= 0 or section_a <= 0:
        return 0.5
    ratio = max(section_a, section_b) / min(section_a, section_b)
    return max(0.0, 1.0 - abs(ratio - PHI) / PHI)


def zipf_prior(rank: int, total: int, alpha: float = 1.2) -> float:
    """Replay-expectation prior in (0,1] from catalog rank (1-based)."""
    rank = max(int(rank), 1)
    total = max(int(total), 1)
    return float((1.0 / rank**alpha) / (1.0 / 1**alpha) * 0.5 + 0.5 * (1.0 - rank / total))
