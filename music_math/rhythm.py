"""rhythm.py — why bodies move: beat hierarchies and entrainment math.

Human movement locks to pulse at ~100–130 BPM (walking cadence ×2, heart
rate ×~1.5). A groove scores high when:
  - tempo sits near the entrainment zone (Gaussian around 120 BPM),
  - the title cues dance/workout energy,
  - duration fits radio/phrase structure (~3–4 min ≈ 100 bars of 4/4).

Stdlib only.
"""

import math
import re

ENTRAINMENT_CENTER_BPM = 120.0
ENTRAINMENT_WIDTH_BPM = 35.0

_GROOVE_CUES = {
    "dance": 1.0, "club": 0.9, "party": 0.9, "workout": 0.9, "gym": 0.8,
    "banger": 0.9, "beat": 0.6, "groove": 0.8, "remix": 0.5, "dhol": 0.8,
    "bhangra": 0.9, "salsa": 0.9, "samba": 0.9, "edm": 0.8, "house": 0.8,
    "anthem": 0.5, "march": 0.6,
}
_CALM_CUES = {
    "lullaby": 1.0, "sleep": 0.9, "ambient": 0.8, "meditation": 0.9,
    "lofi": 0.6, "acoustic": 0.4, "ballad": 0.5, "slow": 0.7, "night": 0.3,
}


def tempo_fit(bpm: float | None) -> float:
    """Gaussian entrainment fit in [0,1]. Unknown tempo → 0.5."""
    if bpm is None or bpm <= 0:
        return 0.5
    z = (float(bpm) - ENTRAINMENT_CENTER_BPM) / ENTRAINMENT_WIDTH_BPM
    return float(math.exp(-0.5 * z * z))


def groove_score(title: str, duration_s: int = 0) -> float:
    """Groove prior in [0,1] from title energy + duration structure."""
    words = re.findall(r"[a-z]+", (title or "").lower())
    energy = sum(_GROOVE_CUES.get(w, 0.0) for w in words)
    calm = sum(_CALM_CUES.get(w, 0.0) for w in words)
    cue = 0.5 + 0.25 * (energy - calm)
    if duration_s and duration_s > 0:
        # 180–240s is the pop-form sweet spot; very long/short drifts off.
        minutes = duration_s / 60.0
        shape = math.exp(-0.5 * ((minutes - 3.5) / 2.0) ** 2)
        cue = 0.7 * cue + 0.3 * shape
    return max(0.0, min(1.0, cue))
