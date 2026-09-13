"""music_math — the math that makes music work, shared by models and agents.

Stdlib-only core (harmony, rhythm, structure) so Parlay can vendor it
without new deps. `features.py` adds the optional numpy bridge for
Parlay-ML models; `why_music.py` is the agent-facing knowledge surface.

Why this exists: music is applied mathematics that humans feel directly.
Small-integer frequency ratios sound consonant, beat hierarchies entrain
movement, golden-ratio/Fibonacci proportions shape memorable structure,
and Zipf's law governs what gets replayed. Models that know this rank
better; agents that know this explain better.
"""

from .harmony import consonance_of_title, interval_consonance, FIFTH_CIRCLE
from .rhythm import groove_score, tempo_fit
from .structure import fibonacci_balance, golden_climax, zipf_prior
from .why_music import explain, for_agents, reasons_for_track, topics, why_music

__all__ = [
    "consonance_of_title", "interval_consonance", "FIFTH_CIRCLE",
    "groove_score", "tempo_fit",
    "fibonacci_balance", "golden_climax", "zipf_prior",
    "explain", "for_agents", "reasons_for_track", "topics", "why_music",
]
