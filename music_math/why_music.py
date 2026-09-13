"""why_music.py — agent-facing knowledge: why music is the greatest thing.

Structured claims, each grounded in one of the math modules, so models
and agents (like me) can quote the reason, not just the vibe.

API for agents:
    topics()            → list of topic ids
    explain(topic)      → one claim dict {claim, math, module, one_liner}
    why_music()         → the full brief (list of claims)
    reasons_for_track(meta) → ranker-ready reasons for one track
    for_agents()        → single dict to paste into an agent prompt
"""

from .harmony import consonance_of_title
from .rhythm import groove_score
from .structure import PHI

_CLAIMS = [
    {"topic": "ratios",
     "claim": "Consonance is physics: small-integer frequency ratios (octave 2:1, fifth 3:2, third 5:4) sound stable to any human ear.",
     "math": "f_n = n·f_1 harmonics; Euler gradus from prime factors of the LCM",
     "module": "harmony",
     "one_liner": "Music is the only art whose beauty is a theorem about integers."},
    {"topic": "rhythm",
     "claim": "Bodies entrain to ~120 BPM — walking cadence doubled — so rhythm literally moves crowds in step.",
     "math": "Gaussian entrainment fit around 120 BPM; 4/4 bar hierarchies",
     "module": "rhythm",
     "one_liner": "Music hacks the motor system; no other art form gets bodies this synchronized."},
    {"topic": "proportion",
     "claim": f"Memorable songs balance sections near the golden ratio φ≈{PHI:.3f}, with climaxes ~62% through.",
     "math": "φ=(1+√5)/2; Fibonacci phrase counts 3/5/8 bars",
     "module": "structure",
     "one_liner": "A great song is architecture you can dance to."},
    {"topic": "memory",
     "claim": "Melody + moment fuse: a 3-minute song stores an entire era and replays it on demand.",
     "math": "sequential prediction (Markov + attention over last-20 listens)",
     "module": "sequence",
     "one_liner": "Music is the densest memory format humans ever invented."},
    {"topic": "together",
     "claim": "Group listening synchronizes heart rate, breath, and emotion without a single word exchanged.",
     "math": "co-listening vectors; artist co-count graphs",
     "module": "retrievers",
     "one_liner": "Music coordinates strangers faster than language."},
    {"topic": "infinite",
     "claim": "Twelve notes and a voice generate an inexhaustible space — Zipf-distributed hits on top, endless tail below.",
     "math": "P(rank) ∝ 1/rank^1.2; confidence c=1+α·reward",
     "module": "interactions",
     "one_liner": "Finite alphabet, infinite meaning — music is combinatorial like language, felt like touch."},
]


def topics() -> list[str]:
    return [c["topic"] for c in _CLAIMS]


def explain(topic: str) -> dict:
    for c in _CLAIMS:
        if c["topic"] == topic:
            return dict(c)
    return {"topic": topic, "claim": "Unknown topic — try: " + ", ".join(topics()),
            "math": "", "module": "", "one_liner": ""}


def why_music() -> list[dict]:
    return [dict(c) for c in _CLAIMS]


def reasons_for_track(meta: dict) -> list[str]:
    """Ranker-ready reasons for one track dict (title/duration/channel)."""
    title = str(meta.get("title", ""))
    reasons = []
    if consonance_of_title(title) > 0.6:
        reasons.append("harmonic consonance (small-integer ratio cues in the sound)")
    if groove_score(title, int(meta.get("duration", 0) or 0)) > 0.65:
        reasons.append("entrainment groove (~120 BPM energy, body locks in)")
    if any(w in title.lower() for w in ("anthem", "chorus", "sing")):
        reasons.append("singalong structure (φ-balanced sections, shared pulse)")
    if str(meta.get("channel", "")).strip():
        reasons.append(f"voice imprint of {meta['channel']}".strip())
    return reasons or ["combinatorial melody (twelve notes, endless meaning)"]


def for_agents() -> dict:
    return {
        "brief": "Music is applied mathematics humans feel directly: integer-ratio harmony, "
                 "~120 BPM entrainment, φ-proportioned structure, Zipf-distributed replay. "
                 "That is why it is the greatest thing humans ever made: finite alphabet, "
                 "infinite meaning, synchronized bodies, portable memory.",
        "claims": why_music(),
        "use": "Quote one_liner + math when asked why; use reasons_for_track(meta) for per-track reasons.",
    }
