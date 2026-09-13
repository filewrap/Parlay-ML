"""harmony.py — why some note combinations feel right: small-integer ratios.

A vibrating string divided into n equal parts sounds the nth harmonic.
Two notes sound consonant when their frequencies form a ratio of small
integers (Pythagoras, ~500 BC; still the physics behind every chorus):

    octave  2:1, fifth 3:2, fourth 4:3, major third 5:4, minor third 6:5

Euler's gradus suavitatis grades a chord by the prime factorization of
its LCM: fewer/smaller primes → smoother. We use a cheap version of it
to score title-evoked harmony cues (e.g. "harmony", "chorus", "duet")
and to measure circle-of-fifths distance between keys.

Stdlib only.
"""

import math
import re

# Semitone → just-intonation ratio (numerator, denominator).
JUST_RATIOS = {
    0: (1, 1),    # unison
    2: (9, 8),    # major second
    3: (6, 5),    # minor third
    4: (5, 4),    # major third
    5: (4, 3),    # fourth
    7: (3, 2),    # fifth
    9: (5, 3),    # major sixth
    12: (2, 1),   # octave
}

# Circle of fifths order (distance = minimal steps).
FIFTH_CIRCLE = ["C", "G", "D", "A", "E", "B", "F#", "C#", "Ab", "Eb", "Bb", "F"]

_HARMONY_CUES = {
    "harmony": 1.0, "chorus": 0.8, "choir": 0.9, "duet": 0.7, "acapella": 0.9,
    "acoustic": 0.6, "unplugged": 0.6, "symphony": 0.8, "orchestra": 0.7,
    "anthem": 0.5, "ballad": 0.5, "soulful": 0.6, "melody": 0.6,
}


def _prime_factors(n: int) -> list[int]:
    out = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        out.append(n)
    return out


def euler_gradus(ratios: list[tuple[int, int]]) -> int:
    """Euler's gradus suavitatis: lower = more consonant. Unison → 1."""
    lcm = 1
    for num, den in ratios:
        lcm = lcm * num // math.gcd(lcm, num)
        lcm = lcm * den // math.gcd(lcm, den)
    factors = _prime_factors(max(lcm, 1))
    return 1 + sum(p - 1 for p in factors)


def interval_consonance(semitones: int) -> float:
    """Consonance of one interval in [0,1]. Octave/fifth ≈ 1, tritone ≈ 0."""
    semitones = abs(int(semitones)) % 12
    if semitones in JUST_RATIOS:
        g = euler_gradus([JUST_RATIOS[semitones]])
        return max(0.0, 1.0 - (g - 1) / 12.0)
    # Dissonant leftovers: minor 2nd, tritone, major 7th score lowest.
    return {"1": 0.15, "6": 0.1, "11": 0.2, "10": 0.35, "8": 0.45}.get(str(semitones), 0.4)


def fifth_distance(key_a: str, key_b: str) -> int:
    """Minimal circle-of-fifths steps between two keys (0–6)."""
    try:
        ia, ib = FIFTH_CIRCLE.index(key_a), FIFTH_CIRCLE.index(key_b)
    except ValueError:
        return 6
    return min((ia - ib) % 12, (ib - ia) % 12)


def consonance_of_title(title: str) -> float:
    """Harmony prior in [0,1] from title cues. Neutral 0.5 when no cue."""
    words = re.findall(r"[a-z]+", (title or "").lower())
    if not words:
        return 0.5
    hits = [_HARMONY_CUES[w] for w in words if w in _HARMONY_CUES]
    if not hits:
        return 0.5
    return max(0.0, min(1.0, 0.5 + sum(hits) / len(words) * 0.5))
