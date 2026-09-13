"""caption.py — numbers → sentence. The Listener learning to talk about music.

Pure functions of measured features: tempo words, key/mode mood words,
texture words (brightness/flatness/clarity), and arc words (lift, climax
position). No model, no API — just honest translation.
"""

from .store import get as _get_audio


def _tempo_word(bpm: float) -> str:
    if bpm <= 0:
        return "unhurried"
    if bpm < 90:
        return "slow"
    if bpm < 110:
        return "easy"
    if bpm < 135:
        return "driving"
    return "fiery"


def _mood_word(mode: str, valence: float) -> str:
    if mode == "minor":
        return "aching" if valence < 0.35 else ("wistful" if valence < 0.55 else "tender")
    return "glowing" if valence > 0.6 else ("warm" if valence > 0.4 else "bittersweet")


def _texture_word(energy: float, brightness: float, clarity: float) -> str:
    e = "hushed" if energy < 0.4 else ("steady" if energy < 0.7 else "urgent")
    b = "intimate" if brightness < 0.35 else ("natural" if brightness < 0.6 else "open")
    if clarity > 0.6:
        return f"{e} and crystal-clear, {b}"
    if clarity < 0.3:
        return f"{e}, hazy and {b}"
    return f"{e} and {b}"


def _arc_word(lift: float, climax_frac: float) -> str:
    if lift < 0.15:
        return "holds one mood throughout, no surprises"
    pos = "early" if climax_frac < 0.4 else ("at two-thirds" if climax_frac < 0.8 else "right at the end")
    verb = "builds and breaks" if lift > 0.35 else "lifts gently"
    tail = ", never quite resolving" if lift > 0.35 and climax_frac >= 0.8 else ""
    return f"{verb}, peaking {pos}{tail}"


def caption(feat: dict) -> str:
    """One honest sentence about a track from its measured features."""
    bpm = float(feat.get("bpm", 0) or 0)
    key = str(feat.get("key") or feat.get("musical_key") or "?")
    mode = str(feat.get("mode", "major"))
    head = f"{_tempo_word(bpm)} {key} {mode} {_mood_word(mode, float(feat.get('valence', 0.5) or 0.5))} song".replace("  ", " ")
    tex = _texture_word(float(feat.get("energy", 0.5) or 0.5),
                        float(feat.get("brightness", 0.5) or 0.5),
                        float(feat.get("harmonic_clarity", 0.5) or 0.5))
    arc = _arc_word(float(feat.get("lift", 0.0) or 0.0),
                    float(feat.get("climax_frac", 0.5) or 0.5))
    bpm_bit = f" (~{bpm:.0f} BPM)" if bpm > 0 else ""
    out = f"{head}{bpm_bit}, {tex}. It {arc}."
    try:
        vp = float(feat.get("voice_pct", 0) or 0)
    except Exception:
        vp = 0.0
    if vp > 0.05:
        enter = feat.get("voice_enter_s", -1)
        try:
            enter = float(enter)
        except Exception:
            enter = -1
        when = f"enters ~{enter:.0f}s" if enter and enter >= 0 else "enters early"
        reg = str(feat.get("register", "") or "")
        peak = feat.get("peak_f0", 0)
        try:
            peak = float(peak)
        except Exception:
            peak = 0
        cry = f", cries to {peak:.0f} Hz at the peak" if peak > 0 else ""
        out += f" The voice {when}, {reg or 'quiet'} register{cry}."
    return out


def caption_for(song_id: str, title: str = "") -> str:
    """Caption for a stored track; falls back gracefully when unheard."""
    try:
        row = _get_audio(str(song_id))
    except Exception:
        row = None
    if not row:
        return f"{title or song_id}: not heard yet — no words, only guesses."
    s = caption(dict(row))
    arc_row = None
    try:
        from .store import get_arc
        arc_row = get_arc(str(song_id))
    except Exception:
        arc_row = None
    if arc_row and arc_row.get("arousal"):
        s += f" Arc: {arc_row['arousal']}."
    if title:
        s = f"{title}: {s[0].lower() + s[1:]}"
    return s
