"""voice.py — the singer, not the song.

 autocorrelation F0 track → periodicity → vocal mask → vibrato
 (the throat-vs-wood signature) → melody stats → synced words
 (lrclib, free, no key) aligned to sections.

 Numpy + stdlib only. Honest limits, stated where they bite: on a
 sparse guitar+voice mix, autocorrelation sees all pitched sound, so
 "voice" means pitched-presence weighted by vibrato and presence band.
"""

import json
import logging
import re
import urllib.parse
import urllib.request

import numpy as np

logger = logging.getLogger("parlay.audio.voice")

_LRCLIB = "https://lrclib.net/api/search"


def f0_track(y: np.ndarray, sr: int = 16000, frame: int = 1024, hop: int = 256,
             fmin: float = 60, fmax: float = 600) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-frame F0 + periodicity. Short 64ms window tracks vibrato (5-8Hz)
    instead of smearing it; 16ms hop follows the throat closely."""
    y = np.asarray(y, dtype=np.float64)
    n = 1 + max(len(y) - frame, 0) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx]
    fr -= fr.mean(axis=1, keepdims=True)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0 = np.zeros(n)
    per = np.zeros(n)
    for i in range(n):
        x = fr[i]
        e = float(x @ x)
        if e < 1e-9:
            continue
        ac = np.correlate(x, x, mode="full")[frame - 1:]
        ac /= (e + 1e-12)
        if hi >= len(ac):
            continue
        j = int(lo + np.argmax(ac[lo:hi + 1]))
        if 0 < j < len(ac) - 1:
            a, b, cc = ac[j - 1], ac[j], ac[j + 1]
            d = a - 2 * b + cc
            if abs(d) > 1e-9:
                j = j + 0.5 * (a - cc) / d
        f0[i] = sr / max(j, 1e-6)
        per[i] = float(np.clip(ac[int(round(min(max(j, 0), len(ac) - 1)))], 0, 1))
    return f0, per, hop / sr


def vocal_mask(f0: np.ndarray, per: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Voiced frames + vibrato frames (4-9Hz wobble = throat, not wood)."""
    voiced = (per > 0.55) & (f0 > 80) & (f0 < 520)
    w = max(int(0.6 / dt), 4)
    vib = np.zeros(len(f0), dtype=bool)
    for i in range(0, len(f0) - w, max(w // 2, 1)):
        seg = f0[i:i + w]
        v = seg[voiced[i:i + w]]
        if len(v) > w // 2 and v.mean() > 0:
            detr = v - np.linspace(v[0], v[-1], len(v))
            if detr.std() / v.mean() > 0.008:
                vib[i:i + w] = True
    return voiced, vib


def presence_10s(voiced: np.ndarray, dt: float, total_s: float) -> list[float]:
    """Share of voiced frames per 10s slice — where the singer stands."""
    out = []
    per_s = 1 / dt
    for t in range(0, int(total_s), 10):
        i0, i1 = int(t * per_s), int(min((t + 10) * per_s, len(voiced)))
        out.append(round(float(voiced[i0:i1].mean()) if i1 > i0 else 0.0, 3))
    return out


def melody_stats(f0: np.ndarray, voiced: np.ndarray) -> dict:
    """Height and wingspan of the singing, from voiced frames only."""
    v = f0[voiced & (f0 > 80) & (f0 < 600)]
    if len(v) < 8:
        return {"voice_pct": round(float(voiced.mean()), 3), "median_f0": 0.0,
                "f0_lo": 0.0, "f0_hi": 0.0, "peak_f0": 0.0, "register": "silent"}
    lo, med, hi = float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))
    reg = "low" if med < 180 else ("mid" if med < 300 else "high")
    return {"voice_pct": round(float(voiced.mean()), 3), "median_f0": round(med, 1),
            "f0_lo": round(lo, 1), "f0_hi": round(hi, 1),
            "peak_f0": round(float(v.max()), 1), "register": reg}


def analyze_voice(y: np.ndarray, sr: int = 16000) -> dict:
    """Full vocal hearing. Returns everything about the singer, or silence."""
    f0, per, dt = f0_track(np.asarray(y, dtype=np.float32), sr)
    voiced, vib = vocal_mask(f0, per, dt)
    total_s = len(y) / sr
    stats = melody_stats(f0, voiced)
    # First entrance: first 10s slice with real presence.
    pres = presence_10s(voiced, dt, total_s)
    enter = next((i * 10 for i, p in enumerate(pres) if p > 0.08), None)
    # Melody contour: median voiced F0 per 10s (0 = nobody singing).
    contour = []
    per_s = 1 / dt
    for t in range(0, int(total_s), 10):
        i0, i1 = int(t * per_s), int(min((t + 10) * per_s, len(f0)))
        vv = f0[i0:i1][voiced[i0:i1] & (f0[i0:i1] > 80) & (f0[i0:i1] < 600)]
        contour.append(round(float(np.median(vv)) if len(vv) > 2 else 0.0, 1))
    return {**stats, "vibrato_pct": round(float(vib.mean()), 3),
            "voice_enter_s": enter, "presence_10s": pres, "melody_10s": contour}


def fetch_lyrics(artist: str, title: str, timeout: int = 25) -> dict:
    """Synced words from lrclib (free, no key). Returns {plain, synced, source}."""
    clean = re.sub(r"[\(\[].*?[\)\]]", " ", title or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    # Drop a leading "Artist - " when the artist is passed separately too.
    if artist and clean.lower().startswith(artist.lower() + " -"):
        clean = clean[len(artist) + 3:].strip()
    q = urllib.parse.urlencode({"q": f"{artist} {clean}".strip() or clean})
    req = urllib.request.Request(f"{_LRCLIB}?{q}", headers={"User-Agent": "Parlay-ML/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hits = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"plain": "", "synced": [], "source": "", "error": str(e)[:120]}
    if not hits:
        return {"plain": "", "synced": [], "source": "lrclib", "error": "no hits"}
    hits.sort(key=lambda h: (not bool(h.get("syncedLyrics")), len(h.get("trackName", ""))))
    h = hits[0]
    synced = []
    for ln in (h.get("syncedLyrics") or "").splitlines():
        m = re.match(r"\[(\d+):(\d+\.\d+)\](.*)", ln)
        if m and m.group(3).strip():
            synced.append({"t": round(int(m.group(1)) * 60 + float(m.group(2)), 2),
                           "line": m.group(3).strip()})
    return {"plain": h.get("plainLyrics") or "", "synced": synced,
            "source": "lrclib", "track": h.get("trackName", ""),
            "artist": h.get("artistName", "")}


def align_lyrics(synced: list[dict], arousal_10s: list[float]) -> list[dict]:
    """Label each line by the feeling under it: verse / lift / chorus."""
    out = []
    for ln in synced:
        i = min(int(ln["t"] // 10), len(arousal_10s) - 1) if arousal_10s else -1
        a = arousal_10s[i] if 0 <= i else 0.3
        section = "verse" if a < 0.3 else ("lift" if a < 0.5 else "chorus")
        out.append({**ln, "arousal": a, "section": section})
    return out


def fetch_and_store_lyrics(song_id: str, title: str, artist: str) -> dict:
    """Hear the words: fetch synced lyrics and cache them. Network needed."""
    from .store import save_lyrics
    lyrics = fetch_lyrics(artist, title)
    if lyrics.get("synced") or lyrics.get("plain"):
        try:
            save_lyrics(str(song_id), lyrics)
        except Exception as e:
            logger.warning("lyrics cache failed: %s", e)
    return lyrics
