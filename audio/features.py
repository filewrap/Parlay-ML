"""features.py — numpy-only hearing: chroma/key/tempo/brightness.

How the machine listens (all native DSP, seconds per 90s clip on CPU):
  frames  Hann 2048 / hop 1024 @16kHz → magnitude spectrogram (rFFT)
  chroma  FFT bins → pitch classes (A440 ref), magnitude-weighted, folded
  key     Krumhansl-Schmuckler correlation: 12 roots × major/minor
  tempo   spectral-flux onset envelope → autocorrelation, 60–200 BPM
  colour  centroid / rolloff / flatness / ZCR / RMS → brightness, energy

Output is ~24 floats: small enough for SQLite, rich enough to beat
title-keyword guessing (measured BPM replaces `groove_score` priors,
measured key replaces genre-code harmony hints).
"""

import math

import numpy as np

FRAME = 2048
HOP = 1024

# Krumhansl-Schmuckler tonal profiles (hearing science, 1982).
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _stft_mag(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram: (n_frames, FRAME//2+1) float32."""
    if len(y) < FRAME:
        y = np.pad(y, (0, FRAME - len(y)))
    win = np.hanning(FRAME).astype(np.float32)
    n = 1 + (len(y) - FRAME) // HOP
    idx = (np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None])
    frames = y[idx] * win[None, :]
    return np.abs(np.fft.rfft(frames, axis=1))


def _chroma_mean(mag: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Mean 12-dim pitch-class profile from a magnitude spectrogram."""
    n_bins = mag.shape[1]
    freqs = np.arange(n_bins) * (sr / FRAME)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-9) / 440.0)
    pc = np.round(midi).astype(int) % 12
    valid = (freqs >= 55.0) & (freqs <= 2093.0)  # C1..C7, music lives here
    comp = np.log10(1.0 + 50.0 * mag[:, valid].astype(np.float64))
    chroma = np.zeros(12)
    np.add.at(chroma, pc[valid], comp.sum(axis=0))
    s = chroma.sum()
    return (chroma / s).astype(np.float32) if s > 0 else chroma.astype(np.float32)


def estimate_key(chroma: np.ndarray) -> tuple[str, str, float, str]:
    """Key via Krumhansl correlation blended with a triad-template anchor.

    Full-scale Krumhansl profiles overlap heavily between fifth-related
    keys (C vs G share 6 of 7 scale tones), so the dominant shadows the
    tonic. Triad templates (3-hot) anchor it: the chord tones themselves
    must be loud. Blend, then a root-prominence tiebreak.

    Returns (root, mode, strength, alt): `alt` is the runner-up root,
    because folded chroma of stacked harmonics genuinely favours the
    dominant — the true tonic is often #2. Models get both candidates.
    """
    c = np.asarray(chroma, dtype=np.float64)
    if c.sum() <= 0:
        return "C", "major", 0.0, "G"
    raw = c / (c.sum() + 1e-12)
    c = (c - c.mean()) / (c.std() + 1e-9)
    best: tuple[str, str, float] = ("C", "major", -9.0)
    scored: list[tuple[float, int, str]] = []
    for mode, prof, third in (("major", KRUMHANSL_MAJOR, 4), ("minor", KRUMHANSL_MINOR, 3)):
        p = (prof - prof.mean()) / (prof.std() + 1e-9)
        for root in range(12):
            r = float(np.corrcoef(c, np.roll(p, root))[0, 1])
            tones = {root, (root + third) % 12, (root + 7) % 12}
            tri = float(np.mean([raw[i] for i in tones])
                        - np.mean([raw[i] for i in range(12) if i not in tones]))
            score = 0.5 * r + 2.0 * tri
            scored.append((score, root, mode))
            if score > best[2]:
                best = (PITCH_NAMES[root], mode, score)
    # Root-prominence tiebreak: the dominant often shadows the tonic
    # (shared triad notes + strong 2nd harmonic). Among near-tied
    # candidates, prefer the one whose root actually sounds loudest.
    root, mode, r = best
    ranked = sorted(scored, reverse=True)
    alt = next((PITCH_NAMES[rt] for _, rt, _ in ranked if PITCH_NAMES[rt] != root), root)
    near = [(rr, rt, m) for rr, rt, m in scored if rr > r - 0.05]
    if len(near) > 1:
        near.sort(key=lambda t: (raw[t[1]], t[0]), reverse=True)
        r, root_idx, mode = near[0]
        root = PITCH_NAMES[root_idx]
    strength = max(0.0, min(1.0, (r + 0.4) / 1.4))
    return root, mode, round(strength, 3), alt


def estimate_tempo(mag: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """Spectral-flux onset envelope → (bpm, strength). 60–200 range."""
    flux = np.maximum(0.0, np.diff(mag.astype(np.float64), axis=0)).sum(axis=1)
    flux = flux - flux.mean()
    if float(np.dot(flux, flux)) < 1e-12:
        return 0.0, 0.0
    fps = sr / HOP  # frames per second (~15.6)
    lo = max(2, int(round(60.0 * fps / 200.0)))   # 200 BPM → shortest period
    hi = min(len(flux) // 2, int(round(60.0 * fps / 60.0)))  # 60 BPM
    if hi <= lo:
        return 0.0, 0.0
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag = int(lo + np.argmax(ac[lo:hi + 1]))
    # Parabolic interpolation for sub-frame accuracy.
    if 0 < lag < len(ac) - 1:
        a, b, cc = ac[lag - 1], ac[lag], ac[lag + 1]
        denom = a - 2 * b + cc
        if abs(denom) > 1e-9:
            lag = lag + 0.5 * (a - cc) / denom
    bpm = 60.0 * fps / max(lag, 1e-6)
    # Fold into 60–200 (octave errors are the classic tempo trap).
    while bpm < 60.0:
        bpm *= 2.0
    while bpm > 200.0:
        bpm /= 2.0
    return round(float(bpm), 1), round(float(np.clip(ac[int(round(lag))], 0, 1)), 3)


def spectral_colour(mag: np.ndarray, y: np.ndarray, sr: int = 16000) -> dict:
    """Brightness/energy/noisiness descriptors from spectrum + waveform."""
    mean_spec = mag.mean(axis=0).astype(np.float64) + 1e-12
    freqs = np.arange(len(mean_spec)) * (sr / FRAME)
    centroid = float((freqs * mean_spec).sum() / mean_spec.sum())
    cum = np.cumsum(mean_spec)
    rolloff = float(freqs[int(np.searchsorted(cum, 0.85 * cum[-1]))])
    flatness = float(np.exp(np.log(mean_spec).mean()) / mean_spec.mean())
    zc = np.mean(np.abs(np.diff(np.signbit(y)))) * 2.0
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return {
        "centroid_hz": round(centroid, 1),
        "brightness": round(max(0.0, min(1.0, centroid / 6000.0)), 3),
        "rolloff_hz": round(rolloff, 1),
        "flatness": round(float(np.clip(flatness, 0, 1)), 3),
        "zcr": round(float(zc), 3),
        "rms": round(rms, 4),
    }


def analyze(y: np.ndarray, sr: int = 16000) -> dict:
    """Hear one waveform → feature dict. The machine listening."""
    mag = _stft_mag(np.asarray(y, dtype=np.float32))
    chroma = _chroma_mean(mag, sr)
    root, mode, key_strength, alt = estimate_key(chroma)
    bpm, tempo_strength = estimate_tempo(mag, sr)
    colour = spectral_colour(mag, np.asarray(y, dtype=np.float32), sr)
    # Soft-saturation energy: mastered music sits ~0.15–0.30 RMS, quiet
    # bedroom recordings ~0.03. 1-exp(-4·rms) maps that to 0.11–0.70.
    energy = float(1.0 - math.exp(-4.0 * colour["rms"]))
    sweet = math.exp(-0.5 * ((bpm - 120.0) / 40.0) ** 2) if bpm else 0.5
    danceability = round(max(0.0, min(1.0, 0.5 * tempo_strength + 0.3 * energy + 0.2 * sweet)), 3)
    valence = round(max(0.0, min(1.0,
                                 0.45 + (0.15 if mode == "major" else -0.15)
                                 + 0.25 * (colour["brightness"] - 0.5)
                                 + 0.15 * (energy - 0.5))), 3)
    # Harmonic clarity: key strength minus out-of-key clutter.
    top3 = sorted(chroma, reverse=True)[:3]
    clarity = round(max(0.0, min(1.0, 0.5 * key_strength + 0.5 * float(sum(top3)))), 3)
    return {
        "bpm": bpm, "tempo_strength": tempo_strength,
        "key": root, "mode": mode, "key_strength": key_strength,
        "key_alt": alt,
        "chroma": [round(float(v), 4) for v in chroma],
        "danceability": danceability, "valence": valence,
        "energy": round(energy, 3), "harmonic_clarity": clarity,
        **colour,
    }


def summarize(feat: dict) -> dict:
    """Compact DB row from a full analysis (chroma folded to string)."""
    return {
        "bpm": feat["bpm"], "key": feat["key"], "mode": feat["mode"],
        "key_alt": feat.get("key_alt", ""),
        "danceability": feat["danceability"], "valence": feat["valence"],
        "energy": feat["energy"], "brightness": feat["brightness"],
        "harmonic_clarity": feat["harmonic_clarity"],
        "tempo_strength": feat["tempo_strength"],
        "key_strength": feat["key_strength"],
        "chroma": ",".join(f"{v:.3f}" for v in feat["chroma"]),
    }
