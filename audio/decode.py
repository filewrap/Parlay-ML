"""decode.py — wav bytes → float32 mono numpy. No audioop needed.

Parses the PCM frames with numpy.frombuffer (8/16/24/32-bit supported),
resamples by linear interpolation if the file isn't 16kHz, mixes down
to mono. Everything the feature extractor needs, nothing more.
"""

import wave

import numpy as np

TARGET_SR = 16000


def read_wav_mono(path: str, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Read any PCM wav → (float32 mono in [-1,1], sample_rate)."""
    with wave.open(path, "rb") as w:
        n_chan, sampwidth, sr, n_frames, _, _ = w.getparams()
        raw = w.readframes(n_frames)
    if sampwidth == 1:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        y = (v.astype(np.float32)) / 8388608.0
    elif sampwidth == 4:
        y = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")
    if n_chan > 1:
        y = y.reshape(-1, n_chan).mean(axis=1).astype(np.float32)
    if sr != target_sr:
        n_out = int(round(len(y) * target_sr / sr))
        old = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
        new = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
        y = np.interp(new, old, y).astype(np.float32)
        sr = target_sr
    # Guard against DC offset + clipping.
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y, sr
