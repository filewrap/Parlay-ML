"""audio package — let the machine listen.

Native pipeline, no librosa/Essentia (both too heavy for the 8GB box):
ffmpeg + yt-dlp subprocesses for capture, numpy-only DSP for hearing,
SQLite for memory. Fetches 90s clips, hears ~24 numbers, deletes audio.

    fetch.py    capture a 90s clip → mono 16kHz wav (temp file)
    decode.py   wav → float32 numpy (no audioop; gone in Python 3.13+)
    features.py hear chroma/key/tempo/brightness from samples
    store.py    audio_features table in catalog.db
    listener.py priority-queue orchestrator (owner taste first)
"""

from .features import analyze, summarize

__all__ = ["analyze", "summarize"]
