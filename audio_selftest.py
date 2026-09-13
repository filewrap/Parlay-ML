#!/usr/bin/env python3
"""audio_selftest.py — run with plain python, no pytest needed.

Proves the machine can listen: decode roundtrip, A440 root, 120 BPM
clicks, silence safety, store roundtrip, scheduler wiring.
"""

import os
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

SR = 16000


def _sine(freq: float, secs: float = 4.0) -> np.ndarray:
    t = np.arange(int(SR * secs)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def main() -> None:
    from audio.decode import read_wav_mono
    from audio.features import analyze
    from audio.store import count, ensure_schema, get, save

    # 1. Decode roundtrip (16-bit + 8-bit paths).
    y = _sine(440.0)
    for width, dtype, scale in ((2, np.int16, 32767), (1, np.uint8, None)):
        p = tempfile.mktemp(suffix=".wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(width)
            w.setframerate(SR)
            w.writeframes(((y * 127 + 128).astype(dtype).tobytes() if width == 1
                           else (y * scale).astype(dtype).tobytes()))
        y2, sr2 = read_wav_mono(p)
        assert sr2 == SR and abs(len(y2) - len(y)) < 2, (sr2, len(y2))
        os.remove(p)
    print("decode roundtrip ok")

    # 2. A440 → root A (tonic or runner-up: dominant shadowing is allowed).
    from audio.features import estimate_key, _chroma_mean, _stft_mag
    f = analyze(y, SR)
    assert f["key"] == "A" or f.get("key_alt") == "A", f
    print("A440 key ok:", f["key"], f["mode"], "alt:", f.get("key_alt"))

    # 3. Click track → ~120 BPM.
    click = np.zeros(SR * 12, dtype=np.float32)
    for k in range(24):
        click[int(k * 0.5 * SR):int(k * 0.5 * SR) + 200] = 1.0
    f2 = analyze(click, SR)
    assert abs(f2["bpm"] - 120) < 8, f2["bpm"]
    print("click tempo ok:", f2["bpm"])

    # 4. Silence → no crash.
    f3 = analyze(np.zeros(SR * 3, dtype=np.float32), SR)
    assert f3["bpm"] == 0.0
    print("silence ok")

    # 5. Store roundtrip.
    ensure_schema()
    from audio.features import summarize
    save("_selftest", summarize(f))
    row = get("_selftest")
    assert row and row["musical_key"] == f["key"], row
    assert count() >= 1
    print("store ok, count:", count())

    # 6. Scheduler has the 02:00 listening hour.
    from scheduler.jobs import build_scheduler
    s = build_scheduler([1])
    ids = sorted(j.id for j in s.get_jobs())
    assert "audio_listen" in ids, ids
    print("scheduler ok:", ids)

    # 7. Pipeline measured bonus prefers ears over guesses.
    from pipeline.recommender import _measured_math_bonus
    b, heard = _measured_math_bonus("helix_test", {"title": "x", "duration": 200})
    assert heard is True, (b, heard)
    b2, heard2 = _measured_math_bonus("never_heard_xyz", {"title": "Choir Dance Anthem", "duration": 210})
    assert heard2 is False and b2 != 0.0, (b2, heard2)
    print("measured bonus ok:", b, "vs guessed:", b2)

    print("ALL AUDIO SELFTESTS PASSED")


if __name__ == "__main__":
    main()
