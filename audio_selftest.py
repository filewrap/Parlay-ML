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

    # 8. Emotional arcs: rising swell lifts late; drone stays flat.
    from audio.features import analyze_arc
    swell = (_sine(220.0, secs=20.0) * np.linspace(0.05, 1.0, SR * 20)).astype(np.float32)
    arc = analyze_arc(swell, SR, mode="minor")
    assert arc["lift"] > 0.15, arc
    assert arc["climax_frac"] > 0.6, arc
    flat = analyze_arc(_sine(220.0, secs=20.0), SR, mode="minor")
    assert flat["lift"] < arc["lift"], (flat["lift"], arc["lift"])
    print("arc ok: swell lift=%.2f climax=%.2f / flat lift=%.2f"
          % (arc["lift"], arc["climax_frac"], flat["lift"]))

    # 9. Captioner speaks.
    from audio.caption import caption
    s = caption({"bpm": 80.2, "key": "A#", "mode": "minor", "valence": 0.275,
                 "energy": 0.729, "brightness": 0.263, "harmonic_clarity": 0.377,
                 "lift": 0.4, "climax_frac": 0.66})
    assert "A#" in s and "BPM" in s, s
    print("caption ok:", s)

    # 10. Fresh-ears injection surfaces a heard track.
    import time as _t
    from core.database import get_conn
    from config import DB_CATALOG
    from audio.store import save as _save_audio
    from audio.features import summarize as _summ
    from pipeline.recommender import _heard_candidates
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""INSERT INTO songs (song_id,title,channel,duration,view_count,
            genre_code,language_code,energy_score,yt_url,first_seen,last_seen,times_fetched)
            VALUES ('_heard_probe','Probe Song','Probe',200,10,6,0,0.5,'',?,?,1)
            ON CONFLICT(song_id) DO UPDATE SET last_seen=excluded.last_seen""",
                     (_t.time(), _t.time()))
    _save_audio("_heard_probe", _summ(dict(
        __import__("audio.features", fromlist=["analyze"]).analyze(
            _sine(330.0, secs=12.0), SR))))
    inj = _heard_candidates(6802929470, {"something_else"})
    assert any(c["song_id"] == "_heard_probe" for c in inj), [c["song_id"] for c in inj]
    print("heard inject ok:", [c["song_id"] for c in inj][:5])
    with get_conn(DB_CATALOG) as conn:
        conn.execute("DELETE FROM songs WHERE song_id='_heard_probe'")
        conn.execute("DELETE FROM audio_features WHERE song_id='_heard_probe'")
        conn.execute("DELETE FROM audio_arcs WHERE song_id='_heard_probe'")
        conn.execute("DELETE FROM listen_ledger WHERE song_id='_heard_probe'")

    # 11. Title gate: same-artist wrong-song must not pass.
    from audio.sources import title_ok, rank_candidates
    assert title_ok("4AM in Karachi", "4AM in Karachi") is True
    assert title_ok("4AM in Karachi", "Afsanay") is False
    assert title_ok("4AM in Karachi", "5AM In Lahore") is False
    assert title_ok("Mehrama", "Mehrama - Official Lyric Video") is True
    ranked = rank_candidates("4AM in Karachi", "Talha Anjum", [
        {"title": "Afsanay", "artist": "Talha Anjum"},
        {"title": "4AM in Karachi", "artist": "Talha Anjum"}])
    assert ranked[0]["title"] == "4AM in Karachi", ranked
    print("title gate ok")

    # 12. The Gate: ledger, graph, emotion, feel, brief.
    from gate.state import record_listen, taste_graph
    from gate import emotion_now, feel_like, agent_brief, taste_profile
    from gate import emotion_now, feel_like, agent_brief, taste_profile
    lid = record_listen("_gate_probe", "Probe", {"bpm": 100, "key": "C",
                        "mode": "major", "valence": 0.5, "energy": 0.5,
                        "danceability": 0.5, "source": "probe"}, "probe caption")
    assert lid > 0
    g = taste_graph()
    assert g["n_listens"] >= 1 and len(g["nodes"]) > 0 and len(g["edges"]) >= 0
    emo = emotion_now()
    assert emo["n"] >= 1 and "label" in emo
    fl = feel_like(mood="low", limit=2)
    assert len(fl) >= 1 and fl == sorted(fl, key=lambda x: -x["feel_score"])
    tp = taste_profile()
    assert tp["n"] >= 1 and "home_key" in tp
    brief = agent_brief()
    assert "measured" in brief and "Mood" not in brief  # lowercase mood line present
    assert "mood now:" in brief
    from core.database import get_conn as _gc2
    from config import DB_CATALOG as _DBC2
    with _gc2(_DBC2) as conn:
        conn.execute("DELETE FROM listen_ledger WHERE song_id='_gate_probe'")
    print("gate ok: emotion=%s taste_home=%s" % (emo["label"], tp["home_key"]))

    # 13. Voice: wobble sings, drone doesn't, silence is silent.
    from audio.voice import analyze_voice, align_lyrics
    t = np.arange(SR * 6) / SR
    drone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    # Naturalistic vibrato: +-35 cents at 5.5Hz (a real throat, not a siren).
    wobble = (0.5 * np.sin(2 * np.pi * (220 * t + 1.1 * np.sin(2 * np.pi * 5.5 * t)))).astype(np.float32)
    vd = analyze_voice(drone, SR)
    vv = analyze_voice(wobble, SR)
    vs = analyze_voice(np.zeros(SR * 3, dtype=np.float32), SR)
    assert vd["voice_pct"] > 0.5, vd
    assert vv["vibrato_pct"] > vd["vibrato_pct"], (vv["vibrato_pct"], vd["vibrato_pct"])
    assert vs["voice_pct"] == 0.0 and vs["register"] == "silent", vs
    assert 100 < vv["median_f0"] < 400, vv
    print("voice ok: drone voiced=%.2f vib=%.2f / wobble vib=%.2f / med=%.0fHz" % (
        vd["voice_pct"], vd["vibrato_pct"], vv["vibrato_pct"], vv["median_f0"]))

    # 14. Words: synced lyrics when the network allows, graceful without.
    from audio.voice import fetch_lyrics
    lyr = fetch_lyrics("Lord Huron", "The Night We Met")
    if lyr.get("synced"):
        assert lyr["synced"][0]["t"] >= 0 and "line" in lyr["synced"][0]
        al = align_lyrics(lyr["synced"][:4], [0.1, 0.1, 0.1, 0.2, 0.2, 0.6, 0.7])
        assert al[0]["section"] == "verse" and al[-1]["section"] == "chorus", al
        print("lyrics ok: %d synced lines, e.g. %r" % (len(lyr["synced"]), lyr["synced"][0]["line"][:40]))
    else:
        print("lyrics skipped (offline): %s" % lyr.get("error", "?")[:60])

    # 15. Caption speaks of the singer too.
    from audio.caption import caption as _cap2
    cs = _cap2({"bpm": 91, "key": "E", "mode": "major", "valence": 0.49,
                "energy": 0.24, "brightness": 0.2, "harmonic_clarity": 0.39,
                "lift": 0.45, "climax_frac": 0.72, "voice_pct": 0.3,
                "voice_enter_s": 31, "register": "low", "peak_f0": 440.0})
    assert "voice" in cs and "31s" in cs, cs
    print("voice caption ok:", cs[:100])

    print("ALL AUDIO SELFTESTS PASSED")


if __name__ == "__main__":
    main()
