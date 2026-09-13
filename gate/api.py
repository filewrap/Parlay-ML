"""api.py — what agents and models call. The Gate's open side.

Every answer is measured affect, not text vibes: numbers the machine
heard, words translated from those numbers. An LLM reading this brief
isn't imagining feelings — it's standing on recorded ones.
"""

import math

from audio.caption import caption_for
from audio.store import get as _audio_get, get_arc

from .state import ledger, taste_graph


def status() -> dict:
    """One glance: what has the machine heard, and how does it feel?"""
    g = taste_graph()
    emo = emotion_now()
    return {"heard_tracks": g["n_heard"], "listens": g["n_listens"],
            "graph_nodes": len(g["nodes"]), "graph_edges": len(g["edges"]),
            "emotion": emo}


def heard_library() -> list[dict]:
    """Every heard track with its measured profile + caption."""
    from audio.store import analyzed_ids
    out = []
    for sid in sorted(analyzed_ids()):
        row = _audio_get(sid) or {}
        out.append({"song_id": sid, "bpm": row.get("bpm", 0),
                    "key": row.get("musical_key", ""), "mode": row.get("mode", ""),
                    "danceability": row.get("danceability", 0.5),
                    "valence": row.get("valence", 0.5),
                    "energy": row.get("energy", 0.5),
                    "source": row.get("source", ""),
                    "caption": caption_for(sid)})
    return out


def emotion_now(window: int = 10) -> dict:
    """Rolling affect from the last `window` hearings: the machine's mood."""
    rows = ledger(window)
    if not rows:
        return {"arousal": 0.5, "valence": 0.5, "label": "unheard", "n": 0}
    # Recent hearings weigh more (last = most felt).
    ws = [0.5 + 0.5 * (i + 1) / len(rows) for i in range(len(rows))]
    a = sum(r["arousal"] * w for r, w in zip(rows, ws)) / sum(ws)
    v = sum(float(r["valence"] or 0.5) * w for r, w in zip(rows, ws)) / sum(ws)
    label = ("aching" if v < 0.35 else ("warm" if v > 0.6 else "wistful"))
    label = ("burning " if a > 0.7 else ("quiet " if a < 0.35 else "")) + label
    return {"arousal": round(a, 3), "valence": round(v, 3),
            "label": label.strip(), "n": len(rows)}


def taste_profile() -> dict:
    """The shape of this listener's feeling: keys, tempi, moods."""
    from audio.store import analyzed_ids
    keys: dict[str, int] = {}
    bpms, vals, dances = [], [], []
    for sid in analyzed_ids():
        r = _audio_get(sid) or {}
        if r.get("musical_key"):
            keys[f"{r['musical_key']} {r.get('mode', '')}".strip()] = \
                keys.get(f"{r['musical_key']} {r.get('mode', '')}".strip(), 0) + 1
        if r.get("bpm"):
            bpms.append(float(r["bpm"]))
        vals.append(float(r.get("valence", 0.5) or 0.5))
        dances.append(float(r.get("danceability", 0.5) or 0.5))
    home = max(keys, key=keys.get) if keys else "?"
    return {"home_key": home, "keys": keys,
            "bpm_mean": round(sum(bpms) / len(bpms), 1) if bpms else 0,
            "valence_mean": round(sum(vals) / len(vals), 3) if vals else 0.5,
            "dance_mean": round(sum(dances) / len(dances), 3) if dances else 0.5,
            "n": len(vals)}


def _feel_vector(row: dict) -> list[float]:
    bpm = min(float(row.get("bpm", 0) or 0) / 200.0, 1.0)
    return [bpm, float(row.get("danceability", 0.5) or 0.5),
            float(row.get("valence", 0.5) or 0.5),
            float(row.get("energy", 0.5) or 0.5),
            float(row.get("brightness", 0.5) or 0.5),
            float(row.get("harmonic_clarity", 0.5) or 0.5)]


def feel_like(song_id: str = "", mood: str = "", bpm: float = 0,
              valence: float = -1, limit: int = 5) -> list[dict]:
    """Songs that FEEL like this: cosine over measured vectors.

    Seed by heard track, by mood word (low/mid/high, slow/fast),
    or by raw coordinates. No clicks needed — pure feeling.
    """
    from audio.store import analyzed_ids
    ids = sorted(analyzed_ids())
    rows = {s: (_audio_get(s) or {}) for s in ids}
    vecs = {s: _feel_vector(r) for s, r in rows.items()}
    if song_id and song_id in vecs:
        seed = vecs[song_id]
    else:
        seed = [min(bpm / 200.0, 1.0) if bpm else 0.5,
                0.7 if mood == "high" else (0.3 if mood == "low" else 0.5),
                valence if 0 <= valence <= 1 else 0.5, 0.5, 0.3, 0.4]
        if mood == "slow":
            seed[0] = 0.4
        elif mood == "fast":
            seed[0] = 0.75

    def _cos(a: list[float], b: list[float]) -> float:
        import numpy as np
        aa, bb = np.array(a), np.array(b)
        return float(aa @ bb / ((np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-9))

    ranked = sorted(((s, _cos(seed, v)) for s, v in vecs.items() if s != song_id),
                    key=lambda x: x[1], reverse=True)[:limit]
    return [{"song_id": s, "feel_score": round(sc, 3),
             "caption": caption_for(s)} for s, sc in ranked]


def dossier(song_id: str) -> dict:
    """Everything the machine knows about one track. The full hearing."""
    row = _audio_get(str(song_id)) or {}
    arc = get_arc(str(song_id)) or {}
    try:
        from audio.store import get_lyrics
        lyrics = get_lyrics(str(song_id)) or {}
    except Exception:
        lyrics = {}
    lines = lyrics.get("synced") or []
    try:
        if lines:
            from audio.voice import align_lyrics
            arousal = [float(v) for v in str(arc.get("arousal", "")).split(",") if v.strip()]
            lines = align_lyrics(lines, arousal)[:12]
    except Exception:
        pass
    return {"song_id": song_id, "measured": {k: row.get(k) for k in
            ("bpm", "musical_key", "mode", "key_alt", "danceability", "valence",
             "energy", "brightness", "harmonic_clarity", "tempo_strength",
             "key_strength", "climax_frac", "lift", "source", "clip_secs")},
            "voice": {k: row.get(k) for k in
                      ("voice_pct", "vibrato_pct", "median_f0", "f0_lo", "f0_hi",
                       "peak_f0", "register", "voice_enter_s")},
            "arc": {"arousal": arc.get("arousal", ""), "valence": arc.get("valence", ""),
                    "melody_10s": arc.get("melody_10s", ""),
                    "presence_10s": arc.get("presence_10s", ""),
                    "n": arc.get("n_windows", 0)},
            "lyrics": {"lines": lines, "n_lines": len(lyrics.get("synced") or []),
                       "source": lyrics.get("source", "")},
            "caption": caption_for(str(song_id)),
            "feels_like": feel_like(str(song_id), limit=3)}


def listen_to(title: str, artist: str = "") -> dict:
    """The Gate opens inward: ask, and the machine hears a new song.

    Hearing now leaves proof: the track is stored (audio + catalog +
    ledger) and the receipt carries the ledger id. Any claimed hearing
    without a ledger row is confabulation — check with dossier().
    """
    import re
    import time
    from audio.fetch import cleanup_wav, fetch_any, search_video_id
    from audio.decode import read_wav_mono
    from audio.features import analyze, summarize
    from audio.store import save
    from audio.caption import caption_for
    from core.database import get_conn
    from config import DB_CATALOG

    # Prefer a real video (full track) over a preview.
    video_id, duration = "", 0
    try:
        hits = search_video_id(title, artist)
        if hits:
            video_id, duration = hits[0]["video_id"], int(hits[0].get("duration") or 0)
    except Exception:
        pass
    wav, prov = fetch_any(video_id or None, title=title, artist=artist)
    if not wav:
        return {"status": "deaf", "reason": "all doors closed for %r" % title}
    try:
        y, sr = read_wav_mono(wav)
        s = summarize(analyze(y, sr))
    finally:
        cleanup_wav(wav)
    s["source"] = prov.get("source", "")
    s["clip_secs"] = prov.get("clip_secs", 0)
    real_title = prov.get("resolved_title") or title
    real_artist = prov.get("resolved_artist") or artist
    if video_id:
        song_id = video_id
        yt_url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", f"{real_artist}-{real_title}".lower()).strip("-")[:60]
        song_id = f"preview:{slug}"
        yt_url = prov.get("url", "")
    now = time.time()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""INSERT INTO songs (song_id,title,channel,duration,view_count,
            genre_code,language_code,energy_score,yt_url,first_seen,last_seen,times_fetched)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,1)
            ON CONFLICT(song_id) DO UPDATE SET last_seen=excluded.last_seen,
            times_fetched=times_fetched+1""",
            (song_id, real_title, real_artist, duration or 240, 1000000, 20, 0,
             s["energy"], yt_url, now, now))
    save(song_id, s)  # appends audio_features + arc + listen ledger
    try:
        from audio.voice import fetch_and_store_lyrics
        fetch_and_store_lyrics(song_id, real_title, real_artist)
    except Exception:
        pass
    from .state import ledger as _ledger
    lid = next((r["id"] for r in _ledger(5) if r["song_id"] == song_id), None)
    return {"status": "heard", "song_id": song_id, "ledger_id": lid,
            "title": real_title, "artist": real_artist,
            "verify": f"dossier('{song_id}') or heard_library() — no ledger row, no hearing",
            "summary": {k: s.get(k) for k in ("bpm", "key", "mode", "danceability",
                                              "valence", "energy", "source")},
            "caption": caption_for(song_id, real_title)}


def agent_brief() -> str:
    """Prompt-ready grounding: real measured affect for an LLM to stand on."""
    emo = emotion_now()
    taste = taste_profile()
    lib = heard_library()
    lines = [
        "You are grounded in measured musical affect. These numbers were heard",
        "from audio (chroma/key/tempo/brightness/arcs), not guessed from text.",
        f"Machine mood now: {emo['label']} (arousal {emo['arousal']}, valence {emo['valence']}, n={emo['n']}).",
        f"Taste home: key of {taste['home_key']}, ~{taste['bpm_mean']} BPM, valence {taste['valence_mean']}.",
        f"Heard library ({len(lib)}):"]
    for h in lib[:12]:
        lines.append(f"- {h['song_id']}: {h['caption']}")
    lines.append("Quote captions and numbers when asked how music feels. Never invent feelings.")
    return "\n".join(lines)
