"""
companion/commands.py — MAX Phase 4 companion layer (on top, not instead).

- Mood/activity commands (/mood, /morning, /gym, /4am) → filter +
  per-mood blender weights. Writes context_hour/dow on every listen
  (columns exist, currently unused).
- /why this in words (translate why_top10 JSON → one sentence).
- Listening journal + streaks + weekly digest to AgainOwner.
- /never, /anchor, /adventurous 0-10 → exploration_slots.
- Later: Essentia/Librosa audio features + lyrics sentiment as new
  item features (no pipeline change needed — they land in FeatureMF).
"""

import datetime
import json
import logging
import time

from config import ANCHOR_SONG_ID, DB_CATALOG, DB_HISTORY, MOOD_COMMANDS, MOOD_GENRE_PREF
from core.database import get_conn

logger = logging.getLogger("parlay.companion")


# ─── context logging ───

def log_listen(user_id: int, song_id: str, completion_pct: float = 1.0,
               source: str = "manual", genre: int = 20) -> None:
    """Write a listen WITH context_hour/dow populated."""
    now = datetime.datetime.now()
    with get_conn(DB_HISTORY) as conn:
        conn.execute("""
        INSERT INTO listens (user_id, song_id, started_at, completion_pct, source,
                             context_genre, context_hour, context_dow)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (int(user_id), str(song_id), time.time(), float(completion_pct), source,
              int(genre), now.hour, now.weekday()))


# ─── mood ───

def mood_for_command(cmd: str) -> str:
    c = cmd.strip().lower().lstrip("/").split()[0]
    return c if c in MOOD_COMMANDS else ""


def mood_filter(mood: str) -> dict:
    """Mood → {genre_pref, adventurous_boost} for pipeline filtering."""
    if mood == "4am":
        return {"genres": MOOD_GENRE_PREF["4am"], "adventurous": -0.15, "recency_boost": 0.0}
    if mood == "gym":
        return {"genres": MOOD_GENRE_PREF["gym"], "adventurous": 0.1, "recency_boost": 0.1}
    if mood == "morning":
        return {"genres": MOOD_GENRE_PREF["morning"], "adventurous": 0.0, "recency_boost": 0.05}
    if mood in MOOD_GENRE_PREF:
        return {"genres": MOOD_GENRE_PREF[mood], "adventurous": 0.0, "recency_boost": 0.0}
    return {"genres": [], "adventurous": 0.0, "recency_boost": 0.0}


# ─── prefs (/never, /anchor, /adventurous) ───

def get_prefs(user_id: int) -> dict:
    with get_conn(DB_CATALOG) as conn:
        row = conn.execute("SELECT blacklist_json, anchor_song_id, adventurous FROM companion_prefs WHERE user_id=?",
                           (int(user_id),)).fetchone()
    if not row:
        return {"blacklist": [], "anchor": "", "adventurous": 3}
    try:
        bl = json.loads(row["blacklist_json"] or "[]")
    except Exception:
        bl = []
    return {"blacklist": bl, "anchor": row["anchor_song_id"] or "",
            "adventurous": int(round(float(row["adventurous"] or 0.3) * 10))}


def _save_prefs(user_id: int, blacklist: list, anchor: str, adventurous01: float) -> None:
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO companion_prefs (user_id, blacklist_json, anchor_song_id, adventurous, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            blacklist_json=excluded.blacklist_json, anchor_song_id=excluded.anchor_song_id,
            adventurous=excluded.adventurous, updated_at=excluded.updated_at
        """, (int(user_id), json.dumps(blacklist), anchor, float(adventurous01), time.time()))


def cmd_never(user_id: int, song_id: str) -> dict:
    p = get_prefs(user_id)
    bl = list(dict.fromkeys(list(p["blacklist"]) + [str(song_id)]))[-200:]
    _save_prefs(user_id, bl, p["anchor"], p["adventurous"] / 10.0)
    return {"status": "ok", "blacklist_n": len(bl)}


def cmd_anchor(user_id: int, song_id: str = "") -> dict:
    p = get_prefs(user_id)
    _save_prefs(user_id, p["blacklist"], str(song_id or ANCHOR_SONG_ID), p["adventurous"] / 10.0)
    return {"status": "ok", "anchor": str(song_id or ANCHOR_SONG_ID)}


def cmd_adventurous(user_id: int, level: int) -> dict:
    level = max(0, min(10, int(level)))
    p = get_prefs(user_id)
    _save_prefs(user_id, p["blacklist"], p["anchor"], level / 10.0)
    # exploration_slots ≈ adventurous mapped 0-10 → 0-5 slots of Top-10.
    return {"status": "ok", "adventurous": level, "exploration_slots": int(round(level / 10 * 5))}


def exploration_slots_for(user_id: int, default: int = 3) -> int:
    p = get_prefs(user_id)
    if p["blacklist"] == [] and p["anchor"] == "" and p["adventurous"] == 3:
        return default
    return int(round(p["adventurous"] / 10 * 5))


# ─── /why in words ───

def why_in_words(why: dict, title: str = "") -> str:
    drivers = (why or {}).get("drivers", []) or []
    b = (why or {}).get("breakdown", {}) or {}
    if not drivers:
        return f"{title} fits your general rotation." if title else "General rotation pick."
    m = {"collaborative fit": "listeners with your taste replay it",
         "neural pattern match": "it matches your deeper listening patterns",
         "title similarity to your taste": "its sound/words match what you finish",
         "exploration pick": "it's a fresh stretch beyond your usual",
         "trending now": "it's heating up right now"}
    bits = [m.get(d, d) for d in drivers[:2]]
    top = max(b.items(), key=lambda x: x[1])[0] if b else ""
    s = f"{' · '.join(bits)}"
    if title:
        s = f"{title}: {s}"
    if top:
        s += f" (strongest signal: {top})."
    return s


# ─── journal + streaks + digest ───

def record_journal_day(user_id: int, plays: int = 1, likes: int = 0, genre: int = 20) -> None:
    day = datetime.date.today().isoformat()
    with get_conn(DB_CATALOG) as conn:
        conn.execute("""
        INSERT INTO listening_journal (user_id, day, plays, likes, top_genre)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, day) DO UPDATE SET
            plays = plays + excluded.plays, likes = likes + excluded.likes
        """, (int(user_id), day, int(plays), int(likes), int(genre)))


def streak_days(user_id: int) -> int:
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute("SELECT day FROM listening_journal WHERE user_id=? ORDER BY day DESC LIMIT 30",
                            (int(user_id),)).fetchall()
    days = [r["day"] for r in rows]
    if not days:
        return 0
    streak = 0
    cur = datetime.date.today()
    dset = set(days)
    while cur.isoformat() in dset:
        streak += 1
        cur -= datetime.timedelta(days=1)
    return streak


def weekly_digest(user_id: int) -> str:
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute("SELECT SUM(plays) p, SUM(likes) l FROM listening_journal WHERE user_id=? AND day>=?",
                            (int(user_id), week_ago)).fetchone()
    plays = int((rows["p"] if rows and rows["p"] else 0))
    likes = int((rows["l"] if rows and rows["l"] else 0))
    streak = streak_days(user_id)
    return (f"🎧 Weekly digest: {plays} plays, {likes} likes, "
            f"{streak}-day streak. Keep the loop going.")
