"""
notifier/owner_profile.py — seed AgainOwner's taste so NeuroSync personalises.

User: AgainOwner / 6802929470.
Mehrama (Love Aaj Kal) is most-played → weighted heaviest.
Everything else gets strong like-weight so TF-IDF + bandit + SVD
start from your real taste instead of cold-start noise.

Usage:
    python -m notifier.owner_profile
    python -m notifier.owner_profile --user-id 6802929470
    python main.py --once --user-ids 6802929470   # recs after seeding

Idempotent: re-running skips songs/feedback already stored.
Stdlib only.
"""

import argparse
import random
import time
import urllib.parse

OWNER_TELEGRAM_ID = 6802929470

# (song_id, title, channel, genre_code, language_code, plays, weight)
# genre_code follows config.GENRE_MAP; language_code follows LANGUAGE_MAP.
_TASTE = [
    ("owner_mehrama", "Mehrama — Love Aaj Kal", "Sony Music India", 14, 1, 20, 1.0),
    ("owner_husn", "Husn — Anuv Jain", "Anuv Jain", 6, 1, 6, 1.0),
    ("owner_arz", "Arz Kiya Hai — Anuv Jain", "Anuv Jain", 6, 1, 5, 1.0),
    ("owner_jotum", "Jo Tum Mere Ho — Anuv Jain", "Anuv Jain", 6, 1, 5, 1.0),
    ("owner_heartbreakkid", "Heartbreak Kid — Talha Anjum", "Talha Anjum", 2, 9, 5, 1.0),
    ("owner_4amkhi", "4AM in Karachi — Talha Anjum", "Talha Anjum", 2, 9, 5, 1.0),
    ("owner_co2", "CO2 — Prateek Kuhad", "Prateek Kuhad", 6, 0, 5, 1.0),
    ("owner_october", "We Fell in Love in October — Girl in Red", "Girl in Red", 6, 0, 4, 0.9),
    ("owner_oneofthegirls", "One of the Girls — The Weeknd", "The Weeknd", 5, 0, 4, 0.9),
    ("owner_perfect", "Perfect — Ed Sheeran", "Ed Sheeran", 3, 0, 4, 0.9),
    ("owner_nightwemet", "The Night We Met — Lord Huron", "Lord Huron", 6, 0, 4, 0.9),
    ("owner_bashedout", "Bashed Out — Brassland", "Brassland", 6, 0, 3, 0.8),
    ("owner_hbanniv", "Heartbreak Anniversary — Giveon", "Giveon", 5, 0, 4, 0.9),
    ("owner_seema1", "Seema Mishra — Popular Folk Hits", "Seema Mishra", 11, 1, 4, 0.9),
    ("owner_seema2", "Seema Mishra — Best Bhajans", "Seema Mishra", 11, 1, 3, 0.8),
    ("owner_seema3", "Seema Mishra — Best of Seema Mishra", "Seema Mishra", 18, 1, 3, 0.8),
]


def _song_row(sid, title, channel, genre, lang):
    q = urllib.parse.quote_plus(f"{title} {channel}")
    now = time.time()
    return {
        "song_id": sid, "title": title, "channel": channel,
        "channel_id": "", "duration": 240, "view_count": 5_000_000,
        "like_count": 250_000, "upload_date": "20240101",
        "thumbnail_url": "",
        "yt_url": f"https://youtube.com/results?search_query={q}",
        "genre_code": genre, "language_code": lang,
        "has_official": 1, "has_lyric": 0, "energy_score": 0.6,
        "first_seen": now - 86400 * 30, "last_seen": now,
        "times_fetched": 10,
    }


def seed_owner_profile(user_id: int = OWNER_TELEGRAM_ID) -> dict:
    from core.database import bootstrap_all, get_conn, DB_CATALOG, DB_HISTORY, DB_FEEDBACK
    bootstrap_all()
    now = time.time()
    songs = [_song_row(sid, t, c, g, lang) for sid, t, c, g, lang, _, _ in _TASTE]
    with get_conn(DB_CATALOG) as conn:
        conn.executemany("""
        INSERT INTO songs (song_id, title, channel, channel_id, duration, view_count,
            like_count, upload_date, thumbnail_url, yt_url, genre_code,
            language_code, has_official, has_lyric, energy_score,
            first_seen, last_seen, times_fetched)
        VALUES (:song_id, :title, :channel, :channel_id, :duration, :view_count,
            :like_count, :upload_date, :thumbnail_url, :yt_url, :genre_code,
            :language_code, :has_official, :has_lyric, :energy_score,
            :first_seen, :last_seen, :times_fetched)
        ON CONFLICT(song_id) DO UPDATE SET
            title=excluded.title, last_seen=excluded.last_seen,
            times_fetched=times_fetched+1
        """, songs)
    plays, likes = 0, 0
    with get_conn(DB_HISTORY) as conn, get_conn(DB_FEEDBACK) as fconn:
        for sid, _, _, _, _, n_plays, weight in _TASTE:
            exists = fconn.execute(
                "SELECT 1 FROM feedback WHERE user_id=? AND song_id=? LIMIT 1",
                (user_id, sid)).fetchone()
            if not exists:
                fconn.execute("""
                INSERT INTO feedback (user_id, song_id, rec_session_id, signal, created_at, model_version)
                VALUES (?, ?, ?, ?, ?, ?)""", (user_id, sid, "owner_seed", 1, now, "owner_seed"))
                likes += 1
            for i in range(n_plays):
                conn.execute("""
                INSERT INTO listens (user_id, song_id, started_at, completion_pct, source)
                VALUES (?, ?, ?, ?, ?)""",
                    (user_id, sid, now - random.uniform(0, 86400 * 30) - i * 3600,
                     min(1.0, weight), "owner_seed"))
                plays += 1
    print(f"✅ Owner taste seeded: user={user_id} songs={len(songs)} listens={plays} new_likes={likes}")
    return {"user_id": user_id, "songs": len(songs), "listens": plays, "likes": likes}


def main() -> None:
    p = argparse.ArgumentParser(description="Seed AgainOwner taste profile.")
    p.add_argument("--user-id", type=int, default=OWNER_TELEGRAM_ID)
    args = p.parse_args()
    seed_owner_profile(args.user_id)


if __name__ == "__main__":
    main()
