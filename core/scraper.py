"""
core/scraper.py — Async YouTube Feed Scraper

Scrapes 1k-2k songs per run across multiple search queries.
Uses yt-dlp as the extraction engine with concurrent workers.

Architecture:
  asyncio.Queue  → YT_FETCH_WORKERS concurrent yt-dlp subprocesses
  Each worker extracts metadata (no audio download, just info)
  Results deduplicated via PrimeBloomFilter
  Top 100 ranked by composite raw_score → saved to DB_FEED

Raw score formula:
  raw_score = log10(view_count+1) * like_ratio * freshness_decay * energy_bonus

  freshness_decay = 0.5 ^ (days_since_upload / FEED_HALFLIFE_HOURS * 24)
  energy_bonus    = Fibonacci(title_energy_words) / max(FIB_SEQ)
  like_ratio      = like_count / max(like_count + dislike_estimate, 1)
"""

import asyncio
import json
import math
import re
import time
import subprocess
import logging
from datetime import datetime, timezone
from typing import Optional

from config import (
    YT_SEARCH_QUERIES, YT_FEED_TOP_N, YT_FETCH_WORKERS,
    YT_MIN_VIEWS, YT_MAX_DURATION, FEED_HALFLIFE_HOURS,
    FIB_SEQ, GENRE_MAP, LANGUAGE_MAP,
    YT_DEEP_ARTISTS, YT_DEEP_MOODS, YT_DEEP_CHARTS, YT_RELATED_TEMPLATES,
    YT_DEEP_TOP_N, FEED_SNAPSHOT_RETENTION_DAYS,
)
from core.database import get_conn, DB_FEED, DB_CATALOG
from core.bloom import is_duplicate, mark_seen

logger = logging.getLogger("parlay.scraper")

# ─── Energy keywords (mapped to Fibonacci bonus) ──────────
ENERGY_WORDS = [
    "fire", "banger", "heat", "vibe", "lit", "hard", "beast",
    "crazy", "insane", "anthem", "official", "remix", "live",
]

OFFICIAL_TAGS = {"official", "official video", "official audio", "official mv"}
LYRIC_TAGS = {"lyrics", "lyric video", "official lyrics"}
REMIX_TAGS = {"remix", "mashup", "bootleg"}


# ─── Metadata Extractor ───────────────────────────────────

def _extract_via_ytdlp(query: str, max_results: int = 50) -> list[dict]:
    """
    Run yt-dlp in a subprocess to search YouTube.
    Returns list of raw info dicts.
    No audio downloaded — metadata only (--skip-download --dump-json).
    """
    search_url = f"ytsearch{max_results}:{query}"
    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "--skip-download",
        "--dump-json",
        "--flat-playlist",
        "--no-playlist",
        "--ignore-errors",
        "--extractor-args", "youtube:skip=dash,hls",
        search_url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        items = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                info = json.loads(line)
                items.append(info)
            except json.JSONDecodeError:
                continue
        return items
    except subprocess.TimeoutExpired:
        logger.warning(f"yt-dlp timeout for query: {query}")
        return []
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: pip install yt-dlp")
        return []
    except Exception as e:
        logger.error(f"yt-dlp error [{query}]: {e}")
        return []


def _detect_genre(title: str, description: str = "") -> int:
    # Word-boundary matching avoids substring collisions
    # ("trap" in "trap nation" ok, but "pop" in "popular" must not win).
    text = (title + " " + description).lower()
    # Check multi-word / distinctive keys first, longest first.
    for keyword in sorted(GENRE_MAP.keys(), key=len, reverse=True):
        if keyword == "unknown":
            continue
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):
            return GENRE_MAP[keyword]
    return GENRE_MAP["unknown"]


MOOD_KEYWORDS = {
    "sad", "chill", "night", "morning", "gym", "workout", "romantic",
    "lofi", "acoustic", "vibes", "4am", "soulful", "heartbreak", "folk",
    "bhajan", "party", "focus", "love",
}


def _extract_mood_tags(title: str, description: str = "") -> str:
    text = (title + " " + description).lower()
    hits = sorted({w for w in MOOD_KEYWORDS if re.search(r"\b" + re.escape(w) + r"\b", text)})
    return ",".join(hits[:8])


def _detect_language(title: str) -> int:
    # Rudimentary heuristic: detect non-latin scripts
    if re.search(r'[\u0900-\u097F]', title):   # Devanagari (Hindi)
        return LANGUAGE_MAP.get("hi", 9)
    if re.search(r'[\uAC00-\uD7AF]', title):   # Korean (K-pop)
        return LANGUAGE_MAP.get("ko", 9)
    if re.search(r'[\u4E00-\u9FFF]', title):   # CJK
        return LANGUAGE_MAP.get("ja", 9)
    if re.search(r'[ñáéíóúü]', title.lower()): # Spanish signals
        return LANGUAGE_MAP.get("es", 9)
    return LANGUAGE_MAP.get("en", 0)


def _energy_score(title: str) -> float:
    """
    Energy = sum of Fibonacci weights for matched energy keywords.
    Normalised to [0, 1].
    """
    title_lower = title.lower()
    hits = sum(1 for w in ENERGY_WORDS if w in title_lower)
    fib_val = FIB_SEQ[min(hits, len(FIB_SEQ)-1)]
    return min(fib_val / max(FIB_SEQ), 1.0)


def _freshness_decay(upload_date_str: Optional[str]) -> float:
    """
    Exponential half-life decay: f = 0.5^(age_hours / halflife)
    If upload_date unknown, return 0.5 (neutral).
    """
    if not upload_date_str:
        return 0.5
    try:
        dt = datetime.strptime(upload_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        return 0.5 ** (age_hours / FEED_HALFLIFE_HOURS)
    except Exception:
        return 0.5


def _raw_score(info: dict) -> float:
    """
    Composite raw score used to rank songs in a feed snapshot.

    Formula:
        S = log10(V+1) * L * F * (1 + E)

    Where:
        V = view_count
        L = like_ratio (clamped 0–1; estimated if missing)
        F = freshness_decay  (half-life: FEED_HALFLIFE_HOURS)
        E = energy_score     (Fibonacci-weighted keyword bonus)

    Taking log10 of views compresses the power-law distribution of
    YouTube view counts, preventing megahits from drowning everything else.
    Multiplying by like_ratio and freshness penalises stale/divisive content.
    """
    v = max(info.get("view_count") or 0, 1)
    like = info.get("like_count") or 0
    # YouTube hides dislike counts; we estimate via sqrt heuristic
    est_dislike = math.sqrt(v) * 0.05
    like_ratio = like / max(like + est_dislike, 1)

    freshness = _freshness_decay(info.get("upload_date"))
    energy = _energy_score(info.get("title", ""))

    return math.log10(v) * like_ratio * freshness * (1 + energy)


def parse_info(info: dict) -> Optional[dict]:
    """
    Parse a yt-dlp info dict into our canonical song dict.
    Returns None if song fails quality filters.
    """
    song_id = info.get("id") or info.get("video_id")
    if not song_id:
        return None

    title = (info.get("title") or "").strip()
    if not title:
        return None

    duration = info.get("duration") or 0
    view_count = info.get("view_count") or 0

    # Quality filters
    if view_count < YT_MIN_VIEWS:
        return None
    if 0 < duration > YT_MAX_DURATION:
        return None

    title_lower = title.lower()
    has_official = int(any(t in title_lower for t in OFFICIAL_TAGS))
    has_lyric = int(any(t in title_lower for t in LYRIC_TAGS))
    has_remix = int(any(t in title_lower for t in REMIX_TAGS))
    desc = info.get("description") or ""

    return {
        "song_id":       song_id,
        "title":         title,
        "channel":       info.get("uploader") or info.get("channel") or "",
        "channel_id":    info.get("channel_id") or "",
        "duration":      int(duration),
        "view_count":    int(view_count),
        "like_count":    int(info.get("like_count") or 0),
        "upload_date":   info.get("upload_date") or "",
        "thumbnail_url": info.get("thumbnail") or "",
        "yt_url":        f"https://youtube.com/watch?v={song_id}",
        "genre_code":    _detect_genre(title, desc),
        "language_code": _detect_language(title),
        "has_official":  has_official,
        "has_lyric":     has_lyric,
        "has_remix":     has_remix,
        "energy_score":  _energy_score(title),
        "description":   desc[:2000],
        "mood_tags":     _extract_mood_tags(title, desc),
        "raw_score":     _raw_score(info),
        "first_seen":    time.time(),
        "last_seen":     time.time(),
    }


# ─── Async Scrape Pipeline ────────────────────────────────

async def _scrape_query(query: str, sem: asyncio.Semaphore, results: list):
    """Scrape one query under a semaphore."""
    async with sem:
        loop = asyncio.get_event_loop()
        infos = await loop.run_in_executor(
            None, _extract_via_ytdlp, query, 100
        )
        for info in infos:
            parsed = parse_info(info)
            if parsed:
                results.append(parsed)
        logger.info(f"  ↳ [{query[:40]}] → {len(infos)} raw, parsed {sum(1 for _ in results[-len(infos):] if _ is not None)}")


async def scrape_yt_feed() -> list[dict]:
    """
    Main entry point. Scrapes all YT_SEARCH_QUERIES concurrently
    (up to YT_FETCH_WORKERS at a time), deduplicates, and returns
    the top YT_FEED_TOP_N by raw_score.
    """
    logger.info(f"🎵 Starting YT feed scrape across {len(YT_SEARCH_QUERIES)} queries...")
    sem = asyncio.Semaphore(YT_FETCH_WORKERS)
    all_results: list[dict] = []

    tasks = [
        _scrape_query(query, sem, all_results)
        for query in YT_SEARCH_QUERIES
    ]
    await asyncio.gather(*tasks)

    logger.info(f"  Raw candidates: {len(all_results)}")

    # Dedup by song_id (bloom filter + dict)
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for song in all_results:
        sid = song["song_id"]
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique.append(song)

    logger.info(f"  After dedup: {len(unique)}")

    # Rank by raw_score, take top N
    unique.sort(key=lambda x: x["raw_score"], reverse=True)
    top = unique[:YT_FEED_TOP_N]

    logger.info(f"  Top {YT_FEED_TOP_N} selected. Saving...")
    await _save_feed_snapshot(top)
    return top


async def _save_feed_snapshot(songs: list[dict]) -> str:
    """Save feed snapshot + upsert into catalog. Returns snapshot_id."""
    now = time.time()
    loop = asyncio.get_event_loop()

    def _db_save():
        with get_conn(DB_FEED) as conn:
            cur = conn.execute(
                "INSERT INTO feed_snapshots (fetched_at, query_used) VALUES (?, ?)",
                (now, json.dumps(YT_SEARCH_QUERIES[:3]))  # sample of queries
            )
            snapshot_id = cur.lastrowid
            conn.executemany(
                """INSERT INTO feed_songs
                   (snapshot_id, song_id, rank_in_snapshot, raw_score)
                   VALUES (?, ?, ?, ?)""",
                [(snapshot_id, s["song_id"], i+1, s["raw_score"])
                 for i, s in enumerate(songs)]
            )

        with get_conn(DB_CATALOG) as conn:
            for s in songs:
                conn.execute("""
                INSERT INTO songs (
                    song_id, title, channel, channel_id, duration, view_count,
                    like_count, upload_date, thumbnail_url, yt_url, genre_code,
                    language_code, has_official, has_lyric, has_remix, energy_score,
                    description, mood_tags,
                    first_seen, last_seen, times_fetched
                ) VALUES (
                    :song_id, :title, :channel, :channel_id, :duration, :view_count,
                    :like_count, :upload_date, :thumbnail_url, :yt_url, :genre_code,
                    :language_code, :has_official, :has_lyric, :has_remix, :energy_score,
                    :description, :mood_tags,
                    :first_seen, :last_seen, 1
                ) ON CONFLICT(song_id) DO UPDATE SET
                    view_count   = MAX(excluded.view_count, view_count),
                    like_count   = MAX(excluded.like_count, like_count),
                    last_seen    = excluded.last_seen,
                    description  = excluded.description,
                    mood_tags    = excluded.mood_tags,
                    times_fetched= times_fetched + 1
                """, {k: s.get(k) for k in (
                    "song_id", "title", "channel", "channel_id", "duration", "view_count",
                    "like_count", "upload_date", "thumbnail_url", "yt_url", "genre_code",
                    "language_code", "has_official", "has_lyric", "has_remix", "energy_score",
                    "description", "mood_tags", "first_seen", "last_seen")})
                mark_seen(s["song_id"])
            # Artist graph: artist → song co-counts (free from channel names).
            try:
                chans = [s.get("channel", "") for s in songs if s.get("channel")]
                for i in range(min(len(chans), 60)):
                    for j in range(i + 1, min(len(chans), 60)):
                        a, b = sorted([chans[i][:48], chans[j][:48]])
                        if a and b and a != b:
                            conn.execute("""
                            INSERT INTO artist_graph (artist_a, artist_b, co_count, updated_at)
                            VALUES (?, ?, 1, ?)
                            ON CONFLICT(artist_a, artist_b) DO UPDATE SET
                                co_count = co_count + 1, updated_at = excluded.updated_at
                            """, (a, b, now))
            except Exception as e:
                logger.warning("artist_graph update skipped: %s", e)

    await loop.run_in_executor(None, _db_save)
    logger.info(f"✅  Feed snapshot saved ({len(songs)} songs).")


# ─── MAX Phase 1: nightly deep scrape (2–5k) ────────────

def deep_queries(seed_catalog_titles: list[str] | None = None) -> list[str]:
    """Build deep query list: discographies + moods + charts + related expansion."""
    qs: list[str] = []
    for artist in YT_DEEP_ARTISTS:
        qs.append(f"{artist} discography full songs")
        qs.append(f"{artist} best songs")
    qs.extend(YT_DEEP_MOODS)
    qs.extend(YT_DEEP_CHARTS)
    for seed in (seed_catalog_titles or [])[:10]:
        for tmpl in YT_RELATED_TEMPLATES:
            try:
                qs.append(tmpl.format(seed=seed, artist=seed))
            except Exception:
                pass
    # De-dup preserving order.
    return list(dict.fromkeys(qs))


async def scrape_deep_feed(seed_catalog_titles: list[str] | None = None) -> list[dict]:
    """Nightly deep job: per-artist / per-mood / charts / related expansion.

    Same workers (4–6) and bloom dedup; keeps YT_DEEP_TOP_N by raw_score.
    Target ~2–5k/night; hourly job never OOMs because this runs at 03:00
    under the scheduler lock (never scrape + train at once).
    """
    queries = deep_queries(seed_catalog_titles)
    logger.info("🌙 Deep scrape: %d queries (workers=%d)...", len(queries), YT_FETCH_WORKERS)
    sem = asyncio.Semaphore(YT_FETCH_WORKERS)
    all_results: list[dict] = []
    tasks = [_scrape_query(q, sem, all_results) for q in queries]
    await asyncio.gather(*tasks)
    seen: set[str] = set()
    unique: list[dict] = []
    for song in all_results:
        sid = song["song_id"]
        if sid not in seen and not is_duplicate(sid):
            seen.add(sid)
            unique.append(song)
    unique.sort(key=lambda x: x["raw_score"], reverse=True)
    top = unique[:YT_DEEP_TOP_N]
    logger.info("  Deep: %d raw → %d unique → keeping %d", len(all_results), len(unique), len(top))
    await _save_feed_snapshot(top)
    prune_old_feed_snapshots()
    return top


def prune_old_feed_snapshots(retention_days: int = FEED_SNAPSHOT_RETENTION_DAYS,
                               dry_run: bool = False) -> int:
    """Snapshot retention: 7 days of feed_songs, vacuum catalog weekly."""
    cutoff = time.time() - retention_days * 86400
    purged = 0
    with get_conn(DB_FEED) as conn:
        try:
            rows = conn.execute("SELECT snapshot_id FROM feed_snapshots WHERE fetched_at < ?",
                                (cutoff,)).fetchall()
            ids = [r["snapshot_id"] for r in rows]
            if ids and dry_run:
                purged = len(ids)
            elif ids:
                ph = ",".join(["?"] * len(ids))
                conn.execute(f"DELETE FROM feed_songs WHERE snapshot_id IN ({ph})", ids)
                conn.execute(f"DELETE FROM feed_snapshots WHERE snapshot_id IN ({ph})", ids)
                purged = len(ids)
        except Exception as e:
            logger.warning("prune feed snapshots failed: %s", e)
    logger.info("  Feed retention: %s %d snapshots older than %dd.",
                "would purge" if dry_run else "purged", purged, retention_days)
    return purged
