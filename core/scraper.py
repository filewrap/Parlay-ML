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
    text = (title + " " + description).lower()
    for keyword, code in GENRE_MAP.items():
        if keyword in text:
            return code
    return GENRE_MAP["unknown"]


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
        "genre_code":    _detect_genre(title, info.get("description") or ""),
        "language_code": _detect_language(title),
        "has_official":  has_official,
        "has_lyric":     has_lyric,
        "has_remix":     has_remix,
        "energy_score":  _energy_score(title),
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
                    language_code, has_official, has_lyric, energy_score,
                    first_seen, last_seen, times_fetched
                ) VALUES (
                    :song_id, :title, :channel, :channel_id, :duration, :view_count,
                    :like_count, :upload_date, :thumbnail_url, :yt_url, :genre_code,
                    :language_code, :has_official, :has_lyric, :energy_score,
                    :first_seen, :last_seen, 1
                ) ON CONFLICT(song_id) DO UPDATE SET
                    view_count   = MAX(excluded.view_count, view_count),
                    like_count   = MAX(excluded.like_count, like_count),
                    last_seen    = excluded.last_seen,
                    times_fetched= times_fetched + 1
                """, s)
                mark_seen(s["song_id"])

    await loop.run_in_executor(None, _db_save)
    logger.info(f"✅  Feed snapshot saved ({len(songs)} songs).")
