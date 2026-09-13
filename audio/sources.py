"""sources.py — every door except YouTube's walled front gate.

apiyt's converter covers most videos, but official/VEVO ones 404 and
YouTube watch is bot-walled from datacenter IPs. These sources need no
auth and work from the VPS:

  iTunes Search API  30s m4a previews, huge catalog (incl. Bollywood)
  Deezer API         30s mp3 previews, huge catalog

30 seconds is plenty for ears: tempo needs ~15s, chroma/key stabilize
in ~20s, brightness/energy instantly. Provenance travels with the file
so the store knows full-track vs preview hearing. Stdlib only.
"""

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger("parlay.audio.sources")

_ITUNES = "https://itunes.apple.com/search"
_DEEZER = "https://api.deezer.com/search"


def _get_json(url: str, timeout: int = 25) -> dict | list | None:
    req = urllib.request.Request(url, headers={"User-Agent": "Parlay-ML/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        logger.warning("source fetch failed %s: %s", url[:80], e)
        return None


def _artist_match(want: str, got: str) -> bool:
    w, g = want.lower().strip(), got.lower().strip()
    if not w:
        return True
    return w in g or g in w


_STOP = {"in", "the", "a", "an", "official", "audio", "video", "lyric", "lyrics",
        "visualizer", "feat", "ft", "hq", "hd", "explicit", "remastered"}


def _tokens(s: str) -> set[str]:
    import re
    s = re.sub(r"\(.*?\)", " ", s.lower())
    return {t for t in re.findall(r"[a-z0-9]+", s) if t not in _STOP}


def title_ok(want_title: str, got_title: str) -> bool:
    """Gate against same-artist wrong-song matches (Afsanay != 4AM).

    Needs at least half the wanted title's significant tokens present.
    """
    w, g = _tokens(want_title), _tokens(got_title)
    if not w:
        return True
    need = max(1, len(w) // 2)
    return len(w & g) >= need


def rank_candidates(want_title: str, want_artist: str, cands: list[dict]) -> list[dict]:
    """Best match first: title overlap, then artist, then shorter title."""
    def _key(c: dict):
        w = _tokens(want_title)
        overlap = len(w & _tokens(c["title"])) if w else 1
        return (0 if title_ok(want_title, c["title"]) else 1,
                0 if _artist_match(want_artist, c["artist"]) else 1,
                -overlap, len(c["title"]))
    return sorted(cands, key=_key)


def itunes_preview(title: str, artist: str = "", limit: int = 5) -> list[dict]:
    """30s previews from iTunes. Returns [{title, artist, url, source}]."""
    q = urllib.parse.urlencode({"term": f"{title} {artist}".strip(),
                                "media": "music", "limit": limit})
    d = _get_json(f"{_ITUNES}?{q}")
    out = []
    for r in (d or {}).get("results", []):
        url = r.get("previewUrl") or ""
        if not url:
            continue
        out.append({"title": r.get("trackName", ""), "artist": r.get("artistName", ""),
                    "url": url, "source": "itunes-preview"})
    out.sort(key=lambda r: (not _artist_match(artist, r["artist"]), len(r["title"])))
    return out


def deezer_preview(title: str, artist: str = "", limit: int = 5) -> list[dict]:
    """30s previews from Deezer. Same shape as itunes_preview."""
    q = urllib.parse.urlencode({"q": f"{title} {artist}".strip(), "limit": limit})
    d = _get_json(f"{_DEEZER}?{q}")
    out = []
    for t in (d or {}).get("data", []):
        url = t.get("preview") or ""
        if not url:
            continue
        an = (t.get("artist") or {}).get("name", "")
        # Skip karaoke/backing clones crowding the results.
        if "karaoke" in t.get("title", "").lower() or "backing" in t.get("title", "").lower():
            continue
        out.append({"title": t.get("title", ""), "artist": an,
                    "url": url, "source": "deezer-preview"})
    out.sort(key=lambda r: (not _artist_match(artist, r["artist"]), len(r["title"])))
    return out


def download_preview(url: str, dest: str, timeout: int = 60) -> str | None:
    """Fetch a preview file. Returns dest or None."""
    req = urllib.request.Request(url, headers={"User-Agent": "Parlay-ML/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as fh:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as e:
        logger.warning("preview download failed: %s", e)
        return None
    try:
        import os
        if os.path.getsize(dest) < 20_000:  # error page, not audio
            return None
    except OSError:
        return None
    return dest
