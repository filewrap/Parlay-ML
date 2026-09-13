"""fetch.py — capture an earful of a track, through whatever door is open.

Cascade order (first success wins):
  1. apiyt direct MP3 (full track, best quality) — needs only the service
  2. iTunes 30s preview (no auth, huge catalog)
  3. Deezer 30s preview (no auth, huge catalog)
  4. yt-dlp watch (datacenter-IP walled; works with AUDIO_COOKIES)

Output is always mono 16kHz wav + provenance {source, clip_secs}.
Concurrency is 1 by design: this job runs at 02:00 under the train lock
so it never fights the scraper or trainer for the VPS.
"""

import importlib.util as _ilu
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger("parlay.audio.fetch")

YTDLP = os.environ.get("YTDLP_BIN", "/home/ubuntu/Parlay/venv/bin/yt-dlp")
CLIP = os.environ.get("AUDIO_CLIP_RANGE", "*00:00-01:30")  # 90 seconds
TIMEOUT_S = int(os.environ.get("AUDIO_FETCH_TIMEOUT", "180"))
# YouTube bot-checks datacenter IPs on watch pages. If you have cookies
# from a logged-in browser, point AUDIO_COOKIES at the cookies.txt file
# (see yt-dlp wiki "pass cookies") and fetches will use them.
COOKIES = os.environ.get("AUDIO_COOKIES", "").strip()

_APIYT = {"mod": None}


def _apiyt():
    """apiyt core, loaded by path (never installed, never a dependency)."""
    if _APIYT["mod"] is None:
        spec = _ilu.spec_from_file_location("apiyt_core", "/tmp/apiyt-real/core.py")
        if spec is None or spec.loader is None:
            return None
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _APIYT["mod"] = mod
    return _APIYT["mod"]


def search_video_id(title: str, artist: str = "", limit: int = 6) -> list[dict]:
    """apiyt search → convertible video IDs, title-gated (Afsanay lesson)."""
    from .sources import rank_candidates, title_ok
    mod = _apiyt()
    if mod is None:
        return []
    try:
        results = list(mod.search(f"{title} {artist}".strip(), limit=limit))
    except Exception as e:
        logger.warning("apiyt search failed: %s", e)
        return []
    cands = [{"title": r.get("title", ""), "artist": r.get("channel", ""),
              "video_id": r.get("id", "")} for r in results if r.get("id")]
    out = []
    for c in rank_candidates(title, artist, cands)[:3]:
        if not title_ok(title, c["title"]):
            logger.info("apiyt reject: %s (want %r)", c["title"], title)
            continue
        try:
            info = mod.resolve(c["video_id"])
        except Exception:
            continue
        if info.get("status") == "ok" and info.get("durl"):
            out.append({**c, "duration": next(
                (r.get("duration", 0) for r in results if r.get("id") == c["video_id"]), 0)})
    return out


def _run(cmd: list[str], timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _to_wav(src: str, wav: str, max_secs: int = 0) -> bool:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if max_secs > 0:
        cmd += ["-t", str(max_secs)]
    cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav]
    r = _run(cmd)
    return r.returncode == 0 and os.path.exists(wav)


def fetch_clip(video_id: str, workdir: str | None = None) -> str | None:
    """Download 90s bestaudio → mono 16k wav. Returns wav path or None.

    Caller must delete the wav when done (see listener.py).
    Legacy entry point (YouTube-ID only). Prefer fetch_any() below.
    """
    wav, _prov = _fetch_ytdlp(str(video_id), workdir)
    return wav


def _fetch_ytdlp(video_id: str, workdir: str | None = None) -> tuple[str | None, dict]:
    tmp = workdir or tempfile.mkdtemp(prefix="listen_")
    raw = os.path.join(tmp, "src.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [YTDLP, "--no-warnings", "--quiet", "-f", "bestaudio/best",
           "--download-sections", CLIP, "--force-keyframes-at-cuts"]
    if COOKIES:
        cmd += ["--cookies", COOKIES]
    cmd += ["-o", raw, url]
    try:
        r = _run(cmd)
    except subprocess.TimeoutExpired:
        logger.warning("fetch %s: timeout", video_id)
        shutil.rmtree(tmp, ignore_errors=True)
        return None, {}
    # Find whatever container yt-dlp produced.
    cands = [os.path.join(tmp, f) for f in os.listdir(tmp)
             if not f.endswith(".wav")]
    cands = [f for f in cands if os.path.isfile(f) and not f.endswith(".part")]
    if not cands:
        logger.warning("fetch %s: no audio (%s)", video_id, (r.stderr or "")[-160:])
        shutil.rmtree(tmp, ignore_errors=True)
        return None, {}
    src = max(cands, key=os.path.getsize)
    wav = os.path.join(tmp, "clip.wav")
    if not _to_wav(src, wav):
        logger.warning("ffmpeg %s failed", video_id)
        shutil.rmtree(tmp, ignore_errors=True)
        return None, {}
    try:
        os.remove(src)
    except OSError:
        pass
    return wav, {"source": "ytdlp-watch", "clip_secs": 90}


def _fetch_apiyt(video_id: str, tmp: str) -> tuple[str | None, dict]:
    mod = _apiyt()
    if mod is None:
        return None, {}
    try:
        mp3 = os.path.join(tmp, f"{video_id}.mp3")
        mod.download(str(video_id), mp3)
    except Exception as e:
        logger.info("apiyt %s: %s", video_id, str(e)[:100])
        return None, {}
    wav = os.path.join(tmp, "clip.wav")
    if not _to_wav(mp3, wav, max_secs=90):
        return None, {}
    try:
        os.remove(mp3)
    except OSError:
        pass
    return wav, {"source": "apiyt-mp3", "clip_secs": 90}


def _fetch_preview(title: str, artist: str, tmp: str) -> tuple[str | None, dict]:
    from .sources import (deezer_preview, download_preview, itunes_preview,
                          rank_candidates, title_ok)
    pool: list[dict] = []
    for resolver in (itunes_preview, deezer_preview):
        try:
            pool.extend(resolver(title, artist))
        except Exception:
            continue
    for c in rank_candidates(title, artist, pool)[:3]:
        if not title_ok(title, c["title"]):
            logger.info("preview reject: %s - %s (want %r)", c["artist"], c["title"], title)
            continue
        ext = ".m4a" if "itunes" in c["source"] else ".mp3"
        src = os.path.join(tmp, f"prev{ext}")
        if not download_preview(c["url"], src):
            continue
        wav = os.path.join(tmp, "clip.wav")
        if _to_wav(src, wav):
            try:
                os.remove(src)
            except OSError:
                pass
            logger.info("preview hearing: %s - %s via %s",
                        c["artist"], c["title"], c["source"])
            return wav, {"source": c["source"], "clip_secs": 30,
                         "resolved_title": c["title"], "resolved_artist": c["artist"]}
    return None, {}


def fetch_any(video_id: str | None = None, title: str = "", artist: str = "",
              workdir: str | None = None) -> tuple[str | None, dict]:
    """Full cascade. Returns (wav_path, provenance) — wav or (None, {}).

    Caller owns the wav's temp dir (use cleanup_wav).
    """
    tmp = workdir or tempfile.mkdtemp(prefix="listen_")
    # apiyt first (full track), then previews (fast, reliable), then
    # yt-dlp watch last (walled without AUDIO_COOKIES).
    if video_id:
        wav, prov = _fetch_apiyt(str(video_id), tmp)
        if wav:
            return wav, prov
    if title:
        wav, prov = _fetch_preview(title, artist, tmp)
        if wav:
            return wav, prov
    if video_id:
        wav, prov = _fetch_ytdlp(str(video_id), tmp)
        if wav:
            return wav, prov
    shutil.rmtree(tmp, ignore_errors=True)
    return None, {}


def cleanup_wav(wav_path: str | None) -> None:
    if not wav_path:
        return
    try:
        shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
    except OSError:
        pass
