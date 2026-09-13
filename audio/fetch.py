"""fetch.py — capture a 90s earful of a track.

yt-dlp bestaudio + --download-sections (bounded disk/CPU), then ffmpeg
to mono 16kHz wav for the DSP. Source file deleted; only the wav lives
briefly, then it is deleted too. Only ~24 numbers survive per track.

Concurrency is 1 by design: this job runs at 02:00 under the train lock
so it never fights the scraper or trainer for the VPS.
"""

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


def _run(cmd: list[str], timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def fetch_clip(video_id: str, workdir: str | None = None) -> str | None:
    """Download 90s bestaudio → mono 16k wav. Returns wav path or None.

    Caller must delete the wav when done (see listener.py).
    """
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
        return None
    # Find whatever container yt-dlp produced.
    cands = [os.path.join(tmp, f) for f in os.listdir(tmp)
             if not f.endswith(".wav")]
    cands = [f for f in cands if os.path.isfile(f) and not f.endswith(".part")]
    if not cands:
        logger.warning("fetch %s: no audio (%s)", video_id, (r.stderr or "")[-160:])
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    src = max(cands, key=os.path.getsize)
    wav = os.path.join(tmp, "clip.wav")
    r2 = _run(["ffmpeg", "-y", "-v", "error", "-i", src,
               "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav])
    try:
        os.remove(src)
    except OSError:
        pass
    if r2.returncode != 0 or not os.path.exists(wav):
        logger.warning("ffmpeg %s failed (%s)", video_id, (r2.stderr or "")[-160:])
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    return wav


def cleanup_wav(wav_path: str | None) -> None:
    if not wav_path:
        return
    try:
        shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
    except OSError:
        pass
