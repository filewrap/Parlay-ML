"""
scheduler/jobs.py — APScheduler async job definitions (MAX).

Staggered VPS schedule (never scrape + train at once):
  :00  scrape (hourly fast top-100)
  :10  push   (hourly Top-10 per user → bot)
  :30  light-train (ALS + retrievers + two-tower + sequence)
  02:00 listen (nightly audio: machine hears ~150 tracks)
  03:00 deep-train (all + FeatureMF + SVD + weekly NCF + blender)
  03:30 feed_deep (nightly 2–5k) — runs AFTER deep-train to avoid overlap
  04:00 clean (purge + VACUUM)

OOM guard: each training job logs peak RSS; scheduler refuses overlap
via lock file (config.TRAIN_LOCK_FILE).
"""

import asyncio
import logging
import time

logger = logging.getLogger("parlay.scheduler")

# These are populated by main.py after init
_pipeline = None
_trainer = None
_bot_push_callback = None   # async fn(user_id, recs) — your Parlay bot hook


def register(pipeline, trainer, bot_push_callback=None):
    global _pipeline, _trainer, _bot_push_callback
    _pipeline = pipeline
    _trainer = trainer
    _bot_push_callback = bot_push_callback


def _rss_mb() -> float:
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def _train_locked() -> bool:
    try:
        from config import TRAIN_LOCK_FILE
        if TRAIN_LOCK_FILE.exists():
            age = time.time() - TRAIN_LOCK_FILE.stat().st_mtime
            if age < 4 * 3600:
                return True
    except Exception:
        pass
    return False


async def feed_refresh_job():
    """Hourly :00 — scrape YouTube top-100."""
    if _train_locked():
        logger.warning("  feed_refresh skipped: training lock held (stagger).")
        return
    logger.info("⏰  [JOB] feed_refresh_job starting... RSS=%.0f MB", _rss_mb())
    try:
        from core.scraper import scrape_yt_feed
        songs = await scrape_yt_feed()
        if _pipeline:
            _pipeline.fit_tfidf()
        logger.info("  Feed refreshed: %d songs", len(songs))
    except Exception as e:
        logger.error("  feed_refresh_job FAILED: %s", e, exc_info=True)


async def feed_deep_job():
    """Nightly 03:30 — deep scrape 2–5k (discographies/moods/charts/related)."""
    if _train_locked():
        logger.warning("  feed_deep skipped: training lock held.")
        return
    logger.info("⏰  [JOB] feed_deep_job starting... RSS=%.0f MB", _rss_mb())
    try:
        from core.scraper import scrape_deep_feed
        from core.database import get_conn
        from config import DB_CATALOG
        seeds: list[str] = []
        try:
            with get_conn(DB_CATALOG) as conn:
                rows = conn.execute("SELECT title FROM songs ORDER BY view_count DESC LIMIT 10").fetchall()
                seeds = [r["title"] for r in rows]
        except Exception:
            pass
        songs = await scrape_deep_feed(seeds)
        if _pipeline:
            _pipeline.fit_tfidf()
        logger.info("  Deep feed: %d songs", len(songs))
    except Exception as e:
        logger.error("  feed_deep_job FAILED: %s", e, exc_info=True)


async def rec_push_job(user_ids: list):
    """Hourly :10 — Top-10 per user (precomputed Top-200 + fresh rerank)."""
    logger.info("⏰  [JOB] rec_push_job for %d users... RSS=%.0f MB", len(user_ids), _rss_mb())
    if not _pipeline:
        logger.warning("  Pipeline not initialised.")
        return
    for uid in user_ids:
        try:
            recs = _pipeline.recommend(uid)
            if recs and _bot_push_callback:
                await _bot_push_callback(uid, recs)
            logger.info("  User %s: pushed %d recs", uid, len(recs))
        except Exception as e:
            logger.error("  rec_push_job FAILED for user %s: %s", uid, e, exc_info=True)


async def retrain_job(light: bool = True):
    """:30 light-train or 03:00 deep-train. Never overlaps scrape (lock)."""
    kind = "light-train" if light else "deep-train"
    logger.info("⏰  [JOB] %s starting... RSS=%.0f MB", kind, _rss_mb())
    if not _trainer:
        logger.warning("  Trainer not initialised.")
        return
    if _train_locked():
        logger.warning("  %s skipped: lock held.", kind)
        return
    try:
        loop = asyncio.get_event_loop()
        svd, ncf, metrics = await loop.run_in_executor(
            None, lambda: _trainer.run(use_synthetic=False, light=light)
        )
        if _pipeline and metrics.get("version"):
            try:
                bundle = _trainer.load_max_bundle() if hasattr(_trainer, "load_max_bundle") else {}
            except Exception:
                bundle = {}
            _pipeline.set_models(svd, ncf, version=metrics.get("version", "v?"), max_bundle=bundle)
            _pipeline.fit_tfidf()
        logger.info("  %s complete. Metrics: %s RSS=%.0f MB", kind, metrics, _rss_mb())
    except Exception as e:
        logger.error("  %s FAILED: %s", kind, e, exc_info=True)


async def audio_listen_job():
    """Nightly 02:00 — the machine listens: ~150 priority tracks → audio_features."""
    logger.info("⏰  [JOB] audio_listen_job starting... RSS=%.0f MB", _rss_mb())
    if _train_locked():
        logger.warning("  listen skipped: training lock held.")
        return
    try:
        from audio.listener import listen_night
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, listen_night)
        logger.info("  Listening night complete: %s RSS=%.0f MB", stats, _rss_mb())
    except Exception as e:
        logger.error("  audio_listen_job FAILED: %s", e, exc_info=True)


async def catalog_clean_job():
    """Daily 04:00 — purge stale + VACUUM (weekly vacuum)."""
    logger.info("⏰  [JOB] catalog_clean_job... RSS=%.0f MB", _rss_mb())
    try:
        from core.database import get_conn, DB_CATALOG
        from core.scraper import prune_old_feed_snapshots
        cutoff = time.time() - 7 * 86400
        with get_conn(DB_CATALOG) as conn:
            cur = conn.execute(
                "DELETE FROM songs WHERE last_seen < ? AND times_fetched < 3",
                (cutoff,)
            )
            purged = cur.rowcount
            # Weekly VACUUM (cheap guard: only on Mondays ~04:00).
            try:
                import datetime
                if datetime.datetime.now().weekday() == 0:
                    conn.execute("VACUUM")
            except Exception as e:
                logger.warning("  VACUUM skipped: %s", e)
            logger.info("  Catalog: purged %d stale songs.", purged)
        prune_old_feed_snapshots()
    except Exception as e:
        logger.error("  catalog_clean_job FAILED: %s", e, exc_info=True)


def build_scheduler(user_ids: list):
    """Staggered cron schedule. Call .start() after loop is running."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. `pip install apscheduler`")
        return None

    scheduler = AsyncIOScheduler()
    scheduler.add_job(feed_refresh_job, CronTrigger(minute=0), id="feed_refresh",
                      name="YT Feed Refresh (:00)", replace_existing=True)
    scheduler.add_job(rec_push_job, CronTrigger(minute=10), args=[user_ids], id="rec_push",
                      name="Rec Push (:10)", replace_existing=True)
    scheduler.add_job(retrain_job, CronTrigger(minute=30), kwargs={"light": True}, id="light_train",
                      name="Light Train (:30)", replace_existing=True)
    scheduler.add_job(audio_listen_job, CronTrigger(hour=2, minute=0), id="audio_listen",
                      name="Audio Listen (02:00)", replace_existing=True)
    scheduler.add_job(retrain_job, CronTrigger(hour=3, minute=0), kwargs={"light": False}, id="deep_train",
                      name="Deep Train (03:00)", replace_existing=True)
    scheduler.add_job(feed_deep_job, CronTrigger(hour=3, minute=30), id="feed_deep",
                      name="Deep Feed (03:30)", replace_existing=True)
    scheduler.add_job(catalog_clean_job, CronTrigger(hour=4, minute=0), id="catalog_clean",
                      name="Catalog Clean (04:00)", replace_existing=True)
    logger.info("Scheduler built with 7 staggered jobs (:00/:10/:30/02:00/03:00/03:30/04:00).")
    return scheduler
