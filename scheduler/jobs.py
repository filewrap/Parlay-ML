"""
scheduler/jobs.py — APScheduler async job definitions

Jobs:
  feed_refresh_job  — every hour: scrape YT, save top-100
  rec_push_job      — every hour: generate top-10, push to Parlay bot
  retrain_job       — every 6h:  full model retrain
  catalog_clean_job — every 24h: remove songs not seen in 7+ days

All jobs are async-compatible and registered in the scheduler singleton.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("parlay.scheduler")

# These are populated by main.py after init
_pipeline = None
_trainer  = None
_bot_push_callback = None   # async fn(user_id, recs) — your Parlay bot hook


def register(pipeline, trainer, bot_push_callback=None):
    global _pipeline, _trainer, _bot_push_callback
    _pipeline = pipeline
    _trainer  = trainer
    _bot_push_callback = bot_push_callback


async def feed_refresh_job():
    """Hourly: scrape YouTube and update top-100 feed."""
    logger.info("⏰  [JOB] feed_refresh_job starting...")
    try:
        from core.scraper import scrape_yt_feed
        songs = await scrape_yt_feed()
        if _pipeline:
            _pipeline.fit_tfidf()   # re-fit TF-IDF on updated catalog
        logger.info(f"  Feed refreshed: {len(songs)} songs")
    except Exception as e:
        logger.error(f"  feed_refresh_job FAILED: {e}", exc_info=True)


async def rec_push_job(user_ids: list):
    """
    Hourly: generate top-10 recommendations for each active user.
    Calls bot_push_callback to deliver them to Telegram.
    """
    logger.info(f"⏰  [JOB] rec_push_job for {len(user_ids)} users...")
    if not _pipeline:
        logger.warning("  Pipeline not initialised.")
        return

    for uid in user_ids:
        try:
            recs = _pipeline.recommend(uid)
            if recs and _bot_push_callback:
                await _bot_push_callback(uid, recs)
            logger.info(f"  User {uid}: pushed {len(recs)} recs")
        except Exception as e:
            logger.error(f"  rec_push_job FAILED for user {uid}: {e}", exc_info=True)


async def retrain_job():
    """Every 6h: retrain SVD + NCF on accumulated data."""
    logger.info("⏰  [JOB] retrain_job starting...")
    if not _trainer:
        logger.warning("  Trainer not initialised.")
        return
    try:
        loop = asyncio.get_event_loop()
        svd, ncf, metrics = await loop.run_in_executor(
            None, lambda: _trainer.run(use_synthetic=False)
        )
        if _pipeline:
            _pipeline.set_models(svd, ncf, version=metrics.get("version", "v?"))
            _pipeline.fit_tfidf()
        logger.info(f"  Retrain complete. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"  retrain_job FAILED: {e}", exc_info=True)


async def catalog_clean_job():
    """Every 24h: purge songs not seen in 7 days."""
    logger.info("⏰  [JOB] catalog_clean_job...")
    try:
        from core.database import get_conn, DB_CATALOG
        cutoff = time.time() - 7 * 86400
        with get_conn(DB_CATALOG) as conn:
            cur = conn.execute(
                "DELETE FROM songs WHERE last_seen < ? AND times_fetched < 3",
                (cutoff,)
            )
            logger.info(f"  Catalog: purged {cur.rowcount} stale songs.")
    except Exception as e:
        logger.error(f"  catalog_clean_job FAILED: {e}", exc_info=True)


def build_scheduler(user_ids: list):
    """
    Build and return an APScheduler AsyncIOScheduler.
    Call .start() after event loop is running.
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from config import (
            FEED_REFRESH_INTERVAL, REC_PUSH_INTERVAL,
            MODEL_RETRAIN_INTERVAL, CATALOG_CLEAN_INTERVAL,
        )
    except ImportError:
        logger.error("APScheduler not installed. `pip install apscheduler`")
        return None

    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        feed_refresh_job,
        IntervalTrigger(seconds=FEED_REFRESH_INTERVAL),
        id="feed_refresh",
        name="YT Feed Refresh",
        replace_existing=True,
    )
    scheduler.add_job(
        rec_push_job,
        IntervalTrigger(seconds=REC_PUSH_INTERVAL),
        args=[user_ids],
        id="rec_push",
        name="Rec Push",
        replace_existing=True,
    )
    scheduler.add_job(
        retrain_job,
        IntervalTrigger(seconds=MODEL_RETRAIN_INTERVAL),
        id="retrain",
        name="Model Retrain",
        replace_existing=True,
    )
    scheduler.add_job(
        catalog_clean_job,
        IntervalTrigger(seconds=CATALOG_CLEAN_INTERVAL),
        id="catalog_clean",
        name="Catalog Clean",
        replace_existing=True,
    )

    logger.info("Scheduler built with 4 jobs.")
    return scheduler