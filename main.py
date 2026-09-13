#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║           PARLAY ML — NEUROSYNC ENGINE v1.0              ║
║   main.py — boot, wire, and run the NeuroSync engine     ║
║   Named Feature: NeuroSync (Beta)                        ║
╚══════════════════════════════════════════════════════════╝

NeuroSync: Because good music finds you, not the other way around.

This is the only entrypoint. It wires the modules together in exactly the
order their own docstrings specify:

  1. core.database.bootstrap_all()          → create all 8 SQLite schemas
                                              ("Call once at startup")
  2. trainer.engine.TrainingEngine          → load_latest() checkpoints, or
                                              train (SyntheticDataGenerator
                                              bootstraps 10,000 songs)
  3. pipeline.recommender.NeuroSyncPipeline → set_models() + fit_tfidf()
  4. scheduler.jobs.register()              → populate the module-level
                                              _pipeline / _trainer /
                                              _bot_push_callback globals
  5. scheduler.jobs.build_scheduler()       → 4 interval jobs
  6. scheduler.start()                      → AFTER the event loop is running

Registered jobs (see scheduler/jobs.py):
  feed_refresh   every FEED_REFRESH_INTERVAL   (1h)  → scrape YT top-100
  rec_push       every REC_PUSH_INTERVAL       (1h)  → top-10 per user → bot
  retrain        every MODEL_RETRAIN_INTERVAL  (6h)  → SVD + NCF retrain
  catalog_clean  every CATALOG_CLEAN_INTERVAL  (24h) → purge stale songs

The bot hook
────────────
scheduler/jobs.py delivers by calling `await _bot_push_callback(user_id, recs)`.
Point NeuroSync at your Parlay bot WITHOUT editing any file:

    export PARLAY_PUSH_CALLBACK="mypkg.telegram_bot:push_recommendations"

Built-in Telegram delivery (stdlib only, no new deps):

    export TELEGRAM_BOT_TOKEN="123456:ABC-..."
    export PARLAY_PUSH_CALLBACK="notifier.telegram:push_recommendations"
    python -m notifier.owner_profile          # seed AgainOwner taste once
    python main.py --once --user-ids 6802929470

If PARLAY_PUSH_CALLBACK is unset but TELEGRAM_BOT_TOKEN is set, the
built-in notifier.telegram:push_recommendations is used automatically.
Otherwise the logging stub renders the keyboard layout into the log.

Fixed in this tree (were Known gaps)
───────────────────────────────────
  * scheduler/jobs.py rec_push now passes the coroutine + args directly
    to AsyncIOScheduler, so the hourly push fires.
  * trainer/engine.py seed_db() now uses 5 placeholders for 5 params.

Recipients
──────────
`scheduler/jobs.build_scheduler()` takes a fixed `user_ids` list, so recipients
are resolved once at boot from the listened/feedback data
(`--user-ids 1,2,3` overrides). AgainOwner (6802929470) is always included
in auto-discovery so the first run pushes even before any listen exists.
New users are picked up on the next restart.

Usage
─────
  python main.py                        # daemon mode (hourly push)
  python main.py --once                 # one feed + one rec cycle, then exit
  python main.py --bootstrap            # seed 10k synthetic songs, train, run
  python main.py --train                # retrain on real data at boot, then run
  python main.py --user-ids 1,2,3       # explicit recipients
  python main.py --refresh-feed         # force a YT scrape before starting
"""

import argparse
import asyncio
import importlib
import inspect
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from typing import Awaitable, Callable, Optional

from config import (
    CATALOG_CLEAN_INTERVAL,
    FEED_REFRESH_INTERVAL,
    LOGS_DIR,
    MODELS_DIR,
    MODEL_RETRAIN_INTERVAL,
    REC_PUSH_INTERVAL,
)

from core.database import DB_FEED, DB_FEEDBACK, DB_HISTORY, bootstrap_all, get_conn
from pipeline.recommender import NeuroSyncPipeline
from scheduler import jobs
from trainer.engine import TrainingEngine

# ─── Feature identity ─────────────────────────────────────
FEATURE_NAME = "NeuroSync"
FEATURE_TAG = "Beta"

logger = logging.getLogger("parlay.main")

BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║           PARLAY ML — NEUROSYNC ENGINE v1.0              ║
║   Named Feature: NeuroSync (Beta)                        ║
║   Tiny Parlay-ML rooted into the recommendation core     ║
╚══════════════════════════════════════════════════════════╝
"""


# ─── Logging ──────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """
    Route the `parlay.*` logger tree to stdout + a rotating file in LOGS_DIR.
    Called by core/database.py, core/scraper.py, models/*, pipeline/*,
    trainer/* and scheduler/* all use `logging.getLogger("parlay.<name>")`.
    """
    root = logging.getLogger("parlay")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)-18s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        file_handler = RotatingFileHandler(
            LOGS_DIR / "neurosync.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError as e:                      # read-only / storage edge case
        root.warning("Could not open log file (%s). Console logging only.", e)


# ─── Helpers ──────────────────────────────────────────────

def _fmt_interval(seconds: int) -> str:
    """Render a scheduler interval in human units for the boot table."""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _has_feed_snapshot() -> bool:
    """True once at least one hourly YT feed snapshot has been stored."""
    with get_conn(DB_FEED) as conn:
        row = conn.execute("SELECT 1 FROM feed_snapshots LIMIT 1").fetchone()
    return row is not None


def get_active_users(explicit: Optional[list[int]] = None) -> list[int]:
    """
    Resolve the recipient list for `scheduler.jobs.build_scheduler(user_ids)`.

    There is no separate users table — a user is anyone who has produced a
    listen record (DB_HISTORY.listens) or a +/− feedback signal
    (DB_FEEDBACK.feedback). Pass --user-ids to bypass discovery.
    AgainOwner is always included in auto-discovery so Telegram delivery
    works on a cold boot (see notifier.owner_profile).
    """
    if explicit:
        return sorted({int(u) for u in explicit})

    users: set[int] = set()
    with get_conn(DB_HISTORY) as conn:
        rows = conn.execute("SELECT DISTINCT user_id FROM listens").fetchall()
        users.update(r["user_id"] for r in rows)

    with get_conn(DB_FEEDBACK) as conn:
        rows = conn.execute("SELECT DISTINCT user_id FROM feedback").fetchall()
        users.update(r["user_id"] for r in rows)

    try:
        from config import OWNER_TELEGRAM_ID
        users.add(int(OWNER_TELEGRAM_ID))
    except Exception:
        pass

    return sorted(users)


async def _logging_push_callback(user_id: int, recs: list) -> None:
    """
    Stub bot hook used when PARLAY_PUSH_CALLBACK is unset.

    scheduler/jobs.py calls `await _bot_push_callback(uid, recs)`; this renders
    the exact delivery shape ("10 songs per hour … with buttons as + −") into
    the log so a full cycle can be verified without a running bot.
    """
    logger.info("📲  → user %s : %d recommendations", user_id, len(recs))
    for rec in recs:
        logger.info(
            "      [%2d] %-55s score=%.4f   [ + ]  [ − ]",
            rec.get("rank", 0),
            (rec.get("title") or "")[:55],
            rec.get("final_score", 0.0),
        )
    if recs:
        logger.info("      session=%s  (feedback → record_feedback)", recs[0].get("session_id"))


def resolve_push_callback() -> Callable[[int, list], Awaitable[None]]:
    """
    Load the Parlay bot hook from PARLAY_PUSH_CALLBACK ("module.path:function").
    If unset but TELEGRAM_BOT_TOKEN is configured, use the built-in
    notifier.telegram:push_recommendations so the pipeline is complete
    out of the box. Falls back to the logging stub otherwise.
    """
    spec = os.environ.get("PARLAY_PUSH_CALLBACK", "").strip()
    if not spec:
        try:
            from config import TELEGRAM_BOT_TOKEN
        except Exception:
            TELEGRAM_BOT_TOKEN = ""
        if TELEGRAM_BOT_TOKEN:
            spec = "notifier.telegram:push_recommendations"
        else:
            logger.warning(
                "PARLAY_PUSH_CALLBACK not set — using the logging stub. "
                "Set it to \"module.path:async_fn\" or set TELEGRAM_BOT_TOKEN "
                "to deliver Top-10 to AgainOwner via notifier.telegram."
            )
            return _logging_push_callback

    module_path, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_path)
        callback = getattr(module, attr or "push_recommendations")
    except (ImportError, AttributeError) as e:
        logger.error("Cannot load push callback %r: %s — using stub.", spec, e)
        return _logging_push_callback

    if not inspect.iscoroutinefunction(callback):
        logger.error("Push callback %r is not an async function — using stub.", spec)
        return _logging_push_callback

    logger.info("Bot push callback wired → %s", spec)
    return callback


def _install_shutdown(stop: asyncio.Event) -> None:
    """SIGINT/SIGTERM → set the stop event so the scheduler shuts down cleanly."""
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, ValueError):
            pass    # platform without loop signal support; KeyboardInterrupt still works


def print_banner(user_ids: list[int]) -> None:
    print(BANNER)
    print(f"  Feature   : {FEATURE_NAME} ({FEATURE_TAG})")
    print(f"  Models    : {MODELS_DIR}")
    print(f"  Logs      : {LOGS_DIR / 'neurosync.log'}")
    print(f"  Recipients: {len(user_ids)} user(s) {user_ids[:10]}"
          f"{' …' if len(user_ids) > 10 else ''}")
    print("  Schedule  :")
    print(f"     feed_refresh   every {_fmt_interval(FEED_REFRESH_INTERVAL)}")
    print(f"     rec_push       every {_fmt_interval(REC_PUSH_INTERVAL)}")
    print(f"     retrain        every {_fmt_interval(MODEL_RETRAIN_INTERVAL)}")
    print(f"     catalog_clean  every {_fmt_interval(CATALOG_CLEAN_INTERVAL)}")
    print()


# ─── Boot sequence ────────────────────────────────────────

async def _async_main(args: argparse.Namespace) -> int:
    loop = asyncio.get_running_loop()

    # ── 1. Schema: "Call once at startup to ensure all tables exist." ──
    bootstrap_all()

    # ── 0. Operator commands (no daemon): --status/--rollback/--clean-dry-run ──
    if getattr(args, "status", False):
        import json as _json
        from ops.status import status as _ops_status
        print(_json.dumps(_ops_status(), indent=2, default=str))
        return 0
    if getattr(args, "rollback", False):
        import json as _json
        from trainer.engine import rollback_prod
        print(_json.dumps(rollback_prod(), indent=2, default=str))
        return 0
    if getattr(args, "clean_dry_run", False):
        n = await jobs.catalog_clean_job(dry_run=True)
        print(f"DRY RUN: would remove ~{n} catalog rows + old snapshots. Nothing changed.")
        return 0

    # ── 2. Models: load latest checkpoints, or train (MAX: one frame, all models) ──
    trainer = TrainingEngine()
    svd = ncf = None
    max_bundle: dict = {}

    if args.bootstrap or args.train or args.light_train or args.deep_train or args.smoke:
        logger.info("Training at boot (synthetic=%s light=%s deep=%s smoke=%s)...",
                    args.bootstrap, args.light_train, args.deep_train, args.smoke)
        try:
            light = bool(args.light_train) and not bool(args.deep_train)
            svd, ncf, metrics = await loop.run_in_executor(
                None,
                lambda: trainer.run(
                    use_synthetic=args.bootstrap,
                    n_synthetic_songs=args.synthetic_songs,
                    n_synthetic_users=args.synthetic_users,
                    synthetic_interactions_per_user=args.synthetic_interactions,
                    light=light,
                    force_ncf=bool(args.weekly_ncf),
                    smoke=bool(args.smoke),
                ),
            )
            try:
                max_bundle = {} if args.smoke else trainer.load_max_bundle()
            except Exception:
                max_bundle = {}
        except Exception as e:
            logger.error("Training failed (%s) — aborting.", e, exc_info=True)
            return 1
        logger.info("Training complete → %s", metrics)
        if args.smoke:
            import json as _json
            print("SMOKE:", _json.dumps(metrics, indent=2, default=str))
            return 0
    else:
        svd, ncf = trainer.load_latest()
        try:
            max_bundle = {"als": getattr(trainer, "als", None),
                          "feature_mf": getattr(trainer, "feature_mf", None),
                          "retrievers": getattr(trainer, "retrievers", None),
                          "sequence": getattr(trainer, "sequence", None),
                          "two_tower": getattr(trainer, "two_tower", None),
                          "blender": getattr(trainer, "blender", None)}
        except Exception:
            max_bundle = {}
        if svd is None and ncf is None and not any(max_bundle.values()):
            logger.warning(
                "No checkpoints in %s — the ensemble will run degraded "
                "(content + bandit + recency only). Use --bootstrap or --train.",
                MODELS_DIR,
            )

    version = (
        getattr(svd, "version", None)
        or getattr(ncf, "version", None)
        or getattr(max_bundle.get("als"), "version", None)
        or "v0"
    )

    # ── 3. Pipeline + content scorer (MAX bundle) ────────────
    pipeline = NeuroSyncPipeline(svd_model=svd, ncf_model=ncf, max_bundle=max_bundle)
    pipeline.set_models(svd, ncf, version=version, max_bundle=max_bundle)
    pipeline.fit_tfidf()

    # ── 4. Populate the scheduler/jobs.py globals ─────────────────────
    push_callback = resolve_push_callback()
    jobs.register(pipeline=pipeline, trainer=trainer, bot_push_callback=push_callback)

    # ── Recipients (resolved once — build_scheduler takes a fixed list) ──
    user_ids = get_active_users(args.user_ids)
    logger.info("Resolved %d recipient(s) from listens + feedback.", len(user_ids))

    print_banner(user_ids)

    # ── 5. First feed: seed the catalog on a cold boot ────────────────
    # --once is documented as "one feed + one rec cycle", so it always scrapes.
    if args.once or args.refresh_feed:
        logger.info("Refreshing the YT feed...")
        await jobs.feed_refresh_job()
    elif not _has_feed_snapshot():
        logger.info("No feed snapshot yet — running the first YT feed refresh...")
        await jobs.feed_refresh_job()

    # ── --once: one full cycle through the real job functions, then exit ──
    if args.once:
        logger.info("--once: running a single rec_push cycle (mood=%r).", getattr(args, "mood", ""))
        mood = getattr(args, "mood", "") or ""
        if mood:
            for uid in user_ids:
                try:
                    recs = pipeline.recommend(uid, mood=mood)
                    if recs:
                        await push_callback(uid, recs)
                except Exception as e:
                    logger.error("once mood push failed for %s: %s", uid, e, exc_info=True)
        else:
            await jobs.rec_push_job(user_ids)
        logger.info("--once complete.")
        return 0

    # ── 6. Scheduler: build → start (after the loop is running) → wait ──
    scheduler = jobs.build_scheduler(user_ids)
    if scheduler is None:
        logger.error(
            "Cannot start: APScheduler unavailable. `pip install -r requirements.txt`"
        )
        return 1

    scheduler.start()
    from ops.status import write_pidfile, clear_pidfile
    write_pidfile()
    logger.info("✅  %s (%s) running. Ctrl-C to stop.", FEATURE_NAME, FEATURE_TAG)

    stop = asyncio.Event()
    _install_shutdown(stop)
    try:
        await stop.wait()
    finally:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)
        clear_pidfile()

    return 0


# ─── CLI ──────────────────────────────────────────────────

def _csv_ints(raw: str) -> list[int]:
    try:
        return [int(x) for x in raw.replace(" ", "").split(",") if x]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {raw!r}"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurosync",
        description=f"Parlay-ML {FEATURE_NAME} ({FEATURE_TAG}) engine — hourly "
                    f"feed → 4-way ensemble → top-10 → +/− feedback → retrain.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="run one feed refresh + one rec_push cycle through the real job "
             "functions, then exit (no scheduler).",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="seed the 10k-song SyntheticDataGenerator and train before running.",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="MAX deep-train: ALS + FeatureMF + retrievers + sequence + SVD (+ weekly NCF) on one frame.",
    )
    parser.add_argument(
        "--light-train", action="store_true",
        help="MAX light-train (:30 hourly): ALS + retrievers + two-tower + sequence only.",
    )
    parser.add_argument(
        "--deep-train", action="store_true",
        help="Alias for --train (03:00 deep job).",
    )
    parser.add_argument(
        "--weekly-ncf", action="store_true",
        help="Force torch NCF training even when weekly window has not elapsed.",
    )
    parser.add_argument(
        "--mood", default="", help="Context mood for --once (morning/gym/4am/chill/sad/party/focus).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print ops status JSON (daemon, jobs, registry, pools, disk) and exit.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="60-second training wiring check: sampled frame, toy models, no registry writes.",
    )
    parser.add_argument(
        "--clean-dry-run", action="store_true",
        help="Report what catalog clean would delete. Changes nothing.",
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Point prod at the previous registry entry and exit.",
    )
    parser.add_argument(
        "--refresh-feed", action="store_true",
        help="force a YT feed scrape at boot even if a snapshot already exists.",
    )
    parser.add_argument(
        "--user-ids", type=_csv_ints, default=None, metavar="1,2,3",
        help="explicit recipient ids (default: auto-discovered from listens + feedback).",
    )
    parser.add_argument(
        "--synthetic-songs", type=int, default=10000,
        help="song count for --bootstrap (default: 10000).",
    )
    parser.add_argument(
        "--synthetic-users", type=int, default=100,
        help="user count for --bootstrap (default: 100).",
    )
    parser.add_argument(
        "--synthetic-interactions", type=int, default=200,
        help="interactions per synthetic user for --bootstrap (default: 200).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log verbosity (default: INFO).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
