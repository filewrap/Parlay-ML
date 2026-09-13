"""status.py — the truth in one JSON blob. The agent's first command."""

import os
import shutil
import time

PIDFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "store", "daemon.pid")


def write_pidfile() -> str:
    path = os.path.abspath(PIDFILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{os.getpid()}\n{time.time()}\n")
    return path


def clear_pidfile() -> None:
    try:
        os.remove(os.path.abspath(PIDFILE))
    except OSError:
        pass


def daemon_state() -> dict:
    """Is the engine alive? pidfile + /proc check, no guessing."""
    path = os.path.abspath(PIDFILE)
    try:
        with open(path) as fh:
            pid, started = fh.read().strip().split()
        pid = int(pid)
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().decode("utf-8", "replace")
        alive = "main.py" in cmd
        return {"alive": alive, "pid": pid, "uptime_s": round(time.time() - float(started), 0) if alive else 0}
    except Exception:
        return {"alive": False, "pid": None, "uptime_s": 0}


def status() -> dict:
    """Everything an operator (human or agent) needs in one call."""
    from config import DB_CATALOG, DB_FEED, DB_TRAINING, PRECOMPUTE_DIR, DATA_DIR
    from core.database import get_conn
    from ops.joblog import recent_jobs

    out: dict = {"at": time.time(), "daemon": daemon_state()}
    # Jobs: last run of each known job.
    try:
        jobs = recent_jobs(40)
        last: dict[str, dict] = {}
        for j in jobs:
            last.setdefault(j["job"], {"ok": bool(j["ok"]), "secs": j["secs"],
                                       "ago_s": round(time.time() - j["started_at"], 0),
                                       "note": j["note"]})
        out["jobs"] = last
    except Exception as e:
        out["jobs"] = {"error": str(e)}
    # Registry: prod vs staging.
    try:
        from trainer.engine import get_registry
        out["registry"] = {"prod": get_registry("prod"), "staging": get_registry("staging")}
    except Exception as e:
        out["registry"] = {"error": str(e)}
    # Pools + hearing.
    try:
        with get_conn(DB_CATALOG) as conn:
            out["catalog_songs"] = conn.execute("SELECT COUNT(*) c FROM songs").fetchone()["c"]
            try:
                out["heard_7d"] = conn.execute(
                    "SELECT COUNT(*) c FROM audio_features WHERE analyzed_at > ?",
                    (time.time() - 7 * 86400,)).fetchone()["c"]
            except Exception:
                out["heard_7d"] = 0
        with get_conn(DB_FEED) as conn:
            snap = conn.execute("SELECT fetched_at FROM feed_snapshots ORDER BY fetched_at DESC LIMIT 1").fetchone()
            out["feed_latest_ago_s"] = round(time.time() - snap["fetched_at"], 0) if snap else None
        out["precompute_files"] = len(list(PRECOMPUTE_DIR.glob("top200_*.json")))
    except Exception as e:
        out["pools_error"] = str(e)
    # Disk.
    try:
        du = shutil.disk_usage(str(DATA_DIR))
        out["disk"] = {"free_gb": round(du.free / 1e9, 1), "used_pct": round(100 * du.used / du.total, 1)}
    except Exception as e:
        out["disk"] = {"error": str(e)}
    return out
