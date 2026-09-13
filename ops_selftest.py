#!/usr/bin/env python3
"""ops_selftest.py — plain python, no pytest. Proves the operator layer.

Covers: pidfile lifecycle, job log, manifest columns, safe rollback,
MCP handshake (no models loaded).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    from ops.status import write_pidfile, clear_pidfile, daemon_state
    from ops.joblog import log_job, recent_jobs

    # 1. pidfile roundtrip (points at THIS process, which lacks main.py cmdline).
    write_pidfile()
    d = daemon_state()
    assert d["pid"] == os.getpid() and d["alive"] is False, d
    clear_pidfile()
    assert daemon_state()["alive"] is False
    print("pidfile ok")

    # 2. job log roundtrip.
    log_job("_selftest_job", 1.5, True, "probe")
    assert any(j["job"] == "_selftest_job" and j["ok"] == 1 for j in recent_jobs(50))
    print("joblog ok")

    # 3. manifest columns exist.
    from core.database import get_conn
    from config import DB_TRAINING
    with get_conn(DB_TRAINING) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(training_runs)").fetchall()]
    assert {"git_sha", "config_json", "frame_hash"} <= set(cols), cols
    print("manifest cols ok")

    # 4. rollback is safe: noop stays noop; a real rollback restores itself.
    from trainer.engine import rollback_prod
    r1 = rollback_prod()
    assert r1["status"] in ("noop", "rolled-back"), r1
    if r1["status"] == "rolled-back":
        r2 = rollback_prod()
        assert r2["status"] == "rolled-back" and r2["to_run"] == r1["from_run"], (r1, r2)
        print("rollback ok: went and returned:", r1["to_run"], "->", r2["to_run"])
    else:
        print("rollback ok (noop, single prod entry):", r1["reason"])

    # 5. MCP handshake without loading models.
    sys.path.insert(0, ".")
    import mcp_server
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["protocolVersion"] == "2024-11-05", init
    tools = mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in tools["result"]["tools"]]
    assert {"status", "recommend", "feel_like", "dossier", "listen_to",
            "heard_library", "agent_brief"} <= set(names), names
    err = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "nope", "params": {}})
    assert "error" in err
    print("mcp handshake ok:", names)

    print("ALL OPS SELFTESTS PASSED")


if __name__ == "__main__":
    main()
