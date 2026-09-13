"""state.py — append-only listen ledger + live taste graph.

Every hearing appends one ledger row (when, what, source, mean affect,
caption). The taste graph derives from ledger + audio_features on every
read: nodes for tracks/keys/moods, edges for listen succession and
shared harmonic color. Nothing cached, nothing stale — the graph is
continuous because it is recomputed from the truth each time.
"""

import logging
import time

from config import DB_CATALOG
from core.database import get_conn

logger = logging.getLogger("parlay.gate.state")

SCHEMA = """
CREATE TABLE IF NOT EXISTS listen_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    song_id     TEXT NOT NULL,
    title       TEXT DEFAULT '',
    source      TEXT DEFAULT '',
    clip_secs   REAL DEFAULT 0,
    bpm         REAL DEFAULT 0,
    musical_key TEXT DEFAULT '',
    mode        TEXT DEFAULT '',
    arousal     REAL DEFAULT 0.5,
    valence     REAL DEFAULT 0.5,
    caption     TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_ledger_ts ON listen_ledger(ts DESC);
CREATE INDEX IF NOT EXISTS idx_ledger_song ON listen_ledger(song_id);
"""


def ensure_schema() -> None:
    with get_conn(DB_CATALOG) as conn:
        conn.executescript(SCHEMA)


def record_listen(song_id: str, title: str = "", summary: dict | None = None,
                  caption: str = "") -> int:
    """Append one hearing. Returns ledger id."""
    ensure_schema()
    s = summary or {}
    arousal = float(s.get("energy", 0.5) or 0.5) * 0.6 + float(s.get("danceability", 0.5) or 0.5) * 0.4
    with get_conn(DB_CATALOG) as conn:
        cur = conn.execute("""
        INSERT INTO listen_ledger
            (ts, song_id, title, source, clip_secs, bpm, musical_key, mode,
             arousal, valence, caption)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (time.time(), str(song_id), str(title), str(s.get("source", "")),
              float(s.get("clip_secs", 0)), float(s.get("bpm", 0)),
              str(s.get("key", "")), str(s.get("mode", "")),
              round(max(0.0, min(1.0, arousal)), 3),
              float(s.get("valence", 0.5) or 0.5), str(caption)))
        return int(cur.lastrowid)


def ledger(limit: int = 50) -> list[dict]:
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        rows = conn.execute("SELECT * FROM listen_ledger ORDER BY ts DESC LIMIT ?",
                            (limit,)).fetchall()
    return [dict(r) for r in rows]


def taste_graph() -> dict:
    """Live graph: track nodes, key/mood hubs, succession + color edges."""
    ensure_schema()
    with get_conn(DB_CATALOG) as conn:
        try:
            feats = conn.execute("SELECT * FROM audio_features").fetchall()
        except Exception:
            feats = []
        seq = conn.execute("SELECT song_id FROM listen_ledger ORDER BY ts").fetchall()
    feats = [dict(r) for r in feats]
    by_id = {f["song_id"]: f for f in feats}
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], float] = {}

    def _node(nid: str, kind: str, label: str, **kw):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "kind": kind, "label": label, **kw}
        return nodes[nid]

    def _edge(a: str, b: str, w: float, kind: str):
        if a == b:
            return
        k = (a, b, kind)
        edges[k] = edges.get(k, 0.0) + w

    for f in feats:
        sid = f["song_id"]
        _node(f"t:{sid}", "track", sid, key=f.get("musical_key", ""),
              mode=f.get("mode", ""), valence=f.get("valence", 0.5),
              bpm=f.get("bpm", 0))
        if f.get("musical_key"):
            kid = f"key:{f['musical_key']}-{f.get('mode', '')}"
            _node(kid, "key", kid[4:])
            _edge(f"t:{sid}", kid, 1.0, "in-key")
        mood = ("low" if float(f.get("valence", 0.5) or 0.5) < 0.35
                else ("high" if float(f.get("valence", 0.5) or 0.5) > 0.6 else "mid"))
        _node(f"mood:{mood}", "mood", mood)
        _edge(f"t:{sid}", f"mood:{mood}", 1.0, "feels")
    order = [r["song_id"] for r in seq]
    for a, b in zip(order, order[1:]):
        _edge(f"t:{a}", f"t:{b}", 1.0, "followed")
        fa, fb = by_id.get(a), by_id.get(b)
        if fa and fb and fa.get("musical_key") and fa.get("musical_key") == fb.get("musical_key"):
            _edge(f"t:{a}", f"t:{b}", 0.5, "same-color")
    return {"nodes": list(nodes.values()),
            "edges": [{"a": a, "b": b, "kind": k, "w": round(w, 2)} for (a, b, k), w in edges.items()],
            "n_listens": len(order), "n_heard": len(feats)}
