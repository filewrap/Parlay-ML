#!/usr/bin/env python3
"""mcp_server.py — the Gate, speaking MCP over stdio. No new deps.

Lets any MCP-capable agent touch Parlay-ML: status, recommend, feel,
dossiers, listening, library, grounding brief. Hand-rolled JSON-RPC
(minimal MCP subset: initialize, tools/list, tools/call, ping).

Run:  /home/ubuntu/Parlay/venv/bin/python mcp_server.py
Test: echo '<json>' | python mcp_server.py
"""

import json
import os
import sys
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROTOCOL = "2024-11-05"

_pipeline = None


def _ensure_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    from trainer.engine import TrainingEngine
    from pipeline.recommender import NeuroSyncPipeline
    e = TrainingEngine()
    svd, ncf = e.load_latest()
    bundle = {"als": getattr(e, "als", None), "feature_mf": getattr(e, "feature_mf", None),
              "retrievers": getattr(e, "retrievers", None),
              "sequence": getattr(e, "sequence", None),
              "two_tower": getattr(e, "two_tower", None),
              "blender": getattr(e, "blender", None)}
    p = NeuroSyncPipeline(svd, ncf, bundle)
    p.set_models(svd, ncf, version="mcp", max_bundle=bundle)
    p.fit_tfidf()
    _pipeline = p
    return p


TOOLS = [
    {"name": "status", "description": "Ops truth: daemon, job runs, prod/staging, pools, disk.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "recommend", "description": "Top-10 for a user, with whys. Loads models on first call (~30s).",
     "inputSchema": {"type": "object", "properties": {
         "user_id": {"type": "integer"}, "mood": {"type": "string"}, "top_k": {"type": "integer"}},
         "required": ["user_id"]}},
    {"name": "feel_like", "description": "Songs that FEEL like a heard track or mood. Pure measured affect.",
     "inputSchema": {"type": "object", "properties": {
         "song_id": {"type": "string"}, "mood": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "dossier", "description": "Full hearing of one track: measured profile, arc, caption, kin.",
     "inputSchema": {"type": "object", "properties": {"song_id": {"type": "string"}},
                     "required": ["song_id"]}},
    {"name": "listen_to", "description": "Ask the machine to hear a new song (title + artist).",
     "inputSchema": {"type": "object", "properties": {"title": {"type": "string"},
                                                     "artist": {"type": "string"}},
                     "required": ["title"]}},
    {"name": "heard_library", "description": "Every heard track with profile + caption.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "agent_brief", "description": "Prompt-ready grounding in measured affect. Never invent feelings.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _text(data) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(data, indent=2, default=str)}]}


def call_tool(name: str, args: dict) -> dict:
    if name == "status":
        from ops.status import status as _s
        return _text(_s())
    if name == "recommend":
        p = _ensure_pipeline()
        recs = p.recommend(int(args["user_id"]), top_k=int(args.get("top_k", 10)),
                           mood=str(args.get("mood", "")))
        return _text(recs)
    if name == "feel_like":
        from gate import feel_like as _f
        return _text(_f(str(args.get("song_id", "")), mood=str(args.get("mood", "")),
                        limit=int(args.get("limit", 5))))
    if name == "dossier":
        from gate import dossier as _d
        return _text(_d(str(args["song_id"])))
    if name == "listen_to":
        from gate import listen_to as _l
        return _text(_l(str(args["title"]), str(args.get("artist", ""))))
    if name == "heard_library":
        from gate import heard_library as _h
        return _text(_h())
    if name == "agent_brief":
        from gate import agent_brief as _b
        return {"content": [{"type": "text", "text": _b()}]}
    raise ValueError(f"unknown tool: {name}")


def handle(msg: dict):
    mid = msg.get("id")
    method = msg.get("method", "")

    def _ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _err(code: int, message: str):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    try:
        if method == "initialize":
            return _ok({"protocolVersion": PROTOCOL,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "parlay-ml", "version": "1.0"}})
        if method == "ping":
            return _ok({})
        if method == "tools/list":
            return _ok({"tools": TOOLS})
        if method == "tools/call":
            params = msg.get("params", {})
            try:
                return _ok(call_tool(params.get("name", ""), params.get("arguments", {}) or {}))
            except Exception as e:
                return _err(-32000, f"tool failed: {e}")
        return _err(-32601, f"unknown method: {method}")
    except Exception:
        return _err(-32603, traceback.format_exc(limit=3))


def main() -> None:
    stdin = sys.stdin
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("method", "").startswith("notifications/"):
            continue
        sys.stdout.write(json.dumps(handle(msg), default=str) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
