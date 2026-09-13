"""gate — the opening where Music reaches AI agents and models.

LLMs trade in text tokens; feelings aren't tokens. The Gate breaks that
rule by grounding agents in measured affect: every number below was
heard from audio, not imagined from words.

  state.py    listen ledger (append-only) + continuous taste graph
  api.py      what agents call: status, emotion, taste, feel-like, dossier
  announce.py "Parlay says" — listening events rendered as messages

The graph never goes stale: it is derived live from the ledger, so every
listen rewires it. Continuous updating by construction.
"""

from .announce import announce_listen
from .api import (agent_brief, dossier, emotion_now, feel_like, heard_library,
                  listen_to, status, taste_profile, taste_graph)

__all__ = ["announce_listen", "agent_brief", "dossier", "emotion_now",
           "feel_like", "heard_library", "listen_to", "status",
           "taste_profile", "taste_graph"]
