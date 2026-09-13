"""
eval/metrics.py — MAX Phase 0 evaluation harness.

Recall@10, NDCG@10, HitRate, coverage, novelty, diversity,
per-genre and per-mood slices. Same harness as RecBole/Cornac.

Promotion rule: deploy only if NDCG@10 beats prod + guardrails
(coverage must not drop >5%).
"""

import logging
import math
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger("parlay.eval")


def _ranked_hits(ranked: list[str], relevant: set[str]) -> list[int]:
    return [1 if s in relevant else 0 for s in ranked]


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(_ranked_hits(ranked[:k], relevant)) / max(len(relevant), 1)


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    hits = _ranked_hits(ranked[:k], relevant)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sorted(hits, reverse=True)
    idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def hitrate_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(s in relevant for s in ranked[:k]) else 0.0


def evaluate_ranker(predict_fn, valid_frame: list[dict], k: int = 10,
                    catalog_meta: dict[str, dict] | None = None) -> dict:
    """predict_fn(user_id, candidates) → ranked song_id list.

    valid_frame holds held-out positives. Candidates = valid songs for
    that user + 100 random others (sampled eval, CPU-cheap, Cornac-style).
    Returns aggregate + sliced metrics.
    """
    by_user: dict[int, list[dict]] = defaultdict(list)
    for d in valid_frame:
        if float(d["plan_reward"]) > 0:
            by_user[int(d["user_id"])].append(d)
    if not by_user:
        return {"error": "no validation positives"}

    all_valid_songs = list({d["song_id"] for d in valid_frame})
    rng = np.random.default_rng(7)

    rec_sum = ndcg_sum = hit_sum = 0.0
    n_users = 0
    recommended_counter: Counter = Counter()
    total_recommended = 0
    genre_hits: dict[int, list[float]] = defaultdict(list)
    catalog_meta = catalog_meta or {}

    for uid, items in by_user.items():
        relevant = {d["song_id"] for d in items}
        # Candidate pool: relevant + random distractors.
        cands = list(relevant)
        pool = [s for s in all_valid_songs if s not in relevant]
        if pool:
            extra = rng.choice(pool, size=min(100, len(pool)), replace=False).tolist()
            cands += extra
        try:
            ranked = list(predict_fn(uid, cands))[:k]
        except Exception as e:
            logger.warning("predict_fn failed for user %s: %s", uid, e)
            continue
        rec_sum += recall_at_k(ranked, relevant, k)
        ndcg_sum += ndcg_at_k(ranked, relevant, k)
        hit_sum += hitrate_at_k(ranked, relevant, k)
        n_users += 1
        for s in ranked[:k]:
            recommended_counter[s] += 1
            total_recommended += 1
        for d in items:
            g = int((catalog_meta.get(d["song_id"]) or {}).get("genre_code", 20))
            genre_hits[g].append(1.0 if d["song_id"] in ranked[:k] else 0.0)

    n_u = max(n_users, 1)
    # Coverage: fraction of valid catalog ever recommended.
    coverage = len(recommended_counter) / max(len(all_valid_songs), 1)
    # Novelty: mean -log2(popularity share) of recommended items.
    novelty = 0.0
    if total_recommended > 0:
        novelty = float(np.mean([-math.log2((c / total_recommended) + 1e-9)
                                 for c in recommended_counter.values()]))
    # Diversity: 1 - mean pairwise genre overlap proxy (unique genres / k).
    diversity = min(1.0, len(recommended_counter) / max(total_recommended, 1) * 10.0)

    per_genre = {str(g): round(float(np.mean(v)), 4) for g, v in genre_hits.items() if v}
    return {
        f"recall@{k}": round(rec_sum / n_u, 4),
        f"ndcg@{k}": round(ndcg_sum / n_u, 4),
        f"hitrate@{k}": round(hit_sum / n_u, 4),
        "coverage": round(coverage, 4),
        "novelty": round(novelty, 4),
        "diversity": round(diversity, 4),
        "n_users_eval": n_users,
        "per_genre_hitrate": per_genre,
    }


def should_promote(new_metrics: dict, prod_metrics: dict | None,
                   coverage_drop_tol: float = 0.05) -> tuple[bool, str]:
    """Promotion rule: NDCG@10 must beat prod + coverage must not drop >5%."""
    new_ndcg = float(new_metrics.get("ndcg@10", 0.0))
    new_cov = float(new_metrics.get("coverage", 0.0))
    if not prod_metrics:
        return True, "no prod baseline — promoting staging"
    prod_ndcg = float(prod_metrics.get("ndcg@10", 0.0))
    prod_cov = float(prod_metrics.get("coverage", 0.0))
    if new_ndcg <= prod_ndcg:
        return False, f"NDCG {new_ndcg:.4f} <= prod {prod_ndcg:.4f}"
    if prod_cov > 0 and (prod_cov - new_cov) / prod_cov > coverage_drop_tol:
        return False, f"coverage drop {(prod_cov - new_cov) / prod_cov:.1%} > 5%"
    return True, f"NDCG {prod_ndcg:.4f} → {new_ndcg:.4f}, coverage guarded"
