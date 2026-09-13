# PARLAY-ML MAX — complete plan

Goal: Parlay-ML as a **music companion**, not just a recommender,
running at maximum output on the real constraint set.

Constraints (non-negotiable):

- 8 GB RAM VPS, **no GPU**, CPU-only training and serving.
- No heavy new deps. numpy + SQLite + stdlib first, torch weekly-only.
- Steal the **math and data-processing** of LightFM / implicit /
  Cornac / RecBole — reimplemented natively, never thin wrappers
  like `LightFM.suggest()`.

Current base: FunkSVD + torch NCF + TF-IDF + Thompson bandit +
recency + MMR, SQLite x8, yt-dlp scraper, APScheduler, Telegram push
to AgainOwner (6802929470).

---

## 0. Operating principles

1. **Confidence, not ratings.** Plays are implicit feedback.
   `confidence c = 1 + alpha * reward` with
   full-play=3, like=5, partial=1, skip-fast=-2, dislike=-5.
   This is the `implicit` worldview and everything below eats it.
2. **One data pipeline.** Every model trains and evals on the same
   interaction frame. No per-model SQL dialects.
3. **Retrievers feed a ranker.** 5+ cheap retrievers build a ~2000
   candidate pool; one learned blender picks Top-10. Never one
   model doing both jobs.
4. **Temporal truth.** No random 80/20 splits (they leak the future).
   Leave-last-N-per-user + global time cutoff, RecBole-style.
5. **Stagger the VPS.** Never scrape + train at once. Workers 4–6,
   not 8. Precompute nightly, rerank hourly.

---

## Phase 0 — Data foundation (Week 1, do first)

**New: `data/interactions.py`**

- Unified builder: listens (history.db) + feedback (feedback.db)
  → `(user_id, song_id, confidence, timestamp)`.
- Time decay at train time (30-day half-life on confidence),
  not only at score time.
- Negative sampler: 4 negatives per positive
  (50% uniform, 50% popularity-biased). Required for all
  pairwise models below.

**New: `data/split.py`**

- Temporal split + leave-last-5-per-user. Replaces random shuffle
  in `trainer/engine.py`.
- Saves split fingerprint (cutoff ts, hash) into training_log.db
  so runs are comparable.

**New: `eval/metrics.py`**

- Recall@10, NDCG@10, HitRate, coverage, novelty, diversity,
  per-genre and per-mood slices. Same harness as RecBole/Cornac.
- Promotion rule: deploy only if NDCG@10 beats prod + guardrails
  (coverage must not drop >5%).

**Edit: `trainer/engine.py`**

- `TrainingEngine.run()` trains every model on the same frame,
  logs all metrics + model paths + split fingerprint,
  keeps `staging` vs `prod` pointers (model registry in
  training_log.db).

Acceptance: every model trains from one call, eval numbers are
comparable across runs.

---

## Phase 1 — Scrape-max (Week 2, more data, same VPS)

- Keep hourly fast job (top-100). Add nightly **deep job**
  `feed_deep` (2–5k): per-artist discographies (Anuv Jain, Talha
  Anjum, Prateek Kuhad, Seema Mishra), per-mood queries, chart
  queries, related-query expansion ("songs like Mehrama").
- yt-dlp workers 4–6. Keep description text → free genre/mood
  keywords + artist graph (artist → song co-counts).
- Bloom dedup stays. Snapshot retention: 7 days of feed_songs,
  vacuum catalog weekly.
- Schedule: `:00` scrape, `:10` push, `:30` light-train,
  `03:00` deep-train, `04:00` clean.

Acceptance: catalog grows ~2–5k/night, hourly job never OOMs.

---

## Phase 2 — Model-max (Weeks 1–3, all numpy, all CPU-fast)

Each file < ~300 lines, no GPU, minutes on 8 GB.

### 2a. `models/als.py` — Implicit ALS (new primary CF)

The `implicit` library's core idea, native numpy:

- Confidence `C = 1 + alpha * R`, binary preference `P`.
- Alternate: `Xu = (YᵀCY + λI)⁻¹ YᵀCp` via Cholesky solves,
  15–20 iterations. Sparse, vectorized per-user/item.
- 128 factors; 50k songs × 128 float32 ≈ 25 MB. 500k
  interactions ≈ 2–4 min CPU.
- Expected: beats FunkSVD on play-count data outright.

### 2b. `models/feature_mf.py` — LightFM-core (kills cold-start)

- `score(u,i) = (p_u + Σ_f F_f) · (q_i + Σ_g G_g) + b_u + b_i`
  with song features = genre_code, language, energy bucket,
  artist_id; user features = mood/time bucket.
- BPR/WARP pairwise loss + Adagrad, sampled negatives from
  Phase 0. New songs with zero plays still score via features.

### 2c. `models/retrievers.py` — Cornac-style zoo (cheap, strong)

- ItemKNN (cosine on co-listen vectors).
- Co-occurrence (direct pair counts — beat pure collab
  0.647 vs 0.338 R-precision on the MPD lineage).
- Item2Vec (skip-gram over listen sequences, numpy).
- Plus existing popular + fresh. Five retrievers → ~2000 pool
  (up from 500/3 today).

### 2d. `models/sequence.py` — next-song continuity

- Markov + 1-layer SASRec-lite attention over last-20 listens
  (numpy). The "4AM in Karachi at 4am" pattern is sequential,
  not a taste vector. Companion continuity lives here.

### 2e. Torch NCF → weekly bonus only

- Keep `models/ncf_model.py` but train weekly. Add a numpy
  two-tower dot-product as its daily CPU stand-in.

Acceptance: ALS + Feature-MF + retrievers all train on CPU in
minutes and beat the SVD-only baseline on NDCG@10.

---

## Phase 3 — Rank-max: learned ensemble v2

**New: `ranker/blender.py`**

- Replace fixed `W_SVD=0.30…` with logistic regression per
  (user × mood) on validation NDCG. Inputs: every sub-score +
  freshness + artist-repeat penalty.
- Keep MMR diversity + hard rules: user blacklist,
  max-3-per-genre, one **anchor slot** (Mehrama-energy for owner).
- Serve path: nightly precompute Top-200/user to disk; hourly
  job reranks with the fresh feed only.

Acceptance: NDCG@10 up, diversity/coverage guarded, p99 push
latency flat.

---

## Phase 4 — Companion layer (on top, not instead)

- Mood/activity commands (`/mood`, `/morning`, `/gym`, `/4am`)
  → filter + per-mood blender weights. Write `context_hour/dow`
  on every listen (columns exist, currently unused).
- `/why this` in words (translate `why_top10` JSON → one sentence).
- Listening journal + streaks + weekly digest to AgainOwner.
- `/never`, `/anchor`, `/adventurous 0-10` → exploration_slots.
- Audio ears (done, native — no Essentia/Librosa): `audio/` hears 90s
  clips with numpy-only DSP (chroma-12 + Krumhansl key/mode, spectral-flux
  tempo, centroid/rolloff/flatness/ZCR) into `audio_features`; measured
  BPM/key feed `tempo_fit`, FeatureMF side features, and the final blend.
  Nightly 02:00 `audio_listen` job, ~150 tracks/night, owner taste first.
  YouTube watch is bot-walled from datacenter IPs → set `AUDIO_COOKIES`
  (cookies.txt) to unlock catalog fetches; DSP verified on synthetic +
  real audio either way.

---

## Resource budget (8 GB, no GPU)

- ALS f=128, 50k songs ≈ 25 MB/model matrix; interactions
  500k ≈ tens of MB. Two models + TF-IDF + KNN co-matrix
  comfortably < 2 GB peak with batched predict.
- Batch all predict paths (never full-matrix × full-users).
- SQLite WAL +751 indices already present; add
  `idx_listens_user_song` if missing; weekly VACUUM.
- OOM guard: each training job logs peak RSS; scheduler refuses
  overlap via lock file.

---

## Build order

1. **Week 1:** Phase 0 + ALS (2a). Biggest NDCG jump, lowest risk.
2. **Week 2:** Phase 1 + retrievers (2c) + eval/registry discipline.
3. **Week 3:** Feature-MF (2b) + sequence (2d) + blender (Phase 3).
4. **After:** companion commands + weekly digest + audio features.

## Definition of done

- `python main.py --train` trains ALS + Feature-MF + retrievers +
  sequence on one interaction frame, evals temporally, promotes
  only on NDCG gain.
- `python main.py --once --user-ids 6802929470` pushes a
  context-aware Top-10 to Telegram with worded `why`s.
- Nightly deep scrape + retrain completes on 8 GB CPU with no
  overlap failures for 7 straight days.
