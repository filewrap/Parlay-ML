"""
╔══════════════════════════════════════════════════════════╗
║           PARLAY ML — NEUROSYNC ENGINE v1.0              ║
║   Named Feature: NeuroSync (Beta)                        ║
║   Tiny Parlay-ML rooted into the recommendation core     ║
╚══════════════════════════════════════════════════════════╝

NeuroSync: Because good music finds you, not the other way around.
"""

from pathlib import Path
import os

# ─── Root paths ───────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "store"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── SQLite databases ─────────────────────────────────────
DB_FEED      = DATA_DIR / "yt_feed.db"        # hourly YT top 100
DB_HISTORY   = DATA_DIR / "user_history.db"   # per-user listen logs
DB_FEEDBACK  = DATA_DIR / "feedback.db"       # +/- signals
DB_CATALOG   = DATA_DIR / "catalog.db"        # deduplicated song master
DB_RECS      = DATA_DIR / "recommendations.db" # finalised top-10 slots
DB_BANDIT    = DATA_DIR / "bandit.db"         # Thompson sampling state
DB_EMBEDDINGS= DATA_DIR / "embeddings.db"     # latent vectors (numpy blobs)
DB_TRAINING  = DATA_DIR / "training_log.db"   # model training runs

# ─── YouTube scrape config ────────────────────────────────
YT_SEARCH_QUERIES = [
    "top music 2024",
    "trending songs",
    "new music hits",
    "popular songs right now",
    "best songs this week",
    "viral music",
    "top 50 songs",
    "chart hits",
    "popular Hindi songs",
    "lofi hip hop",
    "EDM hits",
    "pop music trending",
    "rap hits 2024",
    "R&B trending",
    "indie music popular",
]
# ─── MAX: deep-scrape expansion (Phase 1) ───────────────
# Per-artist discographies + per-mood + charts + related expansion.
YT_DEEP_ARTISTS = [
    "Anuv Jain", "Talha Anjum", "Prateek Kuhad", "Seema Mishra",
]
YT_DEEP_MOODS = [
    "sad lofi night", "chill morning acoustic", "gym workout hits",
    "4am vibes playlist", "romantic hindi songs", "punjabi rap hits",
]
YT_DEEP_CHARTS = [
    "billboard hot 100 this week", "spotify top 50 global",
    "top hindi songs this week", "coke studio pakistan",
]
YT_RELATED_TEMPLATES = [
    "songs like {seed}", "{artist} similar artists", "{seed} mix",
]
YT_FEED_TOP_N     = 100      # songs kept from hourly feed
YT_DEEP_TOP_N     = 2500     # songs kept from nightly deep feed (2-5k target)
YT_FETCH_WORKERS  = 5        # MAX: 4-6 concurrent workers (was 8, OOM guard)
YT_MIN_VIEWS      = 10_000   # filter garbage
YT_MAX_DURATION   = 600      # max 10 min songs
FEED_SNAPSHOT_RETENTION_DAYS = 7  # keep 7 days of feed_songs, vacuum weekly

# ─── Recommendation pipeline ─────────────────────────────
REC_TOP_K          = 10      # final recommendations sent to user
REC_CANDIDATE_POOL = 2000    # pool fed into ranking (MAX: 5+ retrievers → ~2000)
REC_PRECOMPUTE_TOP_N = 200   # nightly precompute per user (Phase 3)
FEEDBACK_DISLIKE_THRESHOLD = 5   # auto-disable after 5 dislikes

# ─── ML / Model hyperparams ──────────────────────────────
SVD_FACTORS        = 128
SVD_EPOCHS         = 40
SVD_LR             = 0.005
SVD_REG            = 0.02
SVD_RETRAIN_EVERY  = 3600 * 6   # every 6 hours

NCF_EMBEDDING_DIM  = 64
NCF_HIDDEN_LAYERS  = [256, 128, 64]
NCF_DROPOUT        = 0.3
NCF_LR             = 1e-3
NCF_BATCH_SIZE     = 512
NCF_EPOCHS         = 20

# ─── Scoring weights (ensemble) ──────────────────────────
# These blend the 4 sub-scorers into a final rank score
W_SVD         = 0.30   # collaborative filter signal
W_NCF         = 0.25   # neural CF
W_CONTENT     = 0.20   # TF-IDF content similarity
W_BANDIT      = 0.15   # Thompson sampling exploration bonus
W_RECENCY     = 0.10   # time-decay freshness bonus

# ─── Sieve / dedup config ────────────────────────────────
# Prime-number based Bloom filter for O(1) dedup
BLOOM_PRIME_SEEDS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
BLOOM_SIZE        = 2_000_003   # large prime for bloom array size
BLOOM_HASH_COUNT  = 7

# ─── Time-decay (half-life) ───────────────────────────────
LISTEN_HALFLIFE_HOURS = 168   # 1 week half-life for old listens
FEED_HALFLIFE_HOURS   = 24    # 24h for feed freshness
INTERACTION_HALFLIFE_DAYS = 30  # MAX Phase 0: confidence time-decay at train time

# ─── MAX implicit-confidence worldview (Phase 0) ────────
# confidence c = 1 + alpha * reward ; preference p = 1 if reward > 0 else 0
IMPLICIT_ALPHA = 2.0
# Plan-native reward shaping (plays are implicit feedback):
# full-play=3, like=5, partial=1, skip-fast=-2, dislike=-5
PLAN_REWARD_FULL_PLAY    = 3.0
PLAN_REWARD_LIKE         = 5.0
PLAN_REWARD_PARTIAL_PLAY = 1.0
PLAN_REWARD_SKIP_FAST    = -2.0
PLAN_REWARD_DISLIKE      = -5.0
NEGATIVES_PER_POSITIVE = 4  # 50% uniform, 50% popularity-biased

# ─── MAX model hyperparams (Phase 2) ────────────────────
ALS_FACTORS    = 128
ALS_ITERATIONS = 18
ALS_REG        = 0.08
ALS_ALPHA      = 2.0   # confidence scale (same worldview as IMPLICIT_ALPHA)
FEATURE_MF_FACTORS = 64
FEATURE_MF_EPOCHS  = 25
FEATURE_MF_LR      = 0.05
TWO_TOWER_DIM  = 64
SEQ_MAX_LEN    = 20    # last-20 listens for sequence continuity

# ─── MAX blender (Phase 3) ──────────────────────────────
BLENDER_LR       = 0.1
BLENDER_EPOCHS   = 200
BLENDER_L2       = 1e-4
BLENDER_MAX_PER_GENRE = 3
ANCHOR_SONG_ID   = "owner_mehrama"  # anchor slot for owner (Mehrama-energy)
ANCHOR_USER_ID   = 6802929470

# ─── Listener addon: fresh ears + emotional arcs ────────
HEARD_BOOST = 0.06          # extra blend lift for recently-heard tracks
HEARD_WINDOW_DAYS = 7       # "recently heard" window
HEARD_MAX_INJECT = 20       # max heard tracks injected into candidate pool
ARC_WINDOW_S = 10           # emotional-arc window size (seconds)

# ─── Companion (Phase 4) ────────────────────────────────
MOOD_COMMANDS = ["morning", "gym", "4am", "chill", "sad", "party", "focus"]
MOOD_GENRE_PREF = {
    "morning": [6, 3, 9], "gym": [4, 2, 16], "4am": [0, 6, 18],
    "chill": [0, 6, 9], "sad": [6, 18, 5], "party": [3, 4, 12], "focus": [0, 10, 9],
}

# ─── Resource budget (8 GB, no GPU) ─────────────────────
MAX_TRAIN_WORKERS = 5
TRAIN_LOCK_FILE = DATA_DIR / "train.lock"
PRECOMPUTE_DIR = DATA_DIR / "precompute"
for _d in [PRECOMPUTE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Scheduler intervals (seconds) ───────────────────────
FEED_REFRESH_INTERVAL  = 3600       # every hour
REC_PUSH_INTERVAL      = 3600       # push recs every hour
MODEL_RETRAIN_INTERVAL = 3600 * 6   # retrain every 6h
CATALOG_CLEAN_INTERVAL = 3600 * 24  # vacuum catalog daily

# ─── Feature names (for ML feature vectors) ──────────────
METADATA_FEATURES = [
    "view_count_log",
    "like_ratio",
    "duration_norm",
    "channel_sub_log",
    "days_since_upload_norm",
    "title_len_norm",
    "has_official_tag",
    "has_lyric_tag",
    "has_remix_tag",
    "language_code",    # int-encoded
    "genre_code",       # int-encoded from title NLP
    "energy_score",     # Fibonacci-normalised heuristic
]

GENRE_MAP = {
    "lofi": 0, "hip hop": 1, "rap": 2, "pop": 3, "edm": 4,
    "electronic": 4, "rnb": 5, "r&b": 5, "indie": 6, "rock": 7,
    "metal": 8, "jazz": 9, "classical": 10, "country": 11,
    "latin": 12, "k-pop": 13, "hindi": 14, "punjabi": 15,
    "trap": 16, "drill": 17, "soul": 18, "afrobeats": 19,
    "unknown": 20,
}

LANGUAGE_MAP = {
    "en": 0, "hi": 1, "pa": 2, "es": 3,
    "ko": 4, "pt": 5, "fr": 6, "de": 7, "ja": 8, "other": 9,
}

# ─── Reward shaping ───────────────────────────────────────
REWARD_LIKE         = +1.0
REWARD_DISLIKE      = -1.0
REWARD_SKIP_FAST    = -0.5   # skipped < 10s
REWARD_FULL_PLAY    = +0.8   # completed >80% of song
REWARD_PARTIAL_PLAY = +0.2   # 30–80% completion

# ─── Telegram push (AgainOwner) ───────────────────────────
# Stdlib-only Bot API push. No new pip deps.
#   export TELEGRAM_BOT_TOKEN="123456:ABC-..."
#   # optional overrides (defaults to the owner below):
#   export TELEGRAM_CHAT_ID="6802929470"
#   export PARLAY_PUSH_CALLBACK="notifier.telegram:push_recommendations"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
OWNER_TELEGRAM_ID = int(os.environ.get("TELEGRAM_OWNER_ID", "6802929470"))
OWNER_USERNAME = os.environ.get("TELEGRAM_OWNER_USERNAME", "AgainOwner")

# ─── Fibonacci sequence (used for energy normalisation) ───
def fib_sequence(n: int) -> list[int]:
    a, b = 1, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq

FIB_SEQ = fib_sequence(20)

# ─── Sieve of Eratosthenes (for prime-hash Bloom filter) ──
def sieve_of_eratosthenes(limit: int) -> list[int]:
    """Classic prime sieve. Primes used as hash seeds for Bloom filter."""
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

# Pre-compute first 100 primes for hash functions
PRIMES_100 = sieve_of_eratosthenes(541)[:100]
