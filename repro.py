import sqlite3, math, random

print('=== [1] trainer/engine.py seed_db INSERT binding ===')
conn = sqlite3.connect(':memory:')
conn.execute('''CREATE TABLE listens (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, song_id TEXT, started_at REAL, duration_played REAL DEFAULT 0, completion_pct REAL DEFAULT 0.0, source TEXT, context_genre INTEGER DEFAULT 20, context_hour INTEGER DEFAULT 0, context_dow INTEGER DEFAULT 0)''')
rows = [(1,'syn_000001',123.0,0.9,'synthetic')]
try:
    conn.executemany("""INSERT INTO listens (user_id, song_id, started_at, completion_pct, source)
        VALUES (?, ?, ?, ?, 'synthetic')""", rows)
    print('  -> OK (unexpected)')
except Exception as e:
    print('  -> RUNTIME ERROR:', type(e).__name__, e)

print()
print('=== [2] bandit_content.py / tfidf user_profile_vector cross-DB JOIN ===')
history = sqlite3.connect(':memory:')  # simulates user_history.db
history.execute('CREATE TABLE listens (id INTEGER PRIMARY KEY, user_id INTEGER, song_id TEXT, started_at REAL, completion_pct REAL)')
history.execute("INSERT INTO listens VALUES (1,1,'syn_1',1.0,0.9)")
# 'songs' table lives in a DIFFERENT db file (catalog.db), not attached here
try:
    history.execute("""SELECT l.song_id, l.completion_pct, s.title FROM listens l
        LEFT JOIN songs s ON s.song_id = l.song_id WHERE l.user_id = ? ORDER BY l.started_at DESC LIMIT 200""", (1,)).fetchall()
    print('  -> OK (unexpected)')
except Exception as e:
    print('  -> RUNTIME ERROR:', type(e).__name__, e)

print()
print('=== [3] bloom.py claimed false-positive rate ===')
m, k = 2_000_003, 7
for n in (100_000, 1_000_000, 2_000_000):
    p = (1 - math.exp(-k*n/m))**k
    print(f'  n={n:>9,} items -> actual fp_rate = {p:.4%}   (docstring claims 0.8% @ 1M)')

print()
print('=== [4] scraper _detect_genre substring collisions ===')
GENRE_MAP = {"lofi":0,"hip hop":1,"rap":2,"pop":3,"edm":4,"electronic":4,"rnb":5,"r&b":5,"indie":6,"rock":7,"metal":8,"jazz":9,"classical":10,"country":11,"latin":12,"k-pop":13,"hindi":14,"punjabi":15,"trap":16,"drill":17,"soul":18,"afrobeats":19,"unknown":20}
INV = {v:kk for kk,v in GENRE_MAP.items()}
def detect(title):
    t = title.lower()
    for kw, code in GENRE_MAP.items():
        if kw in t: return kw, code
    return 'unknown', 20
for t in ['Trap Nation - Hard Beat', 'Popular Songs 2024', 'Rockstar (Official)', 'Classical Metal Fusion', 'Best Rap Hits']:
    kw, code = detect(t)
    print(f'  {t!r:38} -> matched keyword {kw!r} (code {code})')
