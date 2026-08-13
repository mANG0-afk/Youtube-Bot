"""
Backend for the playlist agent.
  secrets    dbutils.secrets.get(), env-var fallback for local/hosted runs
  warehouse  Databricks SQL against test_projects.gold.* views (SP OAuth M2M)
  app state  SQLite -- candidate sets, YouTube cache, playlist log, aliases
  lastfm     artist.getTopTracks (autocorrect=1), artist.getCorrection
  youtube    search.list (API key), playlists/playlistItems.insert (OAuth)

App state is SQLite, not Delta: these are small per-request writes and Delta's
multi-second commit latency would make the app feel broken. Commit a pre-seeded
app_state.db to the repo so a restart begins with a warm YouTube cache.

Env vars (all nine required):
  DATABRICKS_HOST           https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
  DATABRICKS_CLIENT_ID      service principal application id
  DATABRICKS_CLIENT_SECRET  service principal OAuth secret (starts dose...)
  DATABRICKS_HTTP_PATH      /sql/1.0/warehouses/xxxxxxxx
  LASTFM_API_KEY
  YOUTUBE_API_KEY
  YOUTUBE_OAUTH_CLIENT_ID
  YOUTUBE_OAUTH_CLIENT_SECRET
  YOUTUBE_OAUTH_REFRESH_TOKEN

Optional:
  MUSIC_CATALOG    default test_projects
  APP_STATE_DB     default app_state.db
  YT_DAILY_BUDGET  default 9000
"""

import datetime as _dt
import os
import re
import sqlite3
import threading
import time
import unicodedata
import unittest
import uuid
from contextlib import contextmanager
from typing import Optional

import requests

CATALOG = os.environ.get("MUSIC_CATALOG", "test_projects")
GOLD = f"{CATALOG}.gold"
SQLITE_PATH = os.environ.get("APP_STATE_DB", "app_state.db")

LASTFM_API = "https://ws.audioscrobbler.com/2.0/"
YT_API = "https://www.googleapis.com/youtube/v3"

# A TTL'd cache is still "always live" -- what we rejected was stored data
# winning over a fresh call.
ARTIST_TRACKS_TTL_SEC = 7 * 24 * 3600

# Stop spending YouTube quota below the 10,000/day ceiling, leaving headroom.
DAILY_BUDGET = int(os.environ.get("YT_DAILY_BUDGET", "9000"))

# Durable quota ledger. SQLite alone resets with the filesystem on every
# redeploy, which made today's spend read as 0 on top of real usage.
QUOTA_LOG = os.environ.get("QUOTA_LOG_TABLE", f"{CATALOG}.logging.quota_log")

# thread_id of the turn currently being served. Thread-local, not global:
# Streamlit runs each session's script in its own thread, so a plain global
# would label one visitor's spend with another's conversation.
_ctx = threading.local()


def set_quota_context(thread_id: str) -> None:
    _ctx.thread_id = thread_id


# --------------------------------------------------------------------------
# SECRETS
# --------------------------------------------------------------------------
def _resolve_dbutils():
    """
    Databricks injects `dbutils` into the NOTEBOOK namespace only -- an imported
    module never sees it, so a bare dbutils.secrets.get() here raises NameError.
    Reach into IPython's namespace, then fall back to building DBUtils from the
    Spark session (works in jobs). Returns None outside Databricks, which is
    what makes the env-var fallback below kick in when hosted.
    """
    try:
        import IPython
        ip = IPython.get_ipython()
        if ip and "dbutils" in ip.user_ns:
            return ip.user_ns["dbutils"]
    except Exception:
        pass
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        return DBUtils(SparkSession.builder.getOrCreate())
    except Exception:
        return None


_DBUTILS = _resolve_dbutils()


def secret(scope: str, key: str) -> str:
    """Env fallback is not optional: hosted and local runs have no dbutils."""
    if _DBUTILS is not None:
        try:
            return _DBUTILS.secrets.get(scope=scope, key=key)
        except Exception:
            pass
    env = f"{scope}_{key}".upper().replace("-", "_")
    val = os.environ.get(env)
    if not val:
        raise RuntimeError(f"Secret {scope}/{key} unavailable and ${env} is unset.")
    return val


def workspace_host() -> str:
    """Notebook context if available, else the env var."""
    if _DBUTILS is not None:
        try:
            ctx = _DBUTILS.notebook.entry_point.getDbutils().notebook().getContext()
            return ctx.apiUrl().get()
        except Exception:
            pass
    return os.environ["DATABRICKS_HOST"]


# --------------------------------------------------------------------------
# NORMALIZATION -- must match silver.norm_key() exactly. Drift by one
# character and every artist lookup silently misses.
# --------------------------------------------------------------------------
_ACCENTS = str.maketrans("áàâäãåéèêëíìîïóòôöõúùûüñç",
                         "aaaaaaeeeeiiiiooooouuuunc")


def norm_key(raw: str) -> str:
    s = unicodedata.normalize("NFKD", raw or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("ß", "ss").replace("æ", "ae").replace("ø", "o")
    s = s.translate(_ACCENTS)
    s = s.replace("$", "s").replace("&", "and").replace("+", " and ")
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


# --------------------------------------------------------------------------
# DATABRICKS SQL -- service principal OAuth M2M.
#
# oauth_service_principal returns a provider that mints AND refreshes
# short-lived tokens, so there is no static credential to rotate and no expiry
# to debug later. A PAT would carry every permission your user has; this
# carries only what you granted the service principal.
# --------------------------------------------------------------------------
_conn_lock = threading.Lock()
_conn = None


def _warehouse():
    global _conn
    with _conn_lock:
        if _conn is None:
            from databricks import sql as dbsql
            from databricks.sdk.core import Config, oauth_service_principal

            host = os.environ["DATABRICKS_HOST"].replace("https://", "").rstrip("/")

            def credential_provider():
                # Config() with no args reads DATABRICKS_HOST / CLIENT_ID /
                # CLIENT_SECRET straight from the environment.
                return oauth_service_principal(Config())

            _conn = dbsql.connect(
                server_hostname=host,
                http_path=os.environ["DATABRICKS_HTTP_PATH"],
                credentials_provider=credential_provider,
            )
        return _conn


# One connection is shared process-wide, and a single databricks-sql connection
# is not safe for concurrent cursors -- two threads issuing statements at once
# hang rather than error. Background quota flushes made that a real possibility,
# so all statement execution is serialised. Acquired before _conn_lock, always in
# that order, so the pair cannot deadlock.
_use_lock = threading.Lock()


def query(sql: str, params: Optional[dict] = None) -> list[dict]:
    """Named :param markers, so user input is never interpolated into SQL."""
    global _conn
    with _use_lock:
        for attempt in range(2):
            try:
                with _warehouse().cursor() as cur:
                    cur.execute(sql, parameters=params or {})
                    cols = [c[0] for c in cur.description]
                    return [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception:
                with _conn_lock:
                    _conn = None      # stale/expired connection, rebuild once
                if attempt:
                    raise
    return []


# --------------------------------------------------------------------------
# SQLITE APP STATE
# --------------------------------------------------------------------------
@contextmanager
def _db():
    conn = sqlite3.connect(SQLITE_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_app_state():
    with _db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS candidate_set (
            set_id TEXT PRIMARY KEY, source TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS candidate_track (
            set_id TEXT, position INTEGER, artist_key TEXT, track_key TEXT,
            track_name TEXT, artist_name TEXT, duration_sec INTEGER,
            PRIMARY KEY (set_id, position));

        CREATE TABLE IF NOT EXISTS resolved_set (
            set_id TEXT PRIMARY KEY, candidate_set_id TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS resolved_track (
            set_id TEXT, position INTEGER, video_id TEXT, video_title TEXT,
            track_name TEXT, artist_name TEXT,
            PRIMARY KEY (set_id, position));

        -- video_id NULL is a NEGATIVE cache entry: searched, nothing found.
        -- Without it an unfindable track costs 100 units on every run.
        CREATE TABLE IF NOT EXISTS youtube_resolution (
            artist_key TEXT, track_key TEXT, video_id TEXT, video_title TEXT,
            channel TEXT, resolved_at REAL, PRIMARY KEY (artist_key, track_key));

        CREATE TABLE IF NOT EXISTS artist_top_track (
            artist_key TEXT, rank INTEGER, track_name TEXT, artist_name TEXT,
            playcount INTEGER, fetched_at REAL, PRIMARY KEY (artist_key, rank));

        -- Silver dropped requested_name, so aliases can only be learned at
        -- runtime -- app state, not gold.
        CREATE TABLE IF NOT EXISTS artist_alias (
            alias_key TEXT PRIMARY KEY, artist_key TEXT, artist_name TEXT,
            learned_at REAL);

        -- Written BEFORE the next insert, so a crash mid-playlist is
        -- recoverable instead of duplicating on retry.
        CREATE TABLE IF NOT EXISTS playlist_log (
            playlist_id TEXT, video_id TEXT, resolved_set_id TEXT,
            added_at REAL, PRIMARY KEY (playlist_id, video_id));

        -- When this database began counting for a given Pacific day. Hosts with
        -- an ephemeral filesystem reset the ledger on every redeploy, and
        -- without this row the resulting 0 is indistinguishable from a genuine
        -- 0 -- the app would report a clean budget on top of real spend.
        CREATE TABLE IF NOT EXISTS quota_epoch (
            day TEXT PRIMARY KEY, started_at REAL);

        -- Write-ahead buffer for the Delta ledger. Every unit of YouTube spend
        -- lands here first, synchronously and cheaply; a background flush
        -- appends it to Delta. synced=0 rows are spend Delta has not seen yet,
        -- which is exactly what has to be added to the Delta total to get the
        -- true figure without double counting.
        CREATE TABLE IF NOT EXISTS quota_event (
            event_id TEXT PRIMARY KEY, ts REAL, pacific_day TEXT,
            operation TEXT, units INTEGER, thread_id TEXT,
            synced INTEGER DEFAULT 0);
        CREATE INDEX IF NOT EXISTS ix_quota_event_unsynced
            ON quota_event (synced, pacific_day);
        """)
        c.execute("INSERT OR IGNORE INTO quota_epoch VALUES (?,?)",
                  (_pacific_day_key(), time.time()))
    _backfill_quota_events()


def _backfill_quota_events() -> None:
    """
    One-time migration. Spend used to be derived from youtube_resolution and
    playlist_log rather than logged as events, so on the first run after this
    change today's already-spent units would otherwise read as zero -- the exact
    under-reporting this ledger exists to stop.
    """
    day, since = _pacific_day_key(), _pacific_day_start()
    with _db() as c:
        if c.execute("SELECT 1 FROM quota_event WHERE pacific_day=? LIMIT 1",
                     (day,)).fetchone():
            return
        # Deterministic event_ids, so running this twice cannot double count --
        # the local PK ignores the repeat and the Delta read dedupes by id.
        events = []
        for r in c.execute("SELECT artist_key, track_key FROM youtube_resolution "
                           "WHERE resolved_at > ?", (since,)):
            events.append((f"bf:{day}:search:{r['artist_key']}:{r['track_key']}",
                           since, day, "search", 100, "", 0))
        lists = set()
        for r in c.execute("SELECT playlist_id, video_id FROM playlist_log "
                           "WHERE added_at > ?", (since,)):
            events.append((f"bf:{day}:item:{r['playlist_id']}:{r['video_id']}",
                           since, day, "playlist_item_insert", 50, "", 0))
            lists.add(r["playlist_id"])
        for pid in lists:
            events.append((f"bf:{day}:list:{pid}",
                           since, day, "playlist_insert", 50, "", 0))
        if events:
            c.executemany("INSERT OR IGNORE INTO quota_event "
                          "VALUES (?,?,?,?,?,?,?)", events)


def record_quota(operation: str, units: int) -> None:
    """
    Log one unit of spend. Never raises: quota accounting failing must not fail
    the request that was already paid for.
    """
    try:
        with _db() as c:
            c.execute(
                "INSERT INTO quota_event "
                "(event_id, ts, pacific_day, operation, units, thread_id, synced)"
                " VALUES (?,?,?,?,?,?,0)",
                (uuid.uuid4().hex, time.time(), _pacific_day_key(), operation,
                 int(units), getattr(_ctx, "thread_id", "")))
    except Exception:
        pass


def save_candidate_set(rows: list[dict], source: str) -> str:
    set_id = f"cs_{uuid.uuid4().hex[:8]}"
    with _db() as c:
        c.execute("INSERT INTO candidate_set VALUES (?,?,?)",
                  (set_id, source, time.time()))
        c.executemany(
            "INSERT INTO candidate_track VALUES (?,?,?,?,?,?,?)",
            [(set_id, i, r.get("artist_key"), r.get("track_key"),
              r["track_name"], r["artist_name"], r.get("duration_sec"))
             for i, r in enumerate(rows)])
    return set_id


def load_candidate_set(set_id: str) -> list[dict]:
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM candidate_track WHERE set_id=? ORDER BY position",
            (set_id,))]


def save_resolved_set(candidate_set_id: str, rows: list[dict]) -> str:
    set_id = f"rs_{uuid.uuid4().hex[:8]}"
    with _db() as c:
        c.execute("INSERT INTO resolved_set VALUES (?,?,?)",
                  (set_id, candidate_set_id, time.time()))
        c.executemany(
            "INSERT INTO resolved_track VALUES (?,?,?,?,?,?)",
            [(set_id, i, r["video_id"], r.get("video_title"),
              r["track_name"], r["artist_name"]) for i, r in enumerate(rows)])
    return set_id


def load_resolved_set(set_id: str) -> list[dict]:
    with _db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM resolved_track WHERE set_id=? ORDER BY position",
            (set_id,))]


_PAREN = re.compile(r"\s*[\(\[][^)\]]*[\)\]]")


def _search_query(artist_name: str, track_name: str) -> str:
    """
    Bollywood titles carry film credits: 'Halka Halka (From "Fanney Khan")'.
    That suffix pushes results toward trailers and scene clips, so drop it.
    """
    t = _PAREN.sub("", track_name).strip()
    t = re.sub(r"\s*[-–—]\s*(from|feat\.?|ft\.?)\b.*$", "", t, flags=re.I).strip()
    return f"{artist_name} {t or track_name}"


# --------------------------------------------------------------------------
# QUOTA ACCOUNTING
#
# YouTube quota resets at midnight PACIFIC, not UTC, so the day boundary is
# computed in that zone. Getting this wrong shifts the reset by up to 8 hours
# and the app looks broken for a whole evening.
# --------------------------------------------------------------------------
def _pacific_now():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Los_Angeles"))


def _pacific_day_start() -> float:
    from datetime import time as dtime
    now = _pacific_now()
    return now.replace(hour=0, minute=0, second=0,
                       microsecond=0).timestamp()


def _pacific_day_key() -> str:
    return _pacific_now().strftime("%Y-%m-%d")


def quota_tracking_since() -> Optional[float]:
    """
    When this database started counting today, or None if it has counted all day.

    Returns None once the Delta total has actually been read, because the figure
    is then durable and complete and the ">=" caveat would be false modesty. Only
    the SQLite-only fallback needs the disclaimer.
    """
    if quota_ledger_live():
        return None
    day_start = _pacific_day_start()
    with _db() as c:
        row = c.execute("SELECT started_at FROM quota_epoch WHERE day=?",
                        (_pacific_day_key(),)).fetchone()
    if not row:
        return None
    # 60s of slack: a container that boots seconds after midnight has, for all
    # practical purposes, seen the whole day.
    return None if row["started_at"] <= day_start + 60 else row["started_at"]


_ledger_lock = threading.Lock()
_ledger = {"delta_units": 0, "read_at": 0.0, "available": None}
LEDGER_TTL_SEC = 120


def ensure_quota_log() -> bool:
    """
    Create the Delta ledger if absent. Called from the warm-up thread, never at
    import -- it is a warehouse round-trip. Returns False if the table cannot be
    created or read, in which case everything degrades to SQLite-only counting.
    """
    with _ledger_lock:
        if _ledger["available"] is not None:
            return _ledger["available"]
    ok = False
    try:
        query(f"""
            CREATE TABLE IF NOT EXISTS {QUOTA_LOG} (
                event_id   STRING,
                event_ts   TIMESTAMP,
                pacific_day DATE,
                operation  STRING,
                units      INT,
                thread_id  STRING
            ) USING DELTA
        """)
        ok = True
    except Exception:
        ok = False
    with _ledger_lock:
        _ledger["available"] = ok
    return ok


def _delta_units_today() -> int:
    """
    Last known Delta total for the Pacific day. Reads the cache and NOTHING else.

    Never queries. This sits behind the quota chip, which re-renders on every
    Streamlit interaction, and a cold warehouse round-trip is ~25s -- querying
    here froze the page on whichever click happened to find the cache stale,
    which presented as the quota display having disappeared.
    """
    with _ledger_lock:
        return _ledger["delta_units"]


def refresh_delta_units() -> int:
    """
    Re-read the Delta total. Called from background threads only -- warm-up and
    after a flush, which is exactly when the figure can have changed.
    """
    with _ledger_lock:
        if _ledger["available"] is False:
            return 0
    try:
        # DISTINCT event_id, not a bare SUM: the flush marks rows synced only
        # after a successful append, so a crash in between re-sends them. Dedupe
        # on read makes that harmless instead of inflating the day's spend.
        rows = query(
            f"SELECT COALESCE(SUM(units), 0) AS n FROM ("
            f"  SELECT DISTINCT event_id, units FROM {QUOTA_LOG}"
            f"  WHERE pacific_day = CAST(:d AS DATE))",
            {"d": _pacific_day_key()})
        units = int(rows[0]["n"]) if rows else 0
        with _ledger_lock:
            _ledger["delta_units"] = units
            _ledger["read_at"] = time.time()
        return units
    except Exception:
        return _delta_units_today()


def quota_ledger_live() -> bool:
    """True once the Delta total has actually been read at least once."""
    with _ledger_lock:
        return _ledger["read_at"] > 0


def flush_quota_log() -> int:
    """
    Append unsynced local events to Delta as ONE insert, then mark them synced.

    One commit per flush rather than per event: Delta commits cost seconds, and
    a 10-track resolve would otherwise mean 10 of them. Marked synced only after
    a successful append, so a crash in between re-sends rows and over-counts --
    which errs toward degrading the app early rather than overspending.
    """
    if not ensure_quota_log():
        return 0
    with _db() as c:
        rows = [dict(r) for r in c.execute(
            "SELECT event_id, ts, pacific_day, operation, units, thread_id "
            "FROM quota_event WHERE synced = 0 ORDER BY ts LIMIT 500")]
    if not rows:
        return 0

    tuples, params = [], {}
    for i, r in enumerate(rows):
        tuples.append(f"(:e{i}, CAST(:t{i} AS TIMESTAMP), "
                      f"CAST(:d{i} AS DATE), :o{i}, :u{i}, :h{i})")
        params[f"e{i}"] = r["event_id"]
        params[f"t{i}"] = _dt.datetime.fromtimestamp(
            r["ts"], _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        params[f"d{i}"] = r["pacific_day"]
        params[f"o{i}"] = r["operation"]
        params[f"u{i}"] = int(r["units"])
        params[f"h{i}"] = r["thread_id"] or ""
    try:
        query(f"INSERT INTO {QUOTA_LOG} VALUES {', '.join(tuples)}", params)
    except Exception:
        return 0

    with _db() as c:
        c.executemany("UPDATE quota_event SET synced = 1 WHERE event_id = ?",
                      [(r["event_id"],) for r in rows])
    refresh_delta_units()
    return len(rows)


def flush_quota_async() -> None:
    """Fire-and-forget, so a turn never waits on a Delta commit."""
    threading.Thread(target=lambda: flush_quota_log(), daemon=True).start()


def quota_used_today() -> int:
    """
    Delta total for the Pacific day plus local spend Delta has not seen yet.

    Splitting it this way is what makes the figure survive a redeploy without
    double counting: Delta holds every flushed event from every container
    lifetime, and synced=0 holds exactly the remainder.
    """
    with _db() as c:
        local = c.execute(
            "SELECT COALESCE(SUM(units), 0) AS n FROM quota_event "
            "WHERE synced = 0 AND pacific_day = ?",
            (_pacific_day_key(),)).fetchone()["n"]
    return _delta_units_today() + int(local)


def search_url(artist_name: str, track_name: str) -> str:
    """
    Zero-quota fallback for when the budget is spent. Lands the visitor on the
    right video anyway, so the app degrades instead of failing.
    """
    from urllib.parse import quote_plus
    return ("https://www.youtube.com/results?search_query="
            + quote_plus(_search_query(artist_name, track_name)))


# --------------------------------------------------------------------------
# LAST.FM -- no quota, 5 req/s per IP averaged over 5 min.
# --------------------------------------------------------------------------
class _Limiter:
    def __init__(self, per_sec=4.0):
        self.interval = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


_limiter = _Limiter()


def _lastfm(method: str, **params) -> Optional[dict]:
    params.update(method=method, api_key=secret("lastfm", "api_key"),
                  format="json")
    for attempt in range(3):
        _limiter.wait()
        try:
            r = requests.get(LASTFM_API, params=params, timeout=15)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        data = r.json()
        code = data.get("error")
        if code == 6:              # not found -- permanent, don't retry
            return None
        if code == 29:             # rate limited
            time.sleep(30)
            continue
        if code:
            raise RuntimeError(f"Last.fm {code}: {data.get('message')}")
        return data
    return None


def _as_list(v):
    """Last.fm returns a bare dict, not a list, when there's exactly one item."""
    return [] if v is None else (v if isinstance(v, list) else [v])


def lastfm_top_tracks(artist_name: str, limit: int = 10,
                      use_cache: bool = True) -> tuple[Optional[str], list[dict]]:
    """
    autocorrect=1 is the whole trick: 'ac dc' -> AC/DC, 'asap rock' ->
    A$AP Rocky, and it returns the corrected name so the alias can persist.
    """
    akey = norm_key(artist_name)

    if use_cache:
        with _db() as c:
            rows = [dict(r) for r in c.execute(
                "SELECT * FROM artist_top_track WHERE artist_key=? "
                "AND fetched_at > ? ORDER BY rank LIMIT ?",
                (akey, time.time() - ARTIST_TRACKS_TTL_SEC, limit))]
        if rows:
            return rows[0]["artist_name"], rows

    data = _lastfm("artist.getTopTracks", artist=artist_name,
                   autocorrect=1, limit=max(limit, 20))
    if not data or "toptracks" not in data:
        return None, []

    top = data["toptracks"]
    corrected = top.get("@attr", {}).get("artist") or artist_name

    seen, out = set(), []
    for t in _as_list(top.get("track")):
        name = (t.get("name") or "").strip()
        tkey = norm_key(name)
        if not name or tkey in seen:
            continue
        seen.add(tkey)
        out.append({
            "rank": len(out) + 1,
            "track_key": tkey,
            "track_name": name,
            "artist_key": norm_key(corrected),
            "artist_name": corrected,
            "playcount": int(t.get("playcount") or 0),
            "duration_sec": int(t.get("duration") or 0) or None,
        })

    if out:
        now = time.time()
        with _db() as c:
            c.execute("DELETE FROM artist_top_track WHERE artist_key=?",
                      (norm_key(corrected),))
            c.executemany(
                "INSERT INTO artist_top_track VALUES (?,?,?,?,?,?)",
                [(r["artist_key"], r["rank"], r["track_name"],
                  r["artist_name"], r["playcount"], now) for r in out])
    return corrected, out[:limit]


def lastfm_correct(name: str) -> Optional[str]:
    data = _lastfm("artist.getCorrection", artist=name)
    if not data:
        return None
    return (data.get("corrections", {}).get("correction", {})
                .get("artist", {}).get("name"))


# --------------------------------------------------------------------------
# GOLD QUERIES
# --------------------------------------------------------------------------
def load_genres() -> list[dict]:
    return query(f"""
        SELECT genre_key, display_name, track_count
        FROM {GOLD}.genre WHERE track_count > 0
        ORDER BY track_count DESC
    """)


def load_moods() -> list[str]:
    rows = query(f"SELECT DISTINCT mood FROM {GOLD}.mood_tag ORDER BY mood")
    return [r["mood"] for r in rows]


def genre_top_tracks(genre_key: str, limit: int) -> list[dict]:
    return query(f"""
        SELECT artist_key, track_key, track_name, artist_name, duration_sec
        FROM {GOLD}.genre_track
        WHERE genre_key = :gk
        ORDER BY rank
        LIMIT :lim
    """, {"gk": genre_key, "lim": limit})


def resolve_genre(raw: str) -> Optional[dict]:
    gk = norm_key(raw)
    rows = query(f"SELECT genre_key, display_name FROM {GOLD}.genre "
                 f"WHERE genre_key = :gk", {"gk": gk})
    if rows:
        return rows[0]
    rows = query(f"""
        SELECT genre_key, display_name FROM {GOLD}.genre
        WHERE lower(display_name) LIKE :pat
        ORDER BY track_count DESC LIMIT 1
    """, {"pat": f"%{raw.lower().strip()}%"})
    return rows[0] if rows else None


def mood_tracks(mood: str, limit: int) -> list[dict]:
    """
    Stage 1. Explodes tags inline and joins gold.mood_tag, so the pattern
    vocabulary lives in the view rather than in Python.
    """
    return query(f"""
        WITH tt AS (
          SELECT artist_key, track_key, track_name, artist_name,
                 duration_sec, playcount, lower(tag) AS tag
          FROM {GOLD}.track LATERAL VIEW explode(tags) x AS tag
        )
        SELECT tt.artist_key, tt.track_key, tt.track_name, tt.artist_name,
               tt.duration_sec,
               COUNT(DISTINCT tt.tag) AS hits,
               MAX(m.weight)          AS top_weight
        FROM tt
        JOIN {GOLD}.mood_tag m ON tt.tag LIKE m.pattern
        WHERE m.mood = :mood
        GROUP BY tt.artist_key, tt.track_key, tt.track_name,
                 tt.artist_name, tt.duration_sec, tt.playcount
        ORDER BY hits DESC, top_weight DESC, MAX(tt.playcount) DESC NULLS LAST
        LIMIT :lim
    """, {"mood": mood, "lim": limit})


def mood_artists(mood: str, limit: int = 4) -> list[dict]:
    """Stage 2. Only enriched artists have tags; listeners sit on the same row."""
    return query(f"""
        WITH at AS (
          SELECT artist_key, artist_name, listeners, lower(tag) AS tag
          FROM {GOLD}.artist LATERAL VIEW explode(tags) x AS tag
          WHERE has_enrichment
        )
        SELECT at.artist_key, at.artist_name, MAX(at.listeners) AS listeners,
               COUNT(DISTINCT at.tag) AS hits
        FROM at
        JOIN {GOLD}.mood_tag m ON at.tag LIKE m.pattern
        WHERE m.mood = :mood
        GROUP BY at.artist_key, at.artist_name
        ORDER BY listeners DESC NULLS LAST, hits DESC
        LIMIT :lim
    """, {"mood": mood, "lim": limit})


def resolve_seed_track(raw_track: str,
                       raw_artist: str = "") -> Optional[dict]:
    """
    Find the named song in gold.track so its tags can seed a similarity search.

    track_key is not unique -- covers and same-titled songs collide -- so without
    an artist the most-played match wins. That is the one a person naming a song
    bare almost always means.
    """
    tk = norm_key(raw_track)
    if not tk:
        return None

    params = {"tk": tk}
    where = "track_key = :tk"
    if raw_artist.strip():
        params["ak"] = norm_key(raw_artist)
        where += " AND artist_key = :ak"

    rows = query(f"""
        SELECT artist_key, track_key, track_name, artist_name, tag_count
        FROM {GOLD}.track
        WHERE {where}
        ORDER BY playcount DESC NULLS LAST
        LIMIT 1
    """, params)
    if rows:
        return rows[0] | {"how": "exact"}
    # A bare LIKE would match "love" inside a hundred titles, so anchor it.
    rows = query(f"""
        SELECT artist_key, track_key, track_name, artist_name, tag_count
        FROM {GOLD}.track
        WHERE track_key LIKE :pat
        ORDER BY playcount DESC NULLS LAST
        LIMIT 1
    """, {"pat": f"{tk}%"})
    return (rows[0] | {"how": "prefix"}) if rows else None


def similar_tracks_by_tags(artist_key: str, track_key: str,
                           limit: int) -> list[dict]:
    """
    Rank other tracks by how many tags they share with the seed.

    Explodes and joins rather than passing the tag list in as a parameter: the
    databricks-sql-connector binds a Python list as an empty array<void>, so an
    array_intersect against a bound list silently matches nothing. Resolving the
    seed's tags inside SQL also keeps the whole thing one round-trip.

    Ordering is shared-tag count first, playcount only as a tie-break --
    otherwise every result collapses to the same few hits that happen to carry
    one common tag.
    """
    return query(f"""
        WITH seed AS (
          SELECT DISTINCT lower(tag) AS tag
          FROM {GOLD}.track LATERAL VIEW explode(tags) x AS tag
          WHERE artist_key = :ak AND track_key = :tk
        ),
        cand AS (
          SELECT artist_key, track_key, track_name, artist_name,
                 duration_sec, playcount, lower(tag) AS tag
          FROM {GOLD}.track LATERAL VIEW explode(tags) x AS tag
        )
        SELECT c.artist_key, c.track_key, c.track_name, c.artist_name,
               c.duration_sec, COUNT(DISTINCT c.tag) AS shared_tags
        FROM cand c
        JOIN seed s ON c.tag = s.tag
        WHERE NOT (c.artist_key = :ak AND c.track_key = :tk)
        GROUP BY c.artist_key, c.track_key, c.track_name, c.artist_name,
                 c.duration_sec, c.playcount
        ORDER BY shared_tags DESC, c.playcount DESC NULLS LAST
        LIMIT :lim
    """, {"ak": artist_key, "tk": track_key, "lim": limit})


def lastfm_track_artist(raw_track: str) -> Optional[str]:
    """
    Who performs this song, for songs absent from gold.track. autocorrect=1 so a
    misspelled title still resolves, which is the whole point of asking Last.fm
    rather than giving up.
    """
    data = _lastfm("track.search", track=raw_track, limit=1)
    matches = (((data or {}).get("results", {})
                .get("trackmatches", {}) or {}).get("track") or [])
    if isinstance(matches, dict):
        matches = [matches]
    return matches[0].get("artist") if matches else None


def resolve_artist(raw: str) -> tuple[Optional[dict], list[str]]:
    """
    Four rungs, cheapest first: exact -> learned alias -> LIKE fuzzy ->
    Last.fm getCorrection (then persist the alias).
    """
    key = norm_key(raw)
    if not key:
        return None, []

    rows = query(f"SELECT artist_key, artist_name, listeners, has_enrichment "
                 f"FROM {GOLD}.artist WHERE artist_key = :k", {"k": key})
    if rows:
        return rows[0] | {"how": "exact"}, []

    with _db() as c:
        hit = c.execute("SELECT artist_key, artist_name FROM artist_alias "
                        "WHERE alias_key=?", (key,)).fetchone()
    if hit:
        return dict(hit) | {"how": "alias"}, []

    fuzzy = query(f"""
        SELECT artist_key, artist_name, listeners
        FROM {GOLD}.artist
        WHERE artist_name_lower LIKE :pat OR artist_key LIKE :kpat
        ORDER BY listeners DESC NULLS LAST
        LIMIT 5
    """, {"pat": f"%{raw.lower().strip()}%", "kpat": f"%{key}%"})
    if fuzzy:
        # Accept a single confident hit; otherwise hand the list back so the
        # model asks which one instead of guessing.
        if len(fuzzy) == 1 or norm_key(fuzzy[0]["artist_name"]).startswith(key):
            return fuzzy[0] | {"how": "fuzzy"}, []

    corrected = lastfm_correct(raw)
    if corrected and norm_key(corrected) != key:
        rows = query(f"SELECT artist_key, artist_name, listeners "
                     f"FROM {GOLD}.artist WHERE artist_key = :k",
                     {"k": norm_key(corrected)})
        if rows:
            with _db() as c:
                c.execute("INSERT OR REPLACE INTO artist_alias VALUES (?,?,?,?)",
                          (key, rows[0]["artist_key"], rows[0]["artist_name"],
                           time.time()))
            return rows[0] | {"how": "lastfm-correction"}, []
        # Corrected but not in our catalogue -- still usable, Last.fm has it.
        return {"artist_key": norm_key(corrected), "artist_name": corrected,
                "how": "lastfm-only"}, []

    return None, [f["artist_name"] for f in fuzzy]


# --------------------------------------------------------------------------
# YOUTUBE
# Reads use the API key. Writes need OAuth on the account that owns the
# playlists -- API keys cannot create playlists, and service accounts have no
# YouTube channel, so neither works for writes.
# --------------------------------------------------------------------------
_yt_write = None


def _youtube_write():
    global _yt_write
    if _yt_write is None:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(
            token=None,
            refresh_token=secret("youtube", "oauth_refresh_token"),
            client_id=secret("youtube", "oauth_client_id"),
            client_secret=secret("youtube", "oauth_client_secret"),
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/youtube"],
        )
        _yt_write = build("youtube", "v3", credentials=creds,
                          cache_discovery=False)
    return _yt_write


def youtube_find_video(artist_key: str, track_key: str, artist_name: str,
                       track_name: str) -> tuple[Optional[str], Optional[str], bool]:
    """(video_id, video_title, was_cached). 100 units only on a cache miss."""
    with _db() as c:
        row = c.execute(
            "SELECT video_id, video_title FROM youtube_resolution "
            "WHERE artist_key=? AND track_key=?",
            (artist_key, track_key)).fetchone()
    if row:
        return row["video_id"], row["video_title"], True

    vid = vtitle = channel = None
    try:
        r = requests.get(f"{YT_API}/search", timeout=15, params={
            "part": "snippet", "type": "video",
            # NO videoCategoryId. Bollywood film songs are categorised
            # Film & Animation (1) or Entertainment (24), not Music (10) --
            # filtering on 10 silently drops almost every result.
            "q": _search_query(artist_name, track_name),
            "maxResults": 1,
            "key": secret("youtube", "api_key"),
        })
    except Exception as e:
        raise RuntimeError(f"youtube_network_error: {e}")

    if r.status_code != 200:
        # Surface the real reason. Any non-200 swallowed here gets reported as
        # "song not found", which sends you hunting the wrong bug.
        raise RuntimeError(f"youtube_api_error {r.status_code}: {r.text[:300]}")

    items = r.json().get("items", [])
    if items:
        vid = items[0]["id"]["videoId"]
        vtitle = items[0]["snippet"]["title"]
        channel = items[0]["snippet"]["channelTitle"]

    # Logged whether or not a video was found: a miss still cost 100 units.
    record_quota("search", 100)
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO youtube_resolution VALUES (?,?,?,?,?,?)",
                  (artist_key, track_key, vid, vtitle, channel, time.time()))
    return vid, vtitle, False


def youtube_create_playlist(title: str, description: str = "") -> str:
    """playlists.insert -- 50 units. Public so visitors can open and play it."""
    resp = _youtube_write().playlists().insert(
        part="snippet,status",
        body={"snippet": {"title": title[:150],
                          "description": description[:1000]},
              "status": {"privacyStatus": "public"}},
    ).execute()
    record_quota("playlist_insert", 50)
    return resp["id"]


def youtube_add_items(playlist_id: str, videos: list[dict],
                      resolved_set_id: str) -> tuple[int, list[str]]:
    """
    playlistItems.insert -- 50 units EACH. Logs every success before the next
    call, so a crash halfway is recoverable rather than duplicating on retry.
    """
    with _db() as c:
        already = {r["video_id"] for r in c.execute(
            "SELECT video_id FROM playlist_log WHERE playlist_id=?",
            (playlist_id,))}

    added, failed = 0, []
    yt = _youtube_write()
    for v in videos:
        if not v.get("video_id") or v["video_id"] in already:
            continue
        try:
            yt.playlistItems().insert(
                part="snippet",
                body={"snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video",
                                   "videoId": v["video_id"]}}},
            ).execute()
            record_quota("playlist_item_insert", 50)
            with _db() as c:
                c.execute("INSERT OR REPLACE INTO playlist_log VALUES (?,?,?,?)",
                          (playlist_id, v["video_id"], resolved_set_id,
                           time.time()))
            added += 1
        except Exception as e:
            failed.append(f"{v.get('track_name')}: {type(e).__name__}")
    return added, failed


def watch_videos_url(video_ids: list[str]) -> str:
    """Zero-quota temporary playlist. Plays without spending a single unit."""
    return "https://www.youtube.com/watch_videos?video_ids=" + \
           ",".join(v for v in video_ids if v)[:2000]


# --------------------------------------------------------------------------
# The normalizer exists here AND in Spark. Pin them together or joins die.
# --------------------------------------------------------------------------
class TestNormKey(unittest.TestCase):
    def test_matches_spark(self):
        cases = {
            "AC/DC": "acdc", "ac dc": "acdc", "A$AP Rocky": "asaprocky",
            "The Beatles": "beatles", "Beyoncé": "beyonce",
            "Guns N' Roses": "gunsnroses", "Kanye West": "kanyewest",
            "Sigur Rós": "sigurros", "Motörhead": "motorhead",
        }
        for raw, want in cases.items():
            self.assertEqual(norm_key(raw), want, raw)


if __name__ == "__main__":
    init_app_state()
    print(f"initialised {SQLITE_PATH}")
    unittest.main()