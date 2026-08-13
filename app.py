"""
Public chat UI. Runs anywhere the nine env vars are set -- Hugging Face Spaces,
Render, or locally.

Locally: put the nine vars in secrets.env next to this file.
On HF Spaces: Settings -> Variables and secrets -> add each as a SECRET.
"""

import hmac
import json
import os
import uuid
from contextlib import contextmanager

import streamlit as st
from dotenv import load_dotenv

# Loads secrets.env locally; a no-op on hosts where env vars already exist.
load_dotenv("secrets.env")

# Streamlit Community Cloud delivers secrets through st.secrets rather than the
# environment, but backend.py and tools.py read os.environ (and are imported by a
# background thread that has no Streamlit context to read st.secrets from). Copy
# them across before anything imports those modules. setdefault, so a real
# environment variable or secrets.env still wins.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass        # no secrets.toml -- normal locally and on env-var hosts

st.set_page_config(page_title="Playlist Agent", page_icon="🎵",
                   layout="centered")

# Per-session caps. The YouTube quota is shared across ALL visitors, so without
# these one person with a refresh key ends the day for everyone.
SESSION_TRACK_LIMIT = int(os.environ.get("SESSION_TRACK_LIMIT", "10"))
SESSION_PLAYLIST_CAP = int(os.environ.get("SESSION_PLAYLIST_CAP", "2"))

EXAMPLES = ["top 10 hip-hop songs",
            "songs of Arijit Singh",
            "something sad for a long drive",
            "make me a rock playlist"]

YELLOW = "#FFD400"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
# Selectors are data-testid attributes rather than generated class names, which
# Streamlit rewrites between versions. Everything degrades to the config.toml
# palette if a selector ever stops matching, so a Streamlit upgrade can make
# this plainer but not broken.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{ --yellow: {YELLOW}; --ink: #0A0A0A; }}

html, body, [data-testid="stAppViewContainer"], button, input, textarea {{
    font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', 'Cascadia Code',
                 Consolas, 'Courier New', monospace !important;
}}

[data-testid="stAppViewContainer"] {{ background: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 2.2rem; max-width: 46rem; }}

/* Top strip: quota left, reset right */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    gap: .75rem; margin-bottom: 1.4rem;
}}
.quota {{
    font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
    color: var(--yellow); border: 1px solid #3A3000; border-radius: 2px;
    padding: .3rem .55rem; background: #1A1600; white-space: nowrap;
}}
.quota b {{ color: #FFF3B0; font-weight: 700; }}
.quota.spent {{ color: #FF8A6B; border-color: #4A2318; background: #1F1210; }}
.quota.owner {{ margin-left: .4rem; color: var(--ink); background: var(--yellow);
                border-color: var(--yellow); font-weight: 700; }}

h1.brand {{
    font-size: 1.85rem !important; font-weight: 700 !important;
    letter-spacing: -.02em; color: var(--yellow) !important;
    margin: 0 0 .3rem 0 !important;
}}
p.tagline {{ color: #8C8C8C; font-size: .82rem; margin: 0 0 1.6rem 0; }}

/* Search bar */
[data-testid="stForm"] {{
    border: 1px solid #2A2A2A; border-radius: 3px; background: #121212;
    padding: .55rem .6rem;
}}
[data-testid="stForm"]:focus-within {{ border-color: var(--yellow); }}
[data-testid="stForm"] input {{
    background: transparent !important; border: none !important;
    color: #EDEDED !important; font-size: .95rem !important;
}}
[data-testid="stForm"] input::placeholder {{ color: #5A5A5A !important; }}

/* Buttons: the submit arrow is solid yellow, example chips are outlined */
.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 2px; font-size: .78rem; font-weight: 500;
    transition: background .12s, color .12s, border-color .12s;
}}
[data-testid="stFormSubmitButton"] > button {{
    background: var(--yellow); color: var(--ink); border: none;
    font-weight: 700; width: 100%;
}}
[data-testid="stFormSubmitButton"] > button:hover {{ background: #FFE45C; }}

.chips .stButton > button {{
    background: transparent; color: #B8B8B8; border: 1px solid #2E2E2E;
    text-align: left; padding: .45rem .6rem;
}}
.chips .stButton > button:hover {{
    border-color: var(--yellow); color: var(--yellow); background: #171400;
}}
.hint {{
    color: #6E6E6E; font-size: .7rem; letter-spacing: .08em;
    text-transform: uppercase; margin: 1.1rem 0 .55rem 0;
}}

/* Top-left warm-up indicator */
.warm {{
    display: flex; align-items: center; gap: .5rem; color: #8C8C8C;
    font-size: .7rem; letter-spacing: .08em; text-transform: uppercase;
    margin-bottom: 1rem;
}}
.spin {{
    width: .78rem; height: .78rem; border: 2px solid #3A3000;
    border-top-color: var(--yellow); border-radius: 50%;
    display: inline-block; animation: spin .7s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* Landing: owner key, then a visitor path */
p.orbar {{
    display: flex; align-items: center; gap: .8rem;
    color: #5A5A5A; font-size: .7rem; letter-spacing: .12em;
    text-transform: uppercase; margin: 1.1rem 0;
}}
p.orbar::before, p.orbar::after {{
    content: ""; flex: 1; height: 1px; background: #232323;
}}
.modenote {{
    margin-top: 1.8rem; padding-top: .9rem; border-top: 1px solid #1E1E1E;
    color: #6E6E6E; font-size: .72rem; line-height: 1.9;
}}
.modenote b {{ color: var(--yellow); font-weight: 700; }}

/* Conversation */
[data-testid="stChatMessage"] {{
    background: #101010; border: 1px solid #232323; border-radius: 3px;
    padding: .8rem .9rem; margin-bottom: .6rem;
}}
[data-testid="stChatMessage"] a {{ color: var(--yellow); }}

.footer {{
    margin-top: 2.6rem; padding-top: .9rem; border-top: 1px solid #1E1E1E;
    color: #4E4E4E; font-size: .68rem; line-height: 1.7;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Owner unlock
# ---------------------------------------------------------------------------
# Nobody is kept out. Visitors get tracks and a ready-to-play preview link they
# can save to their own account, which costs read quota but writes nothing.
# Playlist CREATION uses the owner's OAuth token against the owner's channel, so
# it unlocks only for whoever holds OWNER_KEY. Unset OWNER_KEY and writes are
# simply on, which is what a local run wants.
# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def warmup() -> dict:
    """
    Connect to the warehouse in the background as soon as the URL is hit.

    Cold start is ~41s: importing backend pulls the Databricks SQL/SDK stack
    (14s) and importing tools runs load_genres()/load_moods() against the
    warehouse (21s). That cost used to land on the click that picks a mode, and
    Streamlit keeps the PREVIOUS frame on screen while a slow rerun finishes --
    so the landing screen, button included, sat there looking like the click had
    done nothing. Starting the work when the page first loads overlaps it with
    the user reading the page.

    Once per container via cache_resource. The thread calls no st.* API, so it
    needs no script context, and the returned dict is shared state the script
    polls. Import locking means a mode picked mid-warm simply waits on the same
    work rather than duplicating it.
    """
    import threading

    status = {"ready": False, "error": None}

    def connect():
        try:
            import backend as be
            import tools            # noqa: F401 -- import IS the work
            be.init_app_state()
            status["ready"] = True
        except Exception as e:
            status["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=connect, daemon=True).start()
    return status


WARM = warmup()


@st.fragment(run_every=1.5)
def warm_badge() -> None:
    """
    Self-refreshing so it clears itself without waiting for the user to click.
    A plain element would sit there stale until the next natural rerun.
    """
    if WARM["ready"] or WARM["error"]:
        return
    st.markdown('<div class="warm"><span class="spin"></span>starting up</div>',
                unsafe_allow_html=True)


OWNER_KEY = os.environ.get("OWNER_KEY", "")


def mode():
    """
    "owner", "visitor", or None while the choice is still open. With no OWNER_KEY
    there is nothing to unlock, so a local run skips the choice and gets writes.
    """
    if not OWNER_KEY:
        return "owner"
    return st.session_state.get("mode")


def is_owner() -> bool:
    return mode() == "owner"


def choose_mode() -> None:
    """
    Landing screen. Rendered before get_app(), so somebody who never picks a mode
    never wakes the SQL warehouse or builds a graph.
    """
    st.markdown('<h1 class="brand">PLAYLIST AGENT</h1>'
                '<p class="tagline">Ask by genre, artist, or mood. It finds real '
                'tracks and hands you a YouTube playlist you can play.</p>',
                unsafe_allow_html=True)

    with st.form("unlock", clear_on_submit=True):
        key = st.text_input("Owner key", type="password",
                            label_visibility="collapsed",
                            placeholder="owner key")
        if st.form_submit_button("Unlock owner mode"):
            # compare_digest, not == : constant time, so the key can't be
            # recovered a character at a time from response timing. Both sides
            # are encoded because compare_digest rejects non-ASCII str.
            if hmac.compare_digest(key.encode("utf-8"),
                                   OWNER_KEY.encode("utf-8")):
                st.session_state.mode = "owner"
                st.rerun()
            st.error("Wrong key.")

    st.markdown('<p class="orbar">or</p>', unsafe_allow_html=True)

    if st.button("Continue as visitor  →", use_container_width=True):
        st.session_state.mode = "visitor"
        st.rerun()

    st.markdown(
        '<div class="modenote">'
        '<b>Visitor</b> — track lists plus a ready-to-play YouTube link you can '
        'save to your own account.<br>'
        '<b>Owner</b> — also creates the playlist directly on the owner\'s '
        'channel.<br><br>'
        'Reload the page to switch modes.'
        '</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Connecting to warehouse…")
def get_app(allow_writes: bool):
    """
    Built once per container per capability, not per rerun. Streamlit re-executes
    this whole script on every interaction; without cache_resource the graph
    would be rebuilt and the warehouse reconnected on every keystroke.

    Keyed on allow_writes so the read-only and full graphs are separate cached
    objects. cache_resource is shared across ALL sessions, so a single graph
    holding the write tools would hand them to every visitor.
    """
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    import backend as be
    import tools

    be.init_app_state()
    # check_same_thread=False: Streamlit may touch it from a worker thread.
    # NOT SqliteSaver.from_conn_string -- that is a context manager and would
    # close the connection immediately, losing the paused interrupt state.
    cp = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
    return tools.build_graph(cp, allow_writes=allow_writes), be, tools


def text_of(msg) -> str:
    """
    Reasoning models (gpt-oss) return content as a list of blocks rather than a
    string. Extract only the text parts so the thinking isn't shown to users.
    """
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    parts = []
    for b in (c or []):
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(p for p in parts if p).strip()


def preview_link(out) -> str:
    """
    The play link, built from resolve_to_youtube's own result.

    This is the whole product for a visitor, and the model cannot be trusted to
    reproduce it: asked to render the URL it retyped the video-id list as the
    link label and corrupted it. So the model is told not to emit URLs at all and
    the link is appended here, where it is always exact.
    """
    for m in reversed(out.get("messages", [])):
        if getattr(m, "name", None) != "resolve_to_youtube":
            continue
        try:
            data = json.loads(m.content)
        except (ValueError, TypeError):
            continue
        url = data.get("preview_url")
        if url:
            n = data.get("resolved") or 0
            return f"\n\n▶ **[Play all {n} tracks]({url})** — opens on YouTube; " \
                   "hit Save there to keep it."
    return ""


def written_url(out) -> str:
    """
    Last-resort reply built from the write tool's own result.

    The quota is already spent and the playlist already exists by this point, so
    the user has to get their link even when the model returns an empty message.
    Reading it back out of the tool result is also more trustworthy than asking
    the model to repeat a URL it could mangle.
    """
    for m in reversed(out.get("messages", [])):
        if getattr(m, "name", None) not in ("create_youtube_playlist",
                                            "add_to_youtube_playlist"):
            continue
        try:
            data = json.loads(m.content)
        except (ValueError, TypeError):
            continue
        url = data.get("playlist_url")
        if url:
            added = data.get("tracks_added")
            count = f" with {added} tracks" if added else ""
            return f"Your playlist is ready{count}: {url}"
    return ""


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
# Set MLFLOW_EXPERIMENT to an experiment path (e.g. /Shared/playlist-agent) and
# every turn is logged as a trace: the agent's reasoning, each tool call with its
# arguments, latency and token counts. Leave it unset and nothing is traced, so
# local runs stay offline and free.
#
# MLFLOW_TRACKING_URI defaults to "databricks" because the workspace is already
# authenticated by the DATABRICKS_* vars. A local ./mlruns store would be wiped
# on every Space restart, which is not monitoring.
@st.cache_resource(show_spinner=False)
def init_tracing() -> bool:
    """Once per container. Never fatal -- monitoring must not take the app down."""
    experiment = os.environ.get("MLFLOW_EXPERIMENT", "")
    if not experiment:
        return False
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                               "databricks"))
        mlflow.set_experiment(experiment)
        mlflow.langchain.autolog()
        return True
    except Exception:
        return False


TRACING = init_tracing()


@contextmanager
def trace_turn(kind: str):
    """
    Wraps a turn so autolog's spans hang off a root span carrying the thread id
    -- without it a trace cannot be tied back to the conversation it came from.
    A no-op when tracing is off, so call sites stay unconditional.
    """
    if not TRACING:
        yield
        return
    import mlflow
    with mlflow.start_span(name=kind) as span:
        span.set_attribute("thread_id", st.session_state.thread)
        span.set_attribute("playlists_made", st.session_state.playlists_made)
        yield


warm_badge()

if mode() is None:
    choose_mode()
    st.stop()


try:
    app, be, tools = get_app(is_owner())
except Exception as e:
    st.error("Backend unavailable.")
    st.caption(f"{type(e).__name__}: {e}")
    st.info("If this is the first request in a while, the SQL warehouse may be "
            "starting up — wait about 20 seconds and reload.")
    st.stop()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
if "thread" not in st.session_state:
    st.session_state.thread = f"web-{uuid.uuid4().hex[:10]}"
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.playlists_made = 0
    st.session_state.query = None

cfg = {"configurable": {"thread_id": st.session_state.thread}}


# ---------------------------------------------------------------------------
# Top strip -- quota sits top-left because a degraded mode needs explaining
# before someone concludes the app is broken.
# ---------------------------------------------------------------------------
def quota_chip() -> str:
    try:
        used = be.quota_used_today()
    except Exception:
        return ""
    spent = used > be.DAILY_BUDGET
    label = ("QUOTA SPENT — TRACK LISTS ONLY" if spent
             else f"YOUTUBE QUOTA TODAY <b>{used:,}</b> / {be.DAILY_BUDGET:,}")
    return f'<span class="quota{" spent" if spent else ""}">{label}</span>'


owner_badge = ('<span class="quota owner">OWNER</span>'
               if OWNER_KEY and is_owner() else "")
bar_left, bar_right = st.columns([4, 1], vertical_alignment="center")
bar_left.markdown(
    f'<div class="topbar">{quota_chip()}{owner_badge}</div>',
    unsafe_allow_html=True)
if st.session_state.history or st.session_state.pending:
    if bar_right.button("reset", use_container_width=True):
        for k in ("thread", "history", "pending", "playlists_made", "query"):
            st.session_state.pop(k, None)
        st.rerun()

st.markdown('<h1 class="brand">PLAYLIST AGENT</h1>'
            '<p class="tagline">Ask by genre, artist, or mood. It builds a real '
            'YouTube playlist you can open and play.</p>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Approval gate -- the graph is paused inside a write tool
# ---------------------------------------------------------------------------
if st.session_state.pending:
    p = st.session_state.pending
    with st.chat_message("assistant"):
        st.markdown(f"**Ready to create:** {p.get('title', 'playlist')}")
        for t in p.get("tracks", []):
            st.markdown(f"- {t}")
        st.caption(f"{p.get('track_count', 0)} tracks · "
                   f"~{p.get('estimated_quota_units', 0)} quota units")

        over_cap = st.session_state.playlists_made >= SESSION_PLAYLIST_CAP
        if over_cap:
            st.warning(f"Limit of {SESSION_PLAYLIST_CAP} playlists per session "
                       "reached — the YouTube quota is shared across visitors.")

        c1, c2 = st.columns(2)
        go = c1.button("Create playlist", type="primary", disabled=over_cap,
                       use_container_width=True)
        no = c2.button("Cancel", use_container_width=True)

    if go or no:
        from langgraph.types import Command
        st.session_state.pending = None
        if go:
            st.session_state.playlists_made += 1
        out, reply = None, ""
        with st.spinner("Writing to YouTube…" if go else "Cancelling…"):
            try:
                with trace_turn("approval_resume"):
                    out = app.invoke(Command(resume=bool(go)), cfg)
            except Exception as e:
                reply = f"Something went wrong: {type(e).__name__}"

        if out is not None:
            # A write can pause the graph AGAIN -- the model has been seen
            # calling the other write tool right after a successful one. Carry
            # that interrupt back into this gate. Dropping it left the graph
            # parked mid-tool and showed the user an empty reply.
            if "__interrupt__" in out:
                st.session_state.pending = out["__interrupt__"][0].value
                st.rerun()
            reply = text_of(out["messages"][-1]) or written_url(out) or (
                "Done." if go else "Cancelled — nothing was written.")

        st.session_state.history.append(("assistant", reply))
        st.rerun()

    # Don't accept new input while an approval is outstanding -- the graph is
    # mid-tool and a second invoke would confuse the checkpointer.
    st.stop()


# ---------------------------------------------------------------------------
# Turn. Runs BEFORE the search bar is drawn: a query set by the form or a chip
# arrives on the next rerun, so handling it first means the examples never flash
# on screen during the turn that retires them.
# ---------------------------------------------------------------------------
if st.session_state.query:
    prompt = st.session_state.query
    st.session_state.query = None
    st.session_state.history.append(("user", prompt))

    # Cap the size so "make me a 50 song playlist" can't drain the budget.
    sent = f"{prompt}\n\n(Use at most {SESSION_TRACK_LIMIT} tracks.)"

    with st.spinner("Thinking…"):
        try:
            with trace_turn("chat_turn"):
                out = app.invoke({"messages": [("user", sent)]}, cfg)
        except Exception as e:
            st.session_state.history.append(
                ("assistant", f"Something went wrong: {type(e).__name__}"))
            out = None

    if out is not None:
        if "__interrupt__" in out:
            st.session_state.pending = out["__interrupt__"][0].value
        else:
            st.session_state.history.append(
                ("assistant", text_of(out["messages"][-1]) + preview_link(out)))
    st.rerun()


# ---------------------------------------------------------------------------
# Search bar, with the examples directly beneath it. They vanish after the
# first query -- once there is a conversation they are noise.
# ---------------------------------------------------------------------------
with st.form("search", clear_on_submit=True):
    field, send = st.columns([7, 1], vertical_alignment="center")
    typed = field.text_input("Ask", label_visibility="collapsed",
                             placeholder="What should I put together?")
    submitted = send.form_submit_button("→")

if submitted and typed.strip():
    st.session_state.query = typed.strip()
    st.rerun()

if not st.session_state.history:
    st.markdown('<p class="hint">Try one of these</p>', unsafe_allow_html=True)
    st.markdown('<div class="chips">', unsafe_allow_html=True)
    left, right = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if (left if i % 2 == 0 else right).button(ex, key=f"ex{i}",
                                                 use_container_width=True):
            st.session_state.query = ex
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# History -- newest turn last, below the input
# ---------------------------------------------------------------------------
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

st.markdown(
    '<div class="footer">'
    'Databricks Lakeflow (bronze→silver→gold) · LangGraph · Last.fm · '
    'YouTube Data API<br>'
    'Genre → gold layer &nbsp;·&nbsp; Artist → Last.fm artist.getTopTracks '
    '&nbsp;·&nbsp; Mood → tag search'
    '</div>', unsafe_allow_html=True)
