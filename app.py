"""
Public chat UI. Runs anywhere the nine env vars are set -- Hugging Face Spaces,
Render, or locally.

Locally: put the nine vars in secrets.env next to this file.
On HF Spaces: Settings -> Variables and secrets -> add each as a SECRET.
"""

import hmac
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

# Loads secrets.env locally; a no-op on HF where env vars already exist.
load_dotenv("secrets.env")

st.set_page_config(page_title="Playlist Agent", page_icon="🎵",
                   layout="centered")

# Per-session caps. The YouTube quota is shared across ALL visitors, so without
# these one person with a refresh key ends the day for everyone.
SESSION_TRACK_LIMIT = int(os.environ.get("SESSION_TRACK_LIMIT", "10"))
SESSION_PLAYLIST_CAP = int(os.environ.get("SESSION_PLAYLIST_CAP", "2"))

# ---------------------------------------------------------------------------
# Access gate
# ---------------------------------------------------------------------------
# A public Space has no unlisted mode -- if it is reachable, it is findable. The
# OAuth token writes to a real YouTube channel and the daily quota is shared
# across all visitors, so an unguarded deploy lets a stranger spend both. Unset
# DEMO_PASSWORD and the gate disappears entirely, which is what local runs want.
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")


def authorized() -> bool:
    if not DEMO_PASSWORD:
        return True
    if st.session_state.get("authed"):
        return True

    st.title("🎵 Playlist Agent")
    st.caption("This is a portfolio demo. Enter the password from my resume "
               "to try it.")
    with st.form("gate"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        # compare_digest, not == : constant time, so the password can't be
        # recovered a character at a time from response timing. Both sides are
        # encoded because compare_digest rejects non-ASCII str.
        if hmac.compare_digest(pw.encode("utf-8"),
                               DEMO_PASSWORD.encode("utf-8")):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    return False


# Gate BEFORE get_app(): an unauthorized visitor should never wake the SQL
# warehouse or build the graph.
if not authorized():
    st.stop()


@st.cache_resource(show_spinner="Connecting to warehouse…")
def get_app():
    """
    Built once per container, not per rerun. Streamlit re-executes this whole
    script on every interaction; without cache_resource the graph would be
    rebuilt and the warehouse reconnected on every keystroke.
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
    return tools.build_graph(cp), be, tools


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


try:
    app, be, tools = get_app()
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

cfg = {"configurable": {"thread_id": st.session_state.thread}}


# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------
st.title("🎵 Playlist Agent")
st.caption("Ask for songs by genre, artist, or mood. It builds a real YouTube "
           "playlist you can open and play.")

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "**Genre** → top tracks from a Databricks gold layer  \n"
        "**Artist** → live Last.fm `artist.getTopTracks`  \n"
        "**Mood** → tag search, falling back to top artists for that mood"
    )

    st.divider()
    st.caption("Try one of these")
    for ex in ["top 10 hip-hop songs",
               "songs of Arijit Singh",
               "something sad for a long drive",
               "make me a rock playlist"]:
        st.code(ex, language=None)

    st.divider()
    # Quota is shared, so surfacing it explains degraded mode rather than
    # leaving visitors thinking the app is broken.
    try:
        used = be.quota_used_today()
        st.caption(f"YouTube quota today: {used:,} / {be.DAILY_BUDGET:,}")
        st.progress(min(1.0, used / max(be.DAILY_BUDGET, 1)))
        if used > be.DAILY_BUDGET:
            st.warning("Playlist creation is paused until midnight Pacific. "
                       "You'll still get track lists with links.")
    except Exception:
        pass

    if st.button("Start over"):
        for k in ("thread", "history", "pending", "playlists_made"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()
    st.caption("Databricks Lakeflow (bronze→silver→gold) · LangGraph · Last.fm "
               "· YouTube Data API")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)


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
        with st.spinner("Writing to YouTube…" if go else "Cancelling…"):
            try:
                out = app.invoke(Command(resume=bool(go)), cfg)
                reply = text_of(out["messages"][-1])
            except Exception as e:
                reply = f"Something went wrong: {type(e).__name__}"
        st.session_state.history.append(("assistant", reply))
        st.rerun()

    # Don't accept new input while an approval is outstanding -- the graph is
    # mid-tool and a second invoke would confuse the checkpointer.
    st.stop()


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prompt = st.chat_input("What should I put together?")
if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cap the size so "make me a 50 song playlist" can't drain the budget.
    sent = f"{prompt}\n\n(Use at most {SESSION_TRACK_LIMIT} tracks.)"

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            out = app.invoke({"messages": [("user", sent)]}, cfg)
        except Exception as e:
            msg = f"Something went wrong: {type(e).__name__}"
            st.markdown(msg)
            st.session_state.history.append(("assistant", msg))
            st.stop()

        if "__interrupt__" in out:
            st.session_state.pending = out["__interrupt__"][0].value
            st.rerun()

        reply = text_of(out["messages"][-1])
        st.markdown(reply)
        st.session_state.history.append(("assistant", reply))