"""
Six tools + LangGraph wiring.

Contract every finder honours: persist rows, return a HANDLE plus a short
preview. The preview is for the model to talk about; the handle is what flows
downstream. The LLM never carries the track list, so it cannot corrupt it.

GENRES and MOODS load from gold.genre / gold.mood_tag at import.
"""

import os
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from databricks_langchain import ChatDatabricks

import backend as be

be.init_app_state()

# Loaded once at import -- also your startup health check. Better to fail here
# than on the first user message.
_GENRE_ROWS = be.load_genres()
GENRES = [g["display_name"] for g in _GENRE_ROWS]
MOODS = be.load_moods()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    candidate_set_id: Optional[str]
    resolved_set_id: Optional[str]
    last_playlist_url: Optional[str]


# --------------------------------------------------------------------------
# SCHEMAS -- one per tool. Models route on "which tool" far more reliably
# than on "which combination of optional arguments".
# --------------------------------------------------------------------------
class GenreInput(BaseModel):
    genre: str = Field(description=f"One of: {', '.join(GENRES)}")
    limit: int = Field(10, description="How many tracks, 1-50")


class ArtistInput(BaseModel):
    artist: str = Field(description="Artist or band name as the user wrote it")
    limit: int = Field(10, description="How many tracks, 1-50")


class MoodInput(BaseModel):
    mood: str = Field(description=f"A mood or situation. Known: {', '.join(MOODS)}")
    limit: int = Field(10, description="How many tracks, 1-50")


class ResolveInput(BaseModel):
    candidate_set_id: str = Field(description="Handle from a find_tracks_* tool")


class AppendInput(BaseModel):
    resolved_set_id: str = Field(description="Handle from resolve_to_youtube")
    playlist_id: str = Field(description="Existing YouTube playlist ID")

class CreateInput(BaseModel):
    resolved_set_id: str = Field(description="Handle from resolve_to_youtube")
    title: str = Field("", description="Playlist title. Leave empty to "
                                       "auto-name from the artist, genre or mood.")
    description: str = Field("", description="Optional playlist description")


def _handle(rows: list[dict], source: str) -> dict:
    set_id = be.save_candidate_set(rows, source)
    return {
        "candidate_set_id": set_id,
        "count": len(rows),
        "source": source,
        "preview": [f"{r['track_name']} — {r['artist_name']}" for r in rows[:10]],
    }
def _source_title(resolved_set_id: str) -> str:
    """
    Fall back to the candidate set's source when no title was given.
    source looks like 'artist:AC/DC', 'genre:Hip-Hop', 'mood:sad:track_tags'.
    """
    with be._db() as c:
        row = c.execute(
            "SELECT cs.source FROM resolved_set rs "
            "JOIN candidate_set cs ON cs.set_id = rs.candidate_set_id "
            "WHERE rs.set_id = ?", (resolved_set_id,)).fetchone()
    if not row or not row["source"]:
        return "Playlist"
    kind, _, rest = row["source"].partition(":")
    label = rest.split(":")[0].strip()
    if not label:
        return "Playlist"
    if kind == "artist":
        return f"{label} — Top Tracks"
    if kind == "genre":
        return f"Top {label} Tracks"
    if kind == "mood":
        return f"{label.title()} Mix"
    return label

def _clamp(n: int, lo: int = 1, hi: int = 50) -> int:
    """Bounds live here, not in the schema -- Databricks serving endpoints
    reject minimum/maximum on integer types in tool JSON schemas."""
    try:
        return max(lo, min(hi, int(n)))
    except (TypeError, ValueError):
        return 10


# --------------------------------------------------------------------------
# PATH 1 -- GENRE. Pure SQL against gold.genre_track. No API call.
# --------------------------------------------------------------------------
@tool(args_schema=GenreInput)
def find_tracks_by_genre(genre: str, limit: int = 10) -> dict:
    """Find the top tracks in a genre. Use when the user names a genre."""
    limit = _clamp(limit)
    match = be.resolve_genre(genre)
    if not match:
        return {"error": f"Unknown genre '{genre}'.", "available": GENRES}
    rows = be.genre_top_tracks(match["genre_key"], limit)
    if not rows:
        return {"error": f"No tracks stored for {match['display_name']}."}
    return _handle(rows, f"genre:{match['display_name']}")


# --------------------------------------------------------------------------
# PATH 2 -- ARTIST. Resolve locally, then hit Last.fm live.
#
# Genre rank is NOT artist rank: a track at #12 in hip-hop is ranked against
# all hip-hop, not the artist's own catalogue. So this path never reads
# genre_top_track -- artist.getTopTracks gives a real artist-level ranking.
# --------------------------------------------------------------------------
@tool(args_schema=ArtistInput)
def find_tracks_by_artist(artist: str, limit: int = 10) -> dict:
    """
    Get an artist's top tracks from Last.fm. Use for ANY named artist, band,
    singer, rapper or group -- including ones not in the local catalogue.
    """
    limit = _clamp(limit)
    match, near = be.resolve_artist(artist)

    # Local resolution is only an optimisation and an alias-learner. It misses
    # every correctly-spelled artist outside our catalogue, because
    # artist.getCorrection returns nothing when the spelling is already right.
    # Last.fm's autocorrect=1 handles the name, so always try the API before
    # giving up -- otherwise Arijit Singh looks unknown and the model
    # substitutes a different tool.
    name = match["artist_name"] if match else artist
    corrected, rows = be.lastfm_top_tracks(name, limit)

    if not rows:
        return {
            "error": f"Last.fm has no top tracks for '{name}'.",
            "instruction": "STOP. Tell the user this artist could not be found "
                           "and ask them to check the spelling. Do NOT call "
                           "find_tracks_by_mood or find_tracks_by_genre. Do NOT "
                           "substitute other songs.",
            "did_you_mean": near or [],
        }

    return _handle(rows, f"artist:{corrected or name}")


# --------------------------------------------------------------------------
# PATH 3 -- MOOD. Two stages, both deterministic, both INSIDE the tool.
# Exposing stage 2 as its own tool would hand a fixed business rule to the
# model's judgement.
# --------------------------------------------------------------------------
@tool(args_schema=MoodInput)
def find_tracks_by_mood(mood: str, limit: int = 10) -> dict:
    """
    Find tracks by mood or situation. ONLY for requests that name no artist and
    no genre. Never use this as a fallback when another finder failed.
    """ 
    limit = _clamp(limit)
    key = mood.lower().strip()
    if key not in MOODS:
        return {"error": f"Unknown mood '{mood}'.", "available": MOODS}

    rows = be.mood_tracks(key, limit)           # stage 1: track tags
    stage = "track_tags"

    if len(rows) < limit:                        # stage 2: artist tags + API
        seen = {(r["artist_key"], r["track_key"]) for r in rows}
        artists = be.mood_artists(key, limit=4)
        for a in artists:
            if len(rows) >= limit:
                break
            _, top = be.lastfm_top_tracks(a["artist_name"], limit=2)
            for t in top:
                pair = (t["artist_key"], t["track_key"])
                if pair not in seen:
                    seen.add(pair)
                    rows.append(t)
        if artists:
            stage = "track_tags+artist_tags"

    if not rows:
        return {"error": f"Nothing matched mood '{mood}'.", "available": MOODS}
    return _handle(rows[:limit], f"mood:{key}:{stage}")


# --------------------------------------------------------------------------
# RESOLVE -- quota-aware, separate from the write on purpose. 100 units per
# cache miss, 0 per hit. Reporting hits/misses lets the model say "8 of 10
# found" so the user can bail before writes cost 50 units each.
# --------------------------------------------------------------------------
@tool(args_schema=ResolveInput)
def resolve_to_youtube(candidate_set_id: str) -> dict:
    """
    Resolve a candidate set to YouTube video IDs. Call after a find_tracks_*
    tool and before creating a playlist.
    """
    rows = be.load_candidate_set(candidate_set_id)
    if not rows:
        return {"error": f"Unknown candidate_set_id '{candidate_set_id}'."}

    # Check the budget BEFORE spending, so a public link degrades to free
    # search links instead of erroring once quota is gone.
    if be.quota_used_today() > be.DAILY_BUDGET:
        return {
            "quota_exhausted": True,
            "tracks": [{"track_name": r["track_name"],
                        "artist_name": r["artist_name"],
                        "search_url": be.search_url(r["artist_name"],
                                                    r["track_name"])}
                       for r in rows],
            "instruction": "Daily YouTube quota is spent. List the tracks with "
                           "each search_url as a link, and say playlist "
                           "creation resets at midnight Pacific. Do NOT call "
                           "create_youtube_playlist.",
        }

    found, missing, cached, api_error = [], [], 0, None
    for r in rows:
        try:
            vid, vtitle, was_cached = be.youtube_find_video(
                r["artist_key"] or "", r["track_key"] or "",
                r["artist_name"], r["track_name"])
        except RuntimeError as e:
            api_error = str(e)      # real API failure -- don't mislabel it
            break
        cached += int(was_cached)
        if vid:
            found.append({"video_id": vid, "video_title": vtitle,
                          "track_name": r["track_name"],
                          "artist_name": r["artist_name"]})
        else:
            missing.append(f"{r['track_name']} — {r['artist_name']}")

    if not found:
        out = {"error": "No tracks resolved to YouTube videos.",
               "unresolved": missing}
        if api_error:
            out["api_error"] = api_error
            out["instruction"] = ("This is a YouTube API failure, not missing "
                                  "songs. Report api_error verbatim and stop.")
        return out

    resolved_id = be.save_resolved_set(candidate_set_id, found)
    return {
        "resolved_set_id": resolved_id,
        "resolved": len(found),
        "unresolved": len(missing),
        "unresolved_titles": missing[:5],
        "cache_hits": cached,
        "preview_url": be.watch_videos_url([f["video_id"] for f in found]),
    }


# --------------------------------------------------------------------------
# WRITES -- interrupt() before the side effect. Quota is spent and the real
# account is mutated, so a human confirms. Needs a checkpointer.
# --------------------------------------------------------------------------
@tool(args_schema=CreateInput)
def create_youtube_playlist(resolved_set_id: str, title: str = "",
                            description: str = "") -> dict:
    """
    Create a new YouTube playlist from a resolved set. Spends quota and writes
    to the real account, so the user confirms first.
    """
    if not resolved_set_id.startswith("rs_"):
        # Structural guard: models do skip resolve_to_youtube and pass a cs_ id.
        return {"error": "That is a candidate_set_id. Call resolve_to_youtube "
                         "first and pass the resolved_set_id it returns."}

    rows = be.load_resolved_set(resolved_set_id)
    if not rows:
        return {"error": f"Unknown resolved_set_id '{resolved_set_id}'."}
    
    title = (title or "").strip() or _source_title(resolved_set_id)

    approved = interrupt({
        "action": "create_playlist",
        "title": title,
        "track_count": len(rows),
        "estimated_quota_units": 50 + 50 * len(rows),
        "tracks": [f"{r['track_name']} — {r['artist_name']}" for r in rows],
    })
    if not approved:
        return {"status": "cancelled_by_user"}

    playlist_id = be.youtube_create_playlist(title, description)
    added, failed = be.youtube_add_items(playlist_id, rows, resolved_set_id)
    return {
        "status": "created",
        "playlist_id": playlist_id,
        "playlist_url": f"https://www.youtube.com/playlist?list={playlist_id}",
        "tracks_added": added,
        "failed": failed[:5],
    }


@tool(args_schema=AppendInput)
def add_to_youtube_playlist(resolved_set_id: str, playlist_id: str) -> dict:
    """Append a resolved set to an existing YouTube playlist."""
    if not resolved_set_id.startswith("rs_"):
        return {"error": "That is a candidate_set_id. Call resolve_to_youtube "
                         "first and pass the resolved_set_id it returns."}

    rows = be.load_resolved_set(resolved_set_id)
    if not rows:
        return {"error": f"Unknown resolved_set_id '{resolved_set_id}'."}

    # Structural guard, not a style rule: the model does call this immediately
    # after a successful create_youtube_playlist, trying to add the same tracks
    # to the playlist it just made. That paused the graph on a second interrupt
    # and the user never got their link. Same set into the same playlist is
    # always a no-op, so refuse it and hand back the URL.
    with be._db() as c:
        done = c.execute(
            "SELECT 1 FROM playlist_log WHERE resolved_set_id=? AND playlist_id=?"
            " LIMIT 1", (resolved_set_id, playlist_id)).fetchone()
    if done:
        return {
            "error": "already_added",
            "playlist_url": f"https://www.youtube.com/playlist?list={playlist_id}",
            "instruction": "These tracks are already in that playlist. Do NOT "
                           "write again. Reply to the user with the playlist_url "
                           "above and stop.",
        }

    approved = interrupt({
        "action": "append_to_playlist",
        "playlist_id": playlist_id,
        "track_count": len(rows),
        "estimated_quota_units": 50 * len(rows),
        "tracks": [f"{r['track_name']} — {r['artist_name']}" for r in rows],
    })
    if not approved:
        return {"status": "cancelled_by_user"}

    added, failed = be.youtube_add_items(playlist_id, rows, resolved_set_id)
    return {
        "status": "appended",
        "playlist_url": f"https://www.youtube.com/playlist?list={playlist_id}",
        "tracks_added": added,   # already-present videos are skipped, not duped
        "failed": failed[:5],
    }


# Reads cost YouTube quota but touch nobody's account. Writes mutate a real
# channel, so they are bound only for the owner -- see build_graph(allow_writes).
READ_TOOLS = [
    find_tracks_by_genre,
    find_tracks_by_artist,
    find_tracks_by_mood,
    resolve_to_youtube,
]

WRITE_TOOLS = [
    create_youtube_playlist,
    add_to_youtube_playlist,
]

TOOLS = READ_TOOLS + WRITE_TOOLS


_SYSTEM_BASE = f"""You are a music playlist assistant.

Genres in the catalogue: {', '.join(GENRES)}
Moods supported: {', '.join(MOODS)}

## Choosing a tool

Check in this order and stop at the first match:

1. Does the request name a specific person, band, singer, rapper, duo or group?
   -> find_tracks_by_artist, with that name exactly as written.
   This applies to EVERY named act, including ones you have never heard of and
   ones absent from the genre list. "Arijit Singh", "AC/DC", "Kanye West",
   "asap rock" are all artists. Do NOT reroute to mood because you associate
   the artist with romance, sadness, energy or any other feeling. A name is an
   artist, full stop.

2. Otherwise, does it name one of the genres listed above?
   -> find_tracks_by_genre

3. Otherwise, does it describe a feeling, situation or activity with no artist
   and no genre named ("something sad", "music for driving")?
   -> find_tracks_by_mood

If none match, ask a clarifying question. Do not guess a tool.

## Never substitute

If a tool returns an error, report that error to the user and stop. NEVER call a
different find_tracks_* tool to compensate. If find_tracks_by_artist fails for
an artist, say so -- do not return mood-based or genre-based songs instead.
Returning the wrong kind of result is worse than returning none.

## Truthfulness

Only name tracks a tool actually returned. If a tool reports count 30 with a
preview of 10, say 30 and list the 10 you were given. Never fill a list from
your own knowledge of an artist's discography."""


# Appended for the owner, who has the write tools bound.
_WRITE_MODE = """

## Sequence

Always, in this order:
1. a find_tracks_* tool          -> returns candidate_set_id
2. resolve_to_youtube(candidate_set_id) -> returns resolved_set_id
3. create_youtube_playlist(resolved_set_id) or add_to_youtube_playlist

Never skip step 2. Pass ids exactly as returned; never invent or edit one.
After resolve_to_youtube, share the preview_url so the user can listen before
committing.

## Writes are terminal

create_youtube_playlist and add_to_youtube_playlist END the job. The moment one
returns status "created" or "appended", your next message is plain text for the
user that includes the playlist_url from that result, and you call NO further
tool. Do not call the other write tool. Do not call the same one again. The
playlist already exists and every extra write costs 50 units per track.

## Titles

Omit `title` and it is auto-named from the artist, genre or mood. Pass a title
only when the user asked for a specific name."""


# Appended for visitors, who have no write tools bound at all. Saying so plainly
# matters: a model asked to do something it has no tool for tends to either claim
# it did or describe the call in prose, and both read as a broken app.
_READ_ONLY = """

## Sequence

Always, in this order:
1. a find_tracks_* tool          -> returns candidate_set_id
2. resolve_to_youtube(candidate_set_id) -> returns resolved_set_id

Never skip step 2. Pass ids exactly as returned; never invent or edit one.

## Finish with the preview link

You have NO tool that creates or edits a YouTube playlist. Do not offer to make
one, do not claim to have made one, and never write a tool call as text.

resolve_to_youtube is the last step. Reply with the track list, then say that the
play link below opens all the tracks in order and that YouTube's own Save button
adds them to their library.

Do NOT paste the preview_url, or any other URL, into your reply. The app renders
the link itself from the tool result. Asked to retype a long URL you mangle it,
and a broken link is the one thing this demo cannot afford.

If the user asks you to create or save a playlist for them, say that this demo
resolves tracks and hands back a ready-to-play link they can save themselves."""


def system_prompt(allow_writes: bool) -> str:
    return _SYSTEM_BASE + (_WRITE_MODE if allow_writes else _READ_ONLY)


# Retained for callers that just want the full-capability prompt.
SYSTEM = system_prompt(True)


def build_graph(checkpointer, allow_writes: bool = True):
    """
    allow_writes=False binds only the read tools and swaps in the read-only
    prompt. Withholding the tool is the control; the prompt only explains the
    absence. Telling a model it "may not" use a tool it can still see invites it
    to try anyway, and the approval interrupt would then fire for a visitor who
    has no business writing to the owner's channel.
    """
    from langchain_core.utils.function_calling import convert_to_openai_tool

    def sanitize(s):
        """Databricks serving endpoints reject numeric/length JSON-schema
        constraints. Strip them at any depth -- cheap insurance so a stray
        ge=/max_length= added later can't 400 the whole app."""
        drop = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "multipleOf", "minLength", "maxLength", "minItems", "maxItems"}
        if isinstance(s, dict):
            return {k: sanitize(v) for k, v in s.items() if k not in drop}
        if isinstance(s, list):
            return [sanitize(v) for v in s]
        return s

    bound = TOOLS if allow_writes else READ_TOOLS
    specs = [sanitize(convert_to_openai_tool(t)) for t in bound]
    prompt = system_prompt(allow_writes)

    # llama-4-maverick and gpt-oss/qwen on Databricks stop emitting structured
    # tool_calls once the history holds TWO tool round-trips -- they fall back to
    # Llama's text syntax ("[create_youtube_playlist(resolved_set_id=rs_x)]") and
    # even leak a raw "assistant" header. tools_condition then sees no tool_calls,
    # routes to END, and the write tool never runs, so the approval interrupt
    # never fires. llama-3-3-70b handles the multi-turn tool history correctly.
    llm = ChatDatabricks(
        endpoint=os.environ.get("LLM_ENDPOINT",
                                "databricks-meta-llama-3-3-70b-instruct"),
        temperature=0).bind(tools=specs)

    def agent(state: AgentState):
        return {"messages": [llm.invoke([SystemMessage(prompt)] + state["messages"])]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(bound))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)