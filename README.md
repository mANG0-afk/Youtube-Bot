---
title: Playlist Agent
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.51.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Playlist Agent

Ask by genre, artist, or mood; get real tracks and a playable YouTube playlist.

- **Genre** → top tracks from a Databricks gold layer (Lakeflow bronze→silver→gold)
- **Artist** → live Last.fm `artist.getTopTracks`
- **Mood** → tag search, falling back to top artists for that mood

A LangGraph agent picks the tool, resolves each track to a YouTube video, and
hands back a link that plays the whole set. Every turn is traced to MLflow.

## Modes

Visitors get track lists and a ready-to-play link they can save to their own
account. Playlist *creation* writes to the owner's channel, so it unlocks only
with `OWNER_KEY`.

**Set `OWNER_KEY` on any public deployment.** With it unset every visitor is
treated as the owner — convenient locally, wrong in public.

## Configuration

Required: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`,
`DATABRICKS_HTTP_PATH`, `LASTFM_API_KEY`, `YOUTUBE_API_KEY`,
`YOUTUBE_OAUTH_CLIENT_ID`, `YOUTUBE_OAUTH_CLIENT_SECRET`,
`YOUTUBE_OAUTH_REFRESH_TOKEN`.

Optional: `OWNER_KEY`, `MLFLOW_EXPERIMENT`, `LLM_ENDPOINT`, `MUSIC_CATALOG`,
`YT_DAILY_BUDGET`, `SESSION_TRACK_LIMIT`, `SESSION_PLAYLIST_CAP`.

Locally, put them in `secrets.env` next to `app.py`. On Spaces, add each as a
**Secret** (not a Variable). Then:

```
streamlit run app.py --server.address localhost
```

## Notes

The YouTube quota is billed to this project, not to visitors: ~100 units per
uncached track lookup against a 10,000/day ceiling. `app_state.db` ships a warm
resolution cache so repeat lookups are free, and the app degrades to plain search
links once the budget is spent rather than erroring.