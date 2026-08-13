"""
Tool-routing fixtures. Run: python eval_routing.py

Asserts which finder the model picks for a prompt, and nothing else. No LLM
judge, no YouTube quota, no warehouse writes -- one model call per case, so the
whole suite is cheap enough to run on every prompt edit.

Every case here exists because the routing was once wrong, or is one edit away
from being wrong. The pairs that matter:

  bare name           -> ARTIST, not song       ("play some Adele")
  name + feeling      -> ARTIST, not mood       ("romantic songs by Arijit Singh")
  "like" + title      -> SONG, not artist       ("more songs like Hello by Adele")
  feeling, no name    -> MOOD                   ("something sad for a long drive")

Exit code is 1 on any failure, so this can gate a commit or a deploy.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv("secrets.env")

CASES = [
    # (prompt, expected tool, expected args subset)
    ("songs of Arijit Singh",          "find_tracks_by_artist", {}),
    ("play some Adele",                "find_tracks_by_artist", {}),
    ("romantic songs by Arijit Singh", "find_tracks_by_artist", {}),
    ("asap rock songs",                "find_tracks_by_artist", {}),
    ("top 10 hip-hop songs",           "find_tracks_by_genre",  {}),
    ("make me a rock playlist",        "find_tracks_by_genre",  {}),
    ("something sad for a long drive", "find_tracks_by_mood",   {}),
    ("music for focus",                "find_tracks_by_mood",   {}),
    ("songs like Ms. Jackson",         "find_tracks_like_song",
     {"track": "Ms. Jackson"}),
    ("something similar to Chocolate", "find_tracks_like_song",
     {"track": "Chocolate"}),
    ("more songs like Hello by Adele", "find_tracks_like_song",
     {"track": "Hello", "artist": "Adele"}),
]


def main() -> int:
    from databricks_langchain import ChatDatabricks
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool

    import tools as T

    def sanitize(s):
        drop = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "multipleOf", "minLength", "maxLength", "minItems", "maxItems"}
        if isinstance(s, dict):
            return {k: sanitize(v) for k, v in s.items() if k not in drop}
        if isinstance(s, list):
            return [sanitize(v) for v in s]
        return s

    specs = [sanitize(convert_to_openai_tool(t)) for t in T.READ_TOOLS]
    llm = ChatDatabricks(
        endpoint=os.environ.get("LLM_ENDPOINT",
                                "databricks-meta-llama-3-3-70b-instruct"),
        temperature=0).bind(tools=specs)
    system = SystemMessage(T.system_prompt(False))

    failures = []
    for prompt, want_tool, want_args in CASES:
        reply = llm.invoke([system, HumanMessage(prompt)])
        calls = reply.tool_calls or []
        got_tool = calls[0]["name"] if calls else "(no tool call)"
        got_args = calls[0]["args"] if calls else {}

        bad_args = {k: (want_args[k], got_args.get(k))
                    for k in want_args
                    if str(got_args.get(k, "")).lower() != want_args[k].lower()}

        if got_tool != want_tool or bad_args:
            failures.append((prompt, want_tool, got_tool, bad_args))
            print(f"FAIL  {prompt}")
            print(f"      tool: got {got_tool}, want {want_tool}")
            for k, (want, got) in bad_args.items():
                print(f"      arg {k}: got {got!r}, want {want!r}")
        else:
            print(f"pass  {prompt:<34} -> {got_tool}")

    total = len(CASES)
    print(f"\n{total - len(failures)}/{total} routed correctly")

    # A multi-turn regression too: several Databricks-hosted models stop emitting
    # structured tool_calls after two tool round-trips, which silently breaks the
    # write path. Cheap to assert, expensive to rediscover.
    print("\nchecking structured tool_calls survive a 2-round-trip history...")
    from langchain_core.messages import AIMessage, ToolMessage
    history = [
        system,
        HumanMessage("top 5 hip-hop songs"),
        AIMessage(content="", tool_calls=[{
            "name": "find_tracks_by_genre", "id": "c1",
            "args": {"genre": "hip-hop", "limit": 5}}]),
        ToolMessage(content='{"candidate_set_id": "cs_test", "count": 5, '
                            '"source": "genre:hip-hop", "preview": []}',
                    tool_call_id="c1", name="find_tracks_by_genre"),
    ]
    deep = llm.invoke(history)
    if deep.tool_calls:
        print(f"pass  depth-1 still structured -> {deep.tool_calls[0]['name']}")
    else:
        print("FAIL  model returned prose instead of a tool call:")
        print(f"      {str(deep.content)[:160]}")
        failures.append(("multi-turn depth", "tool_call", "prose", {}))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
