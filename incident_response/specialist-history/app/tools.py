"""Tools for the incident-history specialist.

Reads a synthetic, self-contained `data/past_incidents.json` standing in
for a real postmortem/incident-tracker search (e.g. Jira, an internal
wiki, PagerDuty postmortems). Swap `_load` + the scoring below for a real
search API to point this at production data -- keep the tool
signatures/return shapes stable, they're the actual contract the agent
reasons over.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _load() -> list[dict[str, Any]]:
    return json.loads((_DATA_DIR / "past_incidents.json").read_text(encoding="utf-8"))


def _tokens(*texts: str) -> set[str]:
    joined = " ".join(texts).lower()
    return set(_WORD_RE.findall(joined))


def search_similar_incidents(query: str, tags: list[str] | None = None, top_k: int = 5) -> dict[str, Any]:
    """Finds past incidents most similar to a free-text query, optionally
    boosted by matching tags.

    Scoring is plain keyword overlap (no embeddings): each past incident's
    title + symptoms + service + root_cause is tokenized and scored by how
    many query tokens it shares, with each exact `tags` match adding extra
    weight. This is a cheap first pass -- good enough to shortlist
    candidates, not a substitute for reading the returned incidents.

    Args:
        query: Free text describing the current incident -- e.g. the
            incident title plus its symptoms and affected service, the
            more descriptive the better.
        tags: Optional exact tags to boost matches on (e.g.
            ["checkout-service", "timeout", "deploy-regression"]).
        top_k: Max number of incidents to return.

    Returns:
        status: "success".
        results: past incidents ranked by score (highest first), each with
            its `score` and full record (incident_id, date, title, service,
            symptoms, tags, root_cause, resolution). Only incidents with a
            score above 0 are included.
    """
    query_tokens = _tokens(query)
    tag_set = {t.lower() for t in (tags or [])}

    scored = []
    for inc in _load():
        inc_tokens = _tokens(inc["title"], " ".join(inc["symptoms"]), inc["service"], inc["root_cause"])
        overlap = len(query_tokens & inc_tokens)
        tag_matches = len(tag_set & {t.lower() for t in inc["tags"]})
        score = overlap + 3 * tag_matches
        if score > 0:
            scored.append({"score": score, **inc})

    scored.sort(key=lambda r: r["score"], reverse=True)
    return {"status": "success", "results": scored[:top_k]}
