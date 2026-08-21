"""Tools for the bisection specialist.

Reads two synthetic, self-contained datasets (`data/commits.json`,
`data/deploys.json`) standing in for a real VCS + deploy-tracker
integration (e.g. GitHub API + your CD system's release log). Swap
`_load` for real API calls to point this at production data -- the tool
signatures/return shapes are the actual contract the agent reasons over,
so keep those stable if you do.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> list[dict[str, Any]]:
    return json.loads((_DATA_DIR / name).read_text(encoding="utf-8"))


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def list_recent_deploys(service: str = "", hours_back: int = 72) -> dict[str, Any]:
    """Lists production deploys, newest first, joined with their commit info.

    Args:
        service: Exact service name to filter by (e.g. "checkout-service").
            Empty string returns deploys across every service.
        hours_back: Only include deploys within this many hours of the most
            recent deploy in the dataset (not wall-clock now).

    Returns:
        status: "success".
        deploys: list of {deploy_id, service, environment, deployed_at,
            deployed_by, commit: {sha, author, message, timestamp}},
            newest first.
    """
    deploys = _load("deploys.json")
    commits_by_sha = {c["sha"]: c for c in _load("commits.json")}

    if not deploys:
        return {"status": "success", "deploys": []}

    newest = max(_parse_iso(d["deployed_at"]) for d in deploys)
    cutoff = newest - timedelta(hours=hours_back)

    filtered = [
        d
        for d in deploys
        if (not service or d["service"] == service) and _parse_iso(d["deployed_at"]) >= cutoff
    ]
    filtered.sort(key=lambda d: d["deployed_at"], reverse=True)

    enriched = [
        {**d, "commit": commits_by_sha.get(d["commit_sha"])} for d in filtered
    ]
    return {"status": "success", "deploys": enriched}


def find_suspect_deploys(
    incident_start_iso: str, service: str, lookback_hours: int = 12
) -> dict[str, Any]:
    """Ranks the deploys most likely to have caused an incident.

    Cross-references `service`'s deploy history against when the incident's
    symptoms started: any deploy in the `lookback_hours` window before
    `incident_start_iso` is a suspect, ranked most-recent-before-incident
    first (the classic bisection heuristic -- the last change before
    symptoms appeared is the most likely culprit).

    Args:
        incident_start_iso: ISO 8601 timestamp of when symptoms began, e.g.
            "2026-08-20T14:32:00Z".
        service: The affected service to inspect deploys for.
        lookback_hours: How far before incident_start_iso to consider a
            deploy a candidate suspect.

    Returns:
        status: "success", or "error" with a message if incident_start_iso
            doesn't parse.
        suspects: list of {deploy_id, service, deployed_at, deployed_by,
            minutes_before_incident, commit: {sha, author, message,
            timestamp}}, ordered closest-before-incident first. Empty if no
            deploy to `service` falls in the lookback window.
    """
    try:
        incident_start = _parse_iso(incident_start_iso)
    except ValueError:
        return {
            "status": "error",
            "message": f"incident_start_iso is not a valid ISO 8601 timestamp: {incident_start_iso!r}",
        }

    deploys = _load("deploys.json")
    commits_by_sha = {c["sha"]: c for c in _load("commits.json")}
    window_start = incident_start - timedelta(hours=lookback_hours)

    candidates = []
    for d in deploys:
        if d["service"] != service:
            continue
        deployed_at = _parse_iso(d["deployed_at"])
        if window_start <= deployed_at <= incident_start:
            minutes_before = (incident_start - deployed_at).total_seconds() / 60
            candidates.append(
                {
                    **d,
                    "commit": commits_by_sha.get(d["commit_sha"]),
                    "minutes_before_incident": round(minutes_before, 1),
                }
            )

    candidates.sort(key=lambda c: c["minutes_before_incident"])
    return {"status": "success", "suspects": candidates}
