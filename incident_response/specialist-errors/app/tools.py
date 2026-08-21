"""Tools for the error-correlation specialist.

Reads a synthetic, self-contained `data/error_logs.json` standing in for
a real log/metrics backend (e.g. a Datadog/Sentry/Elasticsearch query).
Swap `_load` + the filtering below for real queries against that backend
to point this at production data -- keep the tool signatures/return
shapes stable, they're the actual contract the agent reasons over.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"


def _load() -> list[dict[str, Any]]:
    return json.loads((_DATA_DIR / "error_logs.json").read_text(encoding="utf-8"))


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def search_errors(service: str = "", since_iso: str = "", error_type: str = "") -> dict[str, Any]:
    """Searches the error log dataset with optional filters.

    Args:
        service: Exact service name to filter by. Empty string means any service.
        since_iso: ISO 8601 timestamp -- only entries at or after this time
            are returned. Empty string means no lower bound.
        error_type: Exact error type to filter by (e.g. "GatewayTimeoutError").
            Empty string means any type.

    Returns:
        status: "success", or "error" if since_iso doesn't parse.
        entries: matching log entries {timestamp, service, error_type,
            message, trace_id, count}, sorted oldest first.
    """
    since = None
    if since_iso:
        try:
            since = _parse_iso(since_iso)
        except ValueError:
            return {"status": "error", "message": f"since_iso is not a valid ISO 8601 timestamp: {since_iso!r}"}

    entries = [
        e
        for e in _load()
        if (not service or e["service"] == service)
        and (not error_type or e["error_type"] == error_type)
        and (since is None or _parse_iso(e["timestamp"]) >= since)
    ]
    entries.sort(key=lambda e: e["timestamp"])
    return {"status": "success", "entries": entries}


def correlate_error_spikes(window_minutes: int = 30) -> dict[str, Any]:
    """Groups the full error log dataset into (service, error_type) signals
    and flags which ones spike together in the same time window.

    Buckets every entry into `window_minutes`-wide windows (aligned to the
    dataset's earliest timestamp), sums `count` per (service, error_type,
    window), then reports windows where more than one distinct
    (service, error_type) signal spiked together -- that co-occurrence is
    the correlation signal: errors across different services spiking in
    the same window are likely one root cause rippling through a call
    chain, not independent problems.

    Args:
        window_minutes: Width of the time bucket used to group signals.

    Returns:
        status: "success".
        spike_windows: list of {window_start, window_end, signals: [{service,
            error_type, total_count, sample_message}]}, only windows with
            2+ distinct signals, ordered by total volume in the window
            (highest first).
    """
    entries = _load()
    if not entries:
        return {"status": "success", "spike_windows": []}

    earliest = min(_parse_iso(e["timestamp"]) for e in entries)
    window = timedelta(minutes=window_minutes)

    buckets: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for e in entries:
        idx = int((_parse_iso(e["timestamp"]) - earliest) / window)
        key = (e["service"], e["error_type"])
        signal = buckets[idx].setdefault(
            key,
            {
                "service": e["service"],
                "error_type": e["error_type"],
                "total_count": 0,
                "sample_message": e["message"],
            },
        )
        signal["total_count"] += e["count"]

    spike_windows = []
    for idx, signals in buckets.items():
        if len(signals) < 2:
            continue
        window_start = earliest + idx * window
        spike_windows.append(
            {
                "window_start": window_start.isoformat().replace("+00:00", "Z"),
                "window_end": (window_start + window).isoformat().replace("+00:00", "Z"),
                "signals": sorted(signals.values(), key=lambda s: s["total_count"], reverse=True),
            }
        )

    spike_windows.sort(key=lambda w: sum(s["total_count"] for s in w["signals"]), reverse=True)
    return {"status": "success", "spike_windows": spike_windows}
