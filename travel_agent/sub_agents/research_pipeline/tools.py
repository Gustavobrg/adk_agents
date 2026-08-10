"""Tools for the research phase (phase 2) -- poi_scout's geocoding step.

search_web itself is not a custom tool: every scout gets it as the
built-in `google_search` tool (see sub_agents/research_pipeline/agent.py).
This module only holds `validate_poi`, the one tool that talks to an
external API outside of search.
"""
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from google.adk.tools.tool_context import ToolContext

from .research_pack import POI, Price

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "poi"


def _geocode(address: str) -> Optional[tuple[float, float]]:
    """Geocodes via Google Geocoding API. Returns None on any failure --
    missing key, network error, or no match -- so the caller always has a
    single "couldn't validate" path instead of distinct error handling per
    failure mode."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return None
    try:
        response = httpx.get(
            GEOCODE_URL,
            params={"address": address, "key": api_key},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError:
        return None
    if data.get("status") != "OK" or not data.get("results"):
        return None
    location = data["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]


def validate_poi(
    tool_context: ToolContext,
    name: str,
    city: str,
    brief_item: Optional[str] = None,
    duration_min: Optional[int] = None,
    ticket_low: Optional[float] = None,
    ticket_high: Optional[float] = None,
    ticket_currency: str = "BRL",
    closed_weekdays: Optional[List[int]] = None,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Geocodes a POI candidate and saves it to state["research_raw"]["pois"].

    Call this once per attraction you've found real information for via
    search. `validated`, `lat`, and `lng` are always computed here from the
    geocoding result -- never pass them in, and never claim a POI is
    validated in your own text; only this tool's return value decides that.
    Calling this again for the same name+city replaces the earlier entry
    (e.g. if you found better ticket/hours info on a second search).

    Args:
        name: The POI's real name, as found via search.
        city: The city it's in, as found via search.
        brief_item: The brief's attraction name this POI satisfies, if any.
        duration_min: Typical visit duration in minutes, if found.
        ticket_low: Lowest ticket price found, if any.
        ticket_high: Highest ticket price found, if any.
        ticket_currency: ISO 4217 currency for the ticket price.
        closed_weekdays: Weekdays it's closed, 0=Monday .. 6=Sunday.
        source_url: URL of the page that grounds this information.

    Returns:
        status, poi_id, validated (bool), lat/lng (if validated), and a
        message to explain the outcome.
    """
    poi_id = f"{_slugify(name)}-{_slugify(city)}"
    coords = _geocode(f"{name}, {city}")

    ticket = None
    if ticket_low is not None and ticket_high is not None:
        ticket = Price(low=ticket_low, high=ticket_high, currency=ticket_currency)

    poi = POI(
        poi_id=poi_id,
        name=name,
        city=city,
        validated=coords is not None,
        lat=coords[0] if coords else None,
        lng=coords[1] if coords else None,
        duration_min=duration_min,
        ticket=ticket,
        closed_weekdays=closed_weekdays or [],
        brief_item=brief_item,
        source_url=source_url,
    )

    raw = tool_context.state.get("research_raw", {})
    pois = [p for p in raw.get("pois", []) if p.get("poi_id") != poi.poi_id]
    pois.append(poi.model_dump(mode="json"))
    tool_context.state["research_raw"] = {**raw, "pois": pois}

    return {
        "status": "success",
        "poi_id": poi.poi_id,
        "validated": poi.validated,
        "lat": poi.lat,
        "lng": poi.lng,
        "message": (
            "Geocoded and validated."
            if poi.validated
            else "Could not geocode this name/city -- report it as a gap, "
            "don't claim it's validated."
        ),
    }
