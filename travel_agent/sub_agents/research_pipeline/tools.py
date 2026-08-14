"""Geocoding helper for the research phase (phase 2).

Turns a POI candidate's name+city into lat/lng, in plain code -- no LLM
decides when to call this. planning_pipeline's k-means clustering (step 2,
geo_clustering) needs those coordinates to cluster POIs geographically; see
sub_agents/research_pipeline/agent.py for where this gets called.
"""
import os
import re
from typing import Optional

import httpx

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "poi"


def geocode(address: str) -> Optional[tuple[float, float]]:
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
