"""Structured output contract for the incident-history specialist.

See specialist-bisection/app/schemas.py for why the incident-coordinator
keeps its own copy of an equivalent shape instead of importing this one.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SimilarIncident(BaseModel):
    incident_id: str
    date: str
    title: str
    root_cause: str
    resolution: str
    similarity_reason: str = Field(description="Why this past incident is relevant to the current one.")


class IncidentHistoryFinding(BaseModel):
    summary: str = Field(description="One or two sentences on whether this incident has happened before and what that suggests.")
    similar_incidents: list[SimilarIncident] = Field(default_factory=list)
    suggested_playbook: str = Field(
        description="Concrete next steps drawn from how similar past incidents were resolved. State plainly if nothing similar was found."
    )
