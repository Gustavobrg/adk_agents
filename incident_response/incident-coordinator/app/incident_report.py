"""IncidentReport -- what investigation.py assembles from the 4
specialists' A2A responses into state["incident_report"].

Each nested model below is this coordinator's OWN copy of the shape a
specialist documents as its A2A output contract (see e.g.
specialist-bisection/app/schemas.py). Not a shared import: in a real A2A
deployment the specialists are separate services, so the coordinator only
depends on the documented wire contract. All fields are optional because
a specialist call can fail (timeout, unreachable, malformed response) --
the aggregator degrades a single failure to `None` plus an error note
rather than failing the whole investigation.
"""
from __future__ import annotations

from pydantic import BaseModel


class BisectionFinding(BaseModel):
    summary: str
    suspect_commits: list[dict] = []
    confidence: str
    reasoning: str


class ErrorCorrelationFinding(BaseModel):
    summary: str
    spiking_signals: list[dict] = []
    likely_origin_service: str | None = None
    correlation_notes: str


class IncidentHistoryFinding(BaseModel):
    summary: str
    similar_incidents: list[dict] = []
    suggested_playbook: str


class CustomerCommsDraft(BaseModel):
    is_draft: bool = True
    severity_estimate: str
    status_update_draft: str
    internal_notes: str
    recommended_channels: list[str] = []


class SpecialistError(BaseModel):
    specialist: str
    error: str


class IncidentReport(BaseModel):
    brief_title: str
    bisection: BisectionFinding | None = None
    error_correlation: ErrorCorrelationFinding | None = None
    incident_history: IncidentHistoryFinding | None = None
    customer_comms: CustomerCommsDraft | None = None
    specialist_errors: list[SpecialistError] = []
