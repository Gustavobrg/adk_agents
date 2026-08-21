"""Structured output contract for the bisection specialist.

This is the JSON shape the agent's final A2A response is validated
against (`output_schema` below). Callers on the other side of the A2A
call (the incident-coordinator) parse the response text against their
OWN copy of an equivalent shape -- see incident-coordinator/app/
incident_report.py. Keeping two copies instead of a shared import is
deliberate: a real A2A specialist is a separately deployed service, and
the coordinator should only depend on the documented wire contract, not
on this project's Python types.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SuspectCommit(BaseModel):
    commit_sha: str
    service: str
    author: str
    message: str
    deployed_at: str
    minutes_before_incident: float | None = None


class BisectionFinding(BaseModel):
    summary: str = Field(description="One or two sentences on the most likely causal change, if any.")
    suspect_commits: list[SuspectCommit] = Field(
        default_factory=list,
        description="Candidate deploys/commits ranked most-likely-culprit first.",
    )
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(description="Why these commits were ranked this way -- timing, content of the change, etc.")
