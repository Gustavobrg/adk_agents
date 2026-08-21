"""IncidentBrief -- what the intake_agent collects before investigation
starts. Deliberately small compared to a full incident-management
schema: enough for the specialists to act on, not a ticketing system.
"""
from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class BriefStatus(str, Enum):
    draft = "draft"
    confirmed = "confirmed"


class IncidentBrief(BaseModel):
    status: BriefStatus = BriefStatus.draft
    title: str | None = Field(default=None, description="Short human title, e.g. 'Checkout failures spiking'.")
    affected_service: str | None = Field(default=None, description="The primary service reporting symptoms, e.g. 'checkout-service'.")
    symptoms: list[str] = Field(default_factory=list, description="Observed symptoms, e.g. ['payment timeouts', 'elevated 5xx on /checkout'].")
    incident_start_iso: str | None = Field(default=None, description="ISO 8601 timestamp of when symptoms began.")
    description: str | None = Field(default=None, description="Free-text extra context -- what's known, what's already been tried, customer impact scale.")

    REQUIRED: ClassVar[tuple[str, ...]] = ("title", "affected_service", "symptoms", "incident_start_iso")

    def missing_required(self) -> list[str]:
        missing = []
        for field in self.REQUIRED:
            value = getattr(self, field)
            if value in (None, "", []):
                missing.append(field)
        return missing

    def ready_for_investigation(self) -> bool:
        return not self.missing_required()
