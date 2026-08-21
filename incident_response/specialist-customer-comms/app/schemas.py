"""Structured output contract for the customer-comms specialist.

See specialist-bisection/app/schemas.py for why the incident-coordinator
keeps its own copy of an equivalent shape instead of importing this one.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerCommsDraft(BaseModel):
    is_draft: bool = Field(
        default=True,
        description="Always true. This specialist only ever produces a draft for human review -- it never sends anything.",
    )
    severity_estimate: Literal["SEV1", "SEV2", "SEV3", "SEV4"]
    status_update_draft: str = Field(description="Customer-facing status update text, ready for a human to review and post.")
    internal_notes: str = Field(description="Notes for the human reviewer -- tone choices made, anything uncertain, what to fill in before sending.")
    recommended_channels: list[str] = Field(default_factory=list)
