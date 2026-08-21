"""Structured output contract for the error-correlation specialist.

See specialist-bisection/app/schemas.py for why the incident-coordinator
keeps its own copy of an equivalent shape instead of importing this one.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorSignal(BaseModel):
    service: str
    error_type: str
    message_sample: str
    total_count: int
    window_start: str
    window_end: str


class ErrorCorrelationFinding(BaseModel):
    summary: str = Field(description="One or two sentences on what's spiking and how services are affecting each other.")
    spiking_signals: list[ErrorSignal] = Field(
        default_factory=list,
        description="The strongest co-occurring error signals found, highest volume first.",
    )
    likely_origin_service: str | None = Field(
        default=None,
        description="Which service's errors look like the root of the chain, if the pattern makes that clear.",
    )
    correlation_notes: str = Field(description="How the signals relate to each other -- e.g. one service's errors triggering another's.")
