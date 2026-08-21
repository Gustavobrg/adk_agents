"""Tools for the customer-comms specialist."""
from __future__ import annotations

from typing import Any

_SEVERITY_GUIDANCE: dict[str, dict[str, Any]] = {
    "SEV1": {
        "description": "Critical, widespread customer impact (e.g. checkout or login down for most users).",
        "tone": "Direct, urgent, no minimizing language. Acknowledge impact clearly in the first sentence.",
        "cadence": "Update every 30 minutes until resolved.",
        "channels": ["status page", "in-app banner", "email to affected customers"],
    },
    "SEV2": {
        "description": "Significant impact to a subset of users or a degraded (not fully down) core flow.",
        "tone": "Direct and clear, acknowledge impact, avoid over-promising a fix time.",
        "cadence": "Update hourly until resolved.",
        "channels": ["status page", "in-app banner"],
    },
    "SEV3": {
        "description": "Minor impact, workaround available, or affecting a small % of traffic.",
        "tone": "Calm, informative, low-alarm.",
        "cadence": "Update once at detection, once at resolution.",
        "channels": ["status page"],
    },
    "SEV4": {
        "description": "Cosmetic or negligible customer-facing impact.",
        "tone": "Low-key, factual.",
        "cadence": "Single update at resolution, if any.",
        "channels": ["status page (optional)"],
    },
}


def get_severity_guidance(severity: str) -> dict[str, Any]:
    """Returns tone, cadence, and channel guidance for a severity level.

    Args:
        severity: One of "SEV1", "SEV2", "SEV3", "SEV4".

    Returns:
        status: "success", or "error" if severity isn't recognized.
        guidance: {description, tone, cadence, channels} for that severity.
    """
    key = severity.strip().upper()
    if key not in _SEVERITY_GUIDANCE:
        return {
            "status": "error",
            "message": f"Unknown severity {severity!r}; expected one of {sorted(_SEVERITY_GUIDANCE)}.",
        }
    return {"status": "success", "guidance": _SEVERITY_GUIDANCE[key]}
