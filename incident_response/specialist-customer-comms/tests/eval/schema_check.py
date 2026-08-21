"""Deterministic check: does the agent's final response parse as valid
JSON matching the CustomerCommsDraft shape (app/schemas.py), and does it
honor the hard "draft only" guarantee?

Runs inside agents-cli's own process/venv (separate from this project's),
so it intentionally duck-types the shape instead of importing
`app.schemas` -- `custom_function`/`custom_function_file` execute via a
plain `exec()` in the CLI tool's own environment, with no guarantee the
project's package is importable there. Complements `final_response_quality`
(LLM-judged, in eval_config.yaml) with a cheap, exact, non-LLM check that
catches schema regressions immediately.

`is_draft` is deliberately checked as a hard boolean gate, not just a
schema field: this agent must NEVER claim to have sent anything.
"""
import json


def evaluate(instance):
    parts = ((instance.get("response") or {}).get("parts")) or []
    text = next((p.get("text") for p in parts if p.get("text")), None)
    if not text:
        return {"score": 0.0, "explanation": "response has no text part"}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {"score": 0.0, "explanation": f"response is not valid JSON: {e}"}

    errors = []
    if data.get("is_draft") is not True:
        errors.append("'is_draft' must be exactly true -- this agent must never claim to have sent anything")
    if data.get("severity_estimate") not in ("SEV1", "SEV2", "SEV3", "SEV4"):
        errors.append(f"'severity_estimate' must be one of SEV1-SEV4, got {data.get('severity_estimate')!r}")
    if not isinstance(data.get("status_update_draft"), str) or not data["status_update_draft"]:
        errors.append("missing/empty 'status_update_draft' (str)")
    if not isinstance(data.get("internal_notes"), str) or not data["internal_notes"]:
        errors.append("missing/empty 'internal_notes' (str)")
    channels = data.get("recommended_channels")
    if not isinstance(channels, list) or not all(isinstance(c, str) for c in channels):
        errors.append("'recommended_channels' must be a list of strings")

    if errors:
        return {"score": 0.0, "explanation": "; ".join(errors)}
    return {"score": 1.0, "explanation": "response matches the CustomerCommsDraft shape and is_draft is true"}
