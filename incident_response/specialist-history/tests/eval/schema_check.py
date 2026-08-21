"""Deterministic check: does the agent's final response parse as valid
JSON matching the IncidentHistoryFinding shape (app/schemas.py)?

Runs inside agents-cli's own process/venv (separate from this project's),
so it intentionally duck-types the shape instead of importing
`app.schemas` -- `custom_function`/`custom_function_file` execute via a
plain `exec()` in the CLI tool's own environment, with no guarantee the
project's package is importable there. Complements `final_response_quality`
(LLM-judged, in eval_config.yaml) with a cheap, exact, non-LLM check that
catches schema regressions immediately.
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
    if not isinstance(data.get("summary"), str) or not data["summary"]:
        errors.append("missing/empty 'summary' (str)")
    if not isinstance(data.get("suggested_playbook"), str) or not data["suggested_playbook"]:
        errors.append("missing/empty 'suggested_playbook' (str)")

    incidents = data.get("similar_incidents")
    if not isinstance(incidents, list):
        errors.append("'similar_incidents' must be a list")
    else:
        for i, inc in enumerate(incidents):
            if not isinstance(inc, dict):
                errors.append(f"similar_incidents[{i}] is not an object")
                continue
            for field in ("incident_id", "date", "title", "root_cause", "resolution", "similarity_reason"):
                if not isinstance(inc.get(field), str) or not inc[field]:
                    errors.append(f"similar_incidents[{i}].{field} missing/empty")

    if errors:
        return {"score": 0.0, "explanation": "; ".join(errors)}
    return {"score": 1.0, "explanation": "response matches the IncidentHistoryFinding shape"}
