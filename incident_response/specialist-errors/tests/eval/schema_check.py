"""Deterministic check: does the agent's final response parse as valid
JSON matching the ErrorCorrelationFinding shape (app/schemas.py)?

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
    if not isinstance(data.get("correlation_notes"), str) or not data["correlation_notes"]:
        errors.append("missing/empty 'correlation_notes' (str)")
    if "likely_origin_service" in data and data["likely_origin_service"] is not None:
        if not isinstance(data["likely_origin_service"], str):
            errors.append("'likely_origin_service' must be a string or null")

    signals = data.get("spiking_signals")
    if not isinstance(signals, list):
        errors.append("'spiking_signals' must be a list")
    else:
        for i, s in enumerate(signals):
            if not isinstance(s, dict):
                errors.append(f"spiking_signals[{i}] is not an object")
                continue
            for field in ("service", "error_type", "message_sample"):
                if not isinstance(s.get(field), str) or not s[field]:
                    errors.append(f"spiking_signals[{i}].{field} missing/empty")
            if not isinstance(s.get("total_count"), (int, float)):
                errors.append(f"spiking_signals[{i}].total_count must be numeric")

    if errors:
        return {"score": 0.0, "explanation": "; ".join(errors)}
    return {"score": 1.0, "explanation": "response matches the ErrorCorrelationFinding shape"}
