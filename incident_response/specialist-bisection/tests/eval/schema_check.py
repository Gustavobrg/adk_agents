"""Deterministic check: does the agent's final response parse as valid
JSON matching the BisectionFinding shape (app/schemas.py)?

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
    if data.get("confidence") not in ("low", "medium", "high"):
        errors.append(f"'confidence' must be low/medium/high, got {data.get('confidence')!r}")
    if not isinstance(data.get("reasoning"), str) or not data["reasoning"]:
        errors.append("missing/empty 'reasoning' (str)")

    suspects = data.get("suspect_commits")
    if not isinstance(suspects, list):
        errors.append("'suspect_commits' must be a list")
    else:
        for i, s in enumerate(suspects):
            if not isinstance(s, dict):
                errors.append(f"suspect_commits[{i}] is not an object")
                continue
            for field in ("commit_sha", "service", "author", "message", "deployed_at"):
                if not isinstance(s.get(field), str) or not s[field]:
                    errors.append(f"suspect_commits[{i}].{field} missing/empty")

    if errors:
        return {"score": 0.0, "explanation": "; ".join(errors)}
    return {"score": 1.0, "explanation": "response matches the BisectionFinding shape"}
