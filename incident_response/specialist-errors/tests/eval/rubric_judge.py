"""LLM judge that grades a response against this case's own
`rubric_groups.criteria` (see tests/eval/datasets/basic-dataset.json).

Not the built-in `final_response_quality` + `metric_spec_parameters.
rubric_group_key` path the eval skill documents: verified locally via a
real `agents-cli eval run` that this agents-cli version's eval_utils.py
never implements `metric_spec_parameters` (see eval_config.yaml's
history) -- an entry shaped that way silently becomes an invalid
`LLMMetric` and 400s with "Unsupported metric type or invalid metric
name". This is the workaround the eval skill's own metrics-guide.md
documents for grading `rubric_groups` with a custom judge instead.
"""
from google import genai
from pydantic import BaseModel


class _CriterionVerdict(BaseModel):
    criterion_index: int
    passed: bool
    reason: str


class _JudgeOutput(BaseModel):
    verdicts: list[_CriterionVerdict]


def evaluate(instance):
    rubrics = [
        r["content"]["property"]["description"]
        for group in (instance.get("rubric_groups") or {}).values()
        for r in group.get("rubrics", [])
    ]
    if not rubrics:
        return {"score": 0.0, "explanation": "no rubric_groups on this case"}

    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics))
    prompt = (
        "You are grading an AI agent's response against a fixed checklist. "
        "For EACH numbered criterion below, decide pass or fail strictly "
        "from the response text -- do not credit a criterion the response "
        "doesn't actually satisfy, and do not penalize it for anything a "
        "criterion doesn't ask about.\n\n"
        f"Criteria:\n{numbered}\n\n"
        f"User prompt sent to the agent:\n{instance.get('prompt', '')}\n\n"
        f"Agent's final response:\n{instance.get('response', '')}\n\n"
        "Return one verdict per criterion, using its index above."
    )

    client = genai.Client()
    api_response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_schema": _JudgeOutput,
        },
    )
    parsed = api_response.parsed
    if not parsed or not parsed.verdicts:
        return {"score": 0.0, "explanation": api_response.text or "judge returned no usable verdict"}

    total = len(parsed.verdicts)
    passed = sum(1 for v in parsed.verdicts if v.passed)
    failed_reasons = [f"[{v.criterion_index}] {v.reason}" for v in parsed.verdicts if not v.passed]
    explanation = f"{passed}/{total} criteria passed." + (
        " Failed: " + "; ".join(failed_reasons) if failed_reasons else ""
    )
    return {"score": round(passed / total, 3) if total else 0.0, "explanation": explanation}
