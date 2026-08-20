"""Local LLM-as-judge for `custom_response_quality` (see eval_config.yaml).

This project runs on OpenRouter, not Vertex AI / AI Studio (see
app/config.py) -- the judge call goes through litellm + OpenRouter too,
using the same MODEL/OPENROUTER_API_KEY as the agent itself.
"""

import json
import os

import litellm


def _judge_model() -> str:
    model_name = os.getenv("MODEL")
    if not model_name:
        raise ValueError("MODEL is not set (e.g. MODEL=anthropic/claude-sonnet-4.5)")
    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"
    return model_name


def evaluate(instance):
    reference = instance.get("reference")
    rubric = (
        "Grade the agent's final response on a 1-5 scale (1 poor, 5 excellent) for "
        "accuracy, relevance, and clarity."
    )
    if reference:
        rubric += (
            " The response should agree with the expected answer below; penalize "
            "factual disagreement with it."
        )
    prompt = (
        f"You are an expert QA evaluator for an enterprise AI assistant. {rubric}\n"
        f"User Prompt: {instance.get('prompt', '')}\n"
        f"Final Response: {instance.get('response', '')}\n"
    )
    if reference:
        prompt += f"Expected Answer (ground truth): {reference}\n"
    prompt += f"Full Agent Trace: {instance.get('agent_data', '')}\n"
    prompt += 'Respond with ONLY JSON: {"score": <1-5 integer>, "explanation": "<reason>"}'

    response = litellm.completion(
        model=_judge_model(),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic grading
        reasoning={"exclude": True},
    )
    text = response.choices[0].message.content or ""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        verdict = json.loads(text[start:end])
        score = int(verdict["score"])
        explanation = str(verdict.get("explanation", ""))
    except (ValueError, KeyError):
        return {"score": 0, "explanation": text}
    return {"score": max(1, min(5, score)), "explanation": explanation}
