"""Manual prompt-response logging to GCS.

The scaffolded prompt-response logging pipeline (LOGS_BUCKET_NAME +
OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload) relies on the
`opentelemetry-instrumentation-google-genai` auto-instrumentation, which only
patches the `google-genai` SDK. This agent calls OpenRouter via LiteLlm
(see config.py), so that instrumentation never sees our LLM calls and the
completion-hook GCS upload never fires. This module replaces it with a
direct write, matching the row shape Terraform's `completions` BigQuery
external table expects (deployment/terraform/single-project/telemetry.tf):
one row per message, each with a `parts` array, `role`, and `index`.
"""

import asyncio
import json
import logging
import os
import time

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse

_BUCKET_NAME = os.getenv("LOGS_BUCKET_NAME")
_logger = logging.getLogger(__name__)


def _content_text(content) -> str:
    if not content or not content.parts:
        return ""
    return "".join(part.text or "" for part in content.parts)


def _write_completion_sync(blob_name: str, rows: list[dict]) -> None:
    from google.cloud import storage

    body = "\n".join(json.dumps(row) for row in rows) + "\n"
    client = storage.Client()
    client.bucket(_BUCKET_NAME).blob(blob_name).upload_from_string(
        body, content_type="application/jsonl"
    )


async def log_completion(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """after_model_callback: writes the prompt+response as completions rows to GCS."""
    if not _BUCKET_NAME or llm_response.partial:
        return

    rows = [
        {
            "role": "user",
            "parts": [{"type": "text", "content": _content_text(callback_context.user_content)}],
            "index": 0,
        },
        {
            "role": "model",
            "parts": [{"type": "text", "content": _content_text(llm_response.content)}],
            "index": 1,
        },
    ]
    blob_name = f"completions/{int(time.time())}-{callback_context.invocation_id}.jsonl"
    try:
        await asyncio.to_thread(_write_completion_sync, blob_name, rows)
    except Exception:
        _logger.exception("Failed to upload completion log to GCS")
