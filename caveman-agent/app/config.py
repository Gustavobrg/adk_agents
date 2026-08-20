"""Model configuration for caveman-agent."""
import os

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

# app/__init__.py imports agent.py (which needs these env vars) before
# fast_api_app.py's own load_dotenv() call runs, so load here too.
load_dotenv()

OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")


def openrouter_model() -> LiteLlm:
    """Build a LiteLlm model backed by OpenRouter.

    MODEL should hold the OpenRouter model slug, e.g. "anthropic/claude-sonnet-4.5".
    The "openrouter/" prefix is what tells LiteLLM to route through OpenRouter.
    """
    model_name = os.getenv("MODEL")
    if not model_name:
        raise ValueError("MODEL is not set (e.g. MODEL=anthropic/claude-sonnet-4.5)")
    if not model_name.startswith("openrouter/"):
        model_name = f"openrouter/{model_name}"

    return LiteLlm(
        model=model_name,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=OPENROUTER_API_BASE,
        num_retries=3,
        # Some OpenRouter-hosted reasoning models (e.g. nvidia/nemotron
        # nano) dump their "thinking" text straight into the normal content
        # string instead of routing it through LiteLLM's reasoning_content
        # field. `exclude: true` stops those reasoning tokens from being
        # sent back at all. https://openrouter.ai/docs/use-cases/reasoning-tokens
        reasoning={"exclude": True},
    )
