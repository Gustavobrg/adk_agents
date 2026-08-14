"""Configuration for travel agent."""
import os
from google.adk.models.lite_llm import LiteLlm


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
        num_retries=30,
        # Some OpenRouter-hosted reasoning models (e.g. nvidia/nemotron
        # nano) don't route their "thinking" text through LiteLLM's
        # reasoning_content field -- they just dump it straight into the
        # normal content string, which ADK then hands to the caller as if
        # it were the real answer. `exclude: true` is OpenRouter's unified
        # reasoning control: the model still reasons internally, but the
        # reasoning tokens are never sent back, so there's nothing to leak.
        # https://openrouter.ai/docs/use-cases/reasoning-tokens
        reasoning={"exclude": True},
    )


def gemini_search_model() -> str:
    """Model id for agents that use the built-in `google_search` tool.

    google_search is a Gemini built-in tool -- it only works when the agent
    talks to a real Gemini backend (GOOGLE_API_KEY, or Vertex AI). It
    doesn't work behind the LiteLlm/OpenRouter setup that `openrouter_model()`
    uses for the rest of the app, so this model id is passed straight
    through (string) instead of wrapped in LiteLlm. Requires GOOGLE_API_KEY
    in .env.
    """
    return os.getenv("GOOGLE_SEARCH_MODEL", "gemini-2.5-flash")
