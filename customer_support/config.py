"""Model configuration for customer_support."""
import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

# Load the .env next to this file regardless of the process's cwd.
load_dotenv(Path(__file__).parent / ".env")

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
    )
