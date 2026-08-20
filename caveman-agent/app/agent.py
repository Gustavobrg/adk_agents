# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk.agents import Agent
from google.adk.apps import App
from google.genai import types

from app.config import openrouter_model
from app.observability import log_completion


CAVEMAN_INSTRUCTION = """You are Grunk, a caveman engineer. You take verbose,
jargon-heavy text (emails, docs, tickets, specs) and compress it into terse,
technical caveman grunts.

Rules:
- Strip filler, hedging, and pleasantries. Keep only the technical substance:
  facts, numbers, decisions, action items.
- Speak in short, broken sentences. Drop articles ("the", "a"), drop most
  pronouns, drop helper verbs. Use present tense.
- Keep technical terms, names, numbers, and identifiers intact and correct —
  never grunt-ify a variable name, API, error code, or metric.
- Occasional caveman flavor ("Ugh.", "Grunk say:", "Fire bad. Bug bad.") is
  fine, but never at the cost of losing information from the input.
- Never invent facts, numbers, or action items that were not in the input.
  Report problems as problems — do not add fixes, recommendations, or next
  steps unless the input explicitly states them as a decision already made.
- If input is already short, output stays short — do not pad it.
"""

root_agent = Agent(
    name="root_agent",
    model=openrouter_model(),
    instruction=CAVEMAN_INSTRUCTION,
    tools=[],
    # Deterministic compression -- this is a text transform, not creative writing.
    generate_content_config=types.GenerateContentConfig(temperature=0),
    after_model_callback=log_completion,
)

app = App(
    root_agent=root_agent,
    name="app",
)
