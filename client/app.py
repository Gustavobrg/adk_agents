"""Cliente Chainlit para o travel_agent implantado no Vertex AI Agent Engine."""

import os
import uuid

import chainlit as cl
import vertexai
from dotenv import load_dotenv

load_dotenv()

_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_RESOURCE_NAME = os.environ["AGENT_ENGINE_RESOURCE_NAME"]

_AUTH_HELP = (
    "Nao foi possivel conectar ao agente implantado. Verifique "
    "GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION / AGENT_ENGINE_RESOURCE_NAME "
    "em client/.env e confirme que `gcloud auth application-default login` "
    "foi executado.\n\nErro: {exc}"
)


@cl.on_chat_start
async def on_chat_start() -> None:
    client = vertexai.Client(project=_PROJECT, location=_LOCATION)
    engine = client.agent_engines.get(name=_RESOURCE_NAME)
    user_id = str(uuid.uuid4())

    try:
        session = await engine.async_create_session(user_id=user_id)
    except Exception as exc:
        await cl.Message(content=_AUTH_HELP.format(exc=exc), author="system").send()
        raise

    cl.user_session.set("engine", engine)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session["id"])


def _event_text(event: dict) -> str:
    parts = (event.get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts if part.get("text"))


@cl.on_message
async def on_message(message: cl.Message) -> None:
    engine = cl.user_session.get("engine")
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("session_id")

    messages_by_author: dict[str, cl.Message] = {}

    try:
        async for event in engine.async_stream_query(
            message=message.content,
            user_id=user_id,
            session_id=session_id,
        ):
            if event.get("partial"):
                continue
            text = _event_text(event)
            if not text:
                continue

            author = event.get("author") or "agent"
            msg = messages_by_author.get(author)
            if msg is None:
                msg = cl.Message(content="", author=author)
                await msg.send()
                messages_by_author[author] = msg
            await msg.stream_token(text)
    except Exception as exc:
        await cl.Message(content=f"Erro ao consultar o agente: {exc}", author="system").send()
        return

    for msg in messages_by_author.values():
        await msg.update()

    if not messages_by_author:
        await cl.Message(content="(sem resposta de texto neste turno)", author="system").send()
