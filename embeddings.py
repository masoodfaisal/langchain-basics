"""Local Hugging Face embeddings for the LangGraph store.

Wired into ``langgraph.json`` via::

    "store": { "index": { "embed": "./embeddings.py:embed", ... } }

Both ``langgraph dev`` (local) and LangSmith Deployment (prod) call this
function to embed memory documents and incoming search queries against the
runtime-managed store. CI does not exercise this path -- tests inject an
unindexed :class:`InMemoryStore` directly.

Pattern follows https://docs.langchain.com/langsmith/semantic-search
("Custom embeddings"). The function must be async, accept a list of
strings, and return a list of float arrays. We point it at the same
self-hosted OpenAI-compatible gateway as ``agent.py`` so embeddings and
chat completions share one endpoint and one set of credentials.


WARNING: sentence-transformers with PyTorch MPS may be faster

"""

from __future__ import annotations

import asyncio
import os

from fastembed import TextEmbedding

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-base-en-v1.5",
)
CACHE_DIR = os.getenv(
    "EMBEDDING_CACHE_DIR",
    ".cache/fastembed",
)

# Downloads the model from Hugging Face during application startup if it
# is not already cached.
_model = TextEmbedding(
    model_name=MODEL_NAME,
    cache_dir=CACHE_DIR,
)


def _encode(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _model.embed(texts)]


async def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await asyncio.to_thread(_encode, texts)



# Configuration (env vars):

# * ``EMBEDDING_MODEL_NAME`` -- model to call for embeddings. Required when
#   the ``store.index`` block is enabled in ``langgraph.json``.
# * ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` -- gateway settings, shared
#   with the chat model. ``EMBEDDING_BASE_URL`` / ``EMBEDDING_API_KEY``
#   override these when the embedding endpoint differs from the chat one.
# * ``OPENAI_VERIFY_SSL`` -- ``false`` disables TLS verification for
#   self-signed dev gateways. Mirrors the chat-model setting.

# If you want to turn semantic search off, drop the ``store`` block from
# ``langgraph.json`` -- the runtime then provisions an unindexed store and
# this module is never imported.
# """

# from __future__ import annotations

# import logging
# import os

# import httpx
# from openai import AsyncOpenAI

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Configuration
# # ---------------------------------------------------------------------------
# _BASE_URL = os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL"))
# _API_KEY = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY"))
# _MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "")

# _verify_env = os.getenv("OPENAI_VERIFY_SSL", "true").strip().lower()
# _VERIFY_SSL = _verify_env not in {"false", "0", "no"}

# if not _MODEL_NAME:
#     # Fail at import time so a misconfigured deployment crashes during
#     # ``langgraph dev`` startup instead of mid-request.
#     raise RuntimeError(
#         "embeddings.py: EMBEDDING_MODEL_NAME must be set when the store "
#         "block in langgraph.json points at this file. Set the env var, or "
#         "remove the store block to disable semantic search."
#     )

# if not _VERIFY_SSL:
#     logger.warning(
#         "OPENAI_VERIFY_SSL is disabled - TLS certificate verification is "
#         "OFF for embedding requests to %s. Only do this for trusted dev "
#         "gateways.",
#         _BASE_URL,
#     )

# # Module-level client: the LangGraph runtime imports this module once per
# # server process, then calls ``embed`` repeatedly. Reusing one
# # ``AsyncOpenAI`` client lets httpx pool connections.
# _client = AsyncOpenAI(
#     base_url=_BASE_URL,
#     api_key=_API_KEY,
#     http_client=httpx.AsyncClient(verify=_VERIFY_SSL),
# )


# # ---------------------------------------------------------------------------
# # Embedding function (the entry point langgraph.json points at)
# # ---------------------------------------------------------------------------
# async def embed(texts: list[str]) -> list[list[float]]:
#     """Return one embedding vector per input string, in input order.

#     Called by the LangGraph store on both writes (``store.aput`` /
#     ``Memo.write``) and reads (``store.asearch`` / ``Memo.search``).
#     """
#     if not texts:
#         return []
#     response = await _client.embeddings.create(
#         model=_MODEL_NAME,
#         input=texts,
#     )
#     return [item.embedding for item in response.data]
