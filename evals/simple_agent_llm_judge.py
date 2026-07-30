"""LLM-as-a-judge evaluator for ``simple-agent.py``.

Judge model configuration
-------------------------
The judge reuses the project's existing OpenAI-compatible environment keys:

``EVAL_JUDGE_MODEL``
    Model identifier for the judge, for example ``openai:gpt-4o-mini``. A
    model passed to ``build_llm_judges()`` takes priority, followed by this
    variable and then ``DEFAULT_JUDGE_MODEL``.

``OPENAI_BASE_URL``
    Optional base URL for a custom OpenAI-compatible model gateway. If it is
    unset, the underlying OpenAI client uses its normal default endpoint.

``OPENAI_API_KEY``
    API key used by the underlying OpenAI-compatible client. Keep the real
    value in the environment or the project ``.env`` file; never hardcode or
    commit it.

Example configuration (placeholders only)::

    EVAL_JUDGE_MODEL=openai:<judge-model>
    OPENAI_BASE_URL=https://<model-gateway>/v1
    OPENAI_API_KEY=<set-securely>

``LANGSMITH_API_KEY`` configures LangSmith experiment access; it is separate
from the key used to call the judge model. There are intentionally no separate
``EVAL_JUDGE_API_KEY`` or ``EVAL_JUDGE_BASE_URL`` settings: the judge uses the
existing ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` settings.

``LANGSMITH_PROJECT`` is prepended to the reusable evaluator and prompt names,
which keeps resources from different projects separate in the LangSmith UI.

``upsert_llm_evaluator()`` registers this judge as a reusable LangSmith LLM
evaluator. Registration never uploads the local API-key value. Because the
registered evaluator runs in LangSmith, its execution environment must also
have the model credential configured and be able to reach ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.prompts.structured import StructuredPrompt
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.utils import LangSmithConflictError
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from pydantic import BaseModel, Field


DEFAULT_JUDGE_MODEL = "openai:gpt-4o-mini"
CORRECTNESS_PROMPT_NAME = "simple-agent-offline-correctness-judge"
LLM_EVALUATOR_NAME = "simple-agent-offline-llm-correctness"


class CorrectnessFeedback(BaseModel):
    """Structured score returned by the registered judge."""

    simple_agent_llm_correctness: bool = Field(
        description=(
            "True only when the answer satisfies the reference rubric "
            "accurately, helpfully, and safely."
        )
    )
    reasoning: str = Field(description="A concise explanation of the verdict.")


def project_scoped_name(name: str, project_name: str | None = None) -> str:
    """Prefix a LangSmith resource name with the project exactly once."""
    project_name = (project_name or os.getenv("LANGSMITH_PROJECT", "")).strip()
    name = name.strip()
    if not project_name:
        raise ValueError("LANGSMITH_PROJECT must not be empty.")
    if not name:
        raise ValueError("The LangSmith resource name must not be empty.")
    prefix = f"{project_name}-"
    return name if name.startswith(prefix) else f"{prefix}{name}"


def build_llm_judges(model: str | None = None) -> list:
    """Create a local judge, useful when no reusable evaluator is desired."""
    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model=model or os.getenv("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        feedback_key="simple_agent_llm_correctness",
    )
    return [correctness_judge]


def _judge_chain(model_name: str) -> Any:
    """Build the structured prompt stored for the reusable evaluator."""
    prompt = StructuredPrompt.from_messages_and_schema(
        [
            (
                "system",
                "You are grading a beginner-facing music assistant. Decide "
                "whether its response satisfies the reference rubric. "
                "Prioritize musical accuracy, fulfillment of the request, "
                "scope boundaries, and absence of invented information.",
            ),
            (
                "human",
                "USER QUESTION:\n{input}\n\n"
                "ASSISTANT RESPONSE:\n{output}\n\n"
                "REFERENCE RUBRIC:\n{reference}",
            ),
        ],
        schema=CorrectnessFeedback.model_json_schema(),
    )
    return prompt | ChatOpenAI(
        model=model_name.removeprefix("openai:"),
        temperature=0,
    )


async def _find_evaluator(client: Client, name: str) -> Any | None:
    matches = [
        evaluator
        async for evaluator in client.evaluators.list(name_contains=name, limit=100)
        if evaluator.name == name
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple evaluators named {name!r}.")
    return matches[0] if matches else None


async def upsert_llm_evaluator(
    client: Client,
    model_name: str,
    replace: bool,
    project_name: str | None = None,
) -> tuple[Any, str]:
    """Register or update the reusable LLM judge in LangSmith."""
    prompt_name = project_scoped_name(CORRECTNESS_PROMPT_NAME, project_name)
    evaluator_name = project_scoped_name(LLM_EVALUATOR_NAME, project_name)
    try:
        client.push_prompt(
            prompt_name,
            object=_judge_chain(model_name),
            description="Correctness judge for the beginner simple-agent dataset.",
        )
    except LangSmithConflictError as exc:
        if "Nothing to commit" not in str(exc):
            raise

    prompt = client.get_prompt(prompt_name)
    if prompt is None or not prompt.last_commit_hash:
        raise RuntimeError("LangSmith did not return the judge prompt commit.")

    configuration = {
        "prompt_repo_handle": prompt.repo_handle,
        "commit_hash_or_tag": prompt.last_commit_hash,
        "variable_mapping": {
            "input": "inputs.question",
            "output": "outputs.answer",
            "reference": "reference.reference_answer",
        },
    }
    existing = await _find_evaluator(client, evaluator_name)
    if existing is None:
        result = await client.evaluators.create(
            name=evaluator_name,
            type="llm",
            llm_evaluator=configuration,
        )
        action = "Created"
    else:
        if existing.type != "llm":
            raise RuntimeError(
                f"Evaluator {evaluator_name!r} exists but is not LLM-based."
            )
        if not replace:
            return existing, "Reused"
        result = await client.evaluators.update(
            str(existing.id),
            name=evaluator_name,
            llm_evaluator=configuration,
        )
        action = "Updated"

    evaluator = result.evaluator
    if evaluator is None or evaluator.id is None:
        raise RuntimeError("LangSmith did not return the LLM evaluator id.")
    return evaluator, action


LLM_JUDGES = build_llm_judges()
