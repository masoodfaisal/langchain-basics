"""LLM-as-a-judge evaluators for offline LangSmith experiments."""

import os

from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT, CORRECTNESS_PROMPT


JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai:o3-mini")


# Reference-free: works with the current dataset immediately.
conciseness_judge = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    model=JUDGE_MODEL,
    feedback_key="llm_conciseness",
)


# Reference-based: compares the response with dataset reference outputs.
correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model=JUDGE_MODEL,
    feedback_key="llm_correctness",
)


LLM_JUDGES = [
    conciseness_judge,
    correctness_judge,
]