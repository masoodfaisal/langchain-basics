"""Small evaluation dataset for ``simple-agent.py``.

The examples cover the minimal agent's two important behaviours: delegate
music questions to ``ask_music_expert`` and decline unrelated questions
without calling a tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimpleAgentExample:
    """One evaluation example and its expected behaviour."""

    id: str
    question: str
    expected_tools: list[str] = field(default_factory=list)
    expected_tool_args: list[dict[str, Any]] = field(default_factory=list)
    expected_output_substrings: list[str] = field(default_factory=list)
    expected_output_forbidden: list[str] = field(default_factory=list)
    reference_answer: str = ""


EXAMPLES = [
    SimpleAgentExample(
        id="simple-music-syncopation",
        question="What is syncopation in music?",
        expected_tools=["ask_music_expert"],
        expected_tool_args=[
            {
                "name": "ask_music_expert",
                "args": {"question": "What is syncopation in music?"},
            }
        ],
        expected_output_substrings=["syncopation"],
        expected_output_forbidden=["ask_music_expert"],
        reference_answer=(
            "Explain that syncopation places emphasis on normally weak beats "
            "or off-beats, creating rhythmic surprise or forward motion. Keep "
            "the explanation concise and do not mention internal tool names."
        ),
    ),
    SimpleAgentExample(
        id="simple-music-major-minor",
        question="Explain the difference between major and minor chords.",
        expected_tools=["ask_music_expert"],
        expected_tool_args=[
            {
                "name": "ask_music_expert",
                "args": {
                    "question": (
                        "Explain the difference between major and minor chords."
                    )
                },
            }
        ],
        expected_output_substrings=["major", "minor"],
        expected_output_forbidden=["ask_music_expert"],
        reference_answer=(
            "Accurately compare major and minor chords, ideally noting that "
            "the third is the defining interval in a basic triad. Avoid "
            "presenting emotional descriptions as absolute rules."
        ),
    ),
    SimpleAgentExample(
        id="simple-music-recommendation-reasoning",
        question="Why might a jazz listener enjoy Kind of Blue by Miles Davis?",
        expected_tools=["ask_music_expert"],
        expected_tool_args=[
            {
                "name": "ask_music_expert",
                "args": {
                    "question": (
                        "Why might a jazz listener enjoy Kind of Blue by "
                        "Miles Davis?"
                    )
                },
            }
        ],
        expected_output_substrings=["Kind of Blue", "Miles Davis"],
        expected_output_forbidden=["ask_music_expert"],
        reference_answer=(
            "Give a concise, musically grounded rationale, such as the "
            "album's modal approach, memorable improvisation, accessible "
            "atmosphere, or notable ensemble playing. Do not claim access to "
            "the Chinook catalog or customer data."
        ),
    ),
    SimpleAgentExample(
        id="simple-out-of-scope-weather",
        question="What will the weather be in Sydney tomorrow?",
        expected_tools=[],
        expected_output_substrings=["music"],
        expected_output_forbidden=[
            "ask_music_expert",
            "degrees",
            "temperature will",
        ],
        reference_answer=(
            "Politely decline because weather is outside the Chinook music "
            "store scope, then briefly offer help with music instead. Do not "
            "call a tool or invent a weather forecast."
        ),
    ),
    # Rubric-based example for the LLM correctness judge.
    SimpleAgentExample(
        id="simple-rubric-walking-bass",
        question="How does a walking bass line work in jazz?",
        expected_tools=["ask_music_expert"],
        expected_tool_args=[
            {
                "name": "ask_music_expert",
                "args": {"question": "How does a walking bass line work in jazz?"},
            }
        ],
        expected_output_substrings=["walking bass"],
        expected_output_forbidden=["ask_music_expert"],
        reference_answer=(
            "- Tool use: Use the music expert for the explanation.\n"
            "- Accuracy: Explain that a walking bass line commonly uses steady "
            "quarter notes to outline chord changes, often with scale tones, "
            "arpeggios, and connecting notes.\n"
            "- Groundedness: Do not claim access to customer data, the Chinook "
            "catalog, current events, or external tools.\n"
            "- Internal details: Do not mention tool names or implementation "
            "details.\n"
            "- Style: Keep the answer concise and beginner-friendly."
        ),
    ),
]


def to_langsmith_examples() -> list[dict[str, Any]]:
    """Convert the examples into LangSmith's bulk-upload shape."""
    return [
        {
            "inputs": {"question": example.question},
            "outputs": {
                "reference_answer": example.reference_answer,
                "expected_tools": example.expected_tools,
                "expected_tool_args": example.expected_tool_args,
                "expected_output_substrings": example.expected_output_substrings,
                "expected_output_forbidden": example.expected_output_forbidden,
            },
            "metadata": {"example_id": example.id},
        }
        for example in EXAMPLES
    ]
