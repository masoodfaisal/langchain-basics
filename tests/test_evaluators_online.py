from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).parents[1] / "evals" / "evaluators-online.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("evaluators_online", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_buyer_meeting_does_not_match_account_request_terms() -> None:
    module = _load_module()

    text = "Prepare this week's Kao–Ulta account-intelligence brief for my buyer meeting."

    assert not module._contains_any(text, module.ACCOUNT_REQUEST_TERMS)


def test_recent_orders_matches_account_request_terms() -> None:
    module = _load_module()

    assert module._contains_any("show me my recent orders", module.ACCOUNT_REQUEST_TERMS)


def test_docs_run_is_not_scored_for_internal_subject_matter() -> None:
    module = _load_module()

    run = {
        "name": "docs_agent",
        "inputs": {"question": "Explain LangChain and LangGraph."},
        "outputs": {"output": "LangChain and LangGraph are frameworks."},
    }

    assert module.perform_eval(run) == {}
