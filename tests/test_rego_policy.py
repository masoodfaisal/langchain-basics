"""Embedded Rego evaluation, including false results and async worker isolation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

import rego_policy
from rego_policy import RegoPolicy


@pytest.fixture
def policy_path(tmp_path):
    source = tmp_path / "policy with spaces.rego"
    source.write_text(
        'package example\nimport rego.v1\ndefault allow := false\n'
        'allow if { input.action == "read" }\n'
        'decision := {"allow": allow}\n'
        'value := input.value\n',
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("input_data, expected", [
    ({"action": "read"}, True),
    ({"action": "delete"}, False),
    ({}, False),
    (None, False),
])
async def test_real_regopy_applies_policy_and_preserves_false(policy_path, input_data, expected):
    policy = RegoPolicy(policy_path)
    assert await policy.evaluate("data.example.allow", input_data) is expected
    assert await policy.evaluate("data.example.decision", input_data) == {"allow": expected}


@pytest.mark.parametrize("value", [
    None, False, 0, "", [], {"allowed": False},
    2**63, 2**64, -(2**100), "before\x00after",
])
async def test_real_regopy_preserves_json_values(policy_path, value):
    assert await RegoPolicy(policy_path).evaluate("data.example.value", {"value": value}) == value


async def test_nan_input_is_rejected_before_native_evaluation(policy_path, monkeypatch):
    def unexpected_interpreter():
        pytest.fail("Non-JSON input must not reach the native interpreter")

    monkeypatch.setattr(rego_policy, "Interpreter", unexpected_interpreter)
    with pytest.raises(ValueError):
        await RegoPolicy(policy_path).evaluate("data.example.value", {"value": float("nan")})


async def test_real_regopy_undefined_query_returns_none(policy_path):
    assert await RegoPolicy(policy_path).evaluate("data.example.missing") is None


async def test_loaded_source_is_immutable_until_new_evaluator(policy_path):
    policy = RegoPolicy(policy_path)
    policy_path.write_text("package example\nallow := false\n", encoding="utf-8")
    assert await policy.evaluate("data.example.allow", {"action": "read"}) is True
    assert await RegoPolicy(policy_path).evaluate("data.example.allow", {"action": "read"}) is False
    policy_path.unlink()
    assert await policy.evaluate("data.example.allow", {"action": "read"}) is True


async def test_concurrent_evaluations_keep_inputs_isolated(policy_path):
    policy = RegoPolicy(policy_path)
    values = [{"request": index, "allow": index % 2 == 0} for index in range(12)]
    results = await asyncio.gather(*[
        policy.evaluate("data.example.value", {"value": value}) for value in values
    ])
    assert results == values


async def test_invalid_policy_is_rejected_without_exposing_source(policy_path):
    policy_path.write_text("package example\nprivate_policy invalid syntax\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as error:
        await RegoPolicy(policy_path).evaluate("data.example.allow")
    assert "private_policy" not in str(error.value)


async def test_builtin_failure_is_an_error_and_does_not_expose_input(policy_path):
    with pytest.raises(RuntimeError) as error:
        await RegoPolicy(policy_path).evaluate(
            "to_number(input.value)", {"value": "private_customer_data"}
        )
    assert "private_customer_data" not in str(error.value)


@pytest.mark.parametrize("query", ["data.example.allow; true", "[1, 2][_]", "input..broken"])
async def test_ambiguous_or_invalid_queries_do_not_produce_an_approval(policy_path, query):
    with pytest.raises(RuntimeError):
        await RegoPolicy(policy_path).evaluate(query, {"action": "read"})


@pytest.mark.parametrize("failure", [
    "engine_error", "nonboolean_ok", "no_results", "multiple_results",
    "missing_expression", "multiple_expressions", "nonboolean_expression",
    "invalid_bindings", "missing_binding", "additional_binding",
])
async def test_malformed_native_results_fail_closed_without_diagnostics(
    policy_path, monkeypatch, failure
):
    result = SimpleNamespace(expressions=[True], bindings={"__guardian_result__": True})
    results = [result]
    success = True
    if failure == "engine_error":
        success = False
    elif failure == "nonboolean_ok":
        success = 1
    elif failure == "no_results":
        results = []
    elif failure == "multiple_results":
        results = [result, result]
    elif failure == "missing_expression":
        result.expressions = []
    elif failure == "multiple_expressions":
        result.expressions = [True, True]
    elif failure == "nonboolean_expression":
        result.expressions = [1]
    elif failure == "invalid_bindings":
        result.bindings = None
    elif failure == "missing_binding":
        result.bindings = {}
    else:
        result.bindings["unexpected"] = True

    class MalformedOutput:
        def ok(self):
            return success

        def __str__(self):
            return "private_engine_diagnostics"

        def __len__(self):
            return len(results)

        def __getitem__(self, index):
            return results[index]

    class BrokenInterpreter(rego_policy.Interpreter):
        def query(self, query):
            return MalformedOutput()

    monkeypatch.setattr(rego_policy, "Interpreter", BrokenInterpreter)
    with pytest.raises(RuntimeError) as error:
        await RegoPolicy(policy_path).evaluate("data.example.allow", {"action": "read"})
    assert "private_engine_diagnostics" not in str(error.value)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(policy_path, timeout):
    with pytest.raises(ValueError):
        RegoPolicy(policy_path, timeout=timeout)


@pytest.fixture
async def slow_interpreter(monkeypatch):
    """Pause native work without blocking the event loop, then always release it."""
    loop = asyncio.get_running_loop()
    state = SimpleNamespace(
        entered=asyncio.Event(),
        finished=asyncio.Event(),
        release=threading.Event(),
        thread_ids=[],
    )
    real_interpreter = rego_policy.Interpreter

    class SlowInterpreter(real_interpreter):
        def query(self, query):
            state.thread_ids.append(threading.get_ident())
            loop.call_soon_threadsafe(state.entered.set)
            try:
                if not state.release.wait(timeout=2):
                    raise RuntimeError("Test worker was not released")
                return super().query(query)
            finally:
                loop.call_soon_threadsafe(state.finished.set)

    monkeypatch.setattr(rego_policy, "Interpreter", SlowInterpreter)
    yield state
    state.release.set()
    if state.entered.is_set():
        await asyncio.wait_for(state.finished.wait(), timeout=1)


async def test_evaluation_leaves_event_loop_responsive(policy_path, slow_interpreter):
    task = asyncio.create_task(RegoPolicy(policy_path).evaluate(
        "data.example.allow", {"action": "read"}
    ))
    await asyncio.wait_for(slow_interpreter.entered.wait(), timeout=1)
    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=0.1)
    assert not task.done()
    assert len(slow_interpreter.thread_ids) == 1
    assert slow_interpreter.thread_ids[0] != threading.get_ident()
    slow_interpreter.release.set()
    assert await task is True


async def test_input_snapshot_is_unaffected_by_caller_mutation(policy_path, slow_interpreter):
    packet = {"value": {"genres": ["jazz"], "metadata": {"session": "original"}}}
    task = asyncio.create_task(RegoPolicy(policy_path).evaluate("data.example.value", packet))
    await asyncio.wait_for(slow_interpreter.entered.wait(), timeout=1)
    packet["value"]["genres"].append("blues")
    packet["value"]["metadata"]["session"] = "changed"
    slow_interpreter.release.set()
    assert await task == {"genres": ["jazz"], "metadata": {"session": "original"}}


async def test_timeout_discards_late_approval(policy_path, slow_interpreter):
    task = asyncio.create_task(RegoPolicy(policy_path, timeout=0.05).evaluate(
        "data.example.allow", {"action": "read"}
    ))
    await asyncio.wait_for(slow_interpreter.entered.wait(), timeout=1)
    with pytest.raises(TimeoutError):
        await task
    # Cancellation of the wait does not stop native work. Its later yes must
    # never turn the failed evaluation into a successful authorization.
    assert not slow_interpreter.finished.is_set()
    slow_interpreter.release.set()
    await asyncio.wait_for(slow_interpreter.finished.wait(), timeout=1)
    with pytest.raises(TimeoutError):
        task.result()


async def test_cancellation_discards_late_approval(policy_path, slow_interpreter):
    task = asyncio.create_task(RegoPolicy(policy_path).evaluate(
        "data.example.allow", {"action": "read"}
    ))
    await asyncio.wait_for(slow_interpreter.entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not slow_interpreter.finished.is_set()
    slow_interpreter.release.set()
    await asyncio.wait_for(slow_interpreter.finished.wait(), timeout=1)
    assert task.cancelled()


@pytest.fixture
def opa_binary():
    executable = os.environ.get("OPA_BINARY") or shutil.which("opa")
    if not executable:
        pytest.skip("Set OPA_BINARY to additionally compare results with reference OPA")
    return executable


@pytest.mark.parametrize("operation, changed_field, value, allowed", [
    ("remember", None, None, True),
    ("remember", ("memory", "customer_id"), 3, False),
    ("remember", ("memory", "namespace"), ["3", "memories"], False),
    ("remember", ("guardian", "policy_id"), "previous-policy", False),
    ("remember", ("guardian", "intent_match"), False, False),
    ("recall", None, None, True),
    ("recall", ("memory", "namespace"), ["3", "memories"], False),
    ("recall", ("tool", "args"), {"query": "music", "limit": 51}, False),
])
async def test_guardian_decisions_match_optional_reference_opa(
    opa_binary, operation, changed_field, value, allowed
):
    policy_path = Path(__file__).resolve().parents[1] / "policies" / "guardian.rego"
    packet = {
        "subject": {"authenticated": True, "customer_id": 2},
        "tool": {
            "name": "remember",
            "args": {"fact": "Prefers jazz recommendations."},
        },
        "memory": {"customer_id": 2, "namespace": ["2", "memories"], "operation": "put"},
        "source_user_message": "I prefer jazz recommendations.",
        "guardian": {"policy_id": "customer-memory-v4", "intent_match": True},
    }
    if operation == "recall":
        packet["tool"] = {"name": "recall", "args": {"query": "music", "limit": 5}}
        packet["memory"]["operation"] = "search"
        del packet["source_user_message"]
        del packet["guardian"]
    if changed_field is not None:
        section, field = changed_field
        packet[section][field] = value
    embedded = await RegoPolicy(policy_path).evaluate("data.chinook.guardian.decision", packet)
    assert embedded == {"allow": allowed, "policy_id": "customer-memory-v4"}
    process = await asyncio.create_subprocess_exec(
        opa_binary, "eval", "--format=json", "--data", str(policy_path),
        "--stdin-input", "data.chinook.guardian.decision",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(5):
            stdout, stderr = await process.communicate(json.dumps(packet).encode())
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
    assert process.returncode == 0, stderr.decode()
    reference = json.loads(stdout)["result"][0]["expressions"][0]["value"]
    assert embedded == reference
