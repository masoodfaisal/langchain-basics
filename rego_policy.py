"""Apply trusted Rego policies in process with RegoPy, without an OPA service.

RegoPy wraps Microsoft's C++ Rego interpreter, not the OPA Go engine. Each query
uses a fresh interpreter and the same policy-source snapshot. Native evaluation
runs in a worker thread; timeout/cancellation stops waiting, not the worker.
The former OPA subprocess implementation is preserved as comments below.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

from regopy import Interpreter, LogLevel


class RegoPolicy:
    """An async adapter for a trusted Rego snapshot.

    Policy paths and queries come from application code, never from the model.
    Recreate this object to reload Rego. Use only trusted, bounded policies:
    Python cannot forcibly stop a native query that outlives its timeout.
    """

    def __init__(
        self,
        policy_path: Path | str,
        *,
        timeout: float = 5.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Rego timeout must be finite and positive")
        self._source = Path(policy_path).read_text(encoding="utf-8")
        self._timeout = timeout

    async def evaluate(
        self, query: str, input_data: dict[str, Any] | None = None,
    ) -> Any:
        """Return one JSON value, or None when undefined; raise on failure."""
        # Copy before yielding: later mutations cannot change the checked input.
        packet_json = json.dumps(
            {} if input_data is None else input_data, allow_nan=False,
        )

        def run_query() -> Any:
            try:
                # Never share a mutable interpreter/input between concurrent calls.
                engine = Interpreter()
                engine.log_level = LogLevel.NONE
                engine.strict_built_in_errors = True
                engine.add_module("guardian", self._source)
                # JSON avoids int64 overflow and NUL truncation in set_input's
                # direct ctypes marshalling; all packet contents remain data.
                engine.set_input_term(packet_json)
                # Binding preserves false, which a direct RegoPy query treats as
                # unsatisfied. The query is trusted; input stays separate data.
                output = engine.query(f"__guardian_result__ = ({query})")
                if output.ok() is not True:
                    raise ValueError
                if str(output) == "undefined":
                    return None
                if len(output) != 1:
                    raise ValueError
                expressions, bindings = output[0].expressions, output[0].bindings
                if (
                    not isinstance(expressions, list) or len(expressions) != 1
                    or expressions[0] is not True
                    or not isinstance(bindings, dict)
                    or set(bindings) != {"__guardian_result__"}
                ):
                    raise ValueError
                return bindings["__guardian_result__"]
            except Exception:
                # Library diagnostics can contain policy/input data.
                raise RuntimeError("Rego policy evaluation failed or returned an invalid result") from None

        async with asyncio.timeout(self._timeout):
            return await asyncio.to_thread(run_query)


# Previous OPA subprocess implementation (reference only; not executed).
# """Apply trusted Rego policies locally with OPA, without a listening server.
#
# This small adapter starts ``opa eval`` per query; it is not an embedded Rego
# interpreter. Policy source is snapshotted at construction so metadata and
# decisions cannot silently use different versions. No Python packages are needed.
# """
#
# from __future__ import annotations
#
# import asyncio
# import json
# import math
# import shutil
# from pathlib import Path
# from tempfile import TemporaryDirectory
# from typing import Any
#
#
# class RegoPolicy:
#     """A policy snapshot with bounded, async OPA evaluation.
#
#     Paths, executable, and queries must come from trusted application code,
#     never from model-generated arguments. Recreate this object to reload Rego.
#     """
#
#     def __init__(
#         self,
#         policy_path: Path | str,
#         *,
#         opa_binary: str = "opa",
#         timeout: float = 5.0,
#     ) -> None:
#         executable = shutil.which(opa_binary)
#         if executable is None:
#             raise FileNotFoundError("OPA executable required; set opa_binary or add opa to PATH")
#         if not math.isfinite(timeout) or timeout <= 0:
#             raise ValueError("OPA timeout must be finite and positive")
#         self._opa_binary = str(Path(executable).resolve())
#         self._source = Path(policy_path).read_text(encoding="utf-8")
#         self._timeout = timeout
#
#     async def evaluate(
#         self, query: str, input_data: dict[str, Any] | None = None,
#     ) -> Any:
#         """Return one query value, or None when undefined; raise on failure."""
#         # OPA's CLI takes raw input JSON, not the REST API's {"input": ...}.
#         stdin = json.dumps({} if input_data is None else input_data, allow_nan=False).encode()
#         with TemporaryDirectory(prefix="guardian-rego-") as directory:
#             policy_path = Path(directory) / "policy.rego"
#             await asyncio.to_thread(policy_path.write_text, self._source, encoding="utf-8")
#             process = await asyncio.create_subprocess_exec(
#                 self._opa_binary, "eval", "--format=json", "--strict",
#                 "--strict-builtin-errors", "--data", str(policy_path),
#                 "--stdin-input", "--", query,
#                 stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE,
#             )
#             try:
#                 async with asyncio.timeout(self._timeout):
#                     stdout, _ = await process.communicate(stdin)
#             except BaseException:
#                 # Kill and drain/reap before removing the policy, even on cancellation.
#                 if process.returncode is None:
#                     try:
#                         process.kill()
#                     except ProcessLookupError:
#                         pass  # The child exited between the check and kill.
#                 await process.communicate()
#                 raise
#             if process.returncode != 0:
#                 # OPA diagnostics may include policy or input data; do not expose them.
#                 raise RuntimeError("OPA policy evaluation failed")
#
#         try:
#             output = json.loads(stdout)
#             if not isinstance(output, dict) or "errors" in output:
#                 raise ValueError
#             results = output.get("result", [])
#             if not isinstance(results, list):
#                 raise ValueError
#             if not results:
#                 return None
#             if len(results) != 1:
#                 raise ValueError
#             expressions = results[0]["expressions"]
#             if not isinstance(expressions, list) or len(expressions) != 1:
#                 raise ValueError
#             return expressions[0]["value"]
#         except (ValueError, KeyError, TypeError, IndexError):
#             raise RuntimeError("OPA returned a malformed or ambiguous result") from None
