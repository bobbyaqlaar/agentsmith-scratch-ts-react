"""
runtime/test/_recoverable_step_fixtures.py — workflow/activity definitions
for test_recoverable_step.py, kept in their own module deliberately.

Temporal's workflow sandbox re-imports whatever module defines a
`@workflow.defn` class, in a restricted environment, to validate
determinism — and that re-import executes this module's top-level code
again. If this module did its own `sys.path.insert(..., Path(__file__).resolve()...)`
at module scope (the obvious way to write it), the sandbox's restriction
on `pathlib.Path.resolve()` fails validation with a confusing
"Failed validating workflow" error that has nothing to do with the actual
test logic — caught by running this for real, not assumed. The fix is
structural: this module does no path manipulation at all; the test file
that imports it does that *before* importing this module, so `sys.path`
is already correct by the time either the normal import or the sandbox's
re-import needs to find `base_workflow`.
"""

from __future__ import annotations

from typing import Any

from temporalio import activity, workflow

from base_workflow import BaseAgentWorkflow  # type: ignore

ATTEMPT_COUNT: dict[str, int] = {}


@activity.defn
async def crm_update_activity(payload: dict) -> dict:
    ATTEMPT_COUNT["n"] = ATTEMPT_COUNT.get("n", 0) + 1
    if "status" not in payload:
        bad_key = next(k for k in payload if k != "customer_id")
        raise ValueError(f"{bad_key} is not a valid property")
    return {"ok": True, "applied": payload}


@workflow.defn
class CRMWorkflow(BaseAgentWorkflow):
    def __init__(self) -> None:
        super().__init__()

    @workflow.run
    async def run(self, payload: dict) -> Any:
        from datetime import timedelta

        return await self.run_with_recoverable_step(
            "crm_update_activity",
            payload,
            tenant_id="acme",
            gate_id="crm-update-gate",
            timeout=timedelta(seconds=30),
        )
