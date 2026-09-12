"""
runtime/test/test_recoverable_step.py — regression test for
run_with_recoverable_step (Product_Archive.md "HITL/DLQ redesign").

Exercises the exact CRM example end to end against a real Temporal test
server (temporalio.testing.WorkflowEnvironment) and a real throwaway
Postgres (for dlq_enqueue_activity's DeadLetterQueue writes): an activity
hallucinates a field name, the workflow stays alive instead of
terminating, a human_fix_payload signal with the corrected JSON resumes
it, and the activity succeeds.

Also locks in a real bug this test caught during development: without
retry_policy=RetryPolicy(maximum_attempts=1) on the gated execute_activity
call, Temporal's default retry policy retries the same failing payload
indefinitely (with backoff) until start_to_close_timeout — the
recoverable-step logic wouldn't even reach the DLQ-enqueue/wait step for
up to 10 minutes. test_retries_exactly_once_not_temporals_default_policy
below asserts the activity is attempted exactly once per human-submitted
payload, not retried by Temporal itself.

The workflow/activity definitions live in the sibling
_recoverable_step_fixtures.py, not in this file — see that module's
docstring for why (Temporal's sandboxed re-import of a workflow's
defining module restricts `pathlib.Path.resolve()`, which this file's own
sys.path setup below would otherwise trip if it lived in the same module).

Requires DATABASE_URL (throwaway Postgres) — same as test_llm_gateway_budget.py's
sibling tests, run via `pytest runtime/test/ -v` per OPERATIONS.md §3.
Downloads/starts a local Temporal test-server binary on first use
(cached by the SDK afterward) — this is the one test in this directory
that needs network access on a cold CI cache.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows"))

pytest.importorskip("temporalio", reason="temporalio not installed")
from temporalio.worker import Worker
from temporalio.testing import WorkflowEnvironment

from base_workflow import dlq_enqueue_activity  # type: ignore
from _recoverable_step_fixtures import ATTEMPT_COUNT, CRMWorkflow, crm_update_activity  # type: ignore

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set"
)


@pytest.mark.asyncio
async def test_workflow_stays_alive_and_resumes_with_human_fix():
    ATTEMPT_COUNT.clear()
    env = await WorkflowEnvironment.start_local()
    async with env:
        async with Worker(
            env.client,
            task_queue="test-recoverable-step-tq",
            workflows=[CRMWorkflow],
            activities=[crm_update_activity, dlq_enqueue_activity],
        ):
            handle = await env.client.start_workflow(
                CRMWorkflow.run,
                {"customer_id": 102, "account_status": "active"},
                id=f"recoverable-step-test-{os.getpid()}",
                task_queue="test-recoverable-step-tq",
            )

            # Wait for the workflow to fail once and start waiting on the
            # signal, then submit the human's fix.
            for _ in range(50):
                if ATTEMPT_COUNT.get("n", 0) >= 1:
                    break
                await asyncio.sleep(0.1)
            await handle.signal(
                "human_fix_payload",
                args=["crm-update-gate", {"customer_id": 102, "status": "active"}],
            )

            result = await handle.result()

    assert result == {"ok": True, "applied": {"customer_id": 102, "status": "active"}}


@pytest.mark.asyncio
async def test_retries_exactly_once_not_temporals_default_policy():
    """Without retry_policy=RetryPolicy(maximum_attempts=1) on the gated
    execute_activity call, Temporal's default policy retries the same
    failing payload indefinitely — this asserts exactly one attempt per
    submitted payload, the real bug a live test run caught mid-build."""
    ATTEMPT_COUNT.clear()
    env = await WorkflowEnvironment.start_local()
    async with env:
        async with Worker(
            env.client,
            task_queue="test-recoverable-step-retry-tq",
            workflows=[CRMWorkflow],
            activities=[crm_update_activity, dlq_enqueue_activity],
        ):
            handle = await env.client.start_workflow(
                CRMWorkflow.run,
                {"customer_id": 102, "account_status": "active"},
                id=f"recoverable-step-retry-test-{os.getpid()}",
                task_queue="test-recoverable-step-retry-tq",
            )

            for _ in range(50):
                if ATTEMPT_COUNT.get("n", 0) >= 1:
                    break
                await asyncio.sleep(0.1)
            await asyncio.sleep(
                1.0
            )  # would balloon past 1 within this window under Temporal's default retry policy

            assert ATTEMPT_COUNT.get("n", 0) == 1, (
                f"expected exactly 1 attempt before the human fix arrives, got {ATTEMPT_COUNT.get('n')} — "
                "Temporal's default retry policy is retrying the same failing payload"
            )

            await handle.signal(
                "human_fix_payload",
                args=["crm-update-gate", {"customer_id": 102, "status": "active"}],
            )
            await handle.result()
