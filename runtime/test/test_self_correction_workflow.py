from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows"))

import base_workflow as bw  # type: ignore
from base_workflow import BaseAgentWorkflow  # type: ignore


class FakeWorkflow:
    def __init__(self, *, corrected_payload: dict[str, Any]) -> None:
        self.corrected_payload = corrected_payload
        self.calls: list[tuple[Any, Any]] = []

    async def execute_activity(self, activity: Any, payload: Any, **_kwargs: Any) -> Any:
        self.calls.append((activity, payload))
        if activity == "crm_update_activity":
            if "status" not in payload:
                raise ValueError("account_status is not a valid property")
            return {"ok": True, "applied": payload}
        return self.corrected_payload


@pytest.mark.asyncio
async def test_run_with_self_correction_retries_corrected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_workflow = FakeWorkflow(
        corrected_payload={"customer_id": 102, "status": "active"}
    )
    monkeypatch.setattr(bw, "_HAS_TEMPORAL", True)
    monkeypatch.setattr(bw, "RetryPolicy", lambda maximum_attempts: maximum_attempts)
    monkeypatch.setattr(bw, "workflow", fake_workflow)

    result = await BaseAgentWorkflow().run_with_self_correction(
        "crm_update_activity",
        {"customer_id": 102, "account_status": "active"},
        tenant_id="acme",
        gate_id="crm-update-gate",
    )

    assert result == {"ok": True, "applied": {"customer_id": 102, "status": "active"}}
    assert [call[0] for call in fake_workflow.calls] == [
        "crm_update_activity",
        bw.self_correct_payload_activity,
        "crm_update_activity",
    ]


@pytest.mark.asyncio
async def test_run_with_self_correction_falls_back_to_recoverable_step_with_last_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_workflow = FakeWorkflow(
        corrected_payload={"customer_id": 102, "account_status": "active"}
    )

    async def execute_activity(activity: Any, payload: Any, **_kwargs: Any) -> Any:
        fake_workflow.calls.append((activity, payload))
        if activity == "crm_update_activity":
            raise ValueError("account_status is not a valid property")
        return fake_workflow.corrected_payload

    fake_workflow.execute_activity = execute_activity  # type: ignore[method-assign]
    monkeypatch.setattr(bw, "_HAS_TEMPORAL", True)
    monkeypatch.setattr(bw, "RetryPolicy", lambda maximum_attempts: maximum_attempts)
    monkeypatch.setattr(bw, "workflow", fake_workflow)

    workflow = BaseAgentWorkflow()
    fallback_calls: list[dict[str, Any]] = []

    async def fallback(**kwargs: Any) -> dict[str, Any]:
        fallback_calls.append(kwargs)
        return {"status": "dead_letter"}

    workflow.run_with_recoverable_step = fallback  # type: ignore[method-assign]

    result = await workflow.run_with_self_correction(
        "crm_update_activity",
        {"customer_id": 102, "account_status": "active"},
        tenant_id="acme",
        gate_id="crm-update-gate",
        max_self_correction_attempts=1,
    )

    assert result == {"status": "dead_letter"}
    assert fallback_calls == [
        {
            "activity_name": "crm_update_activity",
            "payload": {"customer_id": 102, "account_status": "active"},
            "tenant_id": "acme",
            "gate_id": "crm-update-gate",
            "reason": "validation_error",
            "timeout": bw.HITL_SIGNAL_TIMEOUT,
            "max_attempts": bw.RECOVERABLE_STEP_MAX_ATTEMPTS,
        }
    ]


def test_a_failing_correction_activity_falls_through_to_the_human_path(monkeypatch):
    """The Temporal twin of the corrector-failure case, and the one with the
    sharper consequence.

    `self_correct_payload_activity` ends in `json.loads` via
    `propose_corrected_payload`, and it runs with `maximum_attempts=1`. So a
    model answering in prose raised out of `run_with_self_correction` and failed
    the workflow — skipping `run_with_recoverable_step`, which is the human DLQ
    path. The most likely failure of the automatic fixer was the one that
    stopped an application ever reaching a person.
    """
    import asyncio

    from runtime.workflows import base_workflow
    from runtime.workflows.base_workflow import BaseAgentWorkflow

    executed: list[str] = []

    class _Fake:
        async def execute_activity(self, name, payload=None, **_kw):
            label = getattr(name, "__name__", str(name))
            executed.append(label)
            if label == "self_correct_payload_activity":
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            raise ValueError("schema mismatch")

        def info(self):
            class _I:
                workflow_id = "wf-1"
                run_id = "run-1"
            return _I()

        @staticmethod
        def signal(fn=None, **_k):
            return fn if fn is not None else (lambda f: f)

    monkeypatch.setattr(base_workflow, "workflow", _Fake())
    monkeypatch.setattr(base_workflow, "_HAS_TEMPORAL", True)

    wf = BaseAgentWorkflow()
    reached: dict = {}

    async def _recoverable(**kwargs):
        reached.update(kwargs)
        return {"parked": True}

    wf.run_with_recoverable_step = _recoverable

    result = asyncio.run(
        wf.run_with_self_correction(
            "decide_activity", {"amount": "ten"}, tenant_id="t", gate_id="g"
        )
    )

    assert result == {"parked": True}, (
        "a failed correction activity did not reach the human recoverable step"
    )
    assert reached["payload"] == {"amount": "ten"}, (
        "the human was handed the wreckage of a failed correction, not the last "
        "real payload"
    )
