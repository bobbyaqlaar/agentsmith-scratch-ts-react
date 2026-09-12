"""
runtime/test/_gateway_fixtures.py — one bare LLMGateway builder for tests.

Four modules built the same object by hand: `test_guardrail_evidence`,
`test_prompt_guard_modes`, `test_stream_providers` and `test_ttft_stream`. Two
of them were identical apart from what `_invoke` returned. Each copy names
eleven private attributes, so adding a twelfth to `LLMGateway.__init__` meant
finding four call sites, and missing one produced an AttributeError in a test
that had nothing to do with the change.

`LLMGateway.__new__` rather than the constructor is deliberate and stays: the
real `__init__` reads a registry, opens a budget backend and may reach for
Postgres. These tests are about routing and guardrail behaviour, and a
constructor that touches infrastructure would make them integration tests.

NOT used by `test_degrade_ladder._gw`, deliberately. That module exercises the
REAL `_resolve_role` and `_degrade_chain`; this builder mocks both, so sharing
it there would replace the thing under test with a stub and the suite would
pass no matter what the ladder did.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import LLMGateway

DEFAULT_MODEL = {
    "id": "test-model",
    "provider": "openai",
    "cost_per_input_token": 0.0,
    "cost_per_output_token": 0.0,
}


def fake_gateway(
    *,
    role: str = "developer",
    model: Optional[dict] = None,
    budget_cap_usd: float = 10.0,
    free_tier: bool = True,
    invoke: Optional[Any] = None,
    coerce_messages: Optional[list] = None,
) -> LLMGateway:
    """A gateway with its infrastructure stubbed and its routing pinned.

    `coerce_messages` defaults to None meaning DO NOT mock it — the guardrail
    tests feed real PII through `complete()` and depend on the actual coercion
    to carry it to the scrubber. Passing a list mocks the return, which is what
    the streaming tests want, since they assert on transport rather than on
    message contents.

    `invoke` likewise stays unmocked unless supplied, so a test that patches
    the transport itself is not silently short-circuited by a stub above it.
    """
    gw = LLMGateway.__new__(LLMGateway)
    gw.tenant_id = "t"
    gw.models = {role: dict(model or DEFAULT_MODEL)}
    gw.budget_cap_usd = budget_cap_usd
    gw._idempotency = None
    gw.get_budget_status = MagicMock(
        return_value={
            "ok": True,
            "spent_usd": 0,
            "cap_usd": budget_cap_usd,
            "remaining_usd": budget_cap_usd,
        }
    )
    gw._resolve_role = MagicMock(return_value=(role, None))
    gw._record_span_attributes = MagicMock()
    gw._report_run_status = MagicMock()
    gw._degrade_chain = MagicMock(return_value=[role])
    gw._is_free_tier = MagicMock(return_value=free_tier)
    if coerce_messages is not None:
        gw._coerce_messages = MagicMock(return_value=coerce_messages)
    if invoke is not None:
        gw._invoke = AsyncMock(return_value=invoke)
    return gw
