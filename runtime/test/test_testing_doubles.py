"""
runtime/test/test_testing_doubles.py — the shipped test doubles
(TestbedFeedback-2026-07-21 G4).

The double's whole value is that it behaves like the real gateway; if it
drifts, every tenant's suite silently lies. These tests pin the contract,
including the one that matters most: it must NOT be more capable than the
real gateway (that is how the testbed's hand-rolled double hid G1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import BudgetExceededError, CompletionResult
from runtime.testing import FakeGateway, RecordingGateway


@pytest.mark.asyncio
async def test_returns_real_completion_result_shape():
    gw = FakeGateway(responses={"analyst": '{"rating":"LOW"}'})
    result = await gw.complete("hi", model_hint="analyst")
    assert isinstance(result, CompletionResult)
    assert result.text == '{"rating":"LOW"}'
    assert result.model_used == "fake-analyst"
    assert result.ttft_ms is None  # non-streamed calls report no TTFT


@pytest.mark.asyncio
async def test_scripted_sequence_then_sticky_last():
    """'fail once, then succeed forever' — the shape self-correction tests need."""
    gw = FakeGateway(responses={"a": ["broken", '{"ok":true}']})
    assert (await gw.complete("x", model_hint="a")).text == "broken"
    assert (await gw.complete("x", model_hint="a")).text == '{"ok":true}'
    assert (await gw.complete("x", model_hint="a")).text == '{"ok":true}'


@pytest.mark.asyncio
async def test_callable_response_sees_the_prompt():
    gw = FakeGateway(responses={"a": lambda p: f"len={len(p)}"})
    assert (await gw.complete("12345", model_hint="a")).text == "len=5"


@pytest.mark.asyncio
async def test_streaming_reports_ttft_and_records_the_call():
    gw = FakeGateway(responses={"a": "streamed"}, ttft_ms=12.5)
    result = await gw.complete_stream("hi", model_hint="a")
    assert result.ttft_ms == 12.5
    assert gw.calls[-1].streamed is True


@pytest.mark.asyncio
async def test_double_is_not_more_capable_than_the_real_gateway():
    """G4's core rule. A non-streaming provider must behave like the real
    gateway: fall back (post-G1 default) or raise (pre-G1 behavior) —
    never silently pretend it streamed."""
    gw = FakeGateway(responses={"a": "x"}, providers={"a": "bedrock"})
    result = await gw.complete_stream("hi", model_hint="a")
    assert result.ttft_ms is None            # honest about not streaming
    assert gw.calls[-1].streamed is False

    strict = FakeGateway(providers={"a": "bedrock"}, stream_fallback=False)
    with pytest.raises(NotImplementedError):
        await strict.complete_stream("hi", model_hint="a")


@pytest.mark.asyncio
async def test_budget_cap_raises_the_real_error_type():
    gw = FakeGateway(cap_usd=1.0, cost_per_call=0.6)
    await gw.complete("a")
    with pytest.raises(BudgetExceededError):
        await gw.complete("b")
    assert gw.get_budget_status()["spent_usd"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_assertion_helpers():
    gw = FakeGateway()
    await gw.complete("clean prompt", model_hint="intake")
    await gw.complete("more", model_hint="analyst")
    await gw.complete("again", model_hint="intake")

    assert gw.routes_used() == ["intake", "analyst"]
    assert len(gw.calls_for("intake")) == 2
    gw.assert_prompt_excludes("784-1985-1234567-1")
    with pytest.raises(AssertionError, match="reached the model"):
        gw.assert_prompt_excludes("clean")


@pytest.mark.asyncio
async def test_message_list_prompts_are_flattened_for_assertions():
    gw = FakeGateway()
    await gw.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "secret"}],
        model_hint="a",
    )
    assert "secret" in gw.calls[-1].prompt
    with pytest.raises(AssertionError):
        gw.assert_prompt_excludes("secret")


@pytest.mark.asyncio
async def test_recording_gateway_wraps_and_delegates():
    inner = FakeGateway(responses={"a": "inner"})
    rec = RecordingGateway(inner)
    result = await rec.complete("hi", model_hint="a")
    assert result.text == "inner"
    assert rec.calls[-1].model_hint == "a"
    assert rec.degrade_tiers() == [None]
    assert rec.tenant_id == inner.tenant_id  # attribute passthrough
