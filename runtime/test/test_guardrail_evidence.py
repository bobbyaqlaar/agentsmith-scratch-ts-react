"""
runtime/test/test_guardrail_evidence.py — CompletionResult carries the
guardrail evidence the gateway already computed (TestbedFeedback G3).

Before this, the scrub counts reached only logs and span attributes, so an
app that must record WHAT was redacted in its own decision record (any
PDPL/GDPR decision-path app) had to re-run the scrub itself — the KYC
Sentinel testbed did exactly that, paying for the scrub twice per
application.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import CompletionResult, LLMGateway
from runtime.test._gateway_fixtures import fake_gateway

def _gateway() -> LLMGateway:
    """The shared builder, with this module's transport stub."""
    return fake_gateway(invoke=("safe answer", 3, 2))


PII_PROMPT = "Emirates ID 784-1985-1234567-1, card 4111 1111 1111 1111, ali@example.com"




@pytest.mark.asyncio
async def test_complete_returns_scrub_counts(monkeypatch):
    monkeypatch.setenv("INPUT_GUARDRAIL", "default")
    monkeypatch.setenv("PROMPT_GUARD", "off")
    gw = _gateway()

    result = await gw.complete(PII_PROMPT, model_hint="developer")

    assert isinstance(result, CompletionResult)
    assert result.guardrail_counts.get("emirates_id") == 1
    assert result.guardrail_counts.get("card") == 1
    assert result.guardrail_counts.get("email") == 1
    # The evidence describes the prompt the PROVIDER saw — scrubbed.
    sent_messages = gw._invoke.await_args[0][1]
    sent_text = " ".join(m["content"] for m in sent_messages)
    assert "784-1985-1234567-1" not in sent_text
    assert "4111 1111 1111 1111" not in sent_text


@pytest.mark.asyncio
async def test_clean_prompt_reports_no_redactions(monkeypatch):
    monkeypatch.setenv("INPUT_GUARDRAIL", "default")
    monkeypatch.setenv("PROMPT_GUARD", "off")
    result = await _gateway().complete("nothing sensitive here", model_hint="developer")
    assert result.guardrail_counts == {}


@pytest.mark.asyncio
async def test_guardrail_off_reports_nothing(monkeypatch):
    monkeypatch.setenv("INPUT_GUARDRAIL", "off")
    monkeypatch.setenv("PROMPT_GUARD", "off")
    result = await _gateway().complete(PII_PROMPT, model_hint="developer")
    assert result.guardrail_counts == {}


def test_result_defaults_are_independent_instances():
    """Mutable dataclass defaults must not be shared across results."""
    a = CompletionResult(text="a", model_used="m", input_tokens=0, output_tokens=0, cost_usd=0)
    b = CompletionResult(text="b", model_used="m", input_tokens=0, output_tokens=0, cost_usd=0)
    a.guardrail_counts["x"] = 1
    a.prompt_guard_reasons.append("y")
    assert b.guardrail_counts == {} and b.prompt_guard_reasons == []
