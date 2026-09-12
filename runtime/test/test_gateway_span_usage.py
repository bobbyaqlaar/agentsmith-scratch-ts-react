"""
runtime/test/test_gateway_span_usage.py — the gateway's span attributes.

Two defects are pinned here.

`_record_span_attributes` guarded with `if span is None: return`.
`trace.get_current_span()` NEVER returns None — with no active span it returns
a NonRecordingSpan whose `set_attribute` is a silent no-op — so the guard never
fired and every attribute was dropped on any call path not already inside an
`agent_span`. That is `tenant.id`, `llm.model_name`, cost and TTFT, gone with no
signal, which is what made "every span carries tenant.id" untrue in practice.

And usage was computed and discarded. `CompletionResult` has carried
`input_tokens`/`output_tokens` since the gateway was written and not one of them
reached a span, so cost could be charted but never attributed to a prompt.
Streamed calls report no usage at all in v1 and the result carries 0/0 for it —
written to a span as 0 that would make "used no tokens" and "nobody counted"
the same number on any dashboard that sums them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import LLMGateway
from runtime.tracing import agent_span


def _only_span(exporter) -> dict:
    finished = exporter.get_finished_spans()
    assert len(finished) == 1, [s.name for s in finished]
    return dict(finished[0].attributes)


@pytest.fixture(scope="module")
def gateway():
    """In-memory backends, set before construction.

    IDEMPOTENCY_BACKEND defaults to "redis", so a bare LLMGateway() spends
    roughly two minutes probing a Redis that is not there before falling back.
    These tests touch only span attributes; the backends are not the subject.
    """
    import os

    previous = {k: os.environ.get(k) for k in ("IDEMPOTENCY_BACKEND", "BUDGET_BACKEND")}
    os.environ["IDEMPOTENCY_BACKEND"] = "memory"
    os.environ["BUDGET_BACKEND"] = "memory"
    try:
        yield LLMGateway(tenant_id="acme")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_usage_is_recorded_when_the_provider_reported_it(spans, gateway):
    with agent_span("llm.call", tenant_id="acme"):
        gateway._record_span_attributes(
            "analyst", "claude-sonnet-4.5", None, "wf-1", 0.0123,
            input_tokens=1500, output_tokens=270,
        )
    a = _only_span(spans)
    assert a["llm.usage.reported"] is True
    assert a["llm.usage.input_tokens"] == 1500
    assert a["llm.usage.output_tokens"] == 270
    assert a["llm.usage.total_tokens"] == 1770
    assert a["llm.gateway.cost_estimated"] is False
    assert a["tenant.id"] == "acme"
    assert a["llm.model_name"] == "claude-sonnet-4.5"


def test_unreported_usage_is_absent_not_zero(spans, gateway):
    """The streamed path. A 0 here is a measurement of zero; the truth is that
    nobody counted, and a dashboard summing tokens must be able to tell."""
    with agent_span("llm.call", tenant_id="acme"):
        gateway._record_span_attributes(
            "analyst", "claude-sonnet-4.5", None, None, 0.02,
            ttft_ms=180.0,
            input_tokens=None, output_tokens=None,
            cost_estimated=True,
        )
    a = _only_span(spans)
    assert a["llm.usage.reported"] is False
    assert "llm.usage.input_tokens" not in a
    assert "llm.usage.output_tokens" not in a
    assert "llm.usage.total_tokens" not in a
    # The cost that IS present is a reservation ceiling, and says so.
    assert a["llm.gateway.cost_estimated"] is True
    assert a["llm.gateway.ttft_ms"] == 180.0


def test_a_partial_usage_report_counts_as_no_report(spans, gateway):
    """One of the two is not a usable measurement — total would be a guess."""
    with agent_span("llm.call", tenant_id="acme"):
        gateway._record_span_attributes(
            "analyst", "m", None, None, 0.0,
            input_tokens=100, output_tokens=None,
        )
    a = _only_span(spans)
    assert a["llm.usage.reported"] is False
    assert "llm.usage.input_tokens" not in a


def test_zero_tokens_is_recorded_as_a_real_measurement(spans, gateway):
    """The other side of the distinction: a provider that genuinely reports 0
    must not be filed as 'not reported'."""
    with agent_span("llm.call", tenant_id="acme"):
        gateway._record_span_attributes(
            "analyst", "m", None, None, 0.0, input_tokens=0, output_tokens=0
        )
    a = _only_span(spans)
    assert a["llm.usage.reported"] is True
    assert a["llm.usage.total_tokens"] == 0


def test_no_parent_span_emits_the_gateway_s_own(spans, gateway):
    """The actual fix. With nothing recording, the old code wrote attributes to
    a NonRecordingSpan and the LLM call vanished from the trace — no model, no
    cost, no tokens, no latency, on the one operation that matters most.

    Fixing the dead `is None` guard alone would not have changed that: a no-op
    write and an early return lose the attributes equally. Emitting a span does.
    """
    import time

    started = time.time_ns() - 250_000_000  # 250ms ago
    gateway._record_span_attributes(
        "analyst", "claude-sonnet-4.5", None, "wf-3", 0.04,
        input_tokens=10, output_tokens=20, started_ns=started,
    )
    a = _only_span(spans)
    assert a["llm.gateway.span_source"] == "gateway"
    assert a["llm.model_name"] == "claude-sonnet-4.5"
    assert a["llm.usage.total_tokens"] == 30
    assert a["tenant.id"] == "acme"

    span = spans.get_finished_spans()[0]
    assert span.name == "llm.analyst"
    # The duration is REAL, not the instant of reporting. A zero-length span
    # here would be worse than none — it would drag every latency percentile
    # computed from these spans toward zero.
    elapsed_ms = (span.end_time - span.start_time) / 1e6
    assert 200 <= elapsed_ms < 5000, elapsed_ms


def test_without_a_start_time_no_synthetic_span_is_invented(spans, gateway):
    """No start time means no honest duration, and a span reading as
    instantaneous corrupts the percentiles it feeds. Better absent."""
    gateway._record_span_attributes(
        "analyst", "m", None, None, 1.0, input_tokens=1, output_tokens=1
    )
    assert list(spans.get_finished_spans()) == []


def test_attributes_land_on_the_enclosing_span(spans, gateway):
    """The regression the dead guard caused: inside an `agent_span` the
    gateway's facts must actually appear on it, not be swallowed."""
    with agent_span("kyc.screen", tenant_id="acme"):
        gateway._record_span_attributes(
            "analyst", "m", "downgrade", "wf-9", 0.5, input_tokens=2, output_tokens=3
        )
    a = _only_span(spans)
    assert a["agent.step"] == "kyc.screen"
    assert a["llm.gateway.degrade_reason"] == "downgrade"
    assert a["workflow.id"] == "wf-9"
    assert a["llm.usage.total_tokens"] == 5
