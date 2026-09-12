"""
runtime/test/test_retry_visibility.py — a retried call must not look merely slow.

The gateway has retried transient failures with backoff since tenacity was
wired in, and NOTHING said an attempt had happened: no attribute, no event, no
counter. A call retried three times was indistinguishable from one slow call,
which points every investigation at latency when the answer is quota — and on a
free-tier judge, where 429s are routine, that is the common case rather than
the rare one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import LLMGateway
from runtime.tracing import agent_span


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("IDEMPOTENCY_BACKEND", "memory")
    monkeypatch.setenv("BUDGET_BACKEND", "memory")
    return LLMGateway(tenant_id="acme")


# ── the coarse classifier ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message,expected",
    [
        ("LLM API error 429 (model='m'): rate limit exceeded", "rate_limit"),
        ("Rate limit reached for gpt-4", "rate_limit"),
        ("Read timed out after 120s", "timeout"),
        ("httpx.ConnectTimeout", "timeout"),
        ("LLM API error 503 (model='m'): upstream unavailable", "server_error"),
        ("something else entirely", "transient"),
    ],
)
def test_reason_is_a_class_not_a_message(gateway, message, expected):
    """The metric dimension must be bounded. A counter attribute carrying the
    provider's free text creates a time series per distinct string."""
    assert gateway._retry_reason(RuntimeError(message)) == expected


# ── the span event ───────────────────────────────────────────────────────────


class _State:
    """The parts of a tenacity RetryCallState the hook actually reads."""

    def __init__(self, exc, attempt=2, idle_for=1.5):
        self.attempt_number = attempt
        self.idle_for = idle_for
        self.outcome = type("O", (), {"exception": lambda _self: exc})()


def test_a_retry_adds_an_event_to_the_active_span(spans, gateway):
    hook = gateway._on_retry("claude-sonnet-4.5")
    with agent_span("llm.call"):
        hook(_State(RuntimeError("LLM API error 429: rate limit exceeded")))

    events = list(spans.get_finished_spans()[0].events)
    assert len(events) == 1
    event = events[0]
    assert event.name == "llm.retry"
    assert event.attributes["attempt"] == 2
    assert event.attributes["reason"] == "rate_limit"
    assert event.attributes["sleep_s"] == pytest.approx(1.5)
    assert "rate limit exceeded" in event.attributes["error"]


def test_the_full_message_is_on_the_event_not_the_metric(spans, gateway):
    """Cardinality lives where it is safe. The event carries the provider's
    words; the counter carries a class."""
    hook = gateway._on_retry("m")
    with agent_span("llm.call"):
        hook(_State(RuntimeError("LLM API error 429: quota for project X exhausted")))
    event = next(iter(spans.get_finished_spans()[0].events))
    assert "quota for project X exhausted" in event.attributes["error"]


def test_the_hook_never_raises_without_a_span(gateway):
    """Retrying must not depend on tracing being configured."""
    gateway._on_retry("m")(_State(RuntimeError("boom")))


def test_the_hook_survives_an_outcome_with_no_exception(gateway, spans):
    state = _State(None)
    with agent_span("llm.call"):
        gateway._on_retry("m")(state)
    event = next(iter(spans.get_finished_spans()[0].events))
    assert event.attributes["reason"] == "unknown"


# ── the attempt count, through the real decorator ────────────────────────────


@pytest.mark.asyncio
async def test_attempts_are_recorded_after_a_real_retry(spans, gateway, monkeypatch):
    """Driven through tenacity itself rather than a stand-in, so the wiring —
    before_sleep, the statistics dict, the attribute name — is what is tested."""
    import httpx

    calls = {"n": 0}

    class _FlakyClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectTimeout("simulated transient failure")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                request=httpx.Request("POST", "http://x"),
            )

    monkeypatch.setattr(httpx, "AsyncClient", _FlakyClient)
    monkeypatch.setattr(
        "tenacity.wait_exponential.__call__", lambda *a, **k: 0, raising=False
    )

    cfg = {"id": "m", "provider": "openai"}
    monkeypatch.setattr(gateway, "_resolve_endpoint", lambda c: ("http://x", "k"))

    with agent_span("llm.call"):
        await gateway._invoke(cfg, [{"role": "user", "content": "hi"}], 16, 0.0)

    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert calls["n"] == 3, "the client should have been called three times"
    assert attrs["llm.gateway.attempts"] == 3

    events = [e for e in spans.get_finished_spans()[0].events if e.name == "llm.retry"]
    assert len(events) == 2, "two sleeps between three attempts"
    assert [e.attributes["attempt"] for e in events] == [1, 2]


@pytest.mark.asyncio
async def test_a_call_that_did_not_retry_still_says_so(spans, gateway, monkeypatch):
    """1 is recorded, not omitted. "This call did not retry" is a fact on the
    span rather than the absence of one — the same reason an empty retrieval
    still emits a span."""
    import httpx

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}},
                request=httpx.Request("POST", "http://x"),
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(gateway, "_resolve_endpoint", lambda c: ("http://x", "k"))

    with agent_span("llm.call"):
        await gateway._invoke({"id": "m", "provider": "openai"},
                              [{"role": "user", "content": "hi"}], 16, 0.0)

    span = spans.get_finished_spans()[0]
    assert dict(span.attributes)["llm.gateway.attempts"] == 1
    assert [e for e in span.events if e.name == "llm.retry"] == []
