"""
runtime/test/test_context_propagation.py — the request survives a process hop,
and the retrieval path is visible at all.

Before this there was no `inject`, no `extract` and no `traceparent` anywhere in
the codebase. Of the chain the framework advertises —

    API → orchestrator → vector DB → embedding → LLM → database

— only the LLM hop emitted a span, and it was stitched to its parent only when
the tenant remembered to wrap the call. The worker's run-status POST carried no
context, so the Ops Portal's work was a SEPARATE trace; `agent_runs.trace_id`
existed to correlate the two and was NULL for every run ever recorded, because
`_report_run_status` accepted a `trace_id` argument that none of its nine call
sites ever passed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.embeddings import HashEmbedder
from runtime.tenancy import agent_context
from runtime.tracing import (
    agent_span,
    current_trace_id,
    inject_context,
    traceparent,
)
from runtime.vector_store import MemoryVectorStore

TRACEPARENT = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


# ── propagation ──────────────────────────────────────────────────────────────


def test_traceparent_is_w3c_shaped_inside_a_span(spans):
    with agent_span("step"):
        parent = traceparent()
    assert parent and TRACEPARENT.match(parent), parent


def test_traceparent_matches_the_span_it_was_taken_from(spans):
    with agent_span("step"):
        parent = traceparent()
        trace_id = current_trace_id()
    emitted = spans.get_finished_spans()[0]
    assert parent.split("-")[1] == format(emitted.context.trace_id, "032x")
    assert trace_id == format(emitted.context.trace_id, "032x")


def test_nothing_to_propagate_outside_a_span():
    """None, not a zeroed traceparent. An all-zero trace id is the invalid one
    the spec reserves, and sending it would create a link to nothing."""
    assert traceparent() is None
    assert current_trace_id() is None


def test_inject_leaves_existing_headers_alone(spans):
    with agent_span("step"):
        headers = inject_context({"Authorization": "Bearer x"})
    assert headers["Authorization"] == "Bearer x"
    assert TRACEPARENT.match(headers["traceparent"])


def test_inject_is_a_no_op_with_no_active_span():
    """So a caller can wrap its headers unconditionally."""
    assert inject_context({"Authorization": "Bearer x"}) == {"Authorization": "Bearer x"}
    assert inject_context() == {}


def test_the_gateway_sends_traceparent_and_a_trace_id(spans, monkeypatch):
    """The hop that was broken: the worker's POST to the portal started its own
    trace, so the two halves of one request were two unconnected traces."""
    from runtime.llm_gateway import LLMGateway

    monkeypatch.setenv("IDEMPOTENCY_BACKEND", "memory")
    monkeypatch.setenv("BUDGET_BACKEND", "memory")
    monkeypatch.setenv("OPS_PORTAL_URL", "http://portal.invalid")
    monkeypatch.setenv("OPS_PORTAL_SYNC_TOKEN", "t")

    captured: dict = {}

    class _FakeHttpx:
        # `Timeout` as well as `post`. The gateway builds an httpx.Timeout for
        # the call, and a double carrying only `post` raised AttributeError
        # INSIDE the reporter's `except Exception`, so the POST silently did
        # not happen and this test failed with a KeyError on `captured` — the
        # symptom pointing nowhere near the cause. A double that is LESS
        # capable than the real module hides a change instead of checking it.
        Timeout = staticmethod(lambda **kwargs: ("timeout", kwargs))

        @staticmethod
        def post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers or {}
            captured["json"] = json or {}
            captured["timeout"] = timeout

    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    gw = LLMGateway(tenant_id="acme")
    with agent_span("kyc.screen"):
        gw._report_run_status("run-1", "success")

    assert TRACEPARENT.match(captured["headers"]["traceparent"])
    assert captured["json"]["traceId"] == captured["headers"]["traceparent"].split("-")[1]
    assert captured["headers"]["Authorization"] == "Bearer t"
    # Bounded, and bounded per phase. A flat multi-second timeout on a
    # fire-and-forget telemetry POST is latency the LLM call pays twice.
    assert captured["timeout"] is not None, "the report went out with no timeout at all"


# ── the retrieval hop ────────────────────────────────────────────────────────


@pytest.fixture
def store():
    s = MemoryVectorStore(HashEmbedder())
    s.add(["d1", "d2", "d3"], ["alpha text", "beta text", "gamma text"])
    return s


def test_a_retrieval_emits_a_span_at_all(spans, store):
    store.query("alpha", k=2)
    names = [s.name for s in spans.get_finished_spans()]
    assert "agent.retrieval.memory" in names


def test_the_span_records_hit_identities_not_just_a_count(spans, store):
    """`agent.tool.result_count` was all the framework recorded about a
    retrieval, and a count of 3 says nothing when the wrong three came back —
    the single most common question asked of a RAG system."""
    store.query("alpha", k=2)
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["agent.retrieval.hit_count"] == 2
    assert len(attrs["agent.retrieval.hit_ids"]) == 2
    assert attrs["agent.retrieval.top_score"] >= attrs["agent.retrieval.min_score"]
    assert attrs["agent.retrieval.corpus_size"] == 3


def test_retrieved_text_is_never_put_on_the_span(spans, store):
    """Retrieved documents are the most likely place for PII to enter a span,
    and trace_redactor runs after this."""
    store.query("alpha", k=3)
    attrs = dict(spans.get_finished_spans()[0].attributes)
    joined = " ".join(str(v) for v in attrs.values())
    assert "alpha text" not in joined
    assert "beta text" not in joined


def test_an_empty_retrieval_still_reports_that_it_ran(spans):
    """Zero hits is a result. A span that is simply absent reads as "retrieval
    did not happen", which is a different fact and the one that matters when a
    RAG answer is ungrounded."""
    empty = MemoryVectorStore(HashEmbedder())
    empty.query("anything", k=3)
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["agent.retrieval.hit_count"] == 0
    assert attrs["agent.retrieval.corpus_size"] == 0


def test_the_retrieval_span_inherits_identity_and_nests(spans, store, identity_processor):
    with agent_context(role="research", tenant_id="acme"):
        with agent_span("research.step"):
            store.query("alpha", k=1)

    by_name = {s.name: s for s in spans.get_finished_spans()}
    retrieval = by_name["agent.retrieval.memory"]
    step = by_name["agent.research.step"]
    assert retrieval.attributes["tenant.id"] == "acme"
    assert retrieval.attributes["agent.role"] == "research"
    assert retrieval.parent.span_id == step.context.span_id, "retrieval must nest"
