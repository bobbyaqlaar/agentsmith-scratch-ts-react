"""
runtime/test/test_tracing.py — agent-step + tool-call spans
(TestbedFeedback-2026-07-21 G8).

Two guarantees: (1) with a real in-memory tracer, tool calls and agent
steps produce the documented attributes; (2) with no tracer configured
(the common case in unit tests and any un-instrumented deployment), all of
it is a clean no-op that never changes behavior or raises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.tool_registry import (
    ToolNotAllowedError,
    ToolRegistry,
    tool,
)
from runtime.tracing import agent_span, record_tool_call


# ── No-op path (no OTel tracer configured) ───────────────────────────────────


def test_agent_span_noops_without_a_tracer():
    with agent_span("step.x", tenant_id="t", foo="bar") as span:
        span.set_attribute("anything", 1)  # must not raise
    assert span.is_recording() is False


def test_record_tool_call_noops_without_a_tracer():
    record_tool_call("x", allowed=True, duration_ms=1.0)  # must not raise


def test_agent_span_reraises_but_still_noops(caplog):
    with pytest.raises(ValueError):
        with agent_span("step.boom"):
            raise ValueError("business error")


def test_tool_registry_invoke_unchanged_without_tracer():
    reg = ToolRegistry(strict=False)

    @tool("adder", registry=reg)
    def adder(a: int, b: int) -> int:
        return a + b

    assert reg.invoke("adder", {"a": 2, "b": 3}) == 5
    with pytest.raises(ToolNotAllowedError):
        strict = ToolRegistry(strict=True)  # allowlist loaded = None → deny all
        strict.register(adder, name="adder")
        strict.invoke("adder", {"a": 1, "b": 1})


# ── Real in-memory tracer ────────────────────────────────────────────────────


# `_exporter` and `spans` live in conftest.py — the global tracer provider is
# one-shot, so installing it per module made whichever module lost the race
# skip its span assertions silently.


def _attrs(exporter, span_name):
    for s in exporter.get_finished_spans():
        if s.name == span_name:
            return dict(s.attributes)
    raise AssertionError(f"span {span_name!r} not found in {[s.name for s in exporter.get_finished_spans()]}")


def test_agent_span_records_attributes_and_duration(spans):
    with agent_span("research.lookup", tenant_id="acme", kind="tool", result_count=3) as span:
        span.set_attribute("agent.custom", "v")
    a = _attrs(spans, "agent.research.lookup")
    assert a["agent.step"] == "research.lookup"
    assert a["agent.kind"] == "tool"
    assert a["tenant.id"] == "acme"
    assert a["agent.result_count"] == 3
    assert a["agent.custom"] == "v"
    assert a["agent.duration_ms"] >= 0


def test_agent_span_stamps_error_and_reraises(spans):
    with pytest.raises(RuntimeError):
        with agent_span("step.fails"):
            raise RuntimeError("boom")
    assert _attrs(spans, "agent.step.fails")["agent.error"] == "RuntimeError"


def test_each_tool_call_gets_its_own_child_span(spans):
    """Several tools in one step must each produce a span — annotating the
    enclosing step would let them clobber each other's attributes."""
    reg = ToolRegistry(strict=False)

    @tool("sanctions", registry=reg)
    def sanctions(name: str) -> list:
        return [name]

    @tool("registry_lookup", registry=reg)
    def registry_lookup(company: str) -> dict:
        return {"company": company}

    with agent_span("research"):
        reg.invoke("sanctions", {"name": "acme"})
        reg.invoke("registry_lookup", {"company": "acme"})

    a = _attrs(spans, "agent.tool.sanctions")
    assert a["agent.tool.name"] == "sanctions" and a["agent.tool.allowed"] is True
    b = _attrs(spans, "agent.tool.registry_lookup")
    assert b["agent.tool.name"] == "registry_lookup"


def test_denied_tool_records_allowed_false(spans):
    reg = ToolRegistry(strict=True)  # allowlist None → deny by default

    @tool("wire_transfer", registry=reg)
    def wire_transfer(to: str) -> None:
        raise AssertionError("must not execute")

    with agent_span("research"):
        with pytest.raises(ToolNotAllowedError):
            reg.invoke("wire_transfer", {"to": "x"})

    a = _attrs(spans, "agent.tool.wire_transfer")
    assert a["agent.tool.allowed"] is False
    assert a["agent.tool.error"] == "ToolNotAllowedError"


def test_tool_call_outside_a_step_emits_no_lone_span(spans):
    """A tool call with no active step span shouldn't create a root span —
    that would be noise. It runs; it just isn't traced on its own."""
    reg = ToolRegistry(strict=False)

    @tool("solo", registry=reg)
    def solo() -> int:
        return 1

    assert reg.invoke("solo", {}) == 1
    assert not any(s.name == "agent.tool.solo" for s in spans.get_finished_spans())
