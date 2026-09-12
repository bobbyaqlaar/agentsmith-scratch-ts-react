"""
runtime/test/test_tool_payloads.py — tool arguments and results on the span,
through the redactor rather than around it.

`record_tool_call` recorded the tool's name, its allow/deny outcome, its
duration and any error — and nothing about what it was given or what it
returned, which is what an investigation actually needs. The reason it was not
simply added is that tool arguments are a PII channel: a KYC sanctions lookup
takes an applicant's name.

So they go on `input.value` / `output.value`, the OpenInference names that
`trace_redactor._PAYLOAD_ATTRIBUTES` already scrubs, and they are OFF until a
tenant declares otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.tracing import (
    TOOL_INPUT_ATTRIBUTE,
    TOOL_OUTPUT_ATTRIBUTE,
    _serialise_payload,
    agent_span,
    record_tool_call,
    tool_payloads_enabled,
)


@pytest.fixture
def payloads_on(monkeypatch):
    monkeypatch.setenv("TRACE_TOOL_PAYLOADS", "1")
    yield


def _tool_span(exporter):
    return next(s for s in exporter.get_finished_spans() if ".tool." in s.name)


# ── the default ──────────────────────────────────────────────────────────────


def test_payloads_are_off_unless_declared(spans, monkeypatch):
    """A new egress channel. It routes through the redactor, but a tenant
    should say it wants tool arguments leaving the process — in the file where
    the rest of its posture is declared — rather than discover it in Phoenix."""
    monkeypatch.delenv("TRACE_TOOL_PAYLOADS", raising=False)
    assert tool_payloads_enabled() is False

    with agent_span("step"):
        record_tool_call(
            "sanctions_lookup", allowed=True, duration_ms=1.0,
            args={"name": "Jane Doe"}, result=[{"hit": True}],
        )
    attrs = dict(_tool_span(spans).attributes)
    assert TOOL_INPUT_ATTRIBUTE not in attrs
    assert TOOL_OUTPUT_ATTRIBUTE not in attrs
    # The metadata that was always recorded is unaffected.
    assert attrs["agent.tool.name"] == "sanctions_lookup"


def test_the_env_var_and_the_declaration_both_enable_it(spans, payloads_on):
    assert tool_payloads_enabled() is True
    with agent_span("step"):
        record_tool_call("t", allowed=True, duration_ms=1.0, args={"a": 1}, result="ok")
    attrs = dict(_tool_span(spans).attributes)
    assert attrs[TOOL_INPUT_ATTRIBUTE] == '{"a": 1}'
    assert attrs[TOOL_OUTPUT_ATTRIBUTE] == "ok"


# ── the names are the redactor's ─────────────────────────────────────────────


def test_the_attribute_names_are_the_ones_the_redactor_scrubs():
    """The whole design. A new name like `agent.tool.input` would have been a
    new UNSCRUBBED channel; these inherit the profile pipeline — pattern
    scrubbing in staging, truncation plus an encrypted HITL blob in
    production — without this module knowing anything about redaction."""
    from runtime.trace_redactor import _PAYLOAD_ATTRIBUTES

    assert TOOL_INPUT_ATTRIBUTE in _PAYLOAD_ATTRIBUTES
    assert TOOL_OUTPUT_ATTRIBUTE in _PAYLOAD_ATTRIBUTES


def test_the_redactor_actually_scrubs_a_tool_payload():
    """End to end through the real processor: a card number in a tool's
    arguments must not survive the staging profile.

    Run on an ISOLATED TracerProvider, never registered globally. Two
    constraints force that. A finished span's attributes are immutable, so the
    redactor cannot be driven on one captured from the exporter — it has to run
    in-flight as the processor, which is what its own docstring says. And an
    OTel processor can be added to a provider but never removed, so attaching
    it to the shared one scrubbed every later test in this file, which is how
    three of them passed alone and failed together.

    What is under test here is that THESE ATTRIBUTE NAMES reach the scrubber.
    That `record_tool_call` writes them is covered above.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from runtime.trace_redactor import TraceRedactor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(TraceRedactor(profile="staging", tenant_id="acme"))
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with provider.get_tracer("test").start_as_current_span("agent.tool.payment_lookup") as span:
        span.set_attribute(TOOL_INPUT_ATTRIBUTE, '{"card": "4111111111111111", "name": "Jane"}')
        span.set_attribute("agent.tool.name", "payment_lookup")

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert "4111111111111111" not in attrs[TOOL_INPUT_ATTRIBUTE], attrs[TOOL_INPUT_ATTRIBUTE]
    # A non-payload attribute keeps its value.
    assert attrs["agent.tool.name"] == "payment_lookup"


# ── serialisation ────────────────────────────────────────────────────────────


def test_a_huge_payload_is_capped_before_the_redactor_sees_it():
    """The redactor truncates by profile, but a tool returning a megabyte of
    JSON should never become a megabyte string in the first place — that cost
    lands on the call being traced."""
    out = _serialise_payload({"blob": "x" * 50_000})
    assert len(out) < 5_000
    assert "truncated at" in out, "a truncated payload must say so"


def test_an_unserialisable_payload_is_omitted_not_stringified_as_an_error():
    """None rather than str(exc): an exception message where a caller will read
    the tool's input is worse than an absent attribute."""
    class Hostile:
        def __repr__(self):
            raise RuntimeError("nope")

        __str__ = __repr__

    assert _serialise_payload(Hostile()) is None
    assert _serialise_payload(None) is None


def test_non_json_objects_fall_back_to_str():
    class Point:
        def __repr__(self):
            return "Point(1, 2)"

    assert _serialise_payload(Point()) == '"Point(1, 2)"'


# ── through the registry ─────────────────────────────────────────────────────


def test_the_registry_records_arguments_and_result(spans, payloads_on):
    from runtime.tool_registry import ToolRegistry

    registry = ToolRegistry(strict=False)

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry.register(add, name="add", description="Add two numbers.")

    with agent_span("step"):
        registry.invoke("add", {"a": 2, "b": 3})

    attrs = dict(_tool_span(spans).attributes)
    assert attrs[TOOL_INPUT_ATTRIBUTE] == '{"a": 2, "b": 3}'
    assert attrs[TOOL_OUTPUT_ATTRIBUTE] == "5"


def test_arguments_are_recorded_when_the_tool_raises(spans, payloads_on):
    """The case where knowing what it was given matters most. There is no
    result to record, and the absence of one is not an error."""
    from runtime.tool_registry import ToolRegistry

    registry = ToolRegistry(strict=False)

    def explode(x: int) -> int:
        """Always fails."""
        raise ValueError("boom")

    registry.register(explode, name="explode", description="Always fails.")

    with agent_span("step"):
        with pytest.raises(ValueError):
            registry.invoke("explode", {"x": 7})

    attrs = dict(_tool_span(spans).attributes)
    assert attrs[TOOL_INPUT_ATTRIBUTE] == '{"x": 7}'
    assert TOOL_OUTPUT_ATTRIBUTE not in attrs
    assert attrs["agent.tool.error"] == "ValueError"
