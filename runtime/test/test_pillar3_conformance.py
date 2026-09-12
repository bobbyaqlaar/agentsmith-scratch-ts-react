"""
runtime/test/test_pillar3_conformance.py — the runner pillar 3 never had.

Pillar 3 says every span must carry agent.name, agent.role, agent.owner_id,
tenant.id, llm.model_name, project.name and environment. Audited 2026-08-24:
`agent.role`, `agent.owner_id` and `project.name` were written ONLY by the two
demo scripts, never by the runtime a tenant executes; `tenant.id` was a kwarg
applied under `if tenant_id:`, so omitting it produced an unattributed span in
silence. Nothing checked any of it.

These tests are the check. They assert the property over EMITTED SPANS rather
than over the helper that emits them, which is the distinction that let the old
`test_tracing.py` assertion pass while the contract was broken: it asserted
`tenant.id == "acme"` on a call that had passed `tenant_id="acme"`.

Resource attributes are asserted separately from per-span ones because they
have to be — the split is the finding. A worker runs many roles (KYC registers
six activities on one task queue) and, on the shared-pool default, many
tenants, so neither can live on the Resource.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.tenancy import agent_context
from runtime.tracing import (
    agent_span,
    record_tool_call,
    resource_attributes,
)

# The per-span half. The rest is Resource-borne and checked separately.
REQUIRED_PER_SPAN = {"tenant.id", "agent.role"}


@pytest.fixture(autouse=True)
def _identity_processor(identity_processor):
    """The shared fixture from conftest, made autouse for this module — every
    test here asserts on stamped spans. It is opt-in elsewhere because most
    modules assert on spans that should NOT be stamped."""
    return identity_processor


def _all_spans(exporter):
    spans = exporter.get_finished_spans()
    assert spans, "no spans emitted — the assertions below would check nothing"
    return spans


def test_every_span_in_a_bound_context_carries_identity(spans):
    """The property pillar 3 states, over real spans, including nested ones and
    ones the caller never attributed by hand."""
    with agent_context(role="analyst", tenant_id="acme", run_id="run-1"):
        with agent_span("kyc.screen"):
            record_tool_call("sanctions_lookup", allowed=True, duration_ms=12.0)
        with agent_span("kyc.decide"):
            pass

    emitted = _all_spans(spans)
    assert len(emitted) == 3, [s.name for s in emitted]
    for span in emitted:
        missing = REQUIRED_PER_SPAN - set(span.attributes)
        assert not missing, f"{span.name} is missing {sorted(missing)}"
        assert span.attributes["tenant.id"] == "acme"
        assert span.attributes["agent.role"] == "analyst"
        assert span.attributes["run.id"] == "run-1"


def test_identity_is_not_threaded_by_hand(spans):
    """The point of the contextvar: no call site passes tenant_id, and the
    spans are attributed anyway. Every missing attribute found in the audit was
    a caller who did not pass the kwarg."""
    with agent_context(role="research", tenant_id="globex"):
        with agent_span("research.lookup"):
            pass
    span = _all_spans(spans)[0]
    assert span.attributes["tenant.id"] == "globex"
    assert span.attributes["agent.role"] == "research"


def test_one_worker_many_roles_keeps_them_distinct(spans):
    """Why agent.role cannot be a Resource attribute. This is KYC's actual
    shape: one process, six activities, one task queue. A Resource would stamp
    every span with a single role — five confident lies rather than a gap."""
    for role in ("intake", "research", "analyst"):
        with agent_context(role=role, tenant_id="acme"):
            with agent_span(f"{role}.step"):
                pass

    roles = {s.name: s.attributes["agent.role"] for s in _all_spans(spans)}
    assert roles == {
        "agent.intake.step": "intake",
        "agent.research.step": "research",
        "agent.analyst.step": "analyst",
    }


def test_context_does_not_leak_between_activities(spans):
    """A worker thread is reused. One activity's role must not survive into the
    next, or every span after the first is mislabelled."""
    with agent_context(role="intake", tenant_id="acme"):
        with agent_span("first"):
            pass
    with agent_span("after"):
        pass

    after = next(s for s in _all_spans(spans) if s.name == "agent.after")
    assert "agent.role" not in after.attributes
    assert "tenant.id" not in after.attributes


def test_an_unbound_span_is_unattributed_not_mislabelled(spans):
    """Outside any context the attributes are ABSENT. Not "unknown", not a
    default tenant — a gap is visible in a query, a plausible placeholder gets
    aggregated with real data and is not."""
    with agent_span("orphan"):
        pass
    span = _all_spans(spans)[0]
    assert "tenant.id" not in span.attributes
    assert "agent.role" not in span.attributes


def test_resource_carries_the_per_process_half() -> None:
    attrs = resource_attributes(project_name="agentsmith-test")
    assert attrs["service.name"] == "agentsmith-test"
    assert attrs["project.name"] == "agentsmith-test"
    assert attrs["environment"] in {"development", "staging", "production"}
    # The two that vary within a process must NOT be here.
    assert "agent.role" not in attrs
    assert "tenant.id" not in attrs


def test_owner_is_omitted_rather_than_guessed(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_OWNER_ID", raising=False)
    assert "agent.owner_id" not in resource_attributes(project_name="p")
    monkeypatch.setenv("AGENT_OWNER_ID", "bobby@example.com")
    assert resource_attributes(project_name="p")["agent.owner_id"] == "bobby@example.com"
