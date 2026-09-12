"""
runtime/test/test_metrics.py — counters and histograms, which this framework
had none of.

Everything was spans. Error rate, cache hit ratio and p95 TTFT were therefore
obtainable only by scanning spans: expensive at volume, WRONG under sampling —
a sampled trace store answers "what fraction failed?" with the fraction of what
it happened to keep — and worse as traffic grows. The cache hit ratio could not
be computed at all: the gateway knew whether it hit and only logged it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import runtime.metrics as m


@pytest.fixture(scope="module")
def _reader():
    """ONE MeterProvider for the module.

    `set_meter_provider` is one-shot exactly like `set_tracer_provider`: the
    first call wins and later ones are ignored with a warning. A per-test
    provider therefore leaves every test after the first reading a reader that
    was never wired — which is how four of these passed individually and failed
    together, and the same trap the tracing conftest exists to remove.
    """
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    assert metrics.get_meter_provider() is provider, (
        "another MeterProvider was installed first — these assertions would "
        "read an unwired reader and pass over nothing"
    )
    m._reset_cache()
    return reader


@pytest.fixture
def collected(_reader):
    """The shared reader. Instruments are CUMULATIVE, so tests must not share
    an attribute set — each uses its own tenant id and filters on it."""
    return _reader


def _points(reader, name, tenant=None):
    """Data points for one instrument, optionally narrowed to one tenant.

    The narrowing is not cosmetic: the reader is cumulative and shared, so a
    test that matched every point would see its neighbours' recordings too.
    """
    data = reader.get_metrics_data()
    out = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != name:
                    continue
                for point in metric.data.data_points:
                    if tenant is None or point.attributes.get("tenant.id") == tenant:
                        out.append(point)
    return out


def test_a_call_increments_a_counter_with_an_outcome(collected):
    """`outcome` is what makes an error RATE computable without scanning
    anything — success and failure land on the same series."""
    m.record_llm_call(
        tenant_id="t-outcome", model="m", role="analyst", outcome="success", cost_usd=0.5
    )
    m.record_llm_call(tenant_id="t-outcome", model="m", role="analyst", outcome="failed")

    points = _points(collected, "agentsmith.llm.calls", "t-outcome")
    outcomes = {p.attributes["outcome"]: p.value for p in points}
    assert outcomes == {"success": 1, "failed": 1}


def test_the_cache_ratio_is_computable_at_all(collected):
    """It was only ever logged, so no backend could divide it."""
    m.record_cache(tenant_id="t-cache", hit=True)
    m.record_cache(tenant_id="t-cache", hit=True)
    m.record_cache(tenant_id="t-cache", hit=False)

    by_hit = {p.attributes["hit"]: p.value
              for p in _points(collected, "agentsmith.llm.cache", "t-cache")}
    assert by_hit == {True: 2, False: 1}


def test_unreported_tokens_are_not_recorded_as_zero(collected):
    """A streamed call reports no usage and carries 0/0. Adding that zero to a
    histogram would drag every percentile toward zero while looking like real
    data — the same distinction the span attributes make with
    `llm.usage.reported`."""
    m.record_llm_call(
        tenant_id="t-tokens", model="m", role="analyst", outcome="success",
        input_tokens=None, output_tokens=None,
    )
    assert _points(collected, "agentsmith.llm.input_tokens", "t-tokens") == []

    m.record_llm_call(
        tenant_id="t-tokens", model="m", role="analyst", outcome="success",
        input_tokens=100, output_tokens=20,
    )
    assert _points(collected, "agentsmith.llm.input_tokens", "t-tokens")[0].sum == 100


def test_attributes_stay_low_cardinality(collected):
    """A metric attribute with unbounded values creates a time series per value
    and takes the backend down. Run ids, prompt hashes and trace ids belong on
    spans, which carry them already."""
    m.record_llm_call(tenant_id="t-card", model="m", role="analyst", outcome="success")
    attrs = set(_points(collected, "agentsmith.llm.calls", "t-card")[0].attributes)
    assert attrs <= {"tenant.id", "llm.model_name", "llm.gateway.tier", "outcome", "degraded"}


def test_retrieval_is_measured_too(collected):
    m.record_retrieval(tenant_id="t-retr", backend="memory", hits=3, duration_ms=12.0)
    assert _points(collected, "agentsmith.retrieval.queries", "t-retr")[0].value == 1
    assert _points(collected, "agentsmith.retrieval.hits", "t-retr")[0].sum == 3


def test_everything_no_ops_without_a_meter_provider(monkeypatch):
    """Tenant code records unconditionally; a missing SDK must not raise into
    the path being measured."""
    m._reset_cache()
    monkeypatch.setattr(m, "_meter", lambda: None)
    m.record_llm_call(tenant_id="a", model="m", role="r", outcome="success")
    m.record_cache(tenant_id="a", hit=True)
    m.record_retrieval(tenant_id="a", backend="memory", hits=0)


def test_a_broken_instrument_never_raises(collected, monkeypatch):
    class Exploding:
        def add(self, *a, **k):
            raise RuntimeError("backend down")

        def record(self, *a, **k):
            raise RuntimeError("backend down")

    monkeypatch.setattr(m, "_instrument", lambda *a, **k: Exploding())
    m.record_llm_call(tenant_id="a", model="m", role="r", outcome="success", cost_usd=1.0)
    m.record_cache(tenant_id="a", hit=False)


def test_a_retry_is_counted_with_a_bounded_reason(collected):
    """The message goes on the span event; the counter gets a class. A metric
    attribute carrying the provider's free text creates a time series per
    distinct string and takes the backend down."""
    m.record_retry(tenant_id="t-retry", model="m", attempt=2, reason="rate_limit")
    m.record_retry(tenant_id="t-retry", model="m", attempt=3, reason="rate_limit")
    m.record_retry(tenant_id="t-retry", model="m", attempt=2, reason="timeout")

    points = _points(collected, "agentsmith.llm.retries", "t-retry")
    by_reason = {p.attributes["reason"]: p.value for p in points}
    assert by_reason == {"rate_limit": 2, "timeout": 1}
    for point in points:
        assert set(point.attributes) == {"tenant.id", "llm.model_name", "reason"}
