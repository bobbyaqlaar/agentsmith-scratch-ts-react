"""
runtime/test/test_telemetry_wiring.py — a worker process ends up with a REAL
meter, not a proxy.

THE DEFECT. `runtime/metrics.py` shipped counters and histograms for calls,
cache hit/miss, retries, cost and retrieval; every call site in the gateway and
the vector store was correctly placed and correctly attributed; and
`configure_metrics()` had NO CALLER ANYWHERE — not runtime/worker.py, not KYC
Sentinel's worker, not examples/oil-price-agent. Its only three mentions in the
repo were its own definition, its own docstring, and one line of
docs/observability-audit.md.

Without a MeterProvider, `opentelemetry.metrics.get_meter()` returns a
`_ProxyMeter` whose instruments buffer for a real provider that never arrives.
Nothing raises, nothing logs. So the error rate, the cache hit ratio and the
TTFT percentiles — the four numbers the audit says spans are the wrong
instrument for — were computable nowhere, while §5 of that audit read
"✅ Fixed — a meter alongside the tracer".

WHY test_metrics.py DID NOT CATCH IT. It installs its own MeterProvider and an
InMemoryMetricReader in a fixture, then asserts the instruments record. That
proves the instruments work WHEN a provider exists. Nothing proved one ever
did — review-levers: declared-vs-enforced and test-the-contract (not
the helper), which is the same pairing that let pillar 3 pass while unenforced.

IN A SUBPROCESS, deliberately. OTel's global providers are one-shot per
process, so asserting in-process would both poison test_metrics.py's fixture
and prove less: what matters is that a FRESH process — which is what a worker
is — comes up with a real meter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _in_subprocess(body: str, env_extra: dict | None = None) -> str:
    import os

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    # A real endpoint must never be contacted from a unit test.
    for var in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "AGENT_PHOENIX_ENDPOINT",
    ):
        env.pop(var, None)
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, env=env, timeout=120, check=False
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout.strip()


def test_configure_telemetry_installs_a_real_meter_provider() -> None:
    out = _in_subprocess(
        "from runtime.tracing import configure_telemetry\n"
        "configure_telemetry()\n"
        "from opentelemetry import metrics\n"
        "from opentelemetry.sdk.metrics import MeterProvider\n"
        "p = metrics.get_meter_provider()\n"
        "print(type(p).__name__, isinstance(p, MeterProvider))\n"
    )
    name, is_sdk = out.split()
    assert is_sdk == "True", (
        f"the global meter provider is {name!r}, not an SDK MeterProvider — "
        "every counter in runtime/metrics.py is writing into a proxy"
    )


def test_the_gateways_instruments_are_not_proxies_after_configure() -> None:
    """The property as a caller meets it: record a call, then look at what the
    instrument actually is. A `_ProxyCounter` is what "no provider" looks like,
    and it is indistinguishable from a working counter at the call site."""
    out = _in_subprocess(
        "from runtime.tracing import configure_telemetry\n"
        "configure_telemetry()\n"
        "from runtime import metrics\n"
        "metrics.record_llm_call(tenant_id='t', model='m', role='judge',\n"
        "                        outcome='success', cost_usd=1.0)\n"
        "metrics.record_cache(tenant_id='t', hit=True)\n"
        "names = sorted(type(v).__name__ for v in metrics._INSTRUMENTS.values())\n"
        "print(','.join(names))\n"
    )
    assert out, "no instruments were created at all"
    assert "Proxy" not in out, (
        f"instruments are still proxies after configure_telemetry(): {out}"
    )


def test_without_configuration_they_are_proxies() -> None:
    """The control. If this ever stops being true the test above proves nothing
    — it would be asserting a property the runtime has for free."""
    out = _in_subprocess(
        "from runtime import metrics\n"
        "metrics.record_cache(tenant_id='t', hit=True)\n"
        "print(','.join(sorted(type(v).__name__ for v in metrics._INSTRUMENTS.values())))\n"
    )
    assert "Proxy" in out, (
        f"expected proxy instruments with no provider installed, got {out!r}"
    )


def test_tracing_is_still_installed_by_the_same_call() -> None:
    out = _in_subprocess(
        "from runtime.tracing import configure_telemetry\n"
        "configure_telemetry()\n"
        "from opentelemetry import trace\n"
        "from opentelemetry.sdk.trace import TracerProvider\n"
        "print(isinstance(trace.get_tracer_provider(), TracerProvider))\n"
    )
    assert out == "True"


@pytest.mark.parametrize(
    "module", ["runtime.worker", "examples/oil-price-agent/worker.py"]
)
def test_every_worker_entrypoint_configures_telemetry(module: str) -> None:
    """A sweep, because the defect was an absent CALL, not wrong code.

    Reading the source rather than importing: these entrypoints connect to
    Temporal and bind ports.

    THE FRAMEWORK'S OWN ENTRYPOINTS ONLY. This swept a sibling
    `../KYC_Sentinel/worker.py` too, which skipped on every CI runner — the
    framework's CI does not check the tenant out, so that leg had never run
    where it mattered. A tenant asserting its own wiring belongs in the
    tenant's suite; see KYC Sentinel's `test/test_worker_wiring.py`.
    """
    if module == "runtime.worker":
        path = ROOT / "runtime" / "worker.py"
    else:
        path = ROOT / "examples" / "oil-price-agent" / "worker.py"

    text = path.read_text(encoding="utf-8")
    assert "configure_telemetry()" in text, (
        f"{path.name} starts a worker without installing telemetry — this is "
        "how metrics came to have no provider in any deployment"
    )


@pytest.mark.parametrize(
    "module", ["runtime.worker", "examples/oil-price-agent/worker.py"]
)
def test_env_is_loaded_before_telemetry_reads_it(module: str) -> None:
    """`.env` must be loaded BEFORE `configure_telemetry()` resolves an endpoint.

    The endpoint comes from the environment, so a worker that configures
    telemetry first exports nowhere and says nothing about it — it is a
    correctly-installed provider with no destination, which looks identical to a
    working one until someone goes looking for the traces.

    `runtime/worker.py` and KYC's worker both gained `load_env_file()` when the
    runtime was found not to read `.env` at all. The oil-price example did not,
    and it is the file a tenant copies — so the omission propagated by design
    until telemetry made it bite.
    """
    if module == "runtime.worker":
        path = ROOT / "runtime" / "worker.py"
    else:
        path = ROOT / "examples" / "oil-price-agent" / "worker.py"

    # From the AST, not from the text. The first version of this test compared
    # `text.index(...)` and failed on this very file, because the comment
    # EXPLAINING the ordering names `configure_telemetry()` above the
    # `load_env_file()` call it is explaining. A function named in a comment is
    # not a call — the same rule the orphan sweep in scripts/test/ was written
    # around, and it took about ten minutes to break it.
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, node.lineno)

    assert "load_env_file" in calls, f"{path.name} never loads .env"
    assert "configure_telemetry" in calls, f"{path.name} never configures telemetry"
    assert calls["load_env_file"] < calls["configure_telemetry"], (
        f"{path.name} configures telemetry at line {calls['configure_telemetry']} "
        f"but loads .env at line {calls['load_env_file']}, so the OTLP endpoint "
        "declared in .env is invisible to it"
    )
