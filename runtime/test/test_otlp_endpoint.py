"""
runtime/test/test_otlp_endpoint.py — one endpoint resolver, and it agrees with
the TypeScript one it cannot import.

FOUR implementations of "turn the endpoint variable into an OTLP URL" existed,
and they disagreed:

    scripts/local_agent_stack.py   AGENT_PHOENIX_ENDPOINT -> OTEL_EXPORTER_OTLP_ENDPOINT
    scripts/multi_agent_system.py  AGENT_PHOENIX_ENDPOINT only, localhost default
    KYC_Sentinel/worker.py         AGENT_PHOENIX_ENDPOINT only, no default
    portal/lib/tracing.ts          the full chain, and the only correct one

Every Python copy ended `f"{endpoint.rstrip('/')}/v1/traces"`. The portal's did
not, because this repo's own convention — OPERATIONS.md, docker-compose.yml,
SPECS.md §699, `ai-dashboard-start` — sets OTEL_EXPORTER_OTLP_ENDPOINT to a full
`…/v1/traces` URL in the variable the spec defines as a BASE. `local_agent_stack`
falls back to precisely that variable and appends anyway, giving
`/v1/traces/v1/traces` and a 404 nothing surfaces.

The portal's version is the canonical one, ported to runtime/otlp.py. The TS
copy CANNOT be deleted — different language, different process — so the last
test here PARSES it rather than restating it (review-levers: pin-unremovable-duplicates — a duplicate
that cannot be removed must be pinned, and a test that hardcodes the second copy
is just a third copy).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from runtime.otlp import SIGNAL_PATHS, resolve_otlp_endpoint

ROOT = Path(__file__).resolve().parents[2]


def test_nothing_configured_is_none_not_a_localhost_guess() -> None:
    """Two of the four copies defaulted to http://localhost:6006, so a
    production worker with no endpoint set spent every export attempt on a
    connection to itself."""
    assert resolve_otlp_endpoint("traces", {}) is None
    assert resolve_otlp_endpoint("metrics", {}) is None


def test_a_base_url_gains_the_signal_path() -> None:
    assert (
        resolve_otlp_endpoint("traces", {"AGENT_PHOENIX_ENDPOINT": "http://p:6006"})
        == "http://p:6006/v1/traces"
    )
    assert (
        resolve_otlp_endpoint("metrics", {"AGENT_PHOENIX_ENDPOINT": "http://p:6006/"})
        == "http://p:6006/v1/metrics"
    )


def test_a_url_that_already_names_the_path_is_not_doubled() -> None:
    """THE TRAP. This is the framework's own documented value."""
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:6006/v1/traces"}
    assert resolve_otlp_endpoint("traces", env) == "http://localhost:6006/v1/traces"


def test_a_base_naming_another_signal_does_not_stack_paths() -> None:
    """The case the portal's version never had to handle, because it only ever
    resolved traces. Asking for metrics with the documented traces URL set must
    not produce /v1/traces/v1/metrics — a bug that only becomes reachable once
    the two signals share a resolver."""
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:6006/v1/traces"}
    assert resolve_otlp_endpoint("metrics", env) == "http://localhost:6006/v1/metrics"


def test_the_per_signal_variable_is_used_verbatim() -> None:
    """An operator naming the exact URL is the one person who read the spec."""
    env = {
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "https://collector/custom/path",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://ignored:6006",
    }
    assert resolve_otlp_endpoint("traces", env) == "https://collector/custom/path"


def test_precedence_base_over_framework_name() -> None:
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://spec:4318",
        "AGENT_PHOENIX_ENDPOINT": "http://framework:6006",
    }
    assert resolve_otlp_endpoint("traces", env) == "http://spec:4318/v1/traces"


def test_the_framework_name_still_works_alone() -> None:
    """Every workflow template and CI secret in this repo sets this one.
    Dropping it in favour of the spec names would silently stop exporting for
    every existing deployment."""
    env = {"AGENT_PHOENIX_ENDPOINT": "http://phoenix:6006"}
    assert resolve_otlp_endpoint("traces", env) == "http://phoenix:6006/v1/traces"


def test_an_unknown_signal_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown OTLP signal"):
        resolve_otlp_endpoint("profiles", {"AGENT_PHOENIX_ENDPOINT": "http://p:6006"})


# ── The duplicate that cannot be removed, pinned ─────────────────────────────


def _portal_resolver_source() -> str:
    src = (ROOT / "portal" / "lib" / "tracing.ts").read_text(encoding="utf-8")
    start = src.index("export function resolveTracesEndpoint")
    return src[start : src.index("\n}", start)]


def test_the_portal_resolver_reads_the_same_variables_in_the_same_order() -> None:
    """Parsed out of the TypeScript, not restated here.

    If the portal gains a variable, or reorders them, this fails and someone
    has to decide — rather than the two halves of one contract drifting until a
    deployment exports traces from the worker and nothing from the portal.
    """
    body = _portal_resolver_source()
    found = re.findall(r"env\.([A-Z_]+)", body)
    # Deduplicate, keep order of first appearance.
    ordered = list(dict.fromkeys(found))
    assert ordered == [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "AGENT_PHOENIX_ENDPOINT",
    ], f"portal precedence changed: {ordered}"

    python_order = ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT", "AGENT_PHOENIX_ENDPOINT"]
    assert ordered == python_order


def test_the_portal_resolver_still_guards_the_doubled_suffix() -> None:
    """The one behaviour that made the TS copy the canonical one. If it is ever
    removed there, `runtime/otlp.py` keeping the guard would be the drift."""
    body = _portal_resolver_source()
    assert "/v1/traces" in body and "endsWith" in body, (
        "portal/lib/tracing.ts no longer detects an endpoint that already names "
        "the traces path — runtime/otlp.py still does, so the two have drifted"
    )


def test_no_module_hand_rolls_the_signal_path_any_more() -> None:
    """A sweep over the callers that each used to own this.

    The defect was four copies, so the guard has to be about copies — not about
    whether resolve_otlp_endpoint returns the right string, which the tests
    above already cover.
    """
    sources = [
        ROOT / "scripts" / "local_agent_stack.py",
        ROOT / "scripts" / "multi_agent_system.py",
        ROOT / "runtime" / "tracing.py",
        ROOT / "runtime" / "metrics.py",
        ROOT / "examples" / "oil-price-agent" / "worker.py",
    ]
    present = [p for p in sources if p.exists()]
    assert len(present) == len(sources), f"the sweep lost a file: {sources}"

    offenders = []
    for path in present:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue  # prose about the rule is not a breach of it
            for signal_path in SIGNAL_PATHS.values():
                if signal_path in stripped:
                    offenders.append(f"{path.name}:{n}: {stripped}")
    assert not offenders, (
        "these build an OTLP path themselves instead of calling "
        "runtime.otlp.resolve_otlp_endpoint:\n  " + "\n  ".join(offenders)
    )
