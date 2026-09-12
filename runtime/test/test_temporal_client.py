"""
runtime/test/test_temporal_client.py — one way to reach Temporal.

Seven call sites connected independently and disagreed in three ways at once:
the address default, whether TEMPORAL_TLS was read at all, and whether the
connect was bounded. The TLS half was a live defect — OPERATIONS.md documented
`TEMPORAL_TLS="1"` while the only code reading it compared against `"true"`, so
following the documentation produced `use_tls=False` and TLS was silently off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.temporal_client import (
    DEFAULT_ADDRESS,
    temporal_address,
    tls_enabled,
)


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on", " 1 "])
def test_every_documented_spelling_enables_tls(value: str) -> None:
    """`"1"` is what OPERATIONS.md documents; `"true"` is what the example
    scripts checked for. Only one of them worked, and it was not the
    documented one — so a deployment configured from the docs connected to a
    TLS-terminating endpoint without TLS, and nothing reported it."""
    assert tls_enabled({"TEMPORAL_TLS": value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_anything_else_leaves_tls_off(value: str) -> None:
    assert tls_enabled({"TEMPORAL_TLS": value}) is False


def test_tls_defaults_off_when_unset() -> None:
    assert tls_enabled({}) is False


def test_a_missing_address_falls_back_rather_than_raising() -> None:
    """runtime/worker.py used `os.environ["TEMPORAL_ADDRESS"]`, so an unset
    variable surfaced as a KeyError from inside worker startup instead of a
    connection error naming the host."""
    assert temporal_address({}) == DEFAULT_ADDRESS
    assert temporal_address({"TEMPORAL_ADDRESS": ""}) == DEFAULT_ADDRESS
    assert temporal_address({"TEMPORAL_ADDRESS": " temporal.internal:7233 "}) == "temporal.internal:7233"


def test_no_caller_builds_its_own_temporal_connection() -> None:
    """The consolidation, asserted structurally.

    `runtime/temporal_client` is the only place that may call
    `Client.connect`. Anything else drifts back into a per-file opinion about
    TLS and timeouts, which is what produced four different behaviours from
    one environment.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    hits = subprocess.run(
        ["git", "-C", str(repo), "grep", "-l", "Client.connect", "--",
         "runtime/", "examples/", "scripts/"],
        capture_output=True, text=True,
        check=False,
    ).stdout.split()
    allowed = {"runtime/temporal_client.py", "runtime/test/test_temporal_client.py"}
    assert set(hits) <= allowed, (
        f"direct Client.connect outside the shared connector: "
        f"{sorted(set(hits) - allowed)}. Use "
        f"`from runtime.temporal_client import connect`."
    )
