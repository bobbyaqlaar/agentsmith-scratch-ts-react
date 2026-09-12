"""
runtime/test/test_degraded_defaults.py — the defaults that quietly downgrade a
control, and the two things they now do about it.

Several selectors in this framework default to the ephemeral or the fake
option: an in-process budget ledger, an in-process vector index, a hash
embedder with no semantic meaning. Each is the right choice for CI and a
laptop. Each is chosen by DOING NOTHING — so the deployments most likely to be
running one are the deployments that never made a decision about it, and every
one of them fails invisibly: no exception, no empty result, just a control that
has stopped meaning what it says.

The in-process budget backend is the sharpest of the three. Its docstring has
said "not for multi-worker prod fleets" since it was written, and a fleet of N
workers enforces N copies of the cap — a $150 monthly limit becomes $150 per
worker, with every worker correctly reporting itself under budget.

Separately: three selectors read three env vars for the same kind of choice and
disagreed about the same input. VECTOR_BACKEND="" fell back to its default,
while BUDGET_BACKEND="" and IDEMPOTENCY_BACKEND="" raised — and a
declared-but-empty variable is what a k8s manifest or a CI matrix produces for
an input nobody set.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime import environment as env
from runtime import llm_gateway, vector_store


@pytest.fixture(autouse=True)
def _clear_warn_state(monkeypatch):
    """These warnings are once-per-process by design; each test needs a clean one."""
    monkeypatch.setattr(env, "_degraded_warned", set())
    for var in ("BUDGET_BACKEND", "VECTOR_BACKEND", "IDEMPOTENCY_BACKEND", "EMBEDDER"):
        monkeypatch.delenv(var, raising=False)


# ── env_choice: one reading of the same input ─────────────────────────────────


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_a_declared_but_empty_selector_means_unset(monkeypatch, raw) -> None:
    """os.environ.get(var, default) substitutes only when the key is ABSENT."""
    monkeypatch.setenv("X_BACKEND", raw)
    assert env.env_choice("X_BACKEND", default="memory", allowed=("memory", "redis")) == "memory"


def test_a_typo_still_raises(monkeypatch) -> None:
    """The other half of the change. A misspelled backend must not resolve to a
    default the operator did not choose — 'redsi' is a decision that failed,
    not an absent one."""
    monkeypatch.setenv("X_BACKEND", "redsi")
    with pytest.raises(ValueError, match="redsi"):
        env.env_choice("X_BACKEND", default="memory", allowed=("memory", "redis"))


def test_a_real_value_is_honoured_case_insensitively(monkeypatch) -> None:
    monkeypatch.setenv("X_BACKEND", "REDIS")
    assert env.env_choice("X_BACKEND", default="memory", allowed=("memory", "redis")) == "redis"


def test_the_budget_selector_no_longer_crashes_on_an_empty_var(monkeypatch) -> None:
    """This raised ValueError and took gateway construction down with it."""
    monkeypatch.setenv("BUDGET_BACKEND", "")
    assert llm_gateway._make_budget_backend() is not None


# ── warn_degraded_default ─────────────────────────────────────────────────────


def test_the_in_process_spend_cap_is_an_error_in_production(monkeypatch, caplog) -> None:
    """A fleet of N workers enforces N caps, and each reports itself compliant."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.INFO):
        llm_gateway._make_budget_backend()

    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "IN PROCESS" in caplog.text


def test_the_in_process_vector_index_is_an_error_in_production(monkeypatch, caplog) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.INFO):
        vector_store.make_vector_store()

    assert any(
        r.levelno >= logging.ERROR and "VECTOR_BACKEND" in r.getMessage()
        for r in caplog.records
    )


def test_development_is_told_without_being_alarmed(monkeypatch, caplog) -> None:
    """CI and a laptop run on these by design — this must not shout there."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    with caplog.at_level(logging.INFO):
        llm_gateway._make_budget_backend()

    assert caplog.records, "the message is still worth having in development"
    assert all(r.levelno < logging.ERROR for r in caplog.records)


def test_each_degraded_default_is_announced_once(monkeypatch, caplog) -> None:
    """The call sites are constructors; a gateway per worker would repeat it."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            llm_gateway._make_budget_backend()

    assert caplog.text.count("BUDGET_BACKEND is unset") == 1


def test_the_three_warnings_do_not_suppress_each_other(monkeypatch, caplog) -> None:
    """Once PER KEY, not once per process — a deployment running all three
    defaults needs to hear about all three."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.INFO):
        llm_gateway._make_budget_backend()
        vector_store.make_vector_store()

    assert "BUDGET_BACKEND is unset" in caplog.text
    assert "VECTOR_BACKEND is unset" in caplog.text
    assert "EMBEDDER is unset" in caplog.text


# ── compatibility ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alias", ["memory", "mem", "inmemory"])
def test_every_vector_alias_still_resolves(monkeypatch, alias) -> None:
    """A tenant setting VECTOR_BACKEND=inmemory keeps working. The selector was
    unified; the accepted values were not narrowed."""
    monkeypatch.setenv("VECTOR_BACKEND", alias)
    assert isinstance(vector_store.make_vector_store(), vector_store.MemoryVectorStore)


def test_the_defaults_themselves_are_unchanged(monkeypatch) -> None:
    """Pinned deliberately. Tenants run these defaults in production today and
    the framework does not break them outside a major release — the fix is a
    warning. If a default ever changes, that is a MAJOR-version decision and
    this is where it gets made."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert type(llm_gateway._make_budget_backend()).__name__ == "_MemoryBudgetBackend"
    assert isinstance(vector_store.make_vector_store(), vector_store.MemoryVectorStore)
