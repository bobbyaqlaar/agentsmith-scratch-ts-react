"""
runtime/test/test_degrade_ladder.py — budget-breach role resolution
(TestbedFeedback-2026-07-21 G2 + G3).

G2: _resolve_role used to return chain[1] — one rung — so a caller always
asking for the top role degraded to the next PAID tier and then hard-failed
its reservation. SPECS §29's rung 4 ("Local — switch to Ollama") was
unreachable whenever a paid tier sat between the caller's role and the
local one, which is the normal shape of a cost ladder. Found by the KYC
Sentinel testbed's analyst → research → intake chain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import (
    BudgetExceededError,
    BudgetStatus,
    LLMGateway,
)

BREACHED = BudgetStatus(tenant_id="t", spent_usd=6.0, cap_usd=5.0, period_start="2026-07")
HEALTHY = BudgetStatus(tenant_id="t", spent_usd=1.0, cap_usd=5.0, period_start="2026-07")

# The testbed's real shape: frontier → cheap cloud → free local.
THREE_TIER = {
    "analyst": {
        "id": "claude-sonnet-4-6",
        "provider": "anthropic",
        "cost_per_input_token": 3e-6,
        "cost_per_output_token": 1.5e-5,
        "degrade_to": "research",
    },
    "research": {
        "id": "llama-3.3-70b",
        "provider": "groq",
        "cost_per_input_token": 5.9e-7,
        "cost_per_output_token": 7.9e-7,
        "degrade_to": "intake",
    },
    "intake": {
        "id": "falcon3:3b",
        "provider": "ollama",
        "cost_per_input_token": 0.0,
        "cost_per_output_token": 0.0,
        "degrade_to": None,
    },
}


def _gw(models: dict) -> LLMGateway:
    gw = LLMGateway.__new__(LLMGateway)
    gw.tenant_id = "t"
    gw.models = models
    gw.budget_cap_usd = 5.0
    return gw


def test_no_degrade_while_budget_is_healthy():
    assert _gw(THREE_TIER)._resolve_role("analyst", HEALTHY) == ("analyst", None)


def test_breach_walks_past_the_paid_rung_to_the_free_tier():
    """G2: the whole point — analyst must reach 'intake', not stop at the
    still-paid 'research' rung."""
    assert _gw(THREE_TIER)._resolve_role("analyst", BREACHED) == ("intake", "local")


def test_breach_from_the_middle_rung_also_reaches_free():
    assert _gw(THREE_TIER)._resolve_role("research", BREACHED) == ("intake", "local")


def test_free_tier_caller_is_never_blocked_by_a_breach():
    """Using a free tier adds no spend, so a breach must not stop it."""
    assert _gw(THREE_TIER)._resolve_role("intake", BREACHED) == ("intake", None)


def test_chain_without_any_free_tier_falls_back_to_next_paid_rung():
    models = {
        "a": {"id": "a", "cost_per_input_token": 1e-5, "degrade_to": "b"},
        "b": {"id": "b", "cost_per_input_token": 1e-6, "degrade_to": None},
    }
    assert _gw(models)._resolve_role("a", BREACHED) == ("b", "downgrade")


def test_no_cheaper_tier_halts_with_alert():
    models = {"a": {"id": "a", "cost_per_input_token": 1e-5, "degrade_to": None}}
    with pytest.raises(BudgetExceededError, match="no cheaper tier"):
        _gw(models)._resolve_role("a", BREACHED)


def test_broken_chain_link_is_skipped_not_fatal():
    """degrade_to naming a role that isn't registered must not crash the
    ladder — skip it and keep walking."""
    models = {
        "a": {"id": "a", "cost_per_input_token": 1e-5, "degrade_to": "ghost"},
        "ghost": None,
        "local": {"id": "l", "provider": "ollama", "cost_per_input_token": 0.0},
    }
    models = {k: v for k, v in models.items() if v is not None}
    models["a"]["degrade_to"] = "ghost"
    # chain is [a] (ghost unresolvable) -> no rung below a
    with pytest.raises(BudgetExceededError):
        _gw(models)._resolve_role("a", BREACHED)


def test_cycle_in_degrade_chain_terminates():
    models = {
        "a": {"id": "a", "cost_per_input_token": 1e-5, "degrade_to": "b"},
        "b": {"id": "b", "cost_per_input_token": 1e-6, "degrade_to": "a"},
    }
    gw = _gw(models)
    assert gw._degrade_chain("a") == ["a", "b"]  # _seen guard stops the loop
    assert gw._resolve_role("a", BREACHED) == ("b", "downgrade")


def test_a_chain_with_no_configured_tier_refuses_rather_than_returning_nothing():
    """Every role in the degrade chain missing from the registry.

    The loop `continue`s past each one, so it completes having made no attempt
    and caught no exception — and the "all tiers exhausted" branch never fires,
    because nothing failed. `text` was still None, and that fell through to
    output moderation and into a CompletionResult: a caller received an answer
    object for a call that never happened.

    Reachable, despite a first version of the guard's comment calling it
    unreachable — a `degrade_to` pointing at a role that does not exist is
    enough, which is a one-character typo in models.yaml.
    """
    import asyncio

    from _gateway_fixtures import fake_gateway

    gw = fake_gateway(role="developer")
    # The chain names a role the registry does not have. Nothing to call.
    gw._degrade_chain = lambda _role: ["nonexistent-role"]

    with pytest.raises(RuntimeError, match="No configured model tier"):
        asyncio.run(gw.complete("hello", model_hint="developer"))
