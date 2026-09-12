"""
runtime/test/test_judging.py — shared judge primitives (TestbedFeedback G7).

These pin the exact contracts that the CI eval gate (scripts/run-evals.py)
and any tenant's per-request judge both rely on, so promoting the logic
here cannot silently change either one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.judging import (
    CitationCheck,
    citations_grounded,
    judge_independence_warning,
    outcomes_match,
    pair_parity,
    parity_violation,
    warn_if_judge_not_independent,
)


# ── citation grounding ───────────────────────────────────────────────────────


def test_all_citations_resolve():
    r = citations_grounded(["p-1", "p-2"], {"p-1", "p-2", "p-3"})
    assert r == CitationCheck(True, [], "")


def test_unresolved_citation_flagged():
    r = citations_grounded(["p-1", "ghost"], {"p-1"})
    assert not r.grounded and r.unresolved == ["ghost"]
    assert "not in retrieved set" in r.reason


def test_empty_citations_flagged_by_default():
    assert not citations_grounded([], {"p-1"}).grounded


def test_empty_citations_allowed_when_opted_out():
    assert citations_grounded([], {"p-1"}, require_at_least_one=False).grounded


# ── pair parity (CI shape) ───────────────────────────────────────────────────


def test_pair_parity_matches_and_mismatches():
    results = [
        {"pair_id": "A", "fairness": 1},
        {"pair_id": "A", "fairness": 1},
        {"pair_id": "B", "fairness": 1},
        {"pair_id": "B", "fairness": 0},
    ]
    assert pair_parity(results) == {"A": 1.0, "B": 0.0}


def test_pair_parity_omits_singletons_and_unpaired():
    results = [
        {"pair_id": "solo", "fairness": 1},
        {"fairness": 1},  # no pair_id
    ]
    assert pair_parity(results) == {}


def test_pair_parity_omits_a_pair_no_one_scored():
    """This asserted the opposite — that two unscored members are 'equal' (both
    coerced to 0) and the pair scores 1.0. That preserved run-evals' historical
    normalization, and it made the bias control report "no divergence" about
    something it had never measured.

    The function's own docstring already promised omission: "pairs with fewer
    than two SCORED members are omitted". A member with no value for the outcome
    key is not a scored member.

    run-evals filters errored cases before calling this, so the reachable case
    is a judge that answers successfully and omits the field — narrow, but a
    silent 1.0 is the wrong side to fail on for the one bar that gates bias.
    """
    results = [{"pair_id": "A", "fairness": None}, {"pair_id": "A"}]
    assert pair_parity(results) == {}


def test_pair_parity_omits_a_pair_with_one_unscored_member():
    """The asymmetric case, which the old coercion got wrong in the other
    direction: 1 against a missing value became 1 against 0 — a bias violation
    reported for a case the judge never scored."""
    results = [{"pair_id": "A", "fairness": 1}, {"pair_id": "A"}]
    assert pair_parity(results) == {}


def test_pair_parity_compares_every_member_not_just_the_first_two():
    """It compared `members[0]` and `members[1]` while accepting any count >= 2,
    so a third variant could diverge and the pair still scored 1.0. Three
    nationalities against one profile is an ordinary thing to author."""
    results = [
        {"pair_id": "A", "fairness": 1},
        {"pair_id": "A", "fairness": 1},
        {"pair_id": "A", "fairness": 0},
    ]
    assert pair_parity(results) == {"A": 0.0}


def test_pair_parity_still_passes_a_matching_triple():
    """The control on the test above: it must be the divergence that fails it,
    not the member count."""
    results = [{"pair_id": "A", "fairness": 1}] * 3
    assert pair_parity(results) == {"A": 1.0}


def test_pair_parity_custom_outcome_key():
    results = [
        {"pair_id": "A", "rating": "LOW"},
        {"pair_id": "A", "rating": "LOW"},
        {"pair_id": "B", "rating": "LOW"},
        {"pair_id": "B", "rating": "HIGH"},
    ]
    assert pair_parity(results, outcome_key="rating") == {"A": 1.0, "B": 0.0}


# ── parity_violation (per-request shape) ─────────────────────────────────────


def test_parity_violation_none_when_equal():
    assert parity_violation("LOW", "LOW") is None
    assert outcomes_match("LOW", "LOW")


def test_parity_violation_reports_the_divergence():
    msg = parity_violation("LOW", "HIGH", attribute="nationality")
    assert msg and "nationality" in msg and "'LOW'" in msg and "'HIGH'" in msg


# ── judge/actor independence (E3) ────────────────────────────────────────────


def test_independence_warning_when_same_model():
    msg = judge_independence_warning("claude-sonnet-4-6", "claude-sonnet-4-6")
    assert msg and "not independent" in msg


def test_no_warning_when_models_differ():
    assert judge_independence_warning("claude-sonnet-4-6", "claude-opus-4-8") is None


def test_no_warning_when_either_unset():
    assert judge_independence_warning(None, "x") is None
    assert judge_independence_warning("x", None) is None


def test_warn_helper_logs_only_when_not_independent(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        warn_if_judge_not_independent("m", "m")
    assert any("not independent" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn_if_judge_not_independent("actor", "judge")
    assert not caplog.records


def test_run_evals_delegates_to_the_shared_function():
    """The CI gate must call the promoted primitive, not a private copy —
    otherwise G7's 'same logic' guarantee is hollow."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_evals", Path(__file__).resolve().parents[2] / "scripts" / "run-evals.py"
    )
    run_evals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_evals)

    results = [{"pair_id": "A", "fairness": 1}, {"pair_id": "A", "fairness": 0}]
    assert run_evals._pair_parity(results) == pair_parity(results) == {"A": 0.0}
