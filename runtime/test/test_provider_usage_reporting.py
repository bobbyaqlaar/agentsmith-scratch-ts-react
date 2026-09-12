"""
runtime/test/test_provider_usage_reporting.py — an absent token count is None.

The rest of this codebase works hard to keep "the provider reported none"
distinguishable from "the provider reported zero": `agent_runs.input_tokens` is
nullable and has its own test, `_record_span_attributes` writes
`llm.usage.reported`, `runtime/metrics` refuses to put an unreported count into
a histogram, and the portal renders a gap rather than a number.

All of that guarded a distinction the parsers had already thrown away —
`usage.get("prompt_tokens", 0)` — and the loss was not cosmetic: `cost_usd` is
computed from these, so a provider that omits `usage` produced a cost of exactly
$0.00 and the budget reconcile released the whole reservation.

These test the PARSERS. test_llm_gateway_budget.py tests what the gateway does
with a None, and it stubs `_invoke` — so on its own it would keep passing if the
parsers went back to returning 0. Both halves or neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.provider_dispatch import (
    parse_anthropic_completion,
    parse_cloud_response,
    parse_openai_completion,
    parse_response,
)

OPENAI_NO_USAGE = {"choices": [{"message": {"content": "hello"}}]}
OPENAI_ZERO_USAGE = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
OPENAI_REAL_USAGE = {
    "choices": [{"message": {"content": "hello"}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}

ANTHROPIC_NO_USAGE = {"content": [{"text": "hello"}]}
ANTHROPIC_ZERO_USAGE = {"content": [{"text": "hello"}], "usage": {"input_tokens": 0, "output_tokens": 0}}
ANTHROPIC_REAL_USAGE = {"content": [{"text": "hello"}], "usage": {"input_tokens": 11, "output_tokens": 7}}


def test_openai_shape_without_usage_reports_none() -> None:
    text, in_tok, out_tok = parse_openai_completion(OPENAI_NO_USAGE)
    assert text == "hello"
    assert in_tok is None and out_tok is None


def test_openai_shape_with_a_reported_zero_reports_zero() -> None:
    """The other half. If both cases returned None the fix would be the old
    bug wearing the opposite sign — a real zero must stay a real zero, or the
    budget starts charging an estimate for calls that genuinely cost nothing."""
    _, in_tok, out_tok = parse_openai_completion(OPENAI_ZERO_USAGE)
    assert in_tok == 0 and out_tok == 0


def test_openai_shape_with_a_null_usage_block_reports_none() -> None:
    """`"usage": null` is a shape real proxies emit. `data.get("usage", {})`
    returns None for it — the default only applies to a MISSING key — so the
    next `.get` raised AttributeError inside the parser."""
    _, in_tok, out_tok = parse_openai_completion({**OPENAI_NO_USAGE, "usage": None})
    assert in_tok is None and out_tok is None


def test_anthropic_shape_matches_its_sibling() -> None:
    assert parse_anthropic_completion(ANTHROPIC_NO_USAGE)[1:] == (None, None)
    assert parse_anthropic_completion(ANTHROPIC_ZERO_USAGE)[1:] == (0, 0)
    assert parse_anthropic_completion(ANTHROPIC_REAL_USAGE)[1:] == (11, 7)


@pytest.mark.parametrize(
    "provider,data",
    [
        ("openai", OPENAI_NO_USAGE),
        ("openrouter", OPENAI_NO_USAGE),
        ("groq", OPENAI_NO_USAGE),
        ("anthropic", ANTHROPIC_NO_USAGE),
    ],
)
def test_parse_response_preserves_absence_for_every_route(provider, data) -> None:
    assert parse_response(provider, data)[1:] == (None, None)


@pytest.mark.parametrize("provider", ["azure_openai", "bedrock", "huawei_modelarts"])
def test_cloud_adapters_preserve_absence_too(provider) -> None:
    """The adapters delegate to the two parsers above, which is exactly why an
    earlier fix to the module-level parser left byte-identical unhardened
    copies on those routes. Asserted per adapter so a future copy is caught."""
    data = ANTHROPIC_NO_USAGE if provider == "bedrock" else OPENAI_NO_USAGE
    assert parse_cloud_response(provider, data)[1:] == (None, None)


def test_real_counts_still_come_through() -> None:
    """A guard that drops working data gets deleted."""
    assert parse_openai_completion(OPENAI_REAL_USAGE)[1:] == (11, 7)
    assert parse_response("openai", OPENAI_REAL_USAGE)[1:] == (11, 7)


# ── Temperature on the Anthropic wire ────────────────────────────────────────
#
# It was dropped on every Anthropic-shaped route: build_request's direct
# branch, Vertex's anthropic publisher, and Bedrock's own inline copy of the
# body. So a caller asking for 0.0 got the provider default of 1.0, and the
# control that cares most — scripts/eval_judge.py pins JUDGE_TEMPERATURE = 0.0
# so grading is deterministic — was enforced on OpenAI routes and silently not
# on Claude, which is the obvious model to judge with.

from runtime.provider_dispatch import build_request, build_cloud_request

MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_the_direct_anthropic_route_sends_temperature() -> None:
    _, _, body = build_request("anthropic", "claude-x", MESSAGES, "k", 100, temperature=0.0)
    assert body["temperature"] == 0.0, "a judge pinned to 0 would grade at the default of 1.0"


def test_the_openai_route_still_does() -> None:
    _, _, body = build_request("openai", "gpt-x", MESSAGES, "k", 100, temperature=0.0)
    assert body["temperature"] == 0.0


@pytest.mark.parametrize("provider,cfg", [
    ("vertex_ai", {"project": "p", "publisher": "anthropic"}),
    ("bedrock", {}),
])
def test_every_cloud_anthropic_route_sends_temperature(provider, cfg, monkeypatch) -> None:
    """Vertex's anthropic publisher and Bedrock both build this body. Bedrock
    kept its own inline copy for one differing string, which is why the same
    omission had to be made twice more."""
    if provider == "bedrock":
        pytest.importorskip("boto3")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    try:
        _, _, body = build_cloud_request(provider, "claude-x", MESSAGES, cfg, 100, 0.0)
    except Exception as exc:  # credential acquisition, not body building
        pytest.skip(f"{provider} needs live credentials to build a request: {exc}")
    assert body["temperature"] == 0.0


def test_a_temperature_above_the_anthropic_maximum_is_clamped_loudly(caplog) -> None:
    """Anthropic's range is 0..1 and OpenAI's is 0..2, so the same config means
    different things by provider. Sending 1.5 unchanged turns a working call
    into a 400; sending it silently clamped is a substitution. It is clamped
    and said out loud."""
    with caplog.at_level("WARNING"):
        _, _, body = build_request("anthropic", "claude-x", MESSAGES, "k", 100, temperature=1.5)
    assert body["temperature"] == 1.0
    assert any("exceeds the Anthropic" in r.message for r in caplog.records)
