"""
runtime/test/test_registry_shapes.py — models.yaml supports two shapes, and
both must flatten to the same {role: cfg} contract.

The flat `models:` map conflated two questions: WHICH models exist, and WHICH
ROLE uses which. A closed-weight model could therefore only be *present* by
being *wired in*, so the framework default kept every cloud entry commented
out — readable by a human, invisible to the code, unusable without editing
YAML.

`catalog:` + `profiles:` separates them. Everything downstream (llm_gateway's
degrade chain, cost_router's tiers, _shared.judge_model) still consumes a flat
dict, which is why the resolution happens in the loader rather than at each
call site.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from runtime.llm_gateway import _roles_from_doc, _active_profile_name
from runtime.provider_dispatch import (
    API_FORMAT_ANTHROPIC,
    API_FORMAT_OPENAI,
    build_request,
    parse_response,
    resolve_api_format,
)


CATALOG_DOC = {
    "default_profile": "local",
    "catalog": {
        "qwen2.5": {"provider": "ollama", "cost_per_input_token": 0},
        "claude-sonnet-4-6": {"provider": "anthropic"},
        "openrouter-claude": {
            "id": "anthropic/claude-sonnet-4.5",
            "provider": "openrouter",
            "api_format": "openai_chat",
        },
    },
    "profiles": {
        "local": {"architect": {"use": "qwen2.5", "degrade_to": "fast"}},
        "hybrid": {"architect": "claude-sonnet-4-6"},
        "router": {"architect": "openrouter-claude"},
    },
}


def test_flat_shape_is_unchanged() -> None:
    """Existing tenant files must keep working — KYC Sentinel uses this shape."""
    doc = {"models": {"judge": {"id": "falcon3:3b", "provider": "ollama"}}}
    assert _roles_from_doc(doc) == {"judge": {"id": "falcon3:3b", "provider": "ollama"}}


def test_profile_binds_roles_to_catalog_entries(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("AI_STACK_MODE", raising=False)
    roles = _roles_from_doc(CATALOG_DOC)
    assert roles["architect"]["id"] == "qwen2.5"
    assert roles["architect"]["provider"] == "ollama"
    # Per-role extras from the binding survive alongside the catalog entry.
    assert roles["architect"]["degrade_to"] == "fast"


def test_id_defaults_to_the_catalog_key_but_an_explicit_id_wins(monkeypatch) -> None:
    """OpenRouter namespaces its ids ("anthropic/claude-sonnet-4.5"), so the
    alias a profile binds is not the id the provider expects."""
    monkeypatch.setenv("AGENT_MODEL_PROFILE", "router")
    roles = _roles_from_doc(CATALOG_DOC)
    assert roles["architect"]["id"] == "anthropic/claude-sonnet-4.5"


def test_profile_selection_precedence(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("AI_STACK_MODE", raising=False)
    assert _active_profile_name(CATALOG_DOC) == "local"      # default_profile

    monkeypatch.setenv("AI_STACK_MODE", "hybrid")
    assert _active_profile_name(CATALOG_DOC) == "hybrid"     # ai-mode-hybrid

    monkeypatch.setenv("AGENT_MODEL_PROFILE", "router")
    assert _active_profile_name(CATALOG_DOC) == "router"     # explicit wins


def test_an_unknown_mode_falls_back_rather_than_resolving_nothing(monkeypatch) -> None:
    """AI_STACK_MODE is a machine-wide shell variable and may name something
    this registry has no profile for. Silently binding zero roles would leave
    every lookup empty."""
    monkeypatch.delenv("AGENT_MODEL_PROFILE", raising=False)
    monkeypatch.setenv("AI_STACK_MODE", "some-other-mode")
    assert _active_profile_name(CATALOG_DOC) == "local"


def test_a_binding_to_a_missing_catalog_entry_fails_loudly(monkeypatch) -> None:
    """A typo'd alias must not resolve to an empty config that later reads as
    'no model configured'."""
    monkeypatch.delenv("AGENT_MODEL_PROFILE", raising=False)
    monkeypatch.delenv("AI_STACK_MODE", raising=False)
    doc = {
        "default_profile": "local",
        "catalog": {"qwen2.5": {"provider": "ollama"}},
        "profiles": {"local": {"architect": "qwen-2.5"}},  # typo
    }
    with pytest.raises(ValueError, match="not in `catalog`"):
        _roles_from_doc(doc)


# ── Wire format ──────────────────────────────────────────────────────────────


def test_api_format_defaults_from_provider() -> None:
    assert resolve_api_format({"provider": "anthropic"}) == API_FORMAT_ANTHROPIC
    assert resolve_api_format({"provider": "groq"}) == API_FORMAT_OPENAI
    assert resolve_api_format({"provider": "openrouter"}) == API_FORMAT_OPENAI


def test_explicit_api_format_overrides_the_provider_default() -> None:
    """The whole point of the field: one provider, several envelopes."""
    cfg = {"provider": "openrouter", "api_format": API_FORMAT_ANTHROPIC}
    assert resolve_api_format(cfg) == API_FORMAT_ANTHROPIC


def test_claude_via_openrouter_uses_the_openai_envelope() -> None:
    """Claude reached through OpenRouter answers in OpenAI shape. Keying the
    envelope off the vendor in the id would build an Anthropic request against
    an OpenAI-compatible endpoint and then parse the wrong response fields."""
    path, headers, body = build_request(
        provider="openrouter",
        model_id="anthropic/claude-sonnet-4.5",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        api_key="k",
        max_tokens=16,
        api_format=API_FORMAT_OPENAI,
    )
    assert path == "/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    # System stays in the messages list — it is NOT hoisted, as it would be
    # for the Anthropic Messages API.
    assert body["messages"][0]["role"] == "system"

    text, tin, tout = parse_response(
        "openrouter",
        {"choices": [{"message": {"content": "hi"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
        api_format=API_FORMAT_OPENAI,
    )
    assert (text, tin, tout) == ("hi", 3, 4)


def test_direct_anthropic_still_uses_the_messages_envelope() -> None:
    path, headers, body = build_request(
        provider="anthropic",
        model_id="claude-opus-4-8",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        api_key="k",
        max_tokens=16,
    )
    assert path == "/v1/messages"
    assert "x-api-key" in headers
    assert body["system"] == "s"
    assert all(m["role"] != "system" for m in body["messages"])


# ── Empty completions ────────────────────────────────────────────────────────


def test_null_content_parses_to_empty_string_not_none() -> None:
    """OpenAI-compatible providers legitimately return `"content": null` — a
    model that emitted only reasoning tokens, produced nothing before a stop,
    or was cut off by a filter.

    Returning None broke parse_response's own (text, int, int) contract, and
    the None travelled several frames before anything dereferenced it: the PII
    scrubber — a security control — died with
    `TypeError: expected string or bytes-like object, got 'NoneType'`.
    Found by running the KYC pipeline against live OpenRouter routes; no
    fixture produced a null completion because the fake gateway never does.
    """
    text, tin, tout = parse_response(
        "openrouter",
        {"choices": [{"message": {"content": None}}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 0}},
    )
    assert text == ""
    assert (tin, tout) == (5, 0)


def test_anthropic_empty_content_block_parses_to_empty_string() -> None:
    """Same failure mode on the Messages envelope: an empty `content` list
    would have raised IndexError instead."""
    assert parse_response("anthropic", {"content": []})[0] == ""
    assert parse_response("anthropic", {"content": None})[0] == ""


def test_pii_detection_tolerates_none() -> None:
    """Defence in depth. An empty completion is a normal provider outcome, and
    a guardrail is the worst place to discover it isn't a string."""
    from runtime.input_guardrail import detect_pii

    assert detect_pii(None) == {}
    assert detect_pii("") == {}
    assert detect_pii("card 4111 1111 1111 1111") == {"card": 1}
