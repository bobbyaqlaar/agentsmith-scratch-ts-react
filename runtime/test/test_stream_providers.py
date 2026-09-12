"""
runtime/test/test_stream_providers.py — streaming across providers
(TestbedFeedback-2026-07-21 G1).

complete_stream() used to raise NotImplementedError for anthropic and every
cloud-native provider, while SPECS/OPERATIONS/CHANGELOG advertised TTFT
without caveat — so the obvious tenant design (frontier model on the
latency-critical path) could not use the latency budget at all. The KYC
Sentinel testbed hit this on its Analyst route.

These tests pin: Anthropic SSE is parsed, non-text events don't start the
TTFT clock, and a provider with no streaming surface falls back to
complete() instead of raising.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import CompletionResult, LLMGateway
from runtime.test._gateway_fixtures import fake_gateway
from runtime.provider_dispatch import parse_stream_delta, supports_streaming


class _FakeStreamResp:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 400

    async def __aenter__(self) -> "_FakeStreamResp":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    def json(self) -> dict:
        return {}

    @property
    def text(self) -> str:
        return ""

    async def aiter_lines(self):
        for line in self._lines:
            yield line


# A realistic Anthropic Messages stream: the text arrives only in
# content_block_delta; everything else is protocol noise.
ANTHROPIC_SSE = [
    'data: {"type":"message_start","message":{"usage":{"input_tokens":12}}}',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    'data: {"type":"ping"}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Risk"}}',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" LOW"}}',
    'data: {"type":"content_block_stop","index":0}',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}',
    'data: {"type":"message_stop"}',
]


# ── provider_dispatch unit level ─────────────────────────────────────────────


def test_supports_streaming_matrix():
    for p in ("openai", "groq", "ollama", "anthropic"):
        assert supports_streaming(p), p
    for p in ("vertex_ai", "azure_openai", "bedrock", "huawei_modelarts"):
        assert not supports_streaming(p), p


def test_parse_stream_delta_anthropic_only_returns_text_events():
    import json

    texts = [parse_stream_delta("anthropic", json.loads(line[len("data: "):])) for line in ANTHROPIC_SSE]
    assert texts == [None, None, None, "Risk", " LOW", None, None, None]


def test_parse_stream_delta_ignores_tool_use_json_deltas():
    """input_json_delta carries tool arguments, not assistant prose — it
    must not be counted as the first token."""
    event = {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}
    assert parse_stream_delta("anthropic", event) is None


def test_parse_stream_delta_openai_compatible():
    assert parse_stream_delta("openai", {"choices": [{"delta": {"content": "hi"}}]}) == "hi"
    assert parse_stream_delta("groq", {"choices": [{"delta": {}}]}) is None
    assert parse_stream_delta("openai", {"choices": [{"delta": {"role": "assistant"}}]}) is None


# ── gateway level ────────────────────────────────────────────────────────────


def _gateway(provider: str, **cfg_over) -> LLMGateway:
    """Shared builder, pinned to the `analyst` role this module routes through.

    `_coerce_messages` is stubbed here and not in the guardrail tests: these
    assert on transport, so the message body is irrelevant, whereas those feed
    real PII through and need the actual coercion to carry it to the scrubber.
    """
    return fake_gateway(
        role="analyst",
        model={"id": "test-model", "provider": provider, **cfg_over},
        coerce_messages=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.asyncio
async def test_anthropic_streams_and_measures_ttft(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gw = _gateway("anthropic")

    captured: dict = {}
    fake = _FakeStreamResp(ANTHROPIC_SSE)
    mock_client = MagicMock()

    def _stream(method, url, json=None, headers=None):
        captured.update(url=url, body=json, headers=headers)
        return fake

    mock_client.stream = _stream
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await gw.complete_stream("hi", model_hint="analyst")

    assert isinstance(result, CompletionResult)
    assert result.text == "Risk LOW"          # protocol frames contributed nothing
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    # Messages API shape, not chat/completions
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["body"]["stream"] is True
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_streaming_provider_exhaustion_falls_back_to_complete(monkeypatch):
    """Found running KYC Sentinel's Analyst live against a real,
    credit-exhausted Anthropic key: the streaming path's bare
    resp.raise_for_status() raises httpx.HTTPStatusError, whose str() is
    just "Client error '400 Bad Request' for url ..." — the response BODY
    ("Your credit balance is too low...") that _is_provider_exhausted()
    pattern-matches on was never in that string, so a billing/quota failure
    here used to propagate raw instead of degrading to the next tier the
    way complete()'s equivalent failure already does."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    gw = _gateway("anthropic")

    class _ErrorResp(_FakeStreamResp):
        def json(self) -> dict:
            return {"error": {"message": "Your credit balance is too low to access the Anthropic API."}}

    fake = _ErrorResp([], status_code=400)
    mock_client = MagicMock()
    mock_client.stream = lambda method, url, json=None, headers=None: fake
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    fallback = CompletionResult(
        text="degraded ok", model_used="test-model", input_tokens=1, output_tokens=1, cost_usd=0.0
    )
    gw.complete = AsyncMock(return_value=fallback)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await gw.complete_stream("hi", model_hint="analyst")

    assert result is fallback
    gw.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_cloud_provider_falls_back_to_complete_instead_of_raising():
    """G1: a models.yaml provider swap must not take the pipeline down."""
    gw = _gateway("bedrock")
    fallback = CompletionResult(
        text="ok", model_used="test-model", input_tokens=1, output_tokens=1, cost_usd=0.0
    )
    gw.complete = AsyncMock(return_value=fallback)

    result = await gw.complete_stream("hi", model_hint="analyst", max_tokens=99)

    assert result is fallback
    assert result.ttft_ms is None  # honest: no TTFT was measured
    gw.complete.assert_awaited_once()
    # Falls back BEFORE reserving budget or opening a run row — complete()
    # owns the whole call, so there is no double reservation.
    gw._report_run_status.assert_not_called()
    assert gw.complete.await_args.kwargs["max_tokens"] == 99
    assert gw.complete.await_args.kwargs["model_hint"] == "analyst"


@pytest.mark.asyncio
async def test_endpoint_resolution_shared_with_invoke(monkeypatch):
    """_resolve_endpoint is one implementation for both paths — the
    streaming copy used to omit anthropic entirely."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k1")
    monkeypatch.setenv("GROQ_API_KEY", "k2")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    resolve = LLMGateway._resolve_endpoint

    assert resolve({"provider": "anthropic"}) == ("https://api.anthropic.com", "k1")
    assert resolve({"provider": "groq"}) == ("https://api.groq.com/openai/v1", "k2")
    assert resolve({"provider": "ollama"}) == ("http://ollama.internal:11434/v1", "ollama")
    # per-model override wins
    assert resolve({"provider": "anthropic", "endpoint": "https://proxy.internal"})[0] == "https://proxy.internal"


@pytest.mark.asyncio
async def test_api_key_env_override_and_fallback(monkeypatch):
    """A per-role api_key_env (e.g. giving the judge its own Anthropic
    account, distinct from the analyst's) is used when populated, and falls
    back to the provider default when configured but not yet populated —
    a tenant can roll out a dedicated key one role at a time without either
    role going dark in between."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY_JUDGE", raising=False)
    resolve = LLMGateway._resolve_endpoint

    # Not configured at all — default env var, unchanged behavior.
    assert resolve({"provider": "anthropic"})[1] == "shared-key"

    # Configured but not populated yet — falls back rather than going empty.
    assert resolve({"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY_JUDGE"})[1] == "shared-key"

    # Configured and populated — the override wins.
    monkeypatch.setenv("ANTHROPIC_API_KEY_JUDGE", "judge-only-key")
    assert (
        resolve({"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY_JUDGE"})[1]
        == "judge-only-key"
    )
