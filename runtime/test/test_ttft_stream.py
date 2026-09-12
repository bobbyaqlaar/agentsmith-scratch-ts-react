from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import CompletionResult, LLMGateway
from runtime.test._gateway_fixtures import fake_gateway


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


SSE = [
    'data: {"choices":[{"delta":{"content":"Hi"}}]}',
    'data: {"choices":[{"delta":{"content":"!"}}]}',
    "data: [DONE]",
]


@pytest.mark.asyncio
async def test_complete_stream_sets_ttft_ms() -> None:
    gw = LLMGateway.__new__(LLMGateway)
    gw.tenant_id = "t"
    gw.models = {
        "developer": {
            "id": "test-model",
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434/v1",
        }
    }
    gw.budget_cap_usd = 10.0
    gw._idempotency = None
    gw.get_budget_status = MagicMock(return_value={"ok": True, "remaining_usd": 10})
    gw._resolve_role = MagicMock(return_value=("developer", None))
    gw._coerce_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    gw._record_span_attributes = MagicMock()
    gw._report_run_status = MagicMock()
    gw._degrade_chain = MagicMock(return_value=["developer"])
    gw._is_free_tier = MagicMock(return_value=True)

    fake = _FakeStreamResp(SSE)
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=fake)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await gw.complete_stream("hi", model_hint="developer")

    assert isinstance(result, CompletionResult)
    assert "Hi" in result.text
    assert result.ttft_ms is not None
    assert result.ttft_ms >= 0


def _stream_gateway(*, free_tier: bool) -> LLMGateway:
    """Shared builder with a priced Ollama route — TTFT accounting needs
    non-zero token costs, so the default all-zero model will not do."""
    return fake_gateway(
        model={
            "id": "test-model",
            "provider": "ollama",
            "endpoint": "http://127.0.0.1:11434/v1",
            "cost_per_input_token": 0.01,
            "cost_per_output_token": 0.02,
        },
        free_tier=free_tier,
        coerce_messages=[{"role": "user", "content": "hi"}],
    )


@pytest.mark.asyncio
async def test_complete_stream_keeps_budget_reservation_on_success() -> None:
    gw = _stream_gateway(free_tier=False)
    mock_budget = MagicMock()
    mock_budget.try_reserve = MagicMock(return_value=True)
    gw._budget = mock_budget

    fake = _FakeStreamResp(SSE)
    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=fake)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await gw.complete_stream("hi", model_hint="developer", max_tokens=100)

    expected_cost = 100 * (0.01 + 0.02)
    mock_budget.try_reserve.assert_called_once()
    mock_budget.add_spend.assert_not_called()
    assert result.cost_usd == expected_cost
    gw._record_span_attributes.assert_called_once()
    assert gw._record_span_attributes.call_args[0][4] == expected_cost
