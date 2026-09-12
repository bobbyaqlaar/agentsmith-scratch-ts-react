"""
runtime/test/test_moderation_declared.py — declared moderation hook
(TestbedFeedback-2026-07-21 G10).

`MODERATION_HOOK=required` used to be unsatisfiable: the SEC-MOD-001 harness
runs in a different process from the worker, so an imperative
register_output_moderator() call was invisible to it and `required` failed
unconditionally — the one setting regulated tenants are told to use.

A committed declaration fixes that, but only if the RUNTIME loads the same
declaration the harness checks; otherwise CI would certify a config key
production ignores. These tests pin that binding.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import moderation as mod


@pytest.fixture(autouse=True)
def _clean():
    mod.reset_output_moderator()
    yield
    mod.reset_output_moderator()


@pytest.fixture()
def tenant(tmp_path, monkeypatch):
    """A tenant repo on sys.path with an importable classifier module."""
    (tmp_path / ".agenticframework").mkdir()
    (tmp_path / "tenant_mod.py").write_text(
        textwrap.dedent(
            """
            from runtime.moderation import ModerationResult

            def classify(text: str) -> ModerationResult:
                bad = "forbidden" in text.lower()
                return ModerationResult(allowed=not bad, reasons=["policy"] if bad else [])

            def blocks_everything(text: str) -> ModerationResult:
                return ModerationResult(allowed=False, reasons=["always"])

            def explodes(text: str) -> ModerationResult:
                raise RuntimeError("classifier is broken")

            not_callable = "oops"
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _declare(repo: Path, hook: str) -> None:
    (repo / ".agenticframework" / "tenant.yaml").write_text(
        f"tenant:\n  id: t\nmoderation:\n  hook: \"{hook}\"\n"
    )


# ── declaration resolution ───────────────────────────────────────────────────


def test_undeclared_returns_none(tenant, monkeypatch):
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    assert mod.declared_hook_path() is None
    assert mod.load_declared_moderator() is None


def test_declared_in_tenant_yaml(tenant, monkeypatch):
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    _declare(tenant, "tenant_mod:classify")
    assert mod.declared_hook_path() == "tenant_mod:classify"
    assert mod.load_declared_moderator()("clean").allowed is True


def test_env_overrides_tenant_yaml(tenant, monkeypatch):
    _declare(tenant, "tenant_mod:classify")
    monkeypatch.setenv("MODERATION_HOOK_PATH", "tenant_mod:blocks_everything")
    assert mod.declared_hook_path() == "tenant_mod:blocks_everything"


@pytest.mark.parametrize(
    "hook,fragment",
    [
        ("tenant_mod.classify", "module.path:callable"),   # missing colon
        ("tenant_mod:missing", "has no attribute"),
        ("tenant_mod:not_callable", "not callable"),
        ("no_such_module:fn", "cannot import"),
    ],
)
def test_broken_declaration_is_loud(tenant, monkeypatch, hook, fragment):
    """A broken hook must never degrade to a silent skip — that would be a
    regulated tenant running unmoderated while CI looked green."""
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    _declare(tenant, hook)
    with pytest.raises(mod.ModerationHookImportError, match=fragment):
        mod.load_declared_moderator()


# ── runtime binds to the same declaration ────────────────────────────────────


def test_runtime_auto_registers_the_declared_hook(tenant, monkeypatch):
    """The binding that makes the harness check meaningful."""
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    monkeypatch.setenv("MODERATION_HOOK", "required")
    _declare(tenant, "tenant_mod:classify")

    assert mod.get_output_moderator() is None            # nothing imperative
    result = mod.apply_output_moderation("forbidden content")
    assert result.allowed is False                        # declared hook ran
    assert mod.get_output_moderator() is not None         # and stayed registered


def test_required_still_raises_without_a_declaration(tenant, monkeypatch):
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    monkeypatch.setenv("MODERATION_HOOK", "required")
    with pytest.raises(mod.ModerationHookRequiredError):
        mod.apply_output_moderation("anything")


def test_use_declared_false_isolates_the_harness_assertion(tenant, monkeypatch):
    """The runner proves `required` rejects a hook-less tenant; that check
    must not be defeated by a tenant that HAS declared one."""
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    _declare(tenant, "tenant_mod:classify")
    with pytest.raises(mod.ModerationHookRequiredError):
        mod.apply_output_moderation("x", mode="required", use_declared=False)


def test_imperative_registration_wins(tenant, monkeypatch):
    """A worker that registers explicitly is not overridden by config."""
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    _declare(tenant, "tenant_mod:blocks_everything")
    mod.register_output_moderator(
        lambda t: mod.ModerationResult(allowed=True, reasons=[])
    )
    assert mod.apply_output_moderation("forbidden").allowed is True


def test_off_mode_never_loads_the_declaration(tenant, monkeypatch):
    monkeypatch.delenv("MODERATION_HOOK_PATH", raising=False)
    monkeypatch.setenv("MODERATION_HOOK", "off")
    _declare(tenant, "tenant_mod:blocks_everything")
    assert mod.apply_output_moderation("forbidden").allowed is True
    assert mod.get_output_moderator() is None


# ── A blocked call still happened, and still cost money ──────────────────────


def test_a_moderation_block_still_records_the_call(monkeypatch) -> None:
    """The budget is charged before output moderation runs — the provider was
    called and the tokens were paid for. Re-raising the block jumped over
    `_record_span_attributes`, which emits the span attributes AND the
    `agentsmith.llm.*` counters, so the ledger and the telemetry disagreed by
    exactly the blocked calls.

    An `outcome` dimension exists on the call counter to make an error rate
    computable without scanning spans. A moderation block is the one outcome a
    security control produces, and it was the one the counter never saw.
    """
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _gateway_fixtures import fake_gateway

    from runtime.llm_gateway import ModerationBlockedError

    gw = fake_gateway()
    recorded: dict = {}
    gw._record_span_attributes = lambda *a, **k: recorded.update(k, positional=a)

    def _block(text, raise_on_block=False):
        raise ModerationBlockedError("blocked by the tenant's classifier")

    monkeypatch.setattr("runtime.llm_gateway.apply_output_moderation", _block)

    async def _invoke(cfg, messages, max_tokens, temperature):
        return "some output", 10, 20

    gw._invoke = _invoke

    with pytest.raises(ModerationBlockedError):
        asyncio.run(gw.complete("hello"))

    assert recorded, "a blocked call emitted no span attributes at all"
    assert recorded.get("outcome") == "blocked", (
        f"the call was recorded with outcome={recorded.get('outcome')!r} — a "
        "blocked call must not be counted as a success"
    )
    assert recorded.get("input_tokens") == 10, "the tokens it was charged for are absent"


def test_the_streaming_path_records_a_blocked_call_too(monkeypatch) -> None:
    """The sibling. `complete_stream` had the identical shape — moderation
    raising over `_record_span_attributes` — and a fix applied to one of two
    identical neighbours is this codebase's most repeated defect.

    Driven through the real transport stub the TTFT tests use, so the assertion
    is about the gateway's own sequencing rather than about a stand-in for it.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from runtime.llm_gateway import ModerationBlockedError

    import test_ttft_stream as tts

    gw = tts.LLMGateway.__new__(tts.LLMGateway)
    gw.tenant_id = "t"
    gw.models = {"developer": {"id": "test-model", "provider": "ollama",
                               "endpoint": "http://127.0.0.1:11434/v1"}}
    gw.budget_cap_usd = 10.0
    gw._idempotency = None
    gw.get_budget_status = MagicMock(return_value={"ok": True, "remaining_usd": 10})
    gw._resolve_role = MagicMock(return_value=("developer", None))
    gw._coerce_messages = MagicMock(return_value=[{"role": "user", "content": "hi"}])
    gw._report_run_status = MagicMock()
    gw._degrade_chain = MagicMock(return_value=["developer"])
    gw._is_free_tier = MagicMock(return_value=True)

    recorded: dict = {}
    gw._record_span_attributes = lambda *a, **k: recorded.update(k)

    def _block(text, raise_on_block=False):
        raise ModerationBlockedError("blocked")

    monkeypatch.setattr("runtime.llm_gateway.apply_output_moderation", _block)

    fake = tts._FakeStreamResp(tts.SSE)
    client = MagicMock()
    client.stream = MagicMock(return_value=fake)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=client):
        with pytest.raises(ModerationBlockedError):
            asyncio.run(gw.complete_stream("hi", model_hint="developer"))

    assert recorded, "a blocked streamed call emitted no span attributes at all"
    assert recorded.get("outcome") == "blocked", (
        f"streaming recorded outcome={recorded.get('outcome')!r} for a blocked call"
    )
