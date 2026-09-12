from __future__ import annotations

import pytest

from runtime import moderation as mod


@pytest.fixture(autouse=True)
def _reset_moderation(monkeypatch: pytest.MonkeyPatch):
    mod.reset_output_moderator()
    monkeypatch.delenv("MODERATION_HOOK", raising=False)
    yield
    mod.reset_output_moderator()


def test_no_hook_optional_allows_text() -> None:
    result = mod.apply_output_moderation("hello world")
    assert result.allowed is True
    assert result.skipped is True


def test_required_mode_without_hook_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODERATION_HOOK", "required")
    with pytest.raises(mod.ModerationHookRequiredError):
        mod.apply_output_moderation("hello world")


def test_registered_hook_blocks() -> None:
    def classifier(text: str) -> mod.ModerationResult:
        if "badword" in text.lower():
            return mod.ModerationResult(allowed=False, reasons=["policy"])
        return mod.ModerationResult(allowed=True, reasons=[])

    mod.register_output_moderator(classifier)
    ok = mod.apply_output_moderation("clean text")
    assert ok.allowed is True
    assert ok.skipped is False

    blocked = mod.apply_output_moderation("contains badword here")
    assert blocked.allowed is False
    assert "policy" in blocked.reasons


def test_registered_hook_raise_on_block() -> None:
    mod.register_output_moderator(
        lambda t: mod.ModerationResult(allowed=False, reasons=["x"])
    )
    with pytest.raises(mod.ModerationBlockedError):
        mod.apply_output_moderation("anything", raise_on_block=True)


def test_llm_gateway_reexports_register() -> None:
    from runtime import llm_gateway

    assert hasattr(llm_gateway, "register_output_moderator")
