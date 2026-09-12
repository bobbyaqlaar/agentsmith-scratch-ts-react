"""
runtime/test/test_input_guardrail.py — pre-call PII scrubbing
(Product_Archive.md Security & Guardrails / UAE PDPL).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import input_guardrail as ig


@pytest.fixture(autouse=True)
def _reset_guardrail_state(monkeypatch: pytest.MonkeyPatch) -> None:
    ig.reset_input_guardrail()
    monkeypatch.delenv("INPUT_GUARDRAIL", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    yield
    ig.reset_input_guardrail()


def test_default_scrub_masks_emirates_id() -> None:
    text = "Applicant Emirates ID 784-1234-1234567-1 needs review"
    out, counts = ig.scrub_text(text, mode="default")
    assert "784-1234-1234567-1" not in out
    assert "[REDACTED_EMIRATES_ID]" in out
    assert counts.get("emirates_id", 0) >= 1


def test_default_scrub_masks_email_and_phone() -> None:
    text = "Contact ali@example.com or +971501234567"
    out, counts = ig.scrub_text(text, mode="default")
    assert "ali@example.com" not in out
    assert "+971501234567" not in out
    assert "[REDACTED_EMAIL]" in out
    assert "[REDACTED_PHONE]" in out
    assert counts.get("email", 0) >= 1
    assert counts.get("phone", 0) >= 1


def test_off_mode_leaves_text_unchanged() -> None:
    text = "ID 784-1234-1234567-1 and ali@example.com"
    out, counts = ig.scrub_text(text, mode="off")
    assert out == text
    assert counts == {}


def test_scrub_messages_rewrites_content_strings() -> None:
    messages = [
        {"role": "user", "content": "Emirates ID 784-9999-1234567-1"},
        {"role": "system", "content": "You are helpful."},
    ]
    scrubbed, counts = ig.scrub_messages(messages, mode="default")
    assert "784-9999-1234567-1" not in scrubbed[0]["content"]
    assert scrubbed[1]["content"] == "You are helpful."
    assert counts.get("emirates_id", 0) >= 1
    # Original list not mutated
    assert "784-9999-1234567-1" in messages[0]["content"]


def test_custom_callback_replaces_default() -> None:
    def custom(text: str) -> tuple[str, dict[str, int]]:
        return text.replace("SECRET", "[REDACTED_CUSTOM]"), {"custom": 1}

    ig.register_input_guardrail(custom)
    out, counts = ig.scrub_text("keep email ali@example.com SECRET", mode="custom")
    assert "SECRET" not in out
    assert "[REDACTED_CUSTOM]" in out
    assert "ali@example.com" in out  # default patterns not applied in custom-only
    assert counts == {"custom": 1}


def test_resolve_mode_defaults_off_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("INPUT_GUARDRAIL", raising=False)
    assert ig.resolve_mode() == "off"


def test_resolve_mode_defaults_default_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("INPUT_GUARDRAIL", raising=False)
    assert ig.resolve_mode() == "default"


def test_resolve_mode_respects_explicit_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("INPUT_GUARDRAIL", "off")
    assert ig.resolve_mode() == "off"


# ── detect_pii — the output-side companion ───────────────────────────────────


def test_detect_pii_counts_without_rewriting() -> None:
    text = "Emirates ID 784-1985-1234567-1, card 4111 1111 1111 1111."
    counts = ig.detect_pii(text)
    assert counts == {"emirates_id": 1, "card": 1}


def test_detect_pii_agrees_with_scrub_text() -> None:
    """The whole reason detect_pii exists: an output-side check must classify
    text exactly as the pre-call scrub does. A hook that re-derives the
    patterns drifts (KYC Sentinel's did — a card regex with no Luhn call)."""
    text = "Contact a@b.com, ID 784-1985-1234567-1, card 4111-1111-1111-1111."
    _, scrub_counts = ig.scrub_text(text, mode="default")
    assert ig.detect_pii(text) == scrub_counts


def test_detect_pii_rejects_long_non_luhn_digit_runs() -> None:
    assert "card" not in ig.detect_pii("Registry filing 2024 0918 3345 1207 66.")


def test_detect_pii_ignores_guardrail_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """'Does this text contain PII' is a question about the text, not about
    whether prompts are currently being scrubbed — an output check must still
    work in development, where resolve_mode() is off."""
    monkeypatch.setenv("INPUT_GUARDRAIL", "off")
    assert ig.detect_pii("card 4111 1111 1111 1111") == {"card": 1}


# ── Arabic-Indic numerals are how the market writes digits ───────────────────

_ARABIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _arabic(s: str) -> str:
    return s.translate(_ARABIC)


def test_an_emirates_id_in_arabic_numerals_is_scrubbed():
    """`٧٨٤-١٢٣٤-...` is how a person writes an Emirates ID in Arabic, in a
    framework whose market is the UAE.

    The pattern anchors on a literal ASCII `784`, so it matched nothing and the
    identifier went to the model in the clear. What made it invisible is the
    asymmetry: `\\d` DOES match Arabic-Indic digits, so the card pattern caught
    them all along and `runtime/luhn.py` validates them — anyone testing
    "do we handle Arabic numerals" with a card number would have concluded yes.
    """
    out, counts = ig.scrub_text(f"id {_arabic('784-1234-1234567-1')} ok", mode="default")
    assert counts == {"emirates_id": 1}
    assert "[REDACTED_EMIRATES_ID]" in out
    assert "٧٨٤" not in out


def test_the_bare_15_digit_form_too():
    _out, counts = ig.scrub_text(f"id {_arabic('784123412345671')} ok", mode="default")
    assert counts == {"emirates_id": 1}


def test_a_card_in_arabic_numerals_still_luhn_checks():
    """The control: normalising must not have broken the case that worked."""
    _out, counts = ig.scrub_text(f"pay {_arabic('4111111111111111')}", mode="default")
    assert counts == {"card": 1}
    # And a Luhn-invalid run is still left alone, in either script.
    _, none = ig.scrub_text(f"ref {_arabic('4111111111111112')}", mode="default")
    assert none == {}


def test_the_scrubbed_text_keeps_the_users_own_characters():
    """Detection runs on an ASCII-normalised copy; the text written out is the
    original. A scrubber must not quietly rewrite the parts it did not redact."""
    out, _ = ig.scrub_text(f"order {_arabic('12345')} and id {_arabic('784123412345671')}",
                           mode="default")
    assert "١٢٣٤٥" in out, "an untouched Arabic number was rewritten to ASCII"
    assert "[REDACTED_EMIRATES_ID]" in out
