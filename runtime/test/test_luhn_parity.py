"""
runtime/test/test_luhn_parity.py — the pre-call input guardrail and the
post-call trace redactor must agree on what is a card number
(ReviewFindings-2026-07-18 B1: they used to carry divergent private
copies of the Luhn check).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import input_guardrail, trace_redactor
from runtime.luhn import luhn_valid

VALID_CARDS = [
    "4111111111111111",  # Visa test number
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "5500 0000 0000 0004",  # Mastercard test number
    "4111.1111.1111.1111",  # separator neither old copy agreed on
]

INVALID_CANDIDATES = [
    "4111111111111112",  # checksum off by one
    "1234 5678 9012 3456",
    "123456789012",  # 12 digits — too short
    "12345678901234567890",  # 20 digits — too long
    "",
]


def test_valid_cards_pass():
    for c in VALID_CARDS:
        assert luhn_valid(c), c


def test_invalid_candidates_fail():
    for c in INVALID_CANDIDATES:
        assert not luhn_valid(c), c


def test_guard_and_redactor_share_one_implementation():
    """Identity, not just equality — same function object means the two
    PII controls can never drift apart again."""
    assert input_guardrail._luhn_valid is trace_redactor._luhn_valid
    assert input_guardrail._luhn_valid is luhn_valid


def test_guard_and_redactor_agree_end_to_end():
    """A card embedded in text is caught by BOTH the pre-call scrub and
    the post-call redaction — no gap between the two controls."""
    text = "pay with 4111 1111 1111 1111 today"

    scrubbed, counts = input_guardrail.scrub_text(text, mode="default")
    assert counts.get("card") == 1
    assert "4111 1111 1111 1111" not in scrubbed

    redacted = trace_redactor._redact_credit_cards(text)
    assert "4111 1111 1111 1111" not in redacted
