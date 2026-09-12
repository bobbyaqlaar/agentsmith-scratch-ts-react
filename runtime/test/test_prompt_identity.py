"""
runtime/test/test_prompt_identity.py — which prompt produced this answer.

The framework recorded the model, the cost, the latency and the verdict, and
nothing about what was actually sent — so "answers got worse last Tuesday" had
no column to join against. Prompt versioning needs a template engine this
framework does not have (prompts are inline f-strings; the engine is a tracked
future item), so the join column is a digest of the SYSTEM turn: the part that
is stable across requests and edited by a human, which is what a template is
whether or not it is stored as one.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.prompt_identity import (
    prompt_attributes,
    system_fingerprint,
    system_prompt,
)

SYS = "You are a strict KYC analyst."
MSGS = [{"role": "system", "content": SYS}, {"role": "user", "content": "Screen this."}]


def test_the_digest_is_of_the_system_turn():
    assert system_fingerprint(MSGS) == hashlib.sha256(SYS.encode()).hexdigest()


def test_the_digest_is_stable_across_different_user_turns():
    """The whole point: it must change when someone edits the instructions and
    not when the input changes, or it is a unique value per request and joins
    against nothing."""
    other = [{"role": "system", "content": SYS}, {"role": "user", "content": "Different."}]
    assert system_fingerprint(MSGS) == system_fingerprint(other)


def test_the_digest_changes_when_the_system_prompt_is_edited():
    edited = [{"role": "system", "content": SYS + " Be terse."}, MSGS[1]]
    assert system_fingerprint(edited) != system_fingerprint(MSGS)


def test_no_system_turn_is_none_not_the_hash_of_nothing():
    """"This call had no system prompt" and "it had an empty one" are different
    facts, and the second has a digest a reader would try to look up."""
    assert system_fingerprint([{"role": "user", "content": "hi"}]) is None
    assert system_prompt([{"role": "user", "content": "hi"}]) is None


def test_multipart_content_hashes_the_same_as_the_plain_string():
    """Anthropic and OpenAI both allow a list of typed parts. A prompt that
    switches shape without changing a word must not move the join column."""
    multipart = [
        {"role": "system", "content": [{"type": "text", "text": SYS}]},
        MSGS[1],
    ]
    assert system_fingerprint(multipart) == system_fingerprint(MSGS)


def test_several_system_turns_are_one_prompt():
    """A caller that splits instructions across two system turns has ONE
    prompt; hashing only the first would miss edits to the rest."""
    split = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "x"},
    ]
    assert system_prompt(split) == "A\nB"
    assert system_fingerprint(split) != system_fingerprint(
        [{"role": "system", "content": "A"}, {"role": "user", "content": "x"}]
    )


def test_attributes_never_carry_the_prompt_text():
    """The likeliest place for PII to enter a span, and trace_redactor runs
    after this. The digest answers the question the text was wanted for."""
    attrs = prompt_attributes(MSGS, template_id="kyc.analyst.v1")
    joined = " ".join(str(v) for v in attrs.values())
    assert SYS not in joined
    assert "Screen this." not in joined
    assert attrs["prompt.template.id"] == "kyc.analyst.v1"
    assert attrs["prompt.message_count"] == 2
    assert attrs["prompt.system.chars"] == len(SYS)


def test_absent_facts_are_absent_keys():
    """Nothing is defaulted: a missing prompt.system.sha256 reads as "no system
    prompt", not as a value that failed to compute."""
    attrs = prompt_attributes([{"role": "user", "content": "hi"}])
    assert "prompt.system.sha256" not in attrs
    assert "prompt.template.id" not in attrs


def test_the_gateway_puts_the_digest_on_the_span(spans, monkeypatch):
    from runtime.llm_gateway import LLMGateway
    from runtime.tracing import agent_span

    monkeypatch.setenv("IDEMPOTENCY_BACKEND", "memory")
    monkeypatch.setenv("BUDGET_BACKEND", "memory")
    gw = LLMGateway(tenant_id="acme")
    with agent_span("kyc.screen"):
        gw._record_span_attributes(
            "analyst", "m", None, None, 0.0, messages=MSGS,
            prompt_template_id="kyc.analyst.v1",
        )
    attrs = dict(spans.get_finished_spans()[0].attributes)
    assert attrs["prompt.system.sha256"] == system_fingerprint(MSGS)
    assert attrs["prompt.template.id"] == "kyc.analyst.v1"
