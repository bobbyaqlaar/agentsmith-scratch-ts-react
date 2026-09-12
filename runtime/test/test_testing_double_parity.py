"""
runtime/test/test_testing_double_parity.py — the test double must accept what
the real gateway accepts.

runtime/testing.py is a shipped, documented, tenant-facing API (SPECS §528,
OPERATIONS "Testing your tenant app"). FakeGateway converts the prompt to text
on every complete() and stream() call, and that conversion joined
`m.get("content", "")` straight into a str.join — so a message whose content is
a LIST of typed parts, the ordinary Anthropic and OpenAI shape, raised
TypeError out of the double on a prompt the real gateway handles fine.

A test double that refuses inputs the real thing accepts sends the tenant
looking for a bug in their own code.

The flattening is prompt_identity.content_text now — the same function the real
gateway's prompt hashing uses, whose docstring is about exactly this shape.
Two functions answered one question and only one of them knew about half the
inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.prompt_identity import content_text
from runtime.testing import FakeGateway, _as_text

MULTIMODAL = [{"role": "user", "content": [{"type": "text", "text": "review this"}]}]


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ([{"role": "user", "content": "review this"}], "review this"),
        (MULTIMODAL, "review this"),
        ([{"role": "user", "content": None}], ""),
        ([{"role": "user"}], ""),
        (
            [
                {"role": "system", "content": "be careful"},
                {"role": "user", "content": [{"type": "text", "text": "and this"}]},
            ],
            "be careful\nand this",
        ),
        ([{"role": "user", "content": [{"type": "image", "source": {}}]}], ""),
        ("a bare string prompt", "a bare string prompt"),
    ],
)
def test_every_prompt_shape_converts_without_raising(prompt, expected):
    assert _as_text(prompt) == expected


@pytest.mark.asyncio
async def test_the_double_accepts_a_multimodal_prompt_end_to_end():
    """Not just the helper — the path a tenant's test actually takes.

    This raised TypeError from inside FakeGateway.complete(), which is the
    first thing a tenant writing a multimodal test would have hit.
    """
    gateway = FakeGateway(default_response="ok")
    result = await gateway.complete(prompt=MULTIMODAL, model_hint="developer")

    assert result.text == "ok"
    assert gateway.calls[0].prompt == "review this", (
        "the recorded prompt lost its text"
    )


@pytest.mark.asyncio
async def test_recorded_prompts_stay_assertable(monkeypatch):
    """FakeGateway's recorded prompt is what tenants assert on. A multimodal
    prompt has to arrive as the text inside it, not as a repr of the blocks."""
    gateway = FakeGateway(default_response="ok")
    await gateway.complete(
        prompt=[
            {"role": "system", "content": "you are a KYC analyst"},
            {"role": "user", "content": [{"type": "text", "text": "screen this"}]},
        ],
        model_hint="developer",
    )
    recorded = gateway.calls[0].prompt
    assert "you are a KYC analyst" in recorded
    assert "screen this" in recorded


def test_the_two_converters_are_one_function():
    """The defect was two implementations of one question.

    Asserted by identity rather than by comparing outputs: two functions that
    agree on the cases someone thought to test are exactly what was here
    before.
    """
    import runtime.testing as testing

    assert testing.content_text is content_text
