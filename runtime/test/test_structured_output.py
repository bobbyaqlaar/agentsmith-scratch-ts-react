from __future__ import annotations

import pytest
from pydantic import BaseModel, RootModel

from runtime.structured_output import StructuredOutputError, parse_llm_json


class Demo(BaseModel):
    answer: str
    score: int


def test_parse_fenced_json() -> None:
    raw = 'Here:\n```json\n{"answer":"ok","score":1}\n```'
    assert parse_llm_json(raw, Demo).answer == "ok"


def test_parse_bare_json() -> None:
    raw = '{"answer":"bare","score":2}'
    assert parse_llm_json(raw, Demo).answer == "bare"


def test_parse_fenced_without_language_tag() -> None:
    raw = 'Result:\n```\n{"answer":"plain","score":3}\n```'
    assert parse_llm_json(raw, Demo).answer == "plain"


def test_invalid_schema_raises() -> None:
    raw = '{"answer":"ok"}'  # missing required score
    with pytest.raises(StructuredOutputError):
        parse_llm_json(raw, Demo)


def test_malformed_json_raises() -> None:
    raw = "not json at all"
    with pytest.raises(StructuredOutputError):
        parse_llm_json(raw, Demo)


class Row(BaseModel):
    answer: str


class Rows(RootModel[list[Row]]):
    pass


def test_a_bare_json_array_is_parsed() -> None:
    """`[{...},{...}]` — unreachable before, and the failure was silent-shaped.

    `{` was tried before `[`, and an array of objects contains braces, so the
    brace branch always won and returned the array with its brackets stripped.
    The result parsed as nothing and surfaced as "invalid JSON" — a model that
    had answered correctly looked like it had emitted garbage.
    """
    assert len(parse_llm_json('[{"answer":"a"},{"answer":"b"}]', Rows).root) == 2


def test_a_bare_array_and_a_fenced_one_agree() -> None:
    """The same payload was accepted or rejected on how the model formatted it.

    This is the assertion that would have caught the bug: the fenced form
    worked throughout, so any test written with a fence passed while the bare
    path was broken.
    """
    bare = '[{"answer":"a"}]'
    fenced = f"```json\n{bare}\n```"
    assert parse_llm_json(bare, Rows).root == parse_llm_json(fenced, Rows).root


def test_the_answer_fence_wins_over_an_earlier_working_fence() -> None:
    """The ordinary shape for a reasoning model: working first, answer second.

    The first fence used to win regardless of tag, so the working was extracted
    and the answer discarded.
    """
    raw = (
        "Let me work through it:\n"
        "```\nstep 1: check sanctions\nstep 2: rate\n```\n"
        "Final answer:\n"
        '```json\n{"answer":"ok","score":1}\n```'
    )
    assert parse_llm_json(raw, Demo).answer == "ok"


def test_a_non_json_language_tag_does_not_hijack_extraction() -> None:
    """The tag is captured now, not skipped.

    `(?:json|JSON)?` matched a fence opened with any other tag and then left
    that tag inside the captured body, so "```python" yielded "python\nprint(1)".
    """
    raw = '```python\nprint(1)\n```\n```json\n{"answer":"ok","score":1}\n```'
    assert parse_llm_json(raw, Demo).answer == "ok"


def test_an_explicit_json_tag_outranks_an_earlier_untagged_fence() -> None:
    """Both parse, so parse-validation alone cannot choose between them.

    A model that puts a draft in a bare fence and the answer in a tagged one has
    said which is which, and document order would pick the wrong one.
    """
    raw = (
        'Draft:\n```\n{"answer":"draft","score":0}\n```\n'
        'Final:\n```json\n{"answer":"final","score":1}\n```'
    )
    assert parse_llm_json(raw, Demo).answer == "final"


def test_a_failure_names_the_block_it_tried() -> None:
    """With several fences in a response, WHICH one was parsed is the only
    thing an operator needs, and the message used to omit it.

    This assertion replaces one that could not fail: it checked that the
    model's prose was ABSENT from the error, which held trivially because no
    implementation ever put payload text in the message. It asserts the origin
    label now — something that is only true because the code produces it.
    """
    raw = 'Here is the answer:\n```json\n{"answer":"ok",\n```'
    with pytest.raises(StructuredOutputError) as exc:
        parse_llm_json(raw, Demo)
    assert "```json fence #1" in str(exc.value)

    # And the origin is named on a SCHEMA failure too, not only a parse failure.
    with pytest.raises(StructuredOutputError) as exc:
        parse_llm_json('{"answer":"ok"}', Demo)
    assert "bare {...} span" in str(exc.value)


def test_the_error_does_not_quote_the_model_output() -> None:
    """The origin names the block; it never reproduces it.

    An exception string goes wherever exceptions go — logs, alerts, a traceback
    in a ticket. Model output reaches a span through trace_redactor for exactly
    that reason, and an error path must not become the way around it.
    """
    raw = '```json\n{"emirates_id":"784-1234-1234567-1",\n```'
    with pytest.raises(StructuredOutputError) as exc:
        parse_llm_json(raw, Demo)
    assert "784" not in str(exc.value)
    assert "emirates_id" not in str(exc.value)


def test_an_empty_fence_is_not_blamed_for_a_later_failure() -> None:
    """An empty fence is skipped, so it cannot become the reported origin.

    Without the skip it would rank first and the operator would be sent to a
    blank block to explain a parse error caused by the text after it.
    """
    raw = '```\n```\n{"answer":"ok",}'
    with pytest.raises(StructuredOutputError) as exc:
        parse_llm_json(raw, Demo)
    assert "bare {...} span" in str(exc.value)
    assert "fence" not in str(exc.value)


def test_a_truncated_first_fence_falls_through_to_the_next() -> None:
    """Ordering picks a favourite; parsing is what confirms it.

    Two ```json fences where the first is truncated. Tier order alone would
    take it and fail — only checking whether a candidate IS json reaches the
    second. Mutation testing added this: with every other case here, the
    first-ranked candidate already parsed, so removing the check entirely left
    the suite green.
    """
    raw = (
        'Attempt:\n```json\n{"answer":"ok",\n```\n'
        'Corrected:\n```json\n{"answer":"ok","score":2}\n```'
    )
    assert parse_llm_json(raw, Demo).score == 2


def test_a_prose_fence_falls_through_to_bare_json() -> None:
    """A fenced note followed by an unfenced answer — the fence outranks the
    bare span, so only parse-validation gets past it."""
    raw = 'Note:\n```\nscores are 0-5\n```\n{"answer":"ok","score":3}'
    assert parse_llm_json(raw, Demo).score == 3
