"""
runtime/test/test_idempotency_key.py — make_key must produce the same key twice.

The retry this module guards is the retry AFTER A CRASH, and that runs in a new
process. A key that depends on anything process-local is therefore worthless
precisely when it is needed: the cache misses, the work runs again, and the
duplicate LLM call is paid for.

`default=str` accepted everything and was stable for almost none of it.
Measured across three processes before the change:

    {"tags": {"kyc", "sanctions", "pep"}}   3 distinct keys
    {"ctx": <a custom object>}              3 distinct keys

The first is Python's per-process string hash randomisation changing set
iteration order. The second is a memory address in the default repr. Both
returned a key and neither was reproducible.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import os
import pathlib
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.idempotency import UnstableIdempotencyKey, make_key


class Colour(enum.Enum):
    RED = "red"


STABLE_PAYLOADS = {
    "plain": {"prompt": "hi", "n": 2, "ok": True, "z": None},
    "set_of_strings": {"tags": {"kyc", "sanctions", "pep"}},
    "set_of_ints": {"ids": {3, 1, 2}},
    "nested_set": {"a": [{"b": {"z", "y", "x"}}]},
    "frozenset": {"f": frozenset(["b", "a"])},
    "datetime": {"at": datetime.datetime(2026, 8, 30, 12, 0)},
    "uuid": {"id": uuid.UUID("12345678-1234-5678-1234-567812345678")},
    "decimal": {"amt": decimal.Decimal("10.50")},
    "path": {"p": pathlib.Path("/tmp/x")},
    "enum": {"c": Colour.RED},
    "bytes": {"b": b"ab"},
}


def _key_in_subprocess(literal: str, seed: str) -> str:
    """make_key for `literal`, computed in a fresh interpreter.

    A separate process with a different PYTHONHASHSEED is the only way to see
    this: within one process a set iterates the same way every time, so an
    in-process test passes against the broken implementation.
    """
    repo = str(Path(__file__).resolve().parents[2])
    code = (
        f"import sys; sys.path.insert(0, {repo!r})\n"
        "import datetime, decimal, enum, pathlib, uuid\n"
        "from runtime.idempotency import make_key\n"
        "class Colour(enum.Enum):\n    RED = 'red'\n"
        f"print(make_key({literal}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONHASHSEED=seed),
        # check=False: the assertion below reports the child's stderr, which is
        # a far more useful failure than CalledProcessError's repr.
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("name", sorted(STABLE_PAYLOADS))
def test_the_key_is_the_same_in_another_process(name):
    """Three interpreters, three hash seeds, one key."""
    literals = {
        "plain": "{'prompt': 'hi', 'n': 2, 'ok': True, 'z': None}",
        "set_of_strings": "{'tags': {'kyc', 'sanctions', 'pep'}}",
        "set_of_ints": "{'ids': {3, 1, 2}}",
        "nested_set": "{'a': [{'b': {'z', 'y', 'x'}}]}",
        "frozenset": "{'f': frozenset(['b', 'a'])}",
        "datetime": "{'at': datetime.datetime(2026, 8, 30, 12, 0)}",
        "uuid": "{'id': uuid.UUID('12345678-1234-5678-1234-567812345678')}",
        "decimal": "{'amt': decimal.Decimal('10.50')}",
        "path": "{'p': pathlib.Path('/tmp/x')}",
        "enum": "{'c': Colour.RED}",
        "bytes": "{'b': b'ab'}",
    }
    keys = {_key_in_subprocess(literals[name], seed) for seed in ("0", "1", "2")}
    assert len(keys) == 1, f"{name} produced {len(keys)} different keys: {keys}"


def test_a_payload_with_no_stable_form_is_refused_loudly():
    """The alternative is a key that will never be computed again.

    An object with no __str__ stringifies to `<Ctx object at 0x…>`. `default=str`
    turned that into a key, so every retry looked like new work and paid for it.
    An exception at the call site is fixable; a silent recurring charge is not.
    """

    class Ctx:
        pass

    with pytest.raises(UnstableIdempotencyKey, match="stable idempotency key"):
        make_key({"ctx": Ctx()})


def test_the_refusal_says_what_to_do_about_it():
    class Ctx:
        pass

    with pytest.raises(UnstableIdempotencyKey) as exc:
        make_key({"ctx": Ctx()})
    assert "Convert it to a JSON value" in str(exc.value)
    assert "Ctx" in str(exc.value)


def test_set_ordering_does_not_change_the_key():
    """Same members, different insertion order, one key."""
    assert make_key({"t": {"a", "b", "c"}}) == make_key({"t": {"c", "b", "a"}})


def test_different_members_still_change_the_key():
    """The canonicalisation must not flatten distinct payloads together."""
    assert make_key({"t": {"a", "b"}}) != make_key({"t": {"a", "c"}})


def test_a_set_collides_with_its_sorted_list_and_that_is_known():
    """A set canonicalises to a sorted list, so {"a","b"} and ["a","b"] key the
    same. Asserted as the behaviour it IS, not wished away.

    Two payloads that differ only in whether a field is a set or an already
    sorted list are the same request in every use of this that exists; treating
    them as one is the conservative direction, since the cost of a collision
    here is a cache HIT on equivalent work rather than a missed retry.

    This assertion was first written as `!= ... or True`, which cannot fail —
    the exact defect this file's neighbours were written to catch. If the
    canonicalisation ever needs to distinguish them, this is the line that
    fails and says so.
    """
    assert make_key({"t": {"a", "b"}}) == make_key({"t": ["a", "b"]})


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "hi", "n": 2},
        {"a": [1, 2, {"b": "c"}], "z": None, "ok": True},
        ["list", "of", "things"],
        "a bare string",
        42,
        {"unicode": "مرحبا", "emoji": "✅"},
    ],
)
def test_json_payloads_hash_exactly_as_they_did_before(payload):
    """The compatibility constraint, pinned.

    Every deployment has cached entries keyed by the old derivation. Changing
    what a plain JSON payload hashes to would orphan all of them silently — the
    cache would miss on everything and every in-flight retry would redo its
    work. This reproduces the ORIGINAL expression and requires it to agree.
    """
    original = (
        "sha256:"
        + __import__("hashlib")
        .sha256(json.dumps(payload, sort_keys=True, default=str).encode())
        .hexdigest()
    )
    assert make_key(payload) == original
