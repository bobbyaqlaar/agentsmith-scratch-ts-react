"""
runtime/test/test_replay_webhook_signature.py — the replay webhook's signature.

The legacy signature is an HMAC over the request BODY alone, so it never
expires: a captured request stays cryptographically valid for as long as the
secret does.

dead_letter.py's atomic claim is a real mitigation — an entry already
`replayed` refuses a second attempt — but it is not the whole story, because
_release_claim deliberately returns an entry to `pending` when the replay
handler fails. After a replay that could not reach Temporal, a captured request
is live again. That is a security property resting on a status column designed
to revert.

v2 signs `timestamp.body` and is only accepted inside a tolerance window. Both
shapes are accepted so the Ops Portal (IT-operated) and this receiver
(tenant-operated) can be upgraded in either order.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime import replay_webhook_server as rws

SECRET = "shared-secret"
BODY = json.dumps({"taskId": "t-1", "payload": {"a": 1}}).encode()


def _v2(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    signed = timestamp.encode() + b"." + body
    return "sha256=" + hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def _legacy(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_a_fresh_timestamped_signature_is_accepted():
    now = time.time()
    ts = str(int(now))
    ok, reason = rws._verify_timestamped(SECRET, BODY, ts, _v2(BODY, ts), now=now)
    assert ok, reason


@pytest.mark.parametrize("age", [301, 3600, 86400 * 30])
def test_a_stale_request_is_refused(age):
    """The whole point: a captured request stops being useful."""
    now = time.time()
    ts = str(int(now - age))
    ok, reason = rws._verify_timestamped(SECRET, BODY, ts, _v2(BODY, ts), now=now)
    assert not ok
    assert "tolerance" in reason


def test_a_request_from_the_future_is_refused_too():
    """Skew in either direction. A far-future timestamp would otherwise mint a
    request that stays valid for as long as the clock takes to catch up."""
    now = time.time()
    ts = str(int(now + 3600))
    ok, reason = rws._verify_timestamped(SECRET, BODY, ts, _v2(BODY, ts), now=now)
    assert not ok
    assert "tolerance" in reason


def test_ordinary_clock_skew_still_works():
    """Tolerance exists because two machines never agree exactly. A gate that
    fails on a few seconds of drift gets turned off."""
    now = time.time()
    ts = str(int(now - 30))
    ok, _ = rws._verify_timestamped(SECRET, BODY, ts, _v2(BODY, ts), now=now)
    assert ok


def test_the_timestamp_cannot_be_moved_without_breaking_the_signature():
    """The timestamp is INSIDE the signed material.

    If it were only a header, an attacker replaying a captured body could
    simply put a fresh timestamp on it and the window would prove nothing.
    """
    now = time.time()
    original = str(int(now - 3600))
    signature = _v2(BODY, original)  # signed an hour ago

    ok, reason = rws._verify_timestamped(SECRET, BODY, str(int(now)), signature, now=now)
    assert not ok
    assert "does not match" in reason


def test_a_tampered_body_is_refused():
    now = time.time()
    ts = str(int(now))
    signature = _v2(BODY, ts)
    tampered = json.dumps({"taskId": "t-2", "payload": {"a": 1}}).encode()

    ok, reason = rws._verify_timestamped(SECRET, tampered, ts, signature, now=now)
    assert not ok
    assert "does not match" in reason


def test_the_wrong_secret_is_refused():
    now = time.time()
    ts = str(int(now))
    ok, _ = rws._verify_timestamped(
        SECRET, BODY, ts, _v2(BODY, ts, secret="other-secret"), now=now
    )
    assert not ok


@pytest.mark.parametrize("timestamp", ["", "not-a-number", "12:00", "NaN"])
def test_a_malformed_timestamp_is_refused_not_crashed(timestamp):
    """float('') raises. A malformed header must be a 401, not a traceback out
    of the handler — the same lesson as the Content-Length parse above it."""
    ok, reason = rws._verify_timestamped(
        SECRET, BODY, timestamp, _v2(BODY, timestamp), now=time.time()
    )
    assert not ok
    assert "timestamp" in reason


def test_the_reason_separates_stale_from_forged():
    """An operator diagnosing clock skew and one diagnosing an attack need
    different answers; "invalid signature" for both is how the first gets
    mistaken for the second."""
    now = time.time()
    stale_ts = str(int(now - 3600))
    _, stale = rws._verify_timestamped(SECRET, BODY, stale_ts, _v2(BODY, stale_ts), now=now)

    fresh_ts = str(int(now))
    _, forged = rws._verify_timestamped(SECRET, BODY, fresh_ts, "sha256=deadbeef", now=now)

    assert stale != forged
    assert "tolerance" in stale
    assert "does not match" in forged


def test_the_legacy_signature_still_verifies():
    """A receiver upgraded before the portal must keep accepting what the
    portal still sends, or upgrading the receiver takes replay down."""
    assert rws._verify_signature(SECRET, BODY, _legacy(BODY))
    assert not rws._verify_signature(SECRET, BODY, _legacy(BODY, secret="other"))
    assert not rws._verify_signature(SECRET, BODY, "md5=" + "0" * 32)


def test_the_tolerance_is_stated_in_seconds_and_is_short():
    """Pinned so that widening it is a decision someone makes on purpose."""
    assert rws.SIGNATURE_TOLERANCE_SECONDS == 300


def test_a_non_finite_timestamp_cannot_skip_the_window():
    """NaN parses as a float and every comparison against it is False, so
    `drift > tolerance` was False and the window check passed without ever
    testing anything. Infinity is caught by the comparison; NaN was not.

    This was a real hole in the first version of _verify_timestamped, found by
    a test written for malformed input rather than by reading the code.
    """
    now = time.time()
    for timestamp in ("NaN", "nan", "-nan"):
        ok, reason = rws._verify_timestamped(
            SECRET, BODY, timestamp, _v2(BODY, timestamp), now=now
        )
        assert not ok, f"{timestamp!r} skipped the tolerance window"
        assert "timestamp" in reason


def test_infinity_is_refused_as_well():
    now = time.time()
    for timestamp in ("inf", "-inf", "1e400"):
        ok, _ = rws._verify_timestamped(
            SECRET, BODY, timestamp, _v2(BODY, timestamp), now=now
        )
        assert not ok, f"{timestamp!r} was accepted"


def test_the_signature_matches_the_portal_byte_for_byte():
    """The portal signs and this module verifies; they are separate codebases
    in separate languages, upgraded by different people.

    portal/test/replaySignature.test.ts pins the same value from the other side.
    A change to either derivation fails a test in its own language rather than
    turning into a 401 nobody can explain.
    """
    body = json.dumps({"taskId": "t-1", "payload": {"a": 1}}, separators=(",", ":")).encode()
    timestamp = "1756500000"
    expected = (
        "sha256="
        + hmac.new(
            SECRET.encode(), timestamp.encode() + b"." + body, hashlib.sha256
        ).hexdigest()
    )
    assert _v2(body, timestamp) == expected

    ok, _ = rws._verify_timestamped(
        SECRET, body, timestamp, expected, now=float(timestamp)
    )
    assert ok


def test_signatures_are_compared_in_constant_time():
    """Asserted STRUCTURALLY, because no behavioural test can see it.

    `==` and hmac.compare_digest return the same answers; they differ only in
    how long a mismatch takes, which leaks the length of the matching prefix to
    anyone able to time the endpoint. A mutation swapping one for the other
    passes every other test in this file — verified, it survived — so the only
    thing standing between this and a plausible "simplification" is an
    assertion about the code itself.
    """
    import ast
    import inspect
    import textwrap

    for function in (rws._verify_signature, rws._verify_timestamped):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "compare_digest" in calls, (
            f"{function.__name__} does not use hmac.compare_digest — a signature "
            f"compared with == leaks its matching prefix through timing"
        )

        compares = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops)
            and any(
                isinstance(side, ast.Name) and "expected" in side.id
                for side in [node.left, *node.comparators]
            )
        ]
        assert not compares, (
            f"{function.__name__} compares the expected digest with ==/!="
        )
