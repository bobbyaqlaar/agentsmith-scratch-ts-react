"""
runtime/test/test_dlq_replay_idempotency.py — a replay must happen once.

`DeadLetterQueue.replay()` called its handler — the side effect that signals a
live workflow — before consulting the entry's status at all. Three consequences,
none of them theoretical:

  * the Ops Portal's replay is an ordinary HTTP POST, so a retry, a double-click
    across two tabs, or a captured-and-resent webhook re-signalled the workflow
    every time. In the CRM example that is the customer's record written twice;
  * an entry a human had DISCARDED could still be replayed — the discard
    decision was advisory;
  * two concurrent replays both proceeded.

`portal/lib/dlq.ts`'s discardDlqEntry has carried `AND status = 'pending'` since
it was written. The runtime it drives did not.

These run against a real Postgres because the property under test is the atomic
claim, which only a database can demonstrate — the same reason
portal/test/auditLog.test.ts needs one for its append-only trigger.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.dead_letter import AlreadyResolvedError, DeadLetterQueue

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set — live Postgres test"
)


class Recorder:
    """A replay handler that counts, and can be told to fail."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, entry) -> None:
        self.calls.append(entry.task_id)
        if self.fail:
            raise RuntimeError("the workflow engine was unreachable")


def _queue(handler=None) -> DeadLetterQueue:
    return DeadLetterQueue(replay_handler=handler)


def _enqueue(dlq: DeadLetterQueue) -> str:
    task_id = f"dlq-replay-test-{uuid.uuid4()}"
    dlq.enqueue(
        payload={"customer_id": 102, "account_status": "active"},
        error="validator rejected account_status",
        tenant_id="test-dlq-replay",
        task_id=task_id,
    )
    return task_id


def _status(dlq: DeadLetterQueue, task_id: str) -> str | None:
    entry = dlq._get(task_id)
    return None if entry is None else entry.status


def test_a_pending_entry_replays_exactly_once() -> None:
    handler = Recorder()
    dlq = _queue(handler)
    task_id = _enqueue(dlq)

    dlq.replay(task_id, override_payload={"customer_id": 102, "status": "active"})
    assert handler.calls == [task_id]
    assert _status(dlq, task_id) == "replayed"


def test_replaying_twice_does_not_signal_twice() -> None:
    handler = Recorder()
    dlq = _queue(handler)
    task_id = _enqueue(dlq)

    dlq.replay(task_id)
    with pytest.raises(AlreadyResolvedError):
        dlq.replay(task_id)

    assert handler.calls == [task_id], "the handler ran again for an already-replayed entry"


def test_a_discarded_entry_cannot_be_replayed() -> None:
    handler = Recorder()
    dlq = _queue(handler)
    task_id = _enqueue(dlq)
    dlq.discard(task_id)

    with pytest.raises(AlreadyResolvedError):
        dlq.replay(task_id)

    assert handler.calls == [], "a discarded entry was replayed — the decision was advisory"
    assert _status(dlq, task_id) == "discarded"


def test_a_failed_handler_returns_the_entry_to_pending() -> None:
    """The claim must not outlive a failed attempt.

    `replayed` has always meant "an attempt reached the engine" — the portal
    declines to set it for that reason — so an entry whose handler raised has to
    be replayable again, or one unreachable Temporal strands it forever.
    """
    handler = Recorder(fail=True)
    dlq = _queue(handler)
    task_id = _enqueue(dlq)

    with pytest.raises(RuntimeError, match="unreachable"):
        dlq.replay(task_id)

    assert handler.calls == [task_id]
    assert _status(dlq, task_id) == "pending"

    # ...and the retry, once the engine is back, works.
    ok = _queue(Recorder())
    ok.replay(task_id)
    assert _status(dlq, task_id) == "replayed"


def test_an_unknown_task_is_a_different_error_from_a_resolved_one() -> None:
    """A caller maps these to different answers — 404 versus 409 — and one that
    cannot tell them apart reports "replay failed" for a replay that already
    happened."""
    dlq = _queue(Recorder())
    with pytest.raises(KeyError):
        dlq.replay(f"no-such-task-{uuid.uuid4()}")
