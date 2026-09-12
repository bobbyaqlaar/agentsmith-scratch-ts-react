"""
runtime/test/test_pg_pool.py — unit tests for the shared Postgres
connection pool (ReviewFindings-2026-07-18 C1).

Uses a fake psycopg2 injected into sys.modules — no real Postgres, same
no-external-infra philosophy as test_llm_gateway_budget.py. The live
Postgres path stays covered by `scripts/verify_system.py
--check-idempotency` / `--check-dlq` against a throwaway database.

What must hold:
  1. Repeated connect/close cycles REUSE one physical connection —
     the entire point of the pool.
  2. proxy.close() returns the connection to the pool; the underlying
     socket is not closed.
  3. A pooled connection gone stale (ping fails) is discarded and a
     fresh one served — a DB restart must not fail a store operation
     that a fresh connection would have served (old per-call behavior).
  4. Pool exhaustion degrades to a direct one-shot connection instead
     of blocking or raising.
  5. The budget backend's operations run through the pool unchanged
     (its `finally: conn.close()` pattern releases, not closes).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fake psycopg2 ─────────────────────────────────────────────────────────────


class FakeCursor:
    def __init__(self, conn: "FakeConnection") -> None:
        self._conn = conn

    def execute(self, sql: str, params=None) -> None:
        if self._conn.broken or self._conn.closed:
            raise RuntimeError("connection is broken")
        self._conn.executed.append(sql)

    def fetchone(self):
        # Single-column zero row: satisfies get_spend (float(row[0])),
        # add_spend (fetchone()[0]) and try_reserve (row is not None).
        return (0.0,)

    @property
    def rowcount(self) -> int:
        return 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self) -> None:
        self.closed = 0  # psycopg2 convention: 0 = open
        self.broken = False
        self.executed: list[str] = []

    def cursor(self):
        return FakeCursor(self)

    def rollback(self) -> None:
        if self.closed:
            raise RuntimeError("connection is closed")

    def close(self) -> None:
        self.closed = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePoolError(Exception):
    pass


class FakePool:
    """Mimics psycopg2.pool.ThreadedConnectionPool closely enough:
    getconn() serves an idle connection or creates one up to maxconn,
    then raises; putconn(close=True) discards."""

    def __init__(self, minconn: int, maxconn: int, dsn: str) -> None:
        self.maxconn = maxconn
        self.dsn = dsn
        self.closed = False
        self._idle: list[FakeConnection] = []
        self._outstanding = 0
        self.created: list[FakeConnection] = []

    def getconn(self) -> FakeConnection:
        if self._idle:
            self._outstanding += 1
            return self._idle.pop()
        if self._outstanding >= self.maxconn:
            raise FakePoolError("connection pool exhausted")
        conn = FakeConnection()
        self.created.append(conn)
        self._outstanding += 1
        return conn

    def putconn(self, conn: FakeConnection, close: bool = False) -> None:
        self._outstanding -= 1
        if close or conn.closed:
            conn.close()
        else:
            self._idle.append(conn)

    def closeall(self) -> None:
        self.closed = True
        for conn in self._idle:
            conn.close()
        self._idle.clear()


@pytest.fixture()
def fake_psycopg2(monkeypatch):
    """Install a fake psycopg2 + psycopg2.pool into sys.modules and reset
    pg_pool's per-process pool registry around each test."""
    direct_connections: list[FakeConnection] = []

    fake_mod = types.ModuleType("psycopg2")

    def _direct_connect(dsn):
        conn = FakeConnection()
        direct_connections.append(conn)
        return conn

    fake_mod.connect = _direct_connect  # type: ignore[attr-defined]
    fake_pool_mod = types.ModuleType("psycopg2.pool")
    fake_pool_mod.ThreadedConnectionPool = FakePool  # type: ignore[attr-defined]
    fake_mod.pool = fake_pool_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "psycopg2", fake_mod)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", fake_pool_mod)

    from runtime import pg_pool

    pg_pool.close_all()
    yield types.SimpleNamespace(
        module=fake_mod, direct_connections=direct_connections, pg_pool=pg_pool
    )
    pg_pool.close_all()


def _the_pool(pg_pool) -> FakePool:
    assert len(pg_pool._pools) == 1, "expected exactly one pool per DSN"
    return next(iter(pg_pool._pools.values()))


# ── Pool behavior ─────────────────────────────────────────────────────────────


def test_connections_are_reused_across_borrow_cycles(fake_psycopg2):
    pg_pool = fake_psycopg2.pg_pool
    for _ in range(5):
        conn = pg_pool.connect("postgres://fake/db")
        with conn, conn.cursor() as cur:
            cur.execute("SELECT 42")
        conn.close()
    pool = _the_pool(pg_pool)
    assert len(pool.created) == 1, "5 sequential operations must share 1 connection"
    assert fake_psycopg2.direct_connections == [], "no direct (unpooled) connects"


def test_close_releases_instead_of_closing(fake_psycopg2):
    pg_pool = fake_psycopg2.pg_pool
    conn = pg_pool.connect("postgres://fake/db")
    underlying = conn._conn
    conn.close()
    assert underlying.closed == 0, "close() must release to pool, not close socket"
    conn.close()  # double-close is a no-op, same as psycopg2


def test_stale_pooled_connection_is_discarded_and_replaced(fake_psycopg2):
    pg_pool = fake_psycopg2.pg_pool
    first = pg_pool.connect("postgres://fake/db")
    stale = first._conn
    first.close()
    stale.broken = True  # simulate Postgres restart while idle in pool

    second = pg_pool.connect("postgres://fake/db")
    assert second._conn is not stale, "ping must weed out the dead connection"
    assert stale.closed == 1, "dead connection must be physically closed"
    with second, second.cursor() as cur:
        cur.execute("SELECT 1")
    second.close()


def test_pool_exhaustion_falls_back_to_direct_connect(fake_psycopg2, monkeypatch):
    monkeypatch.setenv("PG_POOL_MAX", "2")
    pg_pool = fake_psycopg2.pg_pool
    held = [pg_pool.connect("postgres://fake/db") for _ in range(2)]
    overflow = pg_pool.connect("postgres://fake/db")
    assert fake_psycopg2.direct_connections, (
        "borrow beyond PG_POOL_MAX must degrade to a direct connection"
    )
    # Direct fallback's close() really closes (old per-call behavior).
    direct = fake_psycopg2.direct_connections[0]
    overflow.close()
    assert direct.closed == 1
    for c in held:
        c.close()


# ── Store integration: budget backend runs through the pool ──────────────────


def test_budget_backend_reuses_pooled_connection(fake_psycopg2, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://fake/db")
    from runtime.llm_gateway import _PostgresBudgetBackend

    backend = _PostgresBudgetBackend()  # CREATE TABLE via pooled conn
    backend.get_spend("acme")
    backend.add_spend("acme", 1.25)
    backend.try_reserve("acme", 1.0, cap_usd=10.0)

    pool = _the_pool(fake_psycopg2.pg_pool)
    assert len(pool.created) == 1, (
        "init + 3 budget operations must reuse one pooled connection"
    )
    assert fake_psycopg2.direct_connections == []
