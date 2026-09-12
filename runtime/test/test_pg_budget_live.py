"""
runtime/test/test_pg_budget_live.py — Postgres budget backend against a REAL
database (TestCoverageReview-2026-07-21 gap 5).

The atomic-reservation race regression (Product_Archive.md 2.1) was only
tested on the in-memory backend; the actual production backend's single-
statement reserve SQL — and, since ReviewFindings C1, the shared
connection pool under concurrent borrowers — ran in CI without a test.
The `python-behaviour` job already provisions Postgres (DATABASE_URL set),
so `pytest runtime/test/` picks this up there automatically; locally it
skips unless DATABASE_URL points somewhere real.
"""

from __future__ import annotations

import os
import sys
import threading
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if not os.environ.get("DATABASE_URL"):
    pytest.skip("DATABASE_URL not set — live Postgres test", allow_module_level=True)

psycopg2 = pytest.importorskip("psycopg2")

from runtime.llm_gateway import _PostgresBudgetBackend


@pytest.fixture()
def backend():
    b = _PostgresBudgetBackend()
    tenant = f"test-{uuid.uuid4().hex[:12]}"
    yield b, tenant
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn, conn.cursor() as cur:
            cur.execute("DELETE FROM llm_gateway_budget WHERE tenant_id = %s", (tenant,))
    finally:
        conn.close()


def test_reserve_and_reconcile(backend):
    b, tenant = backend
    assert b.get_spend(tenant) == 0.0
    assert b.try_reserve(tenant, 0.5, cap_usd=1.0) is True
    assert b.try_reserve(tenant, 0.6, cap_usd=1.0) is False  # would breach
    # Reconcile: actual cost was lower than the estimate — signed delta
    b.add_spend(tenant, -0.2)
    assert b.get_spend(tenant) == pytest.approx(0.3)
    assert b.try_reserve(tenant, 0.6, cap_usd=1.0) is True


def test_concurrent_reserves_never_exceed_cap(backend):
    """20 threads race to reserve $0.2 against a $1.00 cap — exactly 5 may
    win, regardless of interleaving. This is the single-statement UPDATE's
    row-lock guarantee, now exercised through the shared pg_pool under
    real concurrency (pool max defaults to 5 < 20 borrowers, so the
    exhaustion fallback path runs too)."""
    b, tenant = backend
    results: list[bool] = []
    lock = threading.Lock()

    def one():
        ok = b.try_reserve(tenant, 0.2, cap_usd=1.0)
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 5
    assert b.get_spend(tenant) == pytest.approx(1.0)
