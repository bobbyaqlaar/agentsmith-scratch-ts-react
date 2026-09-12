"""
runtime/test/test_tenancy.py — tenant resolution and identity context.

`.agenticframework/tenant.yaml` declared `tenant.id` and nothing read it, so
every call site supplied its own; KYC Sentinel hardcoded the same string twice.
These pin the resolution order and, more importantly, the refusal at the end of
it — tenant.id partitions the budget ledger, the audit log and cross-tenant
isolation, so a silent default would merge two tenants' records.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.tenancy import (
    TENANT_ENV_VAR,
    TenantUnresolvedError,
    agent_context,
    current_identity,
    current_tenant_id,
    resolve_tenant_id,
    tenant_id_from_config,
)


@pytest.fixture
def scaffold(tmp_path: Path):
    def _write(tenant_id):
        (tmp_path / ".agenticframework").mkdir(exist_ok=True)
        doc = {"tenant": {"id": tenant_id}} if tenant_id is not None else {"tenant": {}}
        (tmp_path / ".agenticframework" / "tenant.yaml").write_text(yaml.dump(doc))
        return tmp_path
    return _write


@pytest.fixture(autouse=True)
def _no_ambient_tenant(monkeypatch):
    monkeypatch.delenv(TENANT_ENV_VAR, raising=False)


def test_reads_the_declaration_nothing_had_read(scaffold):
    root = scaffold("kyc-sentinel")
    assert tenant_id_from_config(root) == "kyc-sentinel"
    assert resolve_tenant_id(root=root) == "kyc-sentinel"


def test_precedence_explicit_over_env_over_config(scaffold, monkeypatch):
    root = scaffold("from-config")
    monkeypatch.setenv(TENANT_ENV_VAR, "from-env")
    assert resolve_tenant_id("explicit", root=root) == "explicit"
    assert resolve_tenant_id(root=root) == "from-env"
    monkeypatch.delenv(TENANT_ENV_VAR)
    assert resolve_tenant_id(root=root) == "from-config"


def test_unresolved_raises_rather_than_defaulting(tmp_path: Path):
    """The whole reason there is no fallback. An 'unknown' tenant would merge
    two tenants' spend and two tenants' audit trail into one bucket."""
    with pytest.raises(TenantUnresolvedError) as exc:
        resolve_tenant_id(root=tmp_path)
    assert "budget ledger" in str(exc.value)


def test_blank_values_do_not_count_as_resolved(scaffold, monkeypatch):
    root = scaffold("real")
    assert resolve_tenant_id("   ", root=root) == "real", "whitespace is not an id"
    monkeypatch.setenv(TENANT_ENV_VAR, "  ")
    assert resolve_tenant_id(root=root) == "real", "a blank env var is not an id"


def test_a_config_without_an_id_is_not_declared(scaffold):
    root = scaffold(None)
    assert tenant_id_from_config(root) is None
    with pytest.raises(TenantUnresolvedError):
        resolve_tenant_id(root=root)


def test_context_binds_and_restores(scaffold):
    assert current_tenant_id() is None
    with agent_context(role="analyst", tenant_id="acme", run_id="r1"):
        assert current_identity() == {
            "tenant.id": "acme", "agent.role": "analyst", "run.id": "r1",
        }
        with agent_context(role="judge"):
            assert current_identity()["agent.role"] == "judge"
            assert current_identity()["tenant.id"] == "acme", "inherits what it did not set"
        assert current_identity()["agent.role"] == "analyst", "restored on exit"
    assert current_identity() == {}, "nothing leaks past the outermost scope"


def test_unset_fields_are_omitted_not_placeheld():
    """A gap is visible in a query; a plausible placeholder gets aggregated
    with real data and is not."""
    with agent_context(role="intake"):
        identity = current_identity()
    assert identity == {"agent.role": "intake"}
    assert "tenant.id" not in identity


def test_the_legacy_tenant_id_env_var_is_still_accepted(scaffold, monkeypatch):
    """`TENANT_ID` is what runtime/worker.py read directly and what the
    dedicated-tenant ConfigMap sets. Accepted rather than renamed: a rename
    would strand any cluster on the previous chart with a worker that refuses
    to start."""
    from runtime.tenancy import LEGACY_TENANT_ENV_VAR

    root = scaffold("from-config")
    monkeypatch.setenv(LEGACY_TENANT_ENV_VAR, "from-legacy-env")
    assert resolve_tenant_id(root=root) == "from-legacy-env"

    # The prefixed name wins where both are set — it is the convention every
    # other framework variable follows.
    monkeypatch.setenv(TENANT_ENV_VAR, "from-new-env")
    assert resolve_tenant_id(root=root) == "from-new-env"
