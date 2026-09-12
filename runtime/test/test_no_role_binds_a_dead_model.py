"""
runtime/test/test_no_role_binds_a_dead_model.py — no profile may route a role to
a catalog entry marked `decommissioned: true`.

THE FAILURE THIS EXISTS FOR. Groq retired its entire Llama family on
2026-08-17 and `llama-3.3-70b-versatile` began returning HTTP 404
model_not_found on every call. The framework's `hybrid` profile bound
`developer` to it, so that role was unreachable for anyone selecting
ai-mode-hybrid — latent rather than loud, because `default_profile` is `local`.

What makes this worth a gate rather than a comment is how it presents on a
judged gate. A 404 is an infrastructure failure, so run-evals reports
`NO VERDICT (judge unreachable)` and exits **0** — correct, since an
unreachable judge is not a quality result. Nothing went red. KYC Sentinel's
suites graded nothing for several days while CI stayed green.

The catalog is an offer, not an archive — runtime/models.yaml says these entries
"stay available to any tenant that wants them" — so a dead id sitting there with
live-looking cost figures is a trap. `decommissioned: true` marks it, and this
test is what makes the marker load-bearing rather than a comment with extra
steps.

Covers the framework registry and any tenant models.yaml in the repo, since a
tenant inherits this catalog and can bind against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def _registries() -> list[Path]:
    found = [REPO / "runtime" / "models.yaml"]
    found += sorted(REPO.glob("examples/*/models.yaml"))
    found += sorted(REPO.glob("templates/*/models.yaml"))
    return [p for p in found if p.exists()]


def test_there_is_a_registry_to_check() -> None:
    """Guard the guard — an empty file list would make everything below pass
    while checking nothing."""
    regs = _registries()
    assert regs, "no models.yaml found at all"
    assert (REPO / "runtime" / "models.yaml") in regs


def test_a_decommissioned_entry_is_actually_marked() -> None:
    """Pins the known-dead id. If someone silently deletes the marker, this
    fails rather than the suite quietly losing its only real subject."""
    cat = yaml.safe_load((REPO / "runtime" / "models.yaml").read_text()).get("catalog", {})
    dead = cat.get("llama-3.3-70b-versatile")
    assert dead is not None, (
        "llama-3.3-70b-versatile was removed from the catalog. That is a fine "
        "decision — delete this test with it, but do not leave the entry "
        "unmarked."
    )
    assert dead.get("decommissioned") is True, (
        "llama-3.3-70b-versatile returns HTTP 404 model_not_found on every "
        "call (Groq retired the Llama family 2026-08-17) and must stay marked."
    )


@pytest.mark.parametrize("registry", _registries(), ids=lambda p: str(p.name))
def test_no_profile_binds_a_decommissioned_model(registry: Path) -> None:
    data = yaml.safe_load(registry.read_text()) or {}
    catalog = data.get("catalog", {}) or {}
    dead = {
        name for name, cfg in catalog.items()
        if isinstance(cfg, dict) and cfg.get("decommissioned") is True
    }
    if not dead:
        pytest.skip(f"{registry.name} declares no decommissioned entries")

    offences = []
    for profile, roles in (data.get("profiles", {}) or {}).items():
        for role, cfg in (roles or {}).items():
            if not isinstance(cfg, dict):
                continue
            # `degrade_to` names a ROLE, not a model, so only `use` is a binding.
            if cfg.get("use") in dead:
                offences.append(f"{profile}.{role} -> {cfg['use']}")

    assert not offences, (
        "role(s) bound to a decommissioned model — every call returns 404, and "
        "on a judged gate that reports NO VERDICT and exits 0 rather than going "
        "red:\n  " + "\n  ".join(offences)
    )
