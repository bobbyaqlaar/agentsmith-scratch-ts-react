"""
runtime/test/test_config.py — the two channels the runtime reads configuration
from, and the precedence between them.

Both existed and neither reached the runtime. `.env` was loaded only by
`scripts/`, so a worker outside an already-exported shell ran on defaults —
which is why the detached eval script opens with `set -a; . ./.env`. And
tenant.yaml declared `budget.monthly_usd_cap`, `workflow.task_queue` and
`tenant.owner` that nothing read: KYC declared a $5 cap while the gateway
enforced $150 whenever AGENT_MONTHLY_USD_CAP was unset, which it is in
production because .env is not deployed to Cloud Run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.config import (
    config_get,
    load_env_file,
    resolve,
    tenant_config,
)


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """`_ENV_FILE` and `_SHADOWED` are process-lifetime by design — a worker
    loads .env once — so tests must clear them or one case's .env leaks into the
    next and the precedence under test is not the one being exercised."""
    import runtime.config as cfg

    cfg._ENV_FILE.clear()
    cfg._SHADOWED.clear()
    yield
    cfg._ENV_FILE.clear()
    cfg._SHADOWED.clear()


@pytest.fixture
def repo(tmp_path: Path):
    """A scaffolded repo root: .agenticframework/tenant.yaml plus optional .env."""
    (tmp_path / ".agenticframework").mkdir()

    def _build(declared: dict | None = None, dotenv: str | None = None) -> Path:
        (tmp_path / ".agenticframework" / "tenant.yaml").write_text(
            yaml.dump(declared or {})
        )
        if dotenv is not None:
            (tmp_path / ".env").write_text(dotenv)
        tenant_config(tmp_path, refresh=True)
        return tmp_path

    return _build


# ── .env loading ─────────────────────────────────────────────────────────────


def test_loads_env_file(repo, monkeypatch):
    root = repo(dotenv="FOO_ONE=alpha\nFOO_TWO=beta\n")
    monkeypatch.delenv("FOO_ONE", raising=False)
    monkeypatch.delenv("FOO_TWO", raising=False)
    import os

    assert load_env_file(root) == 2
    assert os.environ["FOO_ONE"] == "alpha"


def test_the_mirror_never_overwrites_the_environment(repo, monkeypatch):
    """`.env` is mirrored into os.environ for third-party libraries that read it
    directly, and that mirror must not change what a container was configured
    with. `resolve()` reads the parsed dict instead, so the mirror's precedence
    is not the framework's precedence — see the next test."""
    root = repo(dotenv="FOO_SET=from-file\n")
    monkeypatch.setenv("FOO_SET", "from-environment")
    import os

    load_env_file(root)
    assert os.environ["FOO_SET"] == "from-environment"



def test_absent_env_file_is_not_an_error(repo):
    assert load_env_file(repo()) == 0


def test_parses_quotes_comments_and_export(repo, monkeypatch):
    root = repo(dotenv='export FOO_Q="quoted value"\nFOO_C=bare # trailing\n')
    for key in ("FOO_Q", "FOO_C"):
        monkeypatch.delenv(key, raising=False)
    import os

    load_env_file(root)
    assert os.environ["FOO_Q"] == "quoted value"
    assert os.environ["FOO_C"] == "bare", "an inline comment is not part of the value"


# ── tenant.yaml ──────────────────────────────────────────────────────────────


def test_dotted_lookup(repo):
    root = repo({"budget": {"monthly_usd_cap": 5}, "tenant": {"id": "acme"}})
    assert config_get("budget.monthly_usd_cap", root) == 5
    assert config_get("tenant.id", root) == "acme"
    assert config_get("budget.nope", root) is None
    assert config_get("nope.nope", root) is None


# ── precedence ───────────────────────────────────────────────────────────────


def test_dotenv_outranks_the_ambient_shell(repo, monkeypatch):
    """The requirement this module exists for. A value the tenant put in .env
    must win over one that merely happens to be exported in the shell that
    launched the process — the two are indistinguishable in os.environ, which
    is why provenance is tracked rather than inferred."""
    root = repo({}, dotenv="CAP_VAR=7\n")
    monkeypatch.setenv("CAP_VAR", "999")
    load_env_file(root)
    assert resolve("budget.monthly_usd_cap", env_var="CAP_VAR",
                   default=150.0, cast=float, root=root) == 7.0


def test_a_declaration_outranks_the_ambient_shell(repo, monkeypatch):
    """AGENT_OWNER_ID in ~/.zshrc used to beat every tenant's declared
    tenant.owner on the machine. It no longer does."""
    root = repo({"tenant": {"owner": "declared@example.com"}})
    monkeypatch.setenv("AGENT_OWNER_ID", "ambient@example.com")
    assert resolve("tenant.owner", env_var="AGENT_OWNER_ID",
                   default=None, root=root) == "declared@example.com"


def test_an_ignored_ambient_value_is_reported_not_swallowed(repo, monkeypatch):
    """Silently ignoring an operator's variable is its own trap: they export
    something, see no effect, and conclude the framework is broken."""
    from runtime.config import shadowed_env

    root = repo({"tenant": {"owner": "declared@example.com"}})
    monkeypatch.setenv("AGENT_OWNER_ID", "ambient@example.com")
    resolve("tenant.owner", env_var="AGENT_OWNER_ID", default=None, root=root)
    assert shadowed_env().get("AGENT_OWNER_ID") == "ambient@example.com"


def test_explicit_still_beats_everything(repo, monkeypatch):
    root = repo({"budget": {"monthly_usd_cap": 5}}, dotenv="CAP_VAR=7\n")
    monkeypatch.setenv("CAP_VAR", "999")
    load_env_file(root)
    assert resolve("budget.monthly_usd_cap", explicit=1, env_var="CAP_VAR",
                   default=150.0, cast=float, root=root) == 1.0


def test_an_undeclared_key_still_comes_from_the_environment(repo, monkeypatch):
    """Secrets. An API key is declared in no file, so the ambient environment is
    its only source and --set-secrets must keep reaching it."""
    root = repo({})
    monkeypatch.setenv("SOME_API_KEY", "sk-live-xyz")
    assert resolve("secrets.some_api_key", env_var="SOME_API_KEY",
                   default=None, root=root) == "sk-live-xyz"


def test_env_overrides_restores_ambient_precedence(repo, monkeypatch):
    """The declared exception. Raising a cap without a redeploy is legitimate —
    it just has to be said in the file, where it can be reviewed."""
    root = repo({"budget": {"monthly_usd_cap": 5},
                 "env_overrides": ["AGENT_MONTHLY_USD_CAP"]})
    monkeypatch.setenv("AGENT_MONTHLY_USD_CAP", "500")
    assert resolve("budget.monthly_usd_cap", env_var="AGENT_MONTHLY_USD_CAP",
                   default=150.0, cast=float, root=root) == 500.0



def test_the_regression_a_declared_cap_beats_the_code_default(repo, monkeypatch):
    """The live gap this closes. tenant.yaml said $5, the gateway enforced $150
    whenever the env var was unset — and it is unset in production, because
    .env is not deployed. A 30x breach resting on somebody remembering
    --set-env-vars."""
    root = repo({"budget": {"monthly_usd_cap": 5}})
    monkeypatch.delenv("AGENT_MONTHLY_USD_CAP", raising=False)
    cap = resolve("budget.monthly_usd_cap", env_var="AGENT_MONTHLY_USD_CAP",
                  default=150.0, cast=float, root=root)
    assert cap == 5.0, "the declared policy must win over the code default"


def test_falls_back_to_the_default_when_nothing_declares_it(repo, monkeypatch):
    monkeypatch.delenv("CAP_VAR", raising=False)
    assert resolve("budget.monthly_usd_cap", env_var="CAP_VAR", default=150.0,
                   cast=float, root=repo()) == 150.0


def test_no_default_means_raise_not_invent(repo, monkeypatch):
    """For anything an auditor would read as a control, an unresolved value
    must stop the process rather than become a plausible number."""
    monkeypatch.delenv("CAP_VAR", raising=False)
    with pytest.raises(LookupError) as exc:
        resolve("budget.monthly_usd_cap", env_var="CAP_VAR", root=repo())
    assert "tenant.yaml" in str(exc.value)


def test_a_malformed_value_raises_rather_than_falling_through(repo, monkeypatch):
    """Falling back to a lower-precedence source would silently ignore whoever
    set the bad value."""
    root = repo({"budget": {"monthly_usd_cap": 5},
                 "env_overrides": ["CAP_VAR"]})
    monkeypatch.setenv("CAP_VAR", "not-a-number")
    with pytest.raises(ValueError) as exc:
        resolve("budget.monthly_usd_cap", env_var="CAP_VAR", default=150.0,
                cast=float, root=root)
    assert "CAP_VAR" in str(exc.value)



def test_a_blank_override_is_not_an_override(repo, monkeypatch):
    root = repo({"budget": {"monthly_usd_cap": 5}})
    monkeypatch.setenv("CAP_VAR", "   ")
    assert resolve("budget.monthly_usd_cap", env_var="CAP_VAR", default=150.0,
                   cast=float, root=root) == 5.0


# ── Security posture: declared in tenant.yaml, overridden by env ─────────────


@pytest.mark.parametrize(
    "dotted,env_var,declared,expected",
    [
        ("security.prompt_guard", "PROMPT_GUARD", "strict", "strict"),
        ("security.input_guardrail", "INPUT_GUARDRAIL", "off", "off"),
        ("moderation.mode", "MODERATION_HOOK", "required", "required"),
    ],
)
def test_posture_is_readable_from_the_declaration(
    repo, monkeypatch, dotted, env_var, declared, expected
):
    """These were reachable only through environment variables, so a tenant's
    security posture lived nowhere reviewable — not in the repo, not in a diff,
    not in anything an auditor could be handed."""
    section, key = dotted.split(".")
    root = repo({section: {key: declared}})
    monkeypatch.delenv(env_var, raising=False)
    assert resolve(dotted, env_var=env_var, default="", root=root) == expected


def test_a_declared_posture_is_not_overridden_by_an_ambient_export(repo, monkeypatch):
    """The posture a tenant declares is the posture it runs. An exported
    PROMPT_GUARD from someone's shell profile must not quietly relax it."""
    root = repo({"security": {"prompt_guard": "strict"}})
    monkeypatch.setenv("PROMPT_GUARD", "off")
    assert resolve("security.prompt_guard", env_var="PROMPT_GUARD",
                   default="", root=root) == "strict"



def test_a_declared_false_flag_is_honoured_not_treated_as_absent(repo, monkeypatch):
    """`False` is a declared value; `None` is an absent one. Collapsing them
    would make `tool_allowlist_strict: false` fall through to the default,
    which for a deny-by-default guard is the wrong direction to guess."""
    from runtime.config import as_bool

    root = repo({"security": {"tool_allowlist_strict": False}})
    monkeypatch.delenv("TOOL_ALLOWLIST_STRICT", raising=False)
    assert resolve("security.tool_allowlist_strict",
                   env_var="TOOL_ALLOWLIST_STRICT", default=True, root=root) is False
    assert as_bool(False) is False


def test_yaml_bare_off_would_have_been_a_boolean(repo, monkeypatch):
    """Why every mode in the scaffold is quoted. YAML 1.1 parses a bare `off`
    as the boolean false, so `prompt_guard: off` would resolve to False and
    match no mode name at all."""
    import yaml as _yaml

    assert _yaml.safe_load("mode: off")["mode"] is False
    assert _yaml.safe_load('mode: "off"')["mode"] == "off"


def test_as_bool_reads_both_channels(monkeypatch):
    from runtime.config import as_bool

    assert as_bool(True) is True and as_bool(False) is False       # yaml
    assert as_bool("1") and as_bool("true") and as_bool("YES")     # env
    assert not as_bool("0") and not as_bool("") and not as_bool("maybe")


def test_the_enforced_budget_key_is_the_one_the_portal_reports(repo):
    """One concept, one key. `gateway.budget_cap_usd` fed the Ops Portal's
    display while `budget.monthly_usd_cap` fed the gateway's enforcement, so a
    tenant declaring the enforced key showed NO cap on the dashboard while a
    cap was in force. The old key is accepted as a fallback, not a rival."""
    import importlib.util
    import sys as _sys

    path = Path(__file__).resolve().parent.parent.parent / "scripts" / "sync-portal-history.py"
    _sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("syncph", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._budget_cap_usd({"budget": {"monthly_usd_cap": 5}}) == 5.0
    assert mod._budget_cap_usd({"gateway": {"budget_cap_usd": 9}}) == 9.0, "legacy still works"
    assert mod._budget_cap_usd(
        {"budget": {"monthly_usd_cap": 5}, "gateway": {"budget_cap_usd": 9}}
    ) == 5.0, "the enforced key wins when both are present"
    assert mod._budget_cap_usd({}) is None


def test_override_is_the_deliberate_channel_the_strict_rule_leaves_room_for(repo, monkeypatch):
    """The strict rule cannot tell an operator's one-off `PROMPT_GUARD=off`
    from a line that leaked in from a login shell — both are just os.environ.
    So it treats a declared key as declared, and genuine deliberate overrides
    need a channel that is visible in the code doing it and scoped to a block."""
    from runtime.config import override

    root = repo({"security": {"prompt_guard": "strict"}})
    monkeypatch.setenv("PROMPT_GUARD", "off")

    assert resolve("security.prompt_guard", env_var="PROMPT_GUARD",
                   default="", root=root) == "strict"
    with override(**{"security.prompt_guard": "off"}):
        assert resolve("security.prompt_guard", env_var="PROMPT_GUARD",
                       default="", root=root) == "off"
    assert resolve("security.prompt_guard", env_var="PROMPT_GUARD",
                   default="", root=root) == "strict", "restored on exit"


def test_overrides_nest_and_restore(repo):
    from runtime.config import override

    root = repo({"security": {"prompt_guard": "strict"}})
    with override(**{"security.prompt_guard": "warn"}):
        with override(**{"security.prompt_guard": "off"}):
            assert resolve("security.prompt_guard", default="", root=root) == "off"
        assert resolve("security.prompt_guard", default="", root=root) == "warn"
    assert resolve("security.prompt_guard", default="", root=root) == "strict"


def test_one_root_finder_and_a_tenant_beats_its_parent_repo(tmp_path: Path):
    """There were FIVE implementations of "find the repo root" in three
    disagreeing variants — some accepting `.agenticframework` as a marker, some
    only `.git`. A tenant nested inside a parent git repo resolved to the parent
    under one and to the tenant under the other, so tenant.yaml and models.yaml
    could load from different directories in a single process.

    The structural duplicate sweep in pass 3 missed all five: their bodies are
    under its four-statement threshold. A sweep's blind spot is a finding too.
    """
    import os

    from runtime.config import repo_root

    outer = tmp_path / "outer"
    (outer / ".git").mkdir(parents=True)
    tenant = outer / "tenant"
    (tenant / ".agenticframework").mkdir(parents=True)

    assert repo_root(tenant) == tenant, "the tenant wins over the repo containing it"
    assert repo_root(outer) == outer

    # And every module now asks the same function.
    from runtime import llm_gateway, moderation, tracing
    from _shared import _repo_root as shared_root

    cwd = Path.cwd()
    try:
        os.chdir(tenant)
        answers = {
            llm_gateway._repo_root(),
            moderation._repo_root(),
            tracing._repo_root(),
            shared_root(),
            repo_root(),
        }
    finally:
        os.chdir(cwd)
    assert len(answers) == 1, f"root finders disagree: {answers}"
    assert answers.pop() == tenant


def test_one_truthy_catalog_across_the_runtime():
    """`runtime/temporal_client` carried its own `_TRUTHY = {"1","true","yes","on"}`
    and `config.as_bool` grew an identical set under a different name. Two
    catalogs of one fact is how one of them ends up accepting a spelling the
    other rejects — and this one gates TLS."""
    from runtime.config import as_bool
    from runtime.temporal_client import tls_enabled

    for spelling in ("1", "true", "TRUE", "yes", "on"):
        assert as_bool(spelling) is True
        assert tls_enabled({"TEMPORAL_TLS": spelling}) is True
    for spelling in ("0", "", "no", "off", "maybe"):
        assert as_bool(spelling) is False
        assert tls_enabled({"TEMPORAL_TLS": spelling}) is False
