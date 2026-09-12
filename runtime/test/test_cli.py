"""
runtime/test/test_cli.py — the tests the shell functions could never have.

`ai-tenant-init` was a zsh function that `install-ai-stack.sh` appended to
~/.zshrc, so it existed only after an interactive install on macOS or Linux.
Nothing tested it, nothing could: there was no importable artifact, and the
scaffold it wrote lived in two places that drifted apart twice in one day.

These assert on the scaffold as a value.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import cli
from runtime.cli import (
    ISOLATIONS,
    STACKS,
    build_parser,
    init_tenant,
    main,
    tenant_yaml,
)


# ── the scaffold as data ─────────────────────────────────────────────────────


def test_scaffold_parses_and_carries_the_declared_id():
    doc = yaml.safe_load(tenant_yaml("acme"))
    assert doc["tenant"]["id"] == "acme"
    assert doc["workflow"]["task_queue"] == "acme"


def test_every_scaffolded_key_is_one_the_runtime_reads():
    """The entry criterion, learned from shipping five keys that nothing read.
    `environments:` with phoenix_namespace / eval_fail_below / redaction_profile
    used to be here; two of the three were actively misleading."""
    doc = yaml.safe_load(tenant_yaml("acme"))
    assert set(doc) == {
        "tenant", "framework", "security", "moderation", "budget", "workflow", "delivery"
    }
    assert "environments" not in doc


def test_modes_are_strings_not_booleans():
    """YAML 1.1 parses a bare `off` as False, which matches no mode name. Every
    mode in the scaffold is quoted; this is what proves it stayed that way."""
    doc = yaml.safe_load(tenant_yaml("acme"))
    assert doc["security"]["prompt_guard"] == "default"
    assert doc["moderation"]["mode"] == "optional"
    assert isinstance(doc["security"]["prompt_guard"], str)
    # And the flags really are booleans, not the strings "false".
    assert doc["security"]["tool_allowlist_strict"] is False
    assert doc["security"]["ip_redaction"] is False


def test_env_overrides_ships_commented_out():
    """Empty by default: an ambient export must not silently relax a declared
    posture. The line is present as documentation of the escape hatch."""
    text = tenant_yaml("acme")
    doc = yaml.safe_load(text)
    assert "env_overrides" not in doc
    assert "# env_overrides:" in text


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_isolation_round_trips(isolation):
    doc = yaml.safe_load(tenant_yaml("acme", isolation=isolation))
    assert doc["tenant"]["isolation"] == isolation


# ── init_tenant ──────────────────────────────────────────────────────────────


def test_init_writes_the_config(tmp_path: Path):
    written = init_tenant("acme", tmp_path)
    assert ".agenticframework/tenant.yaml" in written
    doc = yaml.safe_load((tmp_path / ".agenticframework" / "tenant.yaml").read_text())
    assert doc["tenant"]["id"] == "acme"


def test_init_never_overwrites_without_force(tmp_path: Path):
    """A tenant.yaml is hand-edited after generation. Silently replacing it
    would discard a declared security posture."""
    init_tenant("acme", tmp_path)
    cfg = tmp_path / ".agenticframework" / "tenant.yaml"
    cfg.write_text("tenant:\n  id: edited-by-hand\n")

    written = init_tenant("acme", tmp_path)
    assert ".agenticframework/tenant.yaml" not in written
    assert "edited-by-hand" in cfg.read_text()

    init_tenant("acme", tmp_path, force=True)
    assert "edited-by-hand" not in cfg.read_text()


def test_ci_callees_ship_with_their_caller(tmp_path: Path):
    """A workflow referenced by `uses:` and not provisioned makes GitHub reject
    the WHOLE caller as invalid, not just the missing job — which is how every
    Python/FastAPI tenant was once scaffolded with unusable CI."""
    from runtime.cli import WORKFLOWS, _templates_dir

    if _templates_dir() is None:
        pytest.skip("no workflow-templates available")

    init_tenant("acme", tmp_path, stack="python-fastapi")
    caller = (tmp_path / ".github" / "workflows" / "ci-python-fastapi.yml").read_text()
    for name in WORKFLOWS:
        if f"./.github/workflows/{name}" in caller:
            assert (tmp_path / ".github" / "workflows" / name).is_file(), (
                f"ci-python-fastapi.yml uses {name} but init did not provision it"
            )


def test_unknown_stack_and_isolation_are_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="stack"):
        init_tenant("acme", tmp_path, stack="cobol")
    with pytest.raises(ValueError, match="isolation"):
        init_tenant("acme", tmp_path, isolation="whatever")


# ── the command surface ──────────────────────────────────────────────────────


def test_parser_covers_the_shell_functions_being_replaced():
    parser = build_parser()
    for argv in (
        ["tenant", "init", "acme"],
        ["tenant", "init", "acme", "--stack", "go", "--isolation", "dedicated"],
        ["shellenv", "--mode", "hybrid"],
        ["version"],
        ["doctor"],
        ["purge-idempotency"],
    ):
        assert parser.parse_args(argv) is not None


def test_every_subcommand_dispatches_somewhere():
    """A subcommand without `set_defaults(func=...)` parses fine and then does
    nothing — argparse does not mind, and `main` would raise AttributeError at
    the moment someone runs it. Registering the parser and forgetting the
    handler is one line apart in build_parser."""
    parser = build_parser()
    subparsers = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers, "the parser has no subcommands — this test is checking nothing"
    names = [name for action in subparsers for name in action.choices]
    assert len(names) >= 5, f"expected the full command set, found {names}"
    for name in names:
        if name == "tenant":
            continue  # a group, not a command — its children are checked below
        args = parser.parse_args([name])
        assert callable(getattr(args, "func", None)), f"`agentsmith {name}` dispatches to nothing"


def test_shellenv_emits_evaluable_exports(capsys):
    """The one thing a child process cannot do for its parent. Five generated
    lines instead of sixty-one hand-maintained ones in a profile."""
    assert main(["shellenv", "--mode", "local"]) == 0
    out = capsys.readouterr().out
    assert 'export AI_STACK_MODE="local"' in out
    for line in out.strip().splitlines():
        assert line.startswith("export "), f"not evaluable: {line!r}"


def test_main_returns_two_on_a_bad_argument(tmp_path: Path, capsys):
    """argparse rejects an unknown --stack itself; this covers the ValueError
    path for a caller reaching init_tenant with one anyway.

    Built through the parser and then mutated, NOT hand-constructed as a
    Namespace: a hand-built one duplicates the parser's argument list and
    breaks the moment an argument is added — which it did, the first time one
    was.
    """
    from runtime.cli import _cmd_tenant_init

    args = build_parser().parse_args(["tenant", "init", "acme", "--root", str(tmp_path)])
    args.stack = "cobol"
    assert _cmd_tenant_init(args) == 2
    assert "stack" in capsys.readouterr().err


def test_stacks_match_the_shipped_ci_templates():
    """Every stack the CLI offers must have a template, or `tenant init` writes
    a repo whose CI does not exist."""
    from runtime.cli import _templates_dir

    templates = _templates_dir()
    if templates is None:
        pytest.skip("no workflow-templates available")
    for stack in STACKS:
        assert (templates / f"ci-{stack}.yml").is_file(), f"no template for {stack}"


# ── the framework-root guard ─────────────────────────────────────────────────


def test_refuses_to_scaffold_into_the_framework_itself():
    """Written after doing exactly this by accident: `tenant init` defaults
    --root to cwd, and running it from the wrong terminal is a two-second
    mistake. Twelve tests went red because the stray tenant.yaml declared a
    security posture that outranked the environment they set."""
    from runtime.cli import FrameworkRootError, init_tenant

    framework = Path(__file__).resolve().parent.parent.parent
    with pytest.raises(FrameworkRootError) as exc:
        init_tenant("acme", framework)
    assert "agentsmith-runtime" in str(exc.value)
    assert "--allow-framework-root" in str(exc.value), "the error must name its own override"


def test_force_does_not_bypass_the_guard(tmp_path: Path):
    """`force` means "replace files I already own". Writing into the wrong
    repository is a different question, and overloading one flag onto both is
    how a guard gets disabled by someone solving an unrelated problem."""
    from runtime.cli import FrameworkRootError, init_tenant

    framework = Path(__file__).resolve().parent.parent.parent
    with pytest.raises(FrameworkRootError):
        init_tenant("acme", framework, force=True)


def test_the_override_works_when_meant(tmp_path: Path):
    from runtime.cli import init_tenant

    (tmp_path / "install-ai-stack.sh").write_text("#!/bin/sh\n")
    (tmp_path / "workflow-templates").mkdir()
    written = init_tenant("acme", tmp_path, allow_framework_root=True)
    assert ".agenticframework/tenant.yaml" in written


def test_two_markers_are_required_not_one(tmp_path: Path):
    """A single marker would refuse in a repo that merely vendored the
    installer, and a guard that fires on legitimate work is one people learn to
    bypass."""
    from runtime.cli import init_tenant, looks_like_framework

    (tmp_path / "install-ai-stack.sh").write_text("#!/bin/sh\n")
    assert looks_like_framework(tmp_path) is None, "one marker must not be enough"
    assert ".agenticframework/tenant.yaml" in init_tenant("acme", tmp_path)

    (tmp_path / "workflow-templates").mkdir()
    assert looks_like_framework(tmp_path) is not None, "two markers must trip it"


def test_the_package_name_marker_reads_the_file_not_the_filename(tmp_path: Path):
    """A tenant has a pyproject.toml too — it is the declared package name that
    identifies the framework, and that string exists in exactly one project."""
    from runtime.cli import looks_like_framework

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "some-tenant"\n')
    (tmp_path / "workflow-templates").mkdir()
    assert looks_like_framework(tmp_path) is None

    (tmp_path / "pyproject.toml").write_text('[project]\nname = "agentsmith-runtime"\n')
    assert looks_like_framework(tmp_path) is not None


def test_cli_exit_code_is_distinct_for_a_refusal(capsys):
    """3, not 1 or 2 — a script wrapping this can tell "wrong directory" from
    "bad argument" without parsing the message."""
    framework = Path(__file__).resolve().parent.parent.parent
    assert main(["tenant", "init", "acme", "--root", str(framework)]) == 3
    assert "refusing to scaffold here" in capsys.readouterr().err


# ── The tenant id has to survive being written and read back ──────────────────


@pytest.mark.parametrize(
    "tenant_id",
    ["acme", "off", "no", "yes", "on", "true", "null", "123", "1.5",
     "kyc-sentinel", "acme.corp", "a_1"],
)
def test_the_scaffolded_id_is_the_id_that_was_asked_for(tenant_id):
    """YAML 1.1 reads bare off/no/yes/on/true as booleans and 123 as an int.

    The id was interpolated unquoted, so `agentsmith tenant init off` wrote
    `id: off`, YAML read False, and runtime/tenancy.py resolved the tenant to
    the string "False" — which then keys the spend ledger, the
    HITL_ENCRYPTION_KEY_<TENANT> variable and every span.

    This file's own docstring already explained the trap and quoted the MODE
    values. The quoting had been applied to the values someone thought about
    rather than to the class of problem.
    """
    import yaml

    document = yaml.safe_load(cli.tenant_yaml(tenant_id))
    assert document["tenant"]["id"] == tenant_id
    assert isinstance(document["tenant"]["id"], str)
    assert document["tenant"]["name"] == tenant_id


def test_the_scaffold_round_trips_through_the_real_resolver(tmp_path, monkeypatch):
    """Not just yaml.safe_load — the function that actually reads this file."""
    import runtime.config as config
    from runtime.tenancy import resolve_tenant_id

    monkeypatch.setattr(config, "_CACHE", {})
    for var in ("AGENT_TENANT_ID", "TENANT_ID"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".agenticframework").mkdir()
    (tmp_path / ".agenticframework" / "tenant.yaml").write_text(
        cli.tenant_yaml("off"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    assert resolve_tenant_id() == "off", "the resolver saw a boolean, not the id"


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "a b", "acme\ncorp: x", 'acme"q', "-leading", "a:b", "x" * 65],
)
def test_an_unusable_tenant_id_is_refused_at_creation(bad):
    """The id is written into YAML, resolved back, and spliced into an
    environment variable NAME. Creation is the only moment it is free to
    change, so it is checked there rather than surfacing later as a broken
    config file or an unsettable key variable."""
    with pytest.raises(ValueError, match="tenant id"):
        cli.validate_tenant_id(bad)


def test_a_non_string_id_is_a_clear_error_not_a_typeerror():
    """The isinstance guard is what separates this from a TypeError out of
    re.match. The regex alone rejects "" and "   ", so mutation testing showed
    the guard doing nothing for every case that WAS tested — None is the case
    that needs it.
    """
    for value in (None, 123, ["acme"]):
        with pytest.raises(ValueError, match="non-empty string"):
            cli.validate_tenant_id(value)


def test_the_refusal_explains_the_shape():
    with pytest.raises(ValueError) as exc:
        cli.validate_tenant_id("a b")
    assert "letters, digits" in str(exc.value)
    assert "HITL_ENCRYPTION_KEY" in str(exc.value)


def test_init_refuses_a_bad_id_before_writing_anything(tmp_path):
    """The guard belongs before the first mkdir, not after a partial scaffold."""
    with pytest.raises(ValueError):
        cli.init_tenant("bad id", tmp_path, allow_framework_root=True)
    assert not (tmp_path / ".agenticframework").exists()


# ── The declared framework version ────────────────────────────────────────────


def test_the_scaffold_declares_the_installed_version():
    """It was the literal "1.3.0" in a default argument — a second copy of the
    version number that would drift from pyproject.toml at the next bump, and
    every tenant scaffolded after that would declare a stale release."""
    import yaml

    from runtime.version import SOURCE_SUFFIX, framework_version

    declared = yaml.safe_load(cli.tenant_yaml("acme"))["framework"]["version"]
    running = framework_version()
    expected = (
        running[: -len(SOURCE_SUFFIX)] if running.endswith(SOURCE_SUFFIX) else running
    )
    assert declared == expected


def test_the_declared_version_follows_the_module_not_a_literal(monkeypatch):
    """The assertion that a hardcoded literal cannot pass.

    Comparing the scaffold's version to framework_version() holds just as well
    when the scaffold hardcodes today's number — mutation testing put "1.3.0"
    back and every version test stayed green, because "1.3.0" is what
    framework_version() currently returns. Moving the module's answer is the
    only way to tell the two apart.
    """
    import yaml

    monkeypatch.setattr(cli, "_default_framework_version", lambda: "9.9.9")
    declared = yaml.safe_load(cli.tenant_yaml("acme"))["framework"]["version"]
    assert declared == "9.9.9", "the scaffold is not reading the version, it is a literal"


def test_the_declared_version_carries_no_source_marker():
    """framework_version() reports `1.3.0+src` from a checkout. That is the
    right answer for "what is running" and the wrong one to write into a
    tenant's config, which declares the RELEASE it targets."""
    import yaml

    declared = yaml.safe_load(cli.tenant_yaml("acme"))["framework"]["version"]
    assert "+src" not in declared
