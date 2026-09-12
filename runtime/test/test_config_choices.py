"""
runtime/test/test_config_choices.py — the settings whose value is a WORD.

security.input_guardrail, security.prompt_guard and moderation.mode each take
one of a fixed set of words, and `off` is a documented value of all three.

YAML 1.1 parses a bare `off` as the boolean False. So a tenant who wrote the
value the documentation told them to write got False, which matches no mode,
and their declaration was discarded in favour of the fallback with nothing
logged. The posture an auditor reads in tenant.yaml was not the posture in
force. `on`, `yes` and `no` coerce the same way.

Found by reading mutmut's survivors: the config key in
input_guardrail.resolve_mode could be replaced with a garbage string and every
test still passed, which meant no test exercised the tenant.yaml path at all.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime import config, environment


@pytest.fixture
def tenant_yaml(tmp_path, monkeypatch):
    """Write a tenant.yaml and resolve against it, with warn-once state cleared."""

    # A FRESH cache dict per test, restored by monkeypatch on teardown.
    # tenant_config caches per repo root, and refresh=True repopulates that
    # shared dict — so without this the probe tenant.yaml stayed cached after
    # the test ended and every later test in the process resolved its guard
    # modes from it. Five prompt_guard tests failed that way, none of them in
    # isolation, which is what test pollution looks like from the outside.
    monkeypatch.setattr(config, "_CACHE", {})

    def _write(body: str, environment_name: str = "production"):
        (tmp_path / ".git").mkdir(exist_ok=True)  # make tmp_path the repo root
        (tmp_path / ".agenticframework").mkdir(exist_ok=True)
        (tmp_path / ".agenticframework" / "tenant.yaml").write_text(body, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        for var in ("INPUT_GUARDRAIL", "PROMPT_GUARD", "MODERATION_HOOK"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ENVIRONMENT", environment_name)
        monkeypatch.setattr(environment, "_degraded_warned", set())
        config.tenant_config(refresh=True)

        # NOT reloaded. These resolvers read config on every call, so a reload
        # buys nothing — and it costs class identity: importlib.reload rebinds
        # PromptGuardBlockedError to a NEW class object while llm_gateway still
        # holds the old one, so the gateway's raise stopped matching the
        # `pytest.raises` in four unrelated tests. Only visible in a full run.
        from runtime import input_guardrail, moderation, prompt_guard

        return (
            input_guardrail.resolve_mode(),
            prompt_guard.resolve_mode(),
            moderation.resolve_mode(),
        )

    return _write


BARE_OFF = """
tenant:
  id: probe
security:
  input_guardrail: off
  prompt_guard: off
moderation:
  mode: off
"""

QUOTED_OFF = """
tenant:
  id: probe
security:
  input_guardrail: "off"
  prompt_guard: "off"
moderation:
  mode: "off"
"""


def test_a_bare_off_is_reported_not_obeyed_and_not_ignored(tenant_yaml, caplog):
    """The whole finding in one case.

    `off` is the documented word. YAML hands over False. The guards must stay
    CLOSED — this is a security posture and a value nobody can parse is not
    permission to stop enforcing — and the tenant must be told, because
    silently running the opposite of a declared policy is the actual defect.
    """
    with caplog.at_level(logging.WARNING):
        modes = tenant_yaml(BARE_OFF)

    assert modes == ("default", "default", "optional"), "a guard was disabled by a boolean"
    assert "YAML boolean" in caplog.text
    for key in ("security.input_guardrail", "security.prompt_guard", "moderation.mode"):
        assert key in caplog.text, f"{key} was discarded without a word"


def test_the_remedy_in_the_warning_actually_works(tenant_yaml):
    """The message tells the tenant to quote it. That has to be true."""
    assert tenant_yaml(QUOTED_OFF) == ("off", "off", "off")


def test_false_does_not_disable_a_guard(tenant_yaml):
    """The trap in the obvious fix.

    Translating the boolean back to a word would read `prompt_guard: false` as
    "disable the guard" — turning a control off by writing something that was
    never a valid value for it. The boolean is reported, never interpreted.
    """
    body = "tenant:\n  id: probe\nsecurity:\n  prompt_guard: false\n"
    assert tenant_yaml(body)[1] == "default"


def test_an_unrecognised_value_is_reported(tenant_yaml, caplog):
    """A typo used to be replaced in silence, so a tenant who wrote `warnn` got
    the blocking default and no sign their policy had been ignored."""
    with caplog.at_level(logging.WARNING):
        modes = tenant_yaml("tenant:\n  id: probe\nsecurity:\n  prompt_guard: warnn\n")

    assert modes[1] == "default"
    assert "warnn" in caplog.text
    assert "accepted" in caplog.text


def test_an_unset_value_says_nothing(tenant_yaml, caplog):
    """Unset is the normal case and must stay quiet, or the warning that
    matters gets filtered out with the noise."""
    with caplog.at_level(logging.WARNING):
        modes = tenant_yaml("tenant:\n  id: probe\n")

    assert modes == ("default", "default", "optional")
    assert caplog.text.strip() == ""


def test_development_still_defaults_the_input_guard_off(tenant_yaml):
    """The per-caller fallback survived the refactor: this one is
    environment-dependent, the other two are not."""
    assert tenant_yaml("tenant:\n  id: probe\n", "development")[0] == "off"


def test_a_valid_word_is_honoured(tenant_yaml):
    """The guard must not swallow the values it exists to pass through."""
    body = (
        "tenant:\n  id: probe\nsecurity:\n  prompt_guard: warn\n"
        "  input_guardrail: custom\nmoderation:\n  mode: required\n"
    )
    assert tenant_yaml(body) == ("custom", "warn", "required")


def test_the_block_alias_still_maps_to_default(tenant_yaml):
    """`block` is an explicit alias that predates this change."""
    assert tenant_yaml("tenant:\n  id: probe\nsecurity:\n  prompt_guard: block\n")[1] == "default"
