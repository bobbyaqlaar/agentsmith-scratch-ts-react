from __future__ import annotations

from pathlib import Path

import pytest

from runtime.tool_registry import (
    ToolNotAllowedError,
    ToolRegistry,
    load_allowlist,
    tool,
)


def test_register_and_invoke_tool() -> None:
    registry = ToolRegistry()

    @tool(name="add", description="Add two numbers", registry=registry)
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in registry.names()
    schema = registry.get_schema("add")
    assert schema["name"] == "add"
    assert "a" in schema["parameters"]["properties"]
    assert registry.invoke("add", {"a": 2, "b": 3}) == 5


def test_allowlist_permits_listed_tool(tmp_path: Path) -> None:
    allow = tmp_path / "tool_allowlist.yaml"
    allow.write_text(
        "version: 1\ntools:\n  - name: echo\n    allowed: true\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(allowlist_path=allow, strict=True)

    @tool(name="echo", description="Echo text", registry=registry)
    def echo(text: str) -> str:
        return text

    assert registry.invoke("echo", {"text": "hi"}) == "hi"


def test_allowlist_denies_unlisted_tool_in_strict(tmp_path: Path) -> None:
    allow = tmp_path / "tool_allowlist.yaml"
    allow.write_text(
        "version: 1\ntools:\n  - name: echo\n    allowed: true\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(allowlist_path=allow, strict=True)

    @tool(name="secret_tool", description="Should be denied", registry=registry)
    def secret_tool() -> str:
        return "nope"

    with pytest.raises(ToolNotAllowedError):
        registry.invoke("secret_tool", {})


def test_load_allowlist_reads_template() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "fixtures" / "security" / "templates" / "tool_allowlist.yaml"
    names = load_allowlist(path)
    assert "example_tool" in names


# ── The registry `@tool(...)` uses when the caller names none (pass 14) ──────
#
# It was `_DEFAULT_REGISTRY = ToolRegistry(strict=False)` at module scope. Two
# things followed from that.
#
# The hardcoded False sat thirty lines under the constructor that resolves
# `security.tool_allowlist_strict` from tenant.yaml and TOOL_ALLOWLIST_STRICT
# from the environment — so a tenant declaring deny-by-default got it on every
# registry EXCEPT the default one, and the bare `@tool(name=...)` form is the
# one SPECS.md and OPERATIONS.md name as the API.
#
# And it was private with no accessor, so a tool registered through that form
# could not be invoked through any registry at all: `tool()` hands back the
# function unchanged and the allowlist only binds inside `invoke`.
#
# Every test above passes `registry=`, which is why neither showed.


import runtime.tool_registry as tr


@pytest.fixture(autouse=True)
def _fresh_default_registry(monkeypatch):
    """The default registry is process-global and cached on first use."""
    monkeypatch.setattr(tr, "_DEFAULT_REGISTRY", None)
    yield
    tr._DEFAULT_REGISTRY = None


def test_the_default_registry_honours_a_declared_strict_allowlist(
    monkeypatch, tmp_path: Path
) -> None:
    allow = tmp_path / "tool_allowlist.yaml"
    allow.write_text("version: 1\ntools:\n  - name: listed\n    allowed: true\n", encoding="utf-8")
    monkeypatch.setenv("TOOL_ALLOWLIST_PATH", str(allow))
    monkeypatch.setenv("TOOL_ALLOWLIST_STRICT", "1")

    @tool(name="unlisted_by_default")
    def unlisted_by_default() -> str:
        return "should be denied"

    with pytest.raises(ToolNotAllowedError):
        tr.default_registry().invoke("unlisted_by_default", {})


def test_the_default_registry_is_reachable_at_all(monkeypatch, tmp_path: Path) -> None:
    """A registration nothing can invoke is a registration that did not happen."""
    allow = tmp_path / "tool_allowlist.yaml"
    allow.write_text("version: 1\ntools:\n  - name: greet\n    allowed: true\n", encoding="utf-8")
    monkeypatch.setenv("TOOL_ALLOWLIST_PATH", str(allow))
    monkeypatch.setenv("TOOL_ALLOWLIST_STRICT", "1")

    @tool(name="greet")
    def greet(who: str) -> str:
        return f"hello {who}"

    assert "greet" in tr.default_registry().names()
    assert tr.default_registry().invoke("greet", {"who": "ops"}) == "hello ops"


def test_importing_the_module_builds_no_registry(monkeypatch) -> None:
    """Constructing one reads tenant.yaml and stats the allowlist path.
    `import runtime.tool_registry` should do neither."""
    assert tr._DEFAULT_REGISTRY is None
    tr.default_registry()
    assert tr._DEFAULT_REGISTRY is not None
