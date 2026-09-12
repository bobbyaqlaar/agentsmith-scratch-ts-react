"""
runtime/test/test_gateway_provider_catalog.py — one provider catalog, honoured.

`provider -> API key env var` existed three times: the dict in
`runtime/provider_dispatch._DEFAULT_API_KEY_ENV`, the deliberate mirror in
`scripts/_shared._FALLBACK_API_KEY_ENV` (pinned by
scripts/test/test_judge_model_resolution.py — that one was already guarded),
and a third copy spelled as literals inside `LLMGateway._resolve_endpoint`'s
if/elif chain, which nothing checked.

The failure that shape produces is quiet: add a provider to the dict, forget the
branch, and its key resolves from `OPENAI_API_KEY` instead. The request goes out
with the wrong tenant's credential or none at all, the provider answers 401, and
`_is_provider_exhausted` treats an auth error as exhaustion and degrades down the
chain — so the operator sees "all model tiers exhausted", which reads as a
capacity problem.

This asserts the gateway resolves EVERY provider in the shared catalog from that
catalog's variable, so a fourth copy cannot come back silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.llm_gateway import LLMGateway
from runtime.provider_dispatch import _DEFAULT_API_KEY_ENV

# Providers whose catalog entry is None need no key (ollama is local; vertex_ai,
# bedrock and huawei_modelarts authenticate with cloud credentials, not a
# bearer token), so there is nothing for the gateway to look up.
KEYED_PROVIDERS = sorted(p for p, env in _DEFAULT_API_KEY_ENV.items() if env)


def test_the_catalog_is_not_empty() -> None:
    """Without this the parametrised test below would pass over no cases —
    a sweep that matches nothing is the failure mode this repo keeps finding."""
    assert len(KEYED_PROVIDERS) >= 5, f"expected the provider catalog, found {KEYED_PROVIDERS}"


@pytest.mark.parametrize("provider", KEYED_PROVIDERS)
def test_every_provider_resolves_its_key_from_the_shared_catalog(provider, monkeypatch) -> None:
    env_var = _DEFAULT_API_KEY_ENV[provider]
    sentinel = f"sentinel-for-{provider}"
    # Clear every key in the catalog first, so a provider falling through to
    # another provider's variable is visible rather than masked by a real key
    # in the developer's environment.
    for other in _DEFAULT_API_KEY_ENV.values():
        if other:
            monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(env_var, sentinel)

    _, api_key = LLMGateway._resolve_endpoint({"provider": provider})
    assert api_key == sentinel, (
        f"provider {provider!r} resolved its key from something other than "
        f"{env_var} — the gateway is not reading the shared catalog"
    )


def test_a_per_role_override_still_wins(monkeypatch) -> None:
    """`api_key_env` is how a tenant gives the judge role its own account
    (RFC-002 judge/actor separation at the account level). Collapsing the
    catalog must not collapse that."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-account")
    monkeypatch.setenv("ANTHROPIC_API_KEY_JUDGE", "judge-account")
    _, api_key = LLMGateway._resolve_endpoint(
        {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY_JUDGE"}
    )
    assert api_key == "judge-account"


def test_an_override_that_is_unset_falls_back_rather_than_going_dark(monkeypatch) -> None:
    """Documented behaviour: a tenant rolling out a dedicated key for one role
    at a time must not have both roles stop working in between."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-account")
    monkeypatch.delenv("ANTHROPIC_API_KEY_JUDGE", raising=False)
    _, api_key = LLMGateway._resolve_endpoint(
        {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY_JUDGE"}
    )
    assert api_key == "shared-account"
