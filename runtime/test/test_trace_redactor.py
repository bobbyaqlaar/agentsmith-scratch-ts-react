"""
runtime/test/test_trace_redactor.py — regression tests for trace redaction
(SPECS.md §27, Product_Archive.md 1.2 / 2.2 / 2.3 / 2.8).

No external infra required — these exercise TraceRedactor's pure scrubbing
logic and the per-span tenant binding directly, without a real OTel SDK
span. opentelemetry-sdk is in requirements.txt, so _HAS_OTEL is expected
True in CI; if it's somehow absent these tests are skipped rather than
failing for an unrelated reason.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import trace_redactor as tr
from runtime.environment import get_environment

pytestmark = pytest.mark.skipif(
    not tr._HAS_OTEL, reason="opentelemetry-sdk not installed"
)


class FakeSpanContext:
    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id


class FakeSpan:
    """Minimal stand-in for opentelemetry.sdk.trace.ReadableSpan — only the
    attributes TraceRedactor.on_end() actually reads."""

    def __init__(self, attributes: dict, trace_id: int = 1, span_id: int = 1):
        self._attributes = dict(attributes)
        self.context = FakeSpanContext(trace_id, span_id)


def _redactor(profile: str, tenant_id: str = "unknown", **kw) -> tr.TraceRedactor:
    return tr.TraceRedactor(profile=profile, tenant_id=tenant_id, **kw)


def test_staging_profile_hashes_secrets_preserving_structure():
    redactor = _redactor("staging")
    span = FakeSpan(
        {
            "input.value": "Authorization: Bearer sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"
        }
    )
    redactor.on_end(span)
    scrubbed = span._attributes["input.value"]
    assert "sk-ant-" not in scrubbed
    assert "[REDACTED:" in scrubbed  # hashed marker, not a flat redaction


def test_production_profile_truncates_and_flattens():
    redactor = _redactor("production", tenant_id="acme")
    long_value = "x" * 200 + " contact someone@example.com for help"
    span = FakeSpan({"input.value": long_value})
    redactor.on_end(span)
    scrubbed = span._attributes["input.value"]
    assert len(scrubbed) <= 50 + len("…[truncated]")
    assert "@example.com" not in scrubbed


def test_development_profile_does_not_scrub():
    redactor = _redactor("development")
    span = FakeSpan({"input.value": "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"})
    redactor.on_end(span)
    assert (
        span._attributes["input.value"] == "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"
    )


def test_unrecognized_profile_value_falls_back_to_strictest():
    """Product_Archive.md 2.8: an unrecognized profile string passed
    directly must resolve to the strictest behavior, not the most
    permissive — mirrors get_environment()'s own fail-closed contract."""
    redactor = _redactor("totally-bogus-value")
    assert redactor.profile == "production"


def test_per_span_tenant_id_used_for_hitl_blob_not_constructor_default(
    tmp_path, monkeypatch
):
    """Product_Archive.md 1.2: the regression this guards against is a
    shared worker pool binding tenant_id once at construction time and
    leaking tenant A's HITL blob under tenant B's key (or vice versa)."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY", "test-key-not-a-real-secret")

    # Constructed with a default/fallback tenant_id of "default-tenant" —
    # the span itself carries a DIFFERENT tenant.id attribute, which must win.
    redactor = _redactor("production", tenant_id="default-tenant")
    payload = "x" * 100  # > 50 chars, triggers the HITL blob path
    span = FakeSpan(
        {"input.value": payload, "tenant.id": "real-tenant"}, trace_id=42, span_id=7
    )
    redactor.on_end(span)

    assert (tmp_path / "real-tenant").is_dir(), (
        "blob was not stored under the span's own tenant.id"
    )
    assert not (tmp_path / "default-tenant").exists(), (
        "blob was incorrectly stored under the constructor fallback tenant_id"
    )


def test_blob_ref_includes_span_id_for_collision_safety(tmp_path, monkeypatch):
    """Product_Archive.md 2.2: two sibling spans in the same trace with
    the same attr_key must not collide on the same blob ref."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY", "test-key-not-a-real-secret")

    redactor = _redactor("production", tenant_id="acme")
    payload = "y" * 100
    span_a = FakeSpan({"input.value": payload}, trace_id=99, span_id=1)
    span_b = FakeSpan({"input.value": payload}, trace_id=99, span_id=2)
    redactor.on_end(span_a)
    redactor.on_end(span_b)

    ref_a = span_a._attributes["input.value.hitl_blob_ref"]
    ref_b = span_b._attributes["input.value.hitl_blob_ref"]
    assert ref_a != ref_b, (
        "sibling spans in the same trace collided on the same blob ref"
    )

    blob_dir = tmp_path / "acme"
    written = {p.stem for p in blob_dir.glob("*.json")}
    assert len(written) == 2, f"expected 2 distinct blob files, got {written}"


def test_missing_hitl_key_logs_error_not_silently_swallowed(
    tmp_path, monkeypatch, caplog
):
    """Product_Archive.md 2.3: a missing HITL_ENCRYPTION_KEY must be
    visible (logged), not a silently dropped blob with a dangling ref."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.delenv("HITL_ENCRYPTION_KEY", raising=False)

    redactor = _redactor("production", tenant_id="acme")
    payload = "z" * 100
    span = FakeSpan({"input.value": payload})

    import logging

    with caplog.at_level(logging.ERROR, logger="runtime.trace_redactor"):
        redactor.on_end(span)

    assert any(
        "NOT written" in rec.message or "HITL" in rec.message for rec in caplog.records
    ), "missing-key failure was not logged at ERROR level"
    # truncation still happens — the span is still safe to export
    assert len(span._attributes["input.value"]) <= 50 + len("…[truncated]")


def test_credit_card_redaction_validates_luhn():
    redactor = _redactor("staging")
    valid_card = "4111-1111-1111-1111"  # Luhn-valid test number
    invalid_card = "1234-5678-9012-3456"  # Luhn-invalid — looks card-shaped but isn't
    span = FakeSpan({"input.value": f"card {valid_card} and also {invalid_card}"})
    redactor.on_end(span)
    scrubbed = span._attributes["input.value"]
    assert valid_card not in scrubbed
    assert invalid_card in scrubbed  # not Luhn-valid -> not touched


def test_get_environment_fail_closed_for_unrecognized_value(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "not-a-real-environment")
    assert get_environment() == "production"


def test_get_environment_fail_closed_for_unset(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert get_environment() == "production"


def test_get_environment_explicit_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert get_environment() == "development"


# ── against a REAL span, which is where this was broken ──────────────────────


def _isolated_provider(profile: str):
    """A TracerProvider that is never registered globally.

    Isolated for two reasons found the hard way: an OTel processor can be added
    to a provider and never removed, so attaching a redactor to the shared one
    scrubs every later test in the run; and `set_tracer_provider` is one-shot,
    so a second global provider is silently ignored.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(tr.TraceRedactor(profile=profile, tenant_id="acme"))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_real_span_is_scrubbed_end_to_end():
    """The test this module did not have, and the reason it mattered.

    Every other test here drives `on_end` with a FakeSpan whose `_attributes`
    is a plain dict. The SDK hands processors a ReadableSpan whose attributes
    are `BoundedAttributes(immutable=True)`, and writing to those raises a bare
    TypeError out of `span.end()` — so against a real span this processor
    redacted NOTHING and made ending a span throw. Every test passed, because
    the double accepted writes the real object refuses.

    That is the first entry in this repo's lessons appendix, in the redaction
    control: a test double must never be more capable than the real thing.
    """
    provider, exporter = _isolated_provider("staging")
    secret = "sk-ant-abcdefghijklmnopqrstuvwxyz0123456789"

    with provider.get_tracer("t").start_as_current_span("llm.call") as span:
        span.set_attribute("input.value", f"key={secret} card=4111111111111111")
        span.set_attribute("agent.tool.name", "payment_lookup")

    attributes = dict(exporter.get_finished_spans()[0].attributes)
    assert secret not in attributes["input.value"]
    assert "4111111111111111" not in attributes["input.value"]
    assert "[REDACTED" in attributes["input.value"]
    # A non-payload attribute keeps its value.
    assert attributes["agent.tool.name"] == "payment_lookup"


def test_ending_a_span_with_the_redactor_installed_does_not_raise():
    """The other half of the same bug, and the worse one operationally: the
    TypeError propagated out of `span.end()`, so a worker with the redactor
    installed — which `configure_tracing()` does by default — would have raised
    on every span it closed."""
    provider, _ = _isolated_provider("production")
    with provider.get_tracer("t").start_as_current_span("llm.call") as span:
        span.set_attribute("input.value", "x" * 200)
    # Reaching here at all is the assertion.


def test_development_profile_leaves_a_real_span_alone():
    """The permissive profile must still be permissive against the real object,
    not merely against the double."""
    provider, exporter = _isolated_provider("development")
    with provider.get_tracer("t").start_as_current_span("llm.call") as span:
        span.set_attribute("input.value", "card=4111111111111111")

    assert "4111111111111111" in dict(exporter.get_finished_spans()[0].attributes)["input.value"]


# ── Sequence attributes (pass 11) ────────────────────────────────────────────
#
# The scrub loop did `if not isinstance(value, str): continue`, and a sequence
# of strings is a first-class OTel attribute type the SDK accepts without
# comment. So `set_attribute("retrieved_docs", [...])` — an entirely ordinary
# thing for a RAG step to do — exported whole, in every profile, production
# included. Verified against a real span before the fix: an email, an API key
# and a valid card number all reached the exporter untouched.
#
# These drive a REAL provider, which is the only way the sequence gap was
# visible: a FakeSpan carrying a plain dict scrubs whatever the loop hands it,
# and the loop was skipping sequences before it got there.
#
# (The earlier note here said the FakeSpan had hidden the redactor being
# entirely inert. That claim was wrong — see _writable_attributes' docstring
# for the correction. A live span's attributes are mutable and the control was
# working. The case for a real provider stands on its own without it.)


def _export_with_profile(profile: str, attributes: dict) -> dict:
    """One span through a real provider with the redactor attached."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(tr.TraceRedactor(profile=profile))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("s") as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    return dict(exporter.get_finished_spans()[0].attributes)


@pytest.mark.parametrize("profile", ["staging", "production"])
def test_a_sequence_attribute_is_scrubbed_like_a_string_one(profile) -> None:
    out = _export_with_profile(
        profile,
        {"docs": ["contact bob@example.com", "key sk-ant-aaaaaaaaaaaaaaaaaaaaaaaa"]},
    )
    joined = " ".join(out["docs"])
    assert "bob@example.com" not in joined, f"{profile}: an email left in a list attribute"
    assert "sk-ant-" not in joined, f"{profile}: an API key left in a list attribute"
    assert "REDACTED" in joined


def test_a_card_number_in_a_sequence_is_redacted() -> None:
    out = _export_with_profile("production", {"docs": ["card 4111 1111 1111 1111"]})
    assert "4111" not in " ".join(out["docs"])


def test_the_sequence_keeps_its_shape_and_length() -> None:
    """A scrubber must not quietly change an attribute's type or drop entries —
    writing through `_attributes._dict` bypasses the SDK's own coercion."""
    out = _export_with_profile("staging", {"docs": ["a@b.com", "plain text", "c@d.com"]})
    assert isinstance(out["docs"], tuple)
    assert len(out["docs"]) == 3
    assert out["docs"][1] == "plain text"


def test_non_string_attributes_are_left_alone() -> None:
    """Ints, bools and mixed sequences are not this scrubber's shape. Guessing
    at one is how a redactor starts corrupting data instead of protecting it."""
    out = _export_with_profile("production", {"count": 7, "ok": True})
    assert out["count"] == 7
    assert out["ok"] is True


def test_production_does_not_truncate_a_digest() -> None:
    """`prompt.system.sha256` is recorded PRECISELY so the prompt itself never
    reaches a span. Truncating it to 50 characters produced a string that is
    not a sha256 of anything and joins with nothing computed elsewhere — the
    one attribute designed to be safe in production was the one production
    broke."""
    digest = "b" * 64
    out = _export_with_profile("production", {"prompt.system.sha256": digest})
    assert out["prompt.system.sha256"] == digest


def test_production_still_truncates_ordinary_free_text() -> None:
    """The exemption is a named set, not a hole: everything else still gets the
    50-character ceiling §27 asks for."""
    out = _export_with_profile("production", {"note": "x" * 200})
    assert len(out["note"]) < 200
    assert "truncated" in out["note"]


# ── The two halves of the PII control now agree on what PII is ───────────────

_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def test_the_redactor_scrubs_the_identifiers_the_input_guard_does():
    """`input_guardrail` stripped Emirates IDs and phone numbers from prompts;
    this half knew about neither, so an identifier that never reached the model
    still reached Phoenix on a span attribute. The framework's own docs call the
    two symmetric, and the tenant whose whole subject is Emirates IDs declared
    no extra patterns — nothing told it that it had to.
    """
    r = tr.TraceRedactor(profile="production")
    out = r._scrub("id 784-1234-1234567-1 phone 0501234567", hash_identifiers=False)
    assert "784-1234-1234567-1" not in out
    assert "0501234567" not in out


def test_the_redactor_handles_arabic_numerals_too():
    r = tr.TraceRedactor(profile="production")
    arabic = "784-1234-1234567-1".translate(_ARABIC_DIGITS)
    assert arabic not in r._scrub(f"id {arabic}", hash_identifiers=False)


def test_one_identifier_written_two_ways_hashes_the_same():
    """Staging hashes rather than blanks, so an operator can correlate. Matching
    on the normalised copy means the ASCII and Arabic forms of one Emirates ID
    correlate to one hash instead of looking like two people."""
    r = tr.TraceRedactor(profile="staging")
    a = r._scrub("id 784-1234-1234567-1", hash_identifiers=True)
    b = r._scrub("id " + "784-1234-1234567-1".translate(_ARABIC_DIGITS), hash_identifiers=True)
    assert a == b


def test_a_plain_number_is_left_alone():
    """The control: over-redaction is its own failure — a span whose every
    number is [REDACTED] tells an operator nothing."""
    r = tr.TraceRedactor(profile="production")
    assert r._scrub("order 12345 qty 7", hash_identifiers=False) == "order 12345 qty 7"


def _fallback_tenant(monkeypatch, tmp_path, **env):
    """The redactor's fallback tenant under a given environment.

    chdir'd to an empty directory so `tenant.yaml` discovery cannot reach the
    framework's own config and make every case below pass for that reason.
    """
    for var in ("AGENT_TENANT_ID", "TENANT_ID"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    return tr.TraceRedactor(profile="production").default_tenant_id


def test_fallback_tenant_reads_the_primary_env_var(monkeypatch, tmp_path):
    """AGENT_TENANT_ID is what tenancy.py names first and the docs call primary.

    This line read os.environ["TENANT_ID"] directly and did not know the primary
    variable existed, so a deployment setting only that one got "unknown" — and
    the fallback is what decides which tenant's key a HITL compliance blob is
    written under.
    """
    assert _fallback_tenant(monkeypatch, tmp_path, AGENT_TENANT_ID="acme") == "acme"
    assert _fallback_tenant(monkeypatch, tmp_path, TENANT_ID="legacy-co") == "legacy-co"
    assert (
        _fallback_tenant(
            monkeypatch, tmp_path, AGENT_TENANT_ID="primary", TENANT_ID="legacy"
        )
        == "primary"
    )


def test_a_set_but_empty_tenant_var_falls_back_to_unknown(monkeypatch, tmp_path):
    """`TENANT_ID=""` is routine in k8s manifests and CI matrices.

    os.environ.get(key, default) substitutes only when the key is ABSENT, so a
    declared-but-empty variable produced "" — a compliance blob keyed and
    encrypted under the empty tenant rather than the intended default.
    """
    assert _fallback_tenant(monkeypatch, tmp_path, TENANT_ID="") == "unknown"
    assert _fallback_tenant(monkeypatch, tmp_path, AGENT_TENANT_ID="   ") == "unknown"


def test_an_unresolvable_tenant_never_takes_down_the_processor(monkeypatch, tmp_path):
    """resolve_tenant_id raises rather than guessing, which is right for a
    worker refusing to start and wrong here: a span processor that raises on an
    unset variable takes telemetry down with it."""
    assert _fallback_tenant(monkeypatch, tmp_path) == "unknown"


# ── Per-tenant HITL encryption keys ───────────────────────────────────────────


def _key_for(monkeypatch, tenant_id, **env):
    from runtime import environment

    monkeypatch.setattr(environment, "_degraded_warned", set())
    for name in list(os.environ):
        if name.startswith("HITL_ENCRYPTION_KEY"):
            monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return tr.HITLBlobStore(tenant_id)._key()


def _sha(value: str) -> bytes:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).digest()


@pytest.mark.parametrize(
    "tenant_id, expected_suffix",
    [
        ("kyc-sentinel", "KYC_SENTINEL"),
        ("acme-corp", "ACME_CORP"),
        ("tenant.one", "TENANT_ONE"),
        ("acme", "ACME"),
    ],
)
def test_the_per_tenant_key_variable_is_a_legal_name(tenant_id, expected_suffix):
    """`HITL_ENCRYPTION_KEY_KYC-SENTINEL` is not something a shell can set.

    The id used to be spliced in raw, so for every hyphenated tenant — which is
    the conventional shape, and this framework's own testbed — the per-tenant
    key was unreachable and the fleet-wide key was used instead.
    """
    assert tr.HITLBlobStore.env_suffix(tenant_id) == expected_suffix
    assert re.fullmatch(r"[A-Za-z0-9_]+", tr.HITLBlobStore.env_suffix(tenant_id))


def test_a_hyphenated_tenant_gets_its_own_key(monkeypatch):
    """The finding, asserted directly: this returned the shared key before."""
    key = _key_for(
        monkeypatch,
        "kyc-sentinel",
        HITL_ENCRYPTION_KEY="fleet-wide",
        HITL_ENCRYPTION_KEY_KYC_SENTINEL="kyc-only",
    )
    assert key == _sha("kyc-only")
    assert key != _sha("fleet-wide")


def test_an_unpunctuated_tenant_keeps_the_name_it_already_had(monkeypatch):
    """Backward compatibility. For an id with no punctuation the normalised and
    legacy spellings are identical, so a working deployment keeps working."""
    assert _key_for(
        monkeypatch, "acme", HITL_ENCRYPTION_KEY_ACME="acme-only"
    ) == _sha("acme-only")


def test_falling_back_to_the_fleet_key_is_reported(monkeypatch, caplog):
    """Still allowed — refusing would break deployments — but not silent.

    The message names the variable to set, because "configure a per-tenant key"
    is not actionable when the reason it was missing was that its name could
    not be typed.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    with caplog.at_level(logging.INFO):
        key = _key_for(monkeypatch, "acme-corp", HITL_ENCRYPTION_KEY="fleet-wide")

    assert key == _sha("fleet-wide")
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert "HITL_ENCRYPTION_KEY_ACME_CORP" in caplog.text


def test_the_key_value_is_hashed_verbatim(monkeypatch):
    """The value is NOT stripped before hashing.

    Whitespace-only means unset, but a real value is used exactly as given:
    trimming it would derive a different key and orphan every blob already
    written with the old one.
    """
    assert _key_for(
        monkeypatch, "acme", HITL_ENCRYPTION_KEY_ACME="  padded  "
    ) == _sha("  padded  ")


def test_a_whitespace_only_key_counts_as_unset(monkeypatch):
    """`HITL_ENCRYPTION_KEY_ACME=" "` is a declared-but-empty variable, not a key."""
    with pytest.raises(RuntimeError, match="No HITL encryption key"):
        _key_for(monkeypatch, "acme", HITL_ENCRYPTION_KEY_ACME="   ")


def test_the_missing_key_error_names_the_settable_variable(monkeypatch):
    """It used to name HITL_ENCRYPTION_KEY_KYC-SENTINEL — an instruction that
    cannot be followed."""
    with pytest.raises(RuntimeError) as exc:
        _key_for(monkeypatch, "kyc-sentinel")
    assert "HITL_ENCRYPTION_KEY_KYC_SENTINEL" in str(exc.value)


# ── The compliance guarantee, asserted rather than assumed ────────────────────
#
# The tests above check that a blob FILE APPEARS under the right tenant and that
# two spans get distinct names. Neither ever opened one. So "the full original
# payload is preserved for compliance review" was asserted as "something was
# written": storing the scrubbed text instead of the original, deriving the key
# differently, or writing a corrupt nonce would all have passed, and the failure
# would surface at a compliance request — the worst moment to discover it.


def _stored_payload(tmp_path, monkeypatch, payload, tenant="acme", key="k"):
    """Run a span through the production redactor and read its blob back."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY", key)
    redactor = _redactor("production", tenant_id=tenant)
    span = FakeSpan({"input.value": payload}, trace_id=11, span_id=2)
    redactor.on_end(span)
    ref = span._attributes["input.value.hitl_blob_ref"]
    return tr.HITLBlobStore(tenant).get(ref), span


def test_the_blob_round_trips_to_the_original_payload(tmp_path, monkeypatch):
    """The whole point of the store, in one assertion."""
    payload = "Applicant Emirates ID 784-1234-1234567-1 and email a.b@example.com " + "z" * 40
    recovered, _ = _stored_payload(tmp_path, monkeypatch, payload)
    assert recovered == payload


def test_the_blob_keeps_what_the_span_had_to_lose(tmp_path, monkeypatch):
    """The two halves must disagree, and that is the design.

    The exported span is scrubbed and truncated; the blob holds the identifiers
    verbatim. A blob that stored the SCRUBBED text would satisfy every earlier
    test in this file and quietly make the control pointless.
    """
    payload = "Emirates ID 784-1234-1234567-1, card 4111 1111 1111 1111 " + "z" * 40
    recovered, span = _stored_payload(tmp_path, monkeypatch, payload)

    assert "784-1234-1234567-1" in recovered
    assert "4111 1111 1111 1111" in recovered

    exported = str(span._attributes["input.value"])
    assert "784-1234-1234567-1" not in exported, "the span exported the raw identifier"
    assert "REDACTED" in exported


def test_a_missing_ref_is_none_not_an_error(tmp_path, monkeypatch):
    """A compliance lookup for a blob that was never written — because its
    write failed and was logged — must be distinguishable from a decrypt
    failure, not collapsed into one."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY", "k")
    assert tr.HITLBlobStore("acme").get("never-written") is None


def test_the_wrong_key_raises_instead_of_returning_nonsense(tmp_path, monkeypatch):
    """AES-GCM authenticates, so a failed tag is real information: this blob
    belongs to another tenant, or the key has rotated. Both need saying."""
    payload = "sensitive " + "z" * 60
    _stored_payload(tmp_path, monkeypatch, payload, key="the-right-key")

    monkeypatch.setenv("HITL_ENCRYPTION_KEY", "the-wrong-key")
    store = tr.HITLBlobStore("acme")
    ref = next(iter(p.stem for p in (tmp_path / "acme").glob("*.json")))
    with pytest.raises(RuntimeError, match="did not decrypt"):
        store.get(ref)


def test_one_tenants_key_does_not_open_anothers_blob(tmp_path, monkeypatch):
    """Per-tenant keys are the isolation claim; this is it end to end."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY_ACME", "acme-key")
    monkeypatch.setenv("HITL_ENCRYPTION_KEY_OTHER", "other-key")
    monkeypatch.delenv("HITL_ENCRYPTION_KEY", raising=False)

    redactor = _redactor("production", tenant_id="acme")
    span = FakeSpan({"input.value": "acme secret " + "z" * 60}, trace_id=5, span_id=1)
    redactor.on_end(span)
    ref = span._attributes["input.value.hitl_blob_ref"]

    assert tr.HITLBlobStore("acme").get(ref) is not None
    # Same ref, other tenant: its own directory has no such blob.
    assert tr.HITLBlobStore("other").get(ref) is None


def test_get_refuses_rather_than_guessing_when_s3_is_configured(tmp_path, monkeypatch):
    """put() writes to S3 when the bucket is set. get() reads the local backend
    only, and says so — returning None there would read as "no such blob" for a
    blob that exists in a bucket this method never looked in."""
    monkeypatch.setenv("HITL_BLOB_DIR", str(tmp_path))
    monkeypatch.setenv("HITL_ENCRYPTION_KEY", "k")
    monkeypatch.setenv("HITL_BLOB_S3_BUCKET", "some-bucket")
    with pytest.raises(NotImplementedError, match="local filesystem"):
        tr.HITLBlobStore("acme").get("any-ref")
