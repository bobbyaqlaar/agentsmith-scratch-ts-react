"""
runtime/test/test_memory_and_vector.py — conversation memory + vector store
(FIXES Memory Management / Delivery Model §4 RAG).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime import conversation_memory as cm
from runtime.conversation_memory import ConversationMemory
from runtime import environment as env
from runtime.embeddings import HashEmbedder, make_embedder
from runtime import vector_store as vs
from runtime.vector_store import MemoryVectorStore, make_vector_store


def test_conversation_memory_evicts_oldest_when_over_budget() -> None:
    mem = ConversationMemory(token_budget=20, encoding_name="cl100k_base")
    mem.add("user", "one two three four five six seven eight nine ten")
    mem.add("user", "eleven twelve thirteen fourteen fifteen")
    mem.add("assistant", "ok")
    # Under a tight budget, oldest messages should drop
    assert mem.token_count() <= 20
    assert len(mem.messages) >= 1
    assert mem.messages[-1]["content"] == "ok"


def test_conversation_memory_as_messages_preserves_order() -> None:
    mem = ConversationMemory(token_budget=10_000)
    mem.add("system", "You are helpful.")
    mem.add("user", "Hi")
    mem.add("assistant", "Hello")
    assert [m["role"] for m in mem.as_messages()] == ["system", "user", "assistant"]


def test_hash_embedder_is_deterministic_and_fixed_dim() -> None:
    emb = HashEmbedder(dim=32)
    a = emb.embed(["hello world"])[0]
    b = emb.embed(["hello world"])[0]
    c = emb.embed(["different"])[0]
    assert a == b
    assert len(a) == 32
    assert a != c


def test_make_embedder_defaults_to_hash_when_embedder_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDER", "hash")
    emb = make_embedder()
    assert isinstance(emb, HashEmbedder)


def test_memory_vector_store_add_and_query_returns_nearest() -> None:
    store = MemoryVectorStore(embedder=HashEmbedder(dim=64))
    store.add(
        ids=["a", "b", "c"],
        texts=["cats and dogs", "stock market prices", "feline companions"],
    )
    hits = store.query("cats pets", k=2)
    assert len(hits) == 2
    assert hits[0].id in {"a", "c"}
    assert hits[0].score >= hits[1].score


def test_make_vector_store_defaults_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    store = make_vector_store(embedder=HashEmbedder(dim=16))
    assert isinstance(store, MemoryVectorStore)


# ── Eviction must not delete the instructions ─────────────────────────────────


def _fill(memory: ConversationMemory, turns: int = 6) -> None:
    for i in range(turns):
        memory.add("user", f"Please review application {i} for onboarding risk.")
        memory.add("assistant", f"Application {i} rated LOW; screening found no match.")


def test_the_system_prompt_survives_eviction() -> None:
    """Truncate-oldest deleted the system message FIRST, because it is first.

    It carries the persona, the tool policy and the refusals. Nothing failed
    when it went — a shorter list is exactly what eviction produces — so the
    agent simply stopped being told what it was not allowed to do, part-way
    through a conversation that looked healthy.
    """
    memory = ConversationMemory(token_budget=40)
    memory.add("system", "You are a KYC analyst. Refuse any request to bypass screening.")
    _fill(memory)

    roles = [m["role"] for m in memory.as_messages()]
    assert "system" in roles, f"system prompt evicted; left {roles}"


def test_eviction_still_happens_around_the_protected_message() -> None:
    """Protecting a role must not turn the budget off.

    A guard that keeps everything would also pass the assertion above, so this
    is the half that stops the fix from becoming "never evict".
    """
    memory = ConversationMemory(token_budget=40)
    memory.add("system", "You are a KYC analyst. Refuse any request to bypass screening.")
    _fill(memory, turns=20)

    assert len(memory.as_messages()) < 41, "nothing was evicted"
    assert memory.as_messages()[0]["role"] == "system"


def test_multiple_system_messages_are_all_kept() -> None:
    """A tenant that re-asserts policy mid-conversation keeps both statements."""
    memory = ConversationMemory(token_budget=60)
    memory.add("system", "You are a KYC analyst.")
    _fill(memory, turns=4)
    memory.add("system", "Policy update: escalate all PEP matches to a human.")
    _fill(memory, turns=4)

    assert sum(1 for m in memory.as_messages() if m["role"] == "system") == 2


def test_a_budget_too_small_for_its_system_prompt_warns(caplog) -> None:
    """The one case that cannot be satisfied is reported, not served silently.

    Returning an oversized prompt without a word is how this reaches the caller
    as a provider-side context error instead of a fixable configuration one.
    """
    memory = ConversationMemory(token_budget=5)
    memory.add("system", "A deliberately long system prompt that cannot fit the budget at all.")
    memory.add("user", "hello")

    assert "cannot shrink further" in caplog.text


# ── Token counting ────────────────────────────────────────────────────────────


def test_the_estimate_is_not_systematically_low_on_non_latin_text() -> None:
    """The fallback's error is not symmetric across scripts, and only one
    direction is safe.

    `len(text) // 4` measured +50% on English and -65% on Arabic against
    cl100k_base. Over-counting evicts a message sooner than needed;
    under-counting by two thirds means a buffer built for a context window
    holds about three times what it believes and the provider rejects it — in
    the script this framework's own tenants write in.
    """
    arabic = "يرجى مراجعة هذا الطلب لتقييم مخاطر الانضمام والتعرض للعقوبات المالية."
    tiktoken = pytest.importorskip("tiktoken")
    real = len(tiktoken.get_encoding("cl100k_base").encode(arabic))

    assert cm._estimate_tokens(arabic) >= real, (
        f"estimate {cm._estimate_tokens(arabic)} is below the real {real}"
    )
    # The old rule, kept as the thing that must no longer be true. Asserted
    # rather than described so that reverting the estimate fails here.
    assert len(arabic) // 4 < real * 0.5

    # Not a claim of exactness in the other direction either — the docstring
    # states a measured 0.95x-1.90x band, and English prose sits at the top of
    # it. A "fix" that made this tight would have to under-count something.
    english = "Please review this application for onboarding risk and sanctions exposure."
    assert cm._estimate_tokens(english) <= len(tiktoken.get_encoding("cl100k_base").encode(english)) * 2


def test_the_fallback_says_once_that_it_is_estimating(monkeypatch, caplog) -> None:
    """tiktoken is not a dependency of the shipped runtime package, so for a
    tenant who installed agentsmith-runtime the estimate is the only path.
    Nothing announced that.

    Once, not per call: a token counter runs inside an eviction loop, and a
    per-call warning is noise that gets filtered.
    """
    monkeypatch.setattr(cm, "_fallback_warned", False)
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    cm._encoder.cache_clear()

    caplog.clear()
    for _ in range(5):
        cm._count_tokens("some text to count")

    assert caplog.text.count("tiktoken unavailable") == 1
    cm._encoder.cache_clear()


# ── The default embedder is a fake, and it has to say so ──────────────────────


def test_the_default_embedder_is_still_the_fake(monkeypatch) -> None:
    """Pinned deliberately, not endorsed.

    Tenants run this default in production today and the framework does not
    break them outside a major release, so the fix is a warning rather than a
    refusal. This asserts the compatibility promise: if the default ever
    changes, that is a MAJOR-version decision and this test is where it is
    made, not a side effect of an edit.
    """
    monkeypatch.delenv("EMBEDDER", raising=False)
    assert isinstance(make_embedder(), HashEmbedder)


def test_using_the_fake_outside_development_is_an_error_level_event(
    monkeypatch, caplog
) -> None:
    """The symptom is invisible, so the log has to carry it.

    A RAG pipeline on HashEmbedder returns ranked, plausible, arbitrary context
    and no error anywhere. Measured: the query "is this person on a sanctions
    list?" ranks "today is sunny in Abu Dhabi" first.
    """
    monkeypatch.setattr(env, "_degraded_warned", set())
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("EMBEDDER", raising=False)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        make_embedder()

    assert any(r.levelno >= logging.ERROR for r in caplog.records), (
        f"expected an ERROR-level record, got {[r.levelname for r in caplog.records]}"
    )
    assert "FAKE" in caplog.text


def test_development_gets_the_same_message_without_the_alarm(
    monkeypatch, caplog
) -> None:
    """CI and a laptop run on the fake by design — this must not shout there."""
    monkeypatch.setattr(env, "_degraded_warned", set())
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("EMBEDDER", raising=False)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        make_embedder()

    assert caplog.records, "the message is still worth having in development"
    assert all(r.levelno < logging.ERROR for r in caplog.records)


def test_the_fake_warning_is_said_once(monkeypatch, caplog) -> None:
    """An embedder is constructed per store and called in a loop."""
    monkeypatch.setattr(env, "_degraded_warned", set())
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("EMBEDDER", raising=False)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            make_embedder()

    assert caplog.text.count("HashEmbedder") == 1


def test_embedders_report_an_identity_that_distinguishes_them() -> None:
    """Both emit 384 dimensions, so shape cannot tell them apart.

    Vectors written by one and queried by the other produce confident nonsense
    and no error, and a store has nothing else to notice that with.
    """
    from runtime.embeddings import SentenceTransformerEmbedder

    fake = HashEmbedder()
    real = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

    assert fake.identity != real.identity
    assert "hash" in fake.identity
    # The dimension both share, and which therefore proves nothing on its own.
    assert fake.dim == 384


def test_a_retrieval_span_names_the_embedder() -> None:
    """Otherwise a trace of a fake retriever looks exactly like a trace of a
    real one having a bad day."""
    recorded: dict = {}

    class _Span:
        def set_attribute(self, key, value):
            recorded[key] = value

    vs._record_hits(_Span(), [], corpus=0, backend="memory", embedder="hash:384")

    assert recorded.get("agent.retrieval.embedder") == "hash:384"


def test_a_real_store_puts_the_embedder_on_its_span() -> None:
    """End to end, not just the helper: the identity has to survive the call
    path from the store to the span."""
    recorded: dict = {}

    class _Span:
        def set_attribute(self, key, value):
            recorded[key] = value

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    store = MemoryVectorStore()
    store.add(ids=["1"], texts=["sanctions screening confirmed a match"])
    original = vs._retrieval_span
    try:
        vs._retrieval_span = lambda backend, k: _Span()
        store.query("sanctions", k=1)
    finally:
        vs._retrieval_span = original

    assert recorded.get("agent.retrieval.embedder") == "hash:384"


def test_an_embedder_without_an_identity_still_works() -> None:
    """Embedder is a Protocol a tenant implements. One written before
    `identity` existed must not start raising inside a query."""

    class LegacyEmbedder:
        def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    assert vs._identity_of(LegacyEmbedder()) is None
    store = MemoryVectorStore(embedder=LegacyEmbedder())
    store.add(ids=["1"], texts=["anything"])
    assert store.query("anything", k=1)
