"""Unit tests for the RAG fixes: applicability check, coverage gaps, retrieval filters and index freshness.

Everything runs offline: LLM calls use small fakes and the vector store is in-memory.
"""

import json
from types import SimpleNamespace

import pytest

from solvemycase.config.settings import Settings
from solvemycase.core.proposed.applicability_judge import (
    CANDIDATE_TEXT_CHARS,
    ApplicabilityJudge,
    format_candidates,
    parse_assessments,
)
from solvemycase.core.proposed.coverage import (
    build_coverage_gap_response,
    generic_next_steps,
    no_verified_citations_note,
    summarize_corpus_coverage,
)
from solvemycase.core.retrieval.reranker import LegalCrossEncoderReranker
from solvemycase.core.telemetry import load_telemetry_events, log_inference_event
from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    ExecutionTrace,
    LegalDomain,
    LegalPrecedent,
    LegalProvision,
    RetrievedContext,
    acts_share_significant_token,
    canonical_act_keys,
    normalize_section_id,
    should_trigger_criminal_route,
)
from solvemycase.data.vectorstore import indexer as indexer_module
from solvemycase.data.vectorstore.indexer import EmbeddingProvider, ensure_index_current, run_indexing_pipeline
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

OFFLINE = Settings(openai_api_key=None)


# --------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------


def _ctx(chunk_id: str, doc_type: DocumentType = DocumentType.STATUTE, **overrides) -> RetrievedContext:
    fields = {
        "chunk_id": chunk_id,
        "doc_type": doc_type,
        "title": f"Title {chunk_id}",
        "citation_or_section": "Section 1",
        "text": f"Text of {chunk_id}.",
        "source_url": f"https://example.org/{chunk_id}",
        "score": 0.5,
    }
    fields.update(overrides)
    return RetrievedContext(**fields)


class FakeChatClient:
    """OpenAI-style chat client returning a fixed content string (or raising)."""

    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _verdicts(**applicable) -> str:
    return json.dumps({"assessments": [{"id": k, "applicable": v, "reason": "r"} for k, v in applicable.items()]})


def _provision(doc_id, act, section, domain, text, category=None, title=None) -> LegalProvision:
    return LegalProvision(
        doc_id=doc_id,
        act_name=act,
        section_number=section,
        title=title,
        text=text,
        source_url=f"https://www.indiacode.nic.in/{doc_id}",
        domain=domain,
        act_category=category,
    )


def _small_corpus():
    provisions = [
        _provision(
            "mva_166", "Motor Vehicles Act, 1988", "166", LegalDomain.MOTOR_VEHICLE_ACCIDENT,
            "Application for compensation arising out of an accident to the Claims Tribunal.",
        ),
        _provision(
            "cpa_35", "Consumer Protection Act, 2019", "35", LegalDomain.CONSUMER_RIGHTS,
            "Manner in which complaint shall be made before the District Commission for defective goods.",
        ),
        _provision(
            "bns_325", "Bharatiya Nyaya Sanhita, 2023", "325", LegalDomain.GENERAL_DISPUTE,
            "Whoever commits mischief by killing, poisoning, maiming or rendering useless any animal shall be punished.",
            category="criminal",
        ),
        _provision(
            "tpa_106", "Transfer of Property Act, 1882", "106", LegalDomain.PROPERTY_CONFLICT,
            "Duration of certain leases in absence of written contract or local usage; notice to quit.",
        ),
    ]
    precedents = [
        LegalPrecedent(
            doc_id="nagaraja",
            chunk_id="nagaraja_000",
            court="Supreme Court of India",
            title="Animal Welfare Board of India v. A. Nagaraja",
            citation="(2014) 7 SCC 547",
            year=2014,
            text="Animals have a right to live with dignity; cruelty to animals under the PCA Act 1960.",
            source_url="https://indiankanoon.org/doc/39696860/",
            domain=LegalDomain.GENERAL_DISPUTE,
        )
    ]
    return provisions, precedents


@pytest.fixture
def small_store():
    provisions, precedents = _small_corpus()
    store = QdrantLegalStore(settings=OFFLINE, vector_dim=64, in_memory=True)
    run_indexing_pipeline(
        store=store,
        settings=OFFLINE,
        embedder=EmbeddingProvider(settings=OFFLINE, dim=64),
        corpus_data={"provisions": provisions, "precedents": precedents},
    )
    return store


# --------------------------------------------------------------------------------------------------------------
# Schema helpers
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("173 BNSS", "173"),
        ("Section 106 BNS", "106"),
        ("Section 173(1) BNSS", "173(1)"),
        ("Sec. 53-A", "53a"),
        ("163A", "163a"),
        ("Section 2(7)", "2(7)"),
        ("Order 39 Rule 1", "39(1)"),
    ],
)
def test_normalize_section_id_does_not_swallow_act_abbreviations(raw, expected):
    assert normalize_section_id(raw) == expected


def test_criminal_route_needs_a_whole_word_indicator():
    consumer = LegalDomain.CONSUMER_RIGHTS
    assert should_trigger_criminal_route(LegalDomain.MOTOR_VEHICLE_ACCIDENT, "Minor scratch, no one hurt.") is True
    assert should_trigger_criminal_route(consumer, "I filed the first complaint before the deadline.") is False
    assert should_trigger_criminal_route(consumer, "The delivery agent threatened me and used abusive words.") is True
    assert should_trigger_criminal_route(LegalDomain.GENERAL_DISPUTE, "The watchman killed my pet cat.") is True
    # general_dispute alone no longer pulls in penal provisions for a civil question.
    assert should_trigger_criminal_route(LegalDomain.GENERAL_DISPUTE, "My wife and I want a mutual divorce.") is False


def test_canonical_act_keys_tell_similar_codes_apart():
    assert canonical_act_keys("Bharatiya Nyaya Sanhita, 2023") == frozenset({"bns"})
    assert canonical_act_keys("Section 173 BNSS") == frozenset({"bnss"})
    assert canonical_act_keys("Code of Civil Procedure, 1908") == frozenset({"cpc"})
    assert canonical_act_keys("Code of Criminal Procedure, 1973") == frozenset({"crpc"})
    assert canonical_act_keys("Section 166 Motor Vehicles Act, 1988") == frozenset({"mva"})
    assert "pca" in canonical_act_keys("Prevention of Cruelty to Animals Act, 1960")
    assert canonical_act_keys("Section 173") == frozenset()


def test_acts_share_significant_token_rejects_cross_code_matches():
    assert acts_share_significant_token("BNS", "Bharatiya Nyaya Sanhita, 2023") is True
    assert acts_share_significant_token("BNSS", "Bharatiya Nyaya Sanhita, 2023") is False
    assert acts_share_significant_token("Bharatiya Nagarik Suraksha Sanhita", "Bharatiya Nyaya Sanhita, 2023") is False
    assert acts_share_significant_token("Code of Civil Procedure", "Code of Criminal Procedure, 1973") is False


# --------------------------------------------------------------------------------------------------------------
# Multi-query reranking
# --------------------------------------------------------------------------------------------------------------


def test_lexical_rerank_scores_by_best_matching_query():
    reranker = LegalCrossEncoderReranker(settings=OFFLINE)
    reranker._load_attempted = True  # Force the deterministic lexical path.
    animal = _ctx("bns_325", text="mischief by killing or maiming any animal", score=0.0)
    human = _ctx("bns_103", text="punishment for murder of a human being", score=0.0)

    scenario_only = reranker.rerank("my neighbour's dog was hurt", [human, animal], top_k=2)
    with_sub_query = reranker.rerank(
        "my neighbour's dog was hurt", [human, animal], top_k=2, extra_queries=["killing or maiming any animal"]
    )

    assert with_sub_query[0].chunk_id == "bns_325"
    assert with_sub_query[0].score > scenario_only[[c.chunk_id for c in scenario_only].index("bns_325")].score


def test_cross_encoder_rerank_takes_max_over_queries():
    class FakeEncoder:
        def __init__(self):
            self.pairs = None

        def predict(self, pairs):
            self.pairs = pairs
            # Logit 4 only for (sub-query, animal doc); everything else scores -4.
            return [4.0 if ("animal" in q and "animal" in d) else -4.0 for q, d in pairs]

    reranker = LegalCrossEncoderReranker(settings=OFFLINE)
    encoder = FakeEncoder()
    reranker._cross_encoder = encoder
    reranker._load_attempted = True
    candidates = [_ctx("human", text="death of a human being"), _ctx("animal", text="killing any animal")]

    ranked = reranker.rerank("watchman killed my cat", candidates, top_k=2, extra_queries=["cruelty to animal"])

    assert [c.chunk_id for c in ranked] == ["animal", "human"]
    assert ranked[0].score == pytest.approx(0.982, abs=1e-3)
    assert len(encoder.pairs) == 4  # 2 queries x 2 candidates in one batched predict call.


def test_rerank_query_list_is_deduplicated_and_capped():
    queries = LegalCrossEncoderReranker._dedupe_queries("Scenario", ["scenario", "", "q1", "Q1"] + [f"x{i}" for i in range(20)])
    assert queries[:2] == ["Scenario", "q1"]
    assert len(queries) == LegalCrossEncoderReranker.MAX_RERANK_QUERIES


# --------------------------------------------------------------------------------------------------------------
# Applicability judge
# --------------------------------------------------------------------------------------------------------------


def test_judge_keeps_applicable_candidates_in_rerank_order():
    client = FakeChatClient(content=_verdicts(S1=False, S2=True, S3=True, S9=True))
    judge = ApplicabilityJudge(settings=OFFLINE, client=client)
    candidates = [_ctx("a"), _ctx("b"), _ctx("c"), _ctx("d")]

    result = judge.assess("facts", LegalDomain.GENERAL_DISPUTE, candidates, top_k=5)

    assert result.mode == "llm"
    assert [c.chunk_id for c in result.applicable] == ["b", "c"]
    # S1 was rejected; S4 was not assessed, so it is not assumed to apply; unknown S9 is ignored.
    assert [c.chunk_id for c in result.rejected] == ["a", "d"]
    assert result.coverage_gap is False
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == OFFLINE.openai_model_fast
    assert client.calls[0]["temperature"] == 0.0


def test_judge_caps_applicable_candidates_at_top_k():
    client = FakeChatClient(content=_verdicts(S1=True, S2=True, S3=True))
    judge = ApplicabilityJudge(settings=OFFLINE, client=client)
    result = judge.assess("facts", None, [_ctx("a"), _ctx("b"), _ctx("c")], top_k=2)
    assert [c.chunk_id for c in result.applicable] == ["a", "b"]


def test_judge_reports_coverage_gap_when_nothing_applies():
    judge = ApplicabilityJudge(settings=OFFLINE, client=FakeChatClient(content=_verdicts(S1=False, S2=False)))
    result = judge.assess("divorce", LegalDomain.GENERAL_DISPUTE, [_ctx("a"), _ctx("b")])
    assert result.coverage_gap is True
    assert result.applicable == []
    assert result.mode == "llm"


def test_judge_with_no_candidates_is_a_gap_without_calling_the_llm():
    client = FakeChatClient(content=_verdicts(S1=True))
    result = ApplicabilityJudge(settings=OFFLINE, client=client).assess("facts", None, [])
    assert result.coverage_gap is True
    assert client.calls == []


@pytest.mark.parametrize(
    "client, expected_mode",
    [
        (FakeChatClient(error=RuntimeError("timeout")), "error"),
        (FakeChatClient(content="not json"), "error"),
        (FakeChatClient(content=json.dumps({"assessments": [{"id": "S99", "applicable": True}]})), "error"),
        (FakeChatClient(content=json.dumps({"assessments": [{"id": "S1", "applicable": False}]})), "error"),
        (FakeChatClient(content=json.dumps({"verdicts": []})), "error"),
        (None, "offline"),
    ],
)
def test_judge_fails_open(client, expected_mode):
    judge = ApplicabilityJudge(settings=OFFLINE, client=client)
    candidates = [_ctx(str(i)) for i in range(8)]

    result = judge.assess("facts", None, candidates, top_k=3)

    assert result.mode == expected_mode
    assert [c.chunk_id for c in result.applicable] == ["0", "1", "2"]
    assert result.coverage_gap is False
    assert result.rejected == []


def test_judge_can_be_disabled():
    settings = Settings(openai_api_key=None, applicability_check_enabled=False)
    client = FakeChatClient(content=_verdicts(S1=False))
    result = ApplicabilityJudge(settings=settings, client=client).assess("facts", None, [_ctx("a")])
    assert result.mode == "disabled"
    assert [c.chunk_id for c in result.applicable] == ["a"]
    assert client.calls == []


def test_candidate_blocks_use_short_ids_and_clip_text():
    long_text = "word " * 400
    statute = _ctx("s", act_name="Bharatiya Nyaya Sanhita, 2023", citation_or_section="Section 325", text=long_text)
    judgment = _ctx("j", doc_type=DocumentType.PRECEDENT, court="Supreme Court of India", title="A v. B")

    rendered = format_candidates([statute, judgment])

    assert rendered.startswith("[S1] STATUTE | Bharatiya Nyaya Sanhita, 2023 | Section 325")
    assert "[S2] JUDGMENT | Supreme Court of India" in rendered
    statute_text = rendered.split("\n\n")[0].split("Text: ", 1)[1]
    assert len(statute_text) <= CANDIDATE_TEXT_CHARS + 4


def test_parse_assessments_accepts_loose_ids_and_booleans():
    content = json.dumps({"assessments": [{"id": "s1", "applicable": "yes"}, {"id": "[S2]", "applicable": 0}]})
    assert parse_assessments(content, 2) == {"S1": (True, ""), "S2": (False, "")}


# --------------------------------------------------------------------------------------------------------------
# Coverage-gap responses
# --------------------------------------------------------------------------------------------------------------


def test_coverage_summary_groups_current_acts_and_skips_repealed_ones():
    docs = [
        _ctx("1", act_name="Bharatiya Nyaya Sanhita, 2023", act_category="criminal"),
        _ctx("2", act_name="Bharatiya Nyaya Sanhita, 2023", act_category="criminal"),
        _ctx("3", act_name="Motor Vehicles Act, 1988", domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT),
        _ctx("4", act_name="Consumer Protection Act, 2019", domain=LegalDomain.CONSUMER_RIGHTS),
        _ctx(
            "5",
            act_name="Indian Penal Code, 1860",
            act_category="criminal",
            title="Indian Penal Code, 1860 - Section 304A [Repealed w.e.f. 1 Jul 2024; see BNS s.106]",
        ),
        _ctx("6", doc_type=DocumentType.PRECEDENT, title="Case A"),
        _ctx("7", doc_type=DocumentType.PRECEDENT, title="Case A"),
        _ctx("8", doc_type=DocumentType.PRECEDENT, title="Case B"),
    ]

    summary = summarize_corpus_coverage(docs)

    assert summary.startswith("Our database currently holds selected sections of these laws: ")
    assert "crimes and police complaints (Bharatiya Nyaya Sanhita, 2023)" in summary
    assert "road accidents and motor insurance claims (Motor Vehicles Act, 1988)" in summary
    assert "consumer complaints (Consumer Protection Act, 2019)" in summary
    assert "Indian Penal Code" not in summary
    assert "plus 2 Supreme Court and High Court judgments" in summary
    assert summary.index("crimes") < summary.index("road accidents") < summary.index("consumer")


def test_coverage_summary_for_empty_corpus():
    assert "empty" in summarize_corpus_coverage([])


def test_coverage_gap_response_cites_nothing_and_explains_coverage():
    response = build_coverage_gap_response("x" * 300, LegalDomain.GENERAL_DISPUTE, "Our database holds A.")

    assert response.statutory_citations == [] and response.precedent_citations == []
    assert response.hallucination_check_passed is True
    assert response.scenario_summary.endswith("...") and len(response.scenario_summary) == 243
    assert "doesn't cover" in response.coverage_note and "Our database holds A." in response.coverage_note
    assert [s.step_number for s in response.action_plan] == [1, 2, 3, 4]
    assert all(s.statutory_basis is None and s.limitation_period is None for s in response.action_plan)
    assert any("15100" in s.description for s in response.action_plan)


def test_generic_steps_and_soft_note_are_safe():
    assert len(generic_next_steps()) == 4
    note = no_verified_citations_note("Our database holds A.")
    assert "cites none" in note and "15100" in note and "Our database holds A." in note


# --------------------------------------------------------------------------------------------------------------
# Vector store: domain filters, BM25 zero scores, index freshness
# --------------------------------------------------------------------------------------------------------------


def test_allowed_domains():
    assert QdrantLegalStore.allowed_domains(None) is None
    assert QdrantLegalStore.allowed_domains(LegalDomain.GENERAL_DISPUTE) is None
    assert QdrantLegalStore.allowed_domains("general_dispute") is None
    assert QdrantLegalStore.allowed_domains(LegalDomain.CONSUMER_RIGHTS) == ["consumer_rights", "general_dispute"]
    assert QdrantLegalStore.allowed_domains("motor_vehicle_accident") == ["motor_vehicle_accident", "general_dispute"]


def test_domain_filtered_search_also_returns_cross_cutting_law(small_store):
    embedder = EmbeddingProvider(settings=OFFLINE, dim=64)
    query = "killing any animal complaint"

    hits = small_store.hybrid_search(query, embedder.get_embeddings([query])[0], top_k=10,
                                     domain_filter=LegalDomain.CONSUMER_RIGHTS)

    domains = {h.domain for h in hits}
    assert domains <= {LegalDomain.CONSUMER_RIGHTS, LegalDomain.GENERAL_DISPUTE}
    assert "bns_325" in {h.chunk_id for h in hits}, "general_dispute (criminal) law is visible to every domain"


def test_general_dispute_search_is_unfiltered(small_store):
    embedder = EmbeddingProvider(settings=OFFLINE, dim=64)
    query = "compensation tribunal accident"
    hits = small_store.hybrid_search(query, embedder.get_embeddings([query])[0], top_k=10,
                                     domain_filter=LegalDomain.GENERAL_DISPUTE)
    assert "mva_166" in {h.chunk_id for h in hits}


def test_criminal_search_returns_only_criminal_provisions(small_store):
    embedder = EmbeddingProvider(settings=OFFLINE, dim=64)
    query = "animal killing mischief"
    hits = small_store.criminal_code_search(query, embedder.get_embeddings([query])[0], top_k=5)
    assert hits and all(h.act_category == "criminal" for h in hits)


def test_bm25_gives_no_credit_to_zero_score_documents(small_store):
    # A query with no word in common with the corpus must not rank documents through BM25 at all.
    tokens_only_query = "zzqx unmatched"
    zero_vector = [0.0] * 64
    zero_vector[0] = 1.0
    hits = small_store.hybrid_search(tokens_only_query, zero_vector, top_k=2, domain_filter=None)
    # Dense search still returns neighbours; each hit's fused score then comes from the dense rank only.
    assert all(h.score <= 1.0 / 61 + 1e-9 for h in hits)


def test_needs_reindex_tracks_corpus_text_and_embedder(small_store):
    provisions, precedents = _small_corpus()
    offline_id = EmbeddingProvider(settings=OFFLINE, dim=64).embedder_id

    assert small_store.needs_reindex(provisions, precedents, offline_id) is False
    assert small_store.needs_reindex(provisions, precedents, "openai:text-embedding-3-small") is True

    edited = [p.model_copy(update={"text": p.text + " (amended)"}) if p.doc_id == "bns_325" else p for p in provisions]
    assert small_store.needs_reindex(edited, precedents, offline_id) is True
    relabelled = [p.model_copy(update={"domain": LegalDomain.CONSUMER_RIGHTS}) if p.doc_id == "tpa_106" else p
                  for p in provisions]
    assert small_store.needs_reindex(relabelled, precedents, offline_id) is True

    empty = QdrantLegalStore(settings=OFFLINE, vector_dim=64, in_memory=True)
    assert empty.needs_reindex(provisions, precedents, offline_id) is True


def test_payload_titles_and_embedder_ids_are_stored(small_store):
    statute = next(d for d in small_store.corpus_documents if d.chunk_id == "mva_166")
    assert statute.title.startswith("Motor Vehicles Act, 1988 - Section 166")
    assert statute.citation_or_section == "Section 166"
    assert small_store.loaded_embedder_ids == {"hash:64"}


def test_ensure_index_current_reindexes_only_when_stale(monkeypatch):
    provisions, precedents = _small_corpus()
    corpus = {"provisions": provisions, "precedents": precedents}
    monkeypatch.setattr(indexer_module, "load_unified_corpus", lambda: corpus)
    store = QdrantLegalStore(settings=OFFLINE, vector_dim=64, in_memory=True)
    embedder = EmbeddingProvider(settings=OFFLINE, dim=64)

    assert ensure_index_current(store, OFFLINE, embedder) is True
    assert len(store.corpus_documents) == len(provisions) + len(precedents)
    assert ensure_index_current(store, OFFLINE, embedder) is False

    corpus["provisions"] = provisions[:-1]
    assert ensure_index_current(store, OFFLINE, embedder) is True
    assert len(store.corpus_documents) == len(provisions) - 1 + len(precedents)


# --------------------------------------------------------------------------------------------------------------
# Embeddings batching
# --------------------------------------------------------------------------------------------------------------


class FakeEmbeddingsClient:
    def __init__(self, fail_on_call=None):
        self.batch_sizes = []
        self._fail_on_call = fail_on_call
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model, input):
        self.batch_sizes.append(len(input))
        if self._fail_on_call is not None and len(self.batch_sizes) == self._fail_on_call:
            raise RuntimeError("rate limited")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for _ in input])


def test_openai_embeddings_are_batched():
    embedder = EmbeddingProvider(settings=OFFLINE, dim=3)
    embedder._openai_client = FakeEmbeddingsClient()

    vectors = embedder.get_embeddings([f"text {i}" for i in range(300)])

    assert embedder._openai_client.batch_sizes == [128, 128, 44]
    assert len(vectors) == 300
    assert embedder.last_embedder_id == f"openai:{OFFLINE.openai_embedding_model}"


def test_failed_batch_falls_back_for_the_whole_call():
    embedder = EmbeddingProvider(settings=OFFLINE, dim=3)
    embedder._openai_client = FakeEmbeddingsClient(fail_on_call=2)

    vectors = embedder.get_embeddings([f"text {i}" for i in range(200)])

    assert len(vectors) == 200
    assert embedder.last_embedder_id == "hash:3", "one call never mixes two embedding spaces"
    assert all(v == embedder._offline_hash_embedding(f"text {i}") for i, v in enumerate(vectors))


# --------------------------------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------------------------------


def test_telemetry_records_coverage_gap_and_applicability_mode(tmp_path):
    log_path = tmp_path / "telemetry.jsonl"
    response = build_coverage_gap_response("divorce question", LegalDomain.GENERAL_DISPUTE, "Summary.")
    trace = ExecutionTrace(
        pipeline_type="proposed_langgraph",
        nodes_visited=["guardrail", "decontextualize", "retrieve_and_rerank", "applicability_check", "synthesis"],
        coverage_gap=True,
        applicability_mode="llm",
        applicability_rejected=["Some Act - Section 1"],
    )

    log_inference_event(response, trace, latency_ms=12.0, scenario="divorce question", settings=OFFLINE,
                        log_path=log_path)

    events = load_telemetry_events(settings=OFFLINE, log_path=log_path)
    assert events[-1]["coverage_gap"] is True
    assert events[-1]["applicability_mode"] == "llm"
    assert events[-1]["verified_citations"] == 0


def test_dual_output_response_coverage_note_is_optional():
    response = DualOutputResponse(
        scenario_summary="s", domain=LegalDomain.GENERAL_DISPUTE, action_plan=[], statutory_citations=[]
    )
    assert response.coverage_note is None
