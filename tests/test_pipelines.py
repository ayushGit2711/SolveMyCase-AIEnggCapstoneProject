"""Unit and integration tests for Approach A (Baseline) and Approach B (Proposed LangGraph)."""

import json
import re
from types import SimpleNamespace

import pytest
from solvemycase.config.settings import Settings
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.guardrails.scope_checker import ScopeChecker
from solvemycase.core.proposed.applicability_judge import ApplicabilityJudge
from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.core.proposed.state import AgentState
from solvemycase.core.proposed.verification_node import VerificationNode
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
    StatutoryCitation,
)
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

COVERED_PATH = [
    "guardrail",
    "decontextualize",
    "retrieve_and_rerank",
    "applicability_check",
    "procedural_planner",
    "verification",
    "synthesis",
]
GAP_PATH = ["guardrail", "decontextualize", "retrieve_and_rerank", "applicability_check", "synthesis"]

_CANDIDATE_HEADER = re.compile(r"^\[(S\d+)\]\s*(.*)$", re.MULTILINE)


class FakeJudgeClient:
    """OpenAI-style client stub for the applicability judge.

    ``keep(header)`` decides applicability from each candidate's header line (doc type, Act, section, title).
    """

    def __init__(self, keep=None, error=None):
        self._keep = keep or (lambda header: True)
        self._error = error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        user_msg = kwargs["messages"][-1]["content"]
        assessments = [
            {"id": short_id, "applicable": bool(self._keep(header)), "reason": "test"}
            for short_id, header in _CANDIDATE_HEADER.findall(user_msg)
        ]
        message = SimpleNamespace(content=json.dumps({"assessments": assessments}))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(scope="module")
def in_memory_store():
    """Provides an in-memory vector store populated with test corpus."""
    from solvemycase.data.ingestion.normalizer import load_legal_precedents, load_legal_provisions_from_parquet
    from solvemycase.data.vectorstore.indexer import EmbeddingProvider
    from pathlib import Path

    offline_settings = Settings(openai_api_key=None)
    store = QdrantLegalStore(settings=offline_settings, vector_dim=1536, in_memory=True)
    embedder = EmbeddingProvider(settings=offline_settings, dim=1536)

    provisions = load_legal_provisions_from_parquet(Path("/nonexistent"))
    precedents = load_legal_precedents()

    prov_texts = [f"{p.act_name} Section {p.section_number}: {p.text}" for p in provisions]
    prec_texts = [f"{pr.title} {pr.citation or ''}: {pr.text}" for pr in precedents]

    store.index_provisions_and_precedents(
        provisions=provisions,
        precedents=precedents,
        provision_embeddings=embedder.get_embeddings(prov_texts),
        precedent_embeddings=embedder.get_embeddings(prec_texts),
    )
    return store


def test_scope_checker_guardrails():
    """Verify input guardrail screens non-legal queries and classifies domains."""
    checker = ScopeChecker(settings=Settings(openai_api_key=None))

    # 1. Non-legal rejection
    cake_res = checker.check_scope("Give me a delicious recipe to bake a chocolate cake.")
    assert cake_res.is_legal is False
    assert cake_res.rejection_reason is not None

    # 2. Motor vehicle accident
    mva_res = checker.check_scope("My car was hit from behind by a speeding truck on the highway, driver fled.")
    assert mva_res.is_legal is True
    assert mva_res.domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT

    # 3. Property dispute
    prop_res = checker.check_scope("The tenant has stopped paying rent for 6 months and refuses to vacate the flat.")
    assert prop_res.is_legal is True
    assert prop_res.domain == LegalDomain.PROPERTY_CONFLICT

    # 4. Consumer dispute
    cpa_res = checker.check_scope("I bought a mobile phone on an e-commerce app that stopped working on day 2, seller refused refund.")
    assert cpa_res.is_legal is True
    assert cpa_res.domain == LegalDomain.CONSUMER_RIGHTS


def test_vanilla_rag_baseline(in_memory_store):
    """Test Approach A baseline RAG pipeline execution and IT-1 run_with_trace parity."""
    offline_settings = Settings(openai_api_key=None)
    baseline = VanillaRAGBaseline(settings=offline_settings, store=in_memory_store)
    scenario = "Speeding truck collided with motorcyclist causing severe spinal injury. Driver ran away."

    response = baseline.run(scenario)
    resp_trace, trace = baseline.run_with_trace(scenario)
    assert response == resp_trace
    assert trace.pipeline_type == "vanilla_rag"
    assert trace.nodes_visited == ["embed_query", "hybrid_search", "monolithic_generate"]
    assert response.domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT
    assert len(response.action_plan) > 0
    assert len(response.statutory_citations) > 0


def test_verification_node_strips_hallucinated_citations():
    """Verify that the verification node catches and strips ungrounded citations."""
    verifier = VerificationNode(settings=Settings(openai_api_key=None))

    # Create real context
    real_context = [
        RetrievedContext(
            chunk_id="bns_106",
            doc_type=DocumentType.STATUTE,
            title="Bharatiya Nyaya Sanhita, 2023 - Section 106",
            citation_or_section="Section 106",
            text="Causing death by negligence... rash and negligent driving of vehicle",
            source_url="https://indiacode.gov.in/handle/123456789/20062",
            act_name="Bharatiya Nyaya Sanhita, 2023",
            score=0.9,
        )
    ]

    # Test state with 1 real statute, 1 fake statute, and 1 fake precedent
    state: AgentState = {
        "scenario": "Accident case",
        "is_legal": True,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "rejection_reason": None,
        "clarification_prompt": None,
        "key_entities": {},
        "statute_queries": [],
        "precedent_queries": [],
        "retrieved_contexts": real_context,
        "draft_action_plan": [
            ProceduralActionStep(
                step_number=1,
                phase=ProceduralPhase.POLICE_ADMINISTRATIVE,
                title="File FIR",
                description="File under Section 106 BNS and fake Section 9999",
                forum_or_authority="Police Station",
                statutory_basis="Section 9999 BNS",
            )
        ],
        "draft_statutory_citations": [
            StatutoryCitation(
                act_name="Bharatiya Nyaya Sanhita, 2023",
                section_number="106",
                summary_of_provision="Death by negligence",
                applicability_to_scenario="Applicable to accident",
                source_url="temp",
            ),
            StatutoryCitation(
                act_name="Bharatiya Nyaya Sanhita, 2023",
                section_number="9999",  # Fake fabricated section!
                summary_of_provision="Imaginary provision",
                applicability_to_scenario="Fabricated",
                source_url="fake",
            ),
        ],
        "draft_precedent_citations": [
            PrecedentCitation(
                case_title="Totally Fake Case v. Nonexistent Insurer",  # Fake precedent!
                court="Supreme Court of India",
                legal_principle="Fabricated ratio",
                source_url="fake",
            )
        ],
        "verified_action_plan": [],
        "verified_statutory_citations": [],
        "verified_precedent_citations": [],
        "hallucination_check_passed": False,
        "unverified_citations_stripped": [],
        "iteration_count": 0,
        "final_response": None,
    }

    audit = verifier.verify(state)

    # Assert that fake section 9999 and fake precedent were stripped!
    assert audit["hallucination_check_passed"] is False
    assert len(audit["unverified_citations_stripped"]) == 2
    assert any("9999" in s for s in audit["unverified_citations_stripped"])
    assert any("Totally Fake Case" in s for s in audit["unverified_citations_stripped"])

    # Assert that legitimate Section 106 was preserved and verified with official source_url
    verified_statutes = audit["verified_statutory_citations"]
    assert len(verified_statutes) == 1
    assert verified_statutes[0].section_number == "106"
    assert verified_statutes[0].source_url == "https://indiacode.gov.in/handle/123456789/20062"

    # Assert that the action step with stripped citation basis had its basis sanitized
    assert audit["verified_action_plan"][0].statutory_basis is None


def test_proposed_langgraph_pipeline(in_memory_store):
    """Test Approach B LangGraph state graph execution end-to-end and IT-1/IT-2 trace parity."""
    offline_settings = Settings(openai_api_key=None)
    graph_runner = LegalAgentGraph(settings=offline_settings, store=in_memory_store)

    # 1. Test rejection flow
    rejected_res, rej_trace = graph_runner.run_with_trace("Tell me the history of ancient Rome.")
    assert len(rejected_res.action_plan) == 0
    assert "Query rejected" in rejected_res.scenario_summary
    assert rej_trace.nodes_visited == ["guardrail", "handle_rejection"]

    # 2. Test successful dispute flow
    scenario = "My landlord locked the flat while I was at work and threw my belongings out without any court order."
    response = graph_runner.run(scenario)
    resp_trace, trace = graph_runner.run_with_trace(scenario)
    assert response == resp_trace
    assert trace.nodes_visited == COVERED_PATH

    # Offline, the applicability check fails open: no abstention without an LLM verdict.
    assert trace.applicability_mode == "offline"
    assert trace.coverage_gap is False
    assert trace.applicability_rejected == []
    assert 0 < len(trace.retrieved_contexts) <= offline_settings.rerank_top_k

    assert response.domain == LegalDomain.PROPERTY_CONFLICT
    assert len(response.action_plan) >= 3
    assert any(step.phase == ProceduralPhase.FORUM_FILING for step in response.action_plan)
    assert response.hallucination_check_passed is True

    # Check that all verified citations have legitimate URLs
    for stat in response.statutory_citations:
        assert stat.source_url.startswith("https://")
    for prec in response.precedent_citations:
        assert prec.source_url.startswith("https://")

    # 3. IT-2: Verify RetrievalQualityGate triggers when retrieval_min_confidence exceeds top rerank score
    high_threshold_settings = Settings(openai_api_key=None, retrieval_min_confidence=1.01)
    gate_graph = LegalAgentGraph(settings=high_threshold_settings, store=in_memory_store)
    _, gate_trace = gate_graph.run_with_trace(scenario)
    assert gate_trace.retrieval_gate_triggered is True


def test_graph_stream_yields_every_stage_and_matches_run(in_memory_store):
    """stream() must report each node in order and end with the same response as run()."""
    graph = LegalAgentGraph(settings=Settings(openai_api_key=None), store=in_memory_store)
    scenario = "A speeding truck hit my scooter and the driver fled. I have fractures and hospital bills."

    updates = list(graph.stream(scenario))
    node_names = [name for name, _ in updates]

    assert node_names == COVERED_PATH
    streamed_response = updates[-1][1]["final_response"]
    assert streamed_response == graph.run(scenario)


def test_graph_stream_rejection_path(in_memory_store):
    """Out-of-scope queries stream guardrail -> handle_rejection with a final response."""
    graph = LegalAgentGraph(settings=Settings(openai_api_key=None), store=in_memory_store)
    updates = list(graph.stream("Please write a poem about the monsoon clouds over Mumbai."))

    assert [name for name, _ in updates] == ["guardrail", "handle_rejection"]
    assert updates[-1][1]["final_response"].action_plan == []


def test_graph_coverage_gap_when_no_candidate_applies(in_memory_store):
    """When the applicability check rejects every candidate, answer honestly instead of citing unrelated law."""
    settings = Settings(openai_api_key=None)
    fake_client = FakeJudgeClient(keep=lambda header: False)
    graph = LegalAgentGraph(
        settings=settings,
        store=in_memory_store,
        applicability_judge=ApplicabilityJudge(settings=settings, client=fake_client),
    )

    response, trace = graph.run_with_trace("My husband wants a divorce and refuses to return my jewellery. Police won't help.")

    assert trace.nodes_visited == GAP_PATH
    assert trace.coverage_gap is True
    assert trace.applicability_mode == "llm"
    assert trace.retrieved_contexts == []
    assert trace.applicability_rejected, "rejected candidate titles are recorded for the trace"
    assert len(fake_client.calls) == 1, "exactly one applicability call per query"

    assert response.statutory_citations == []
    assert response.precedent_citations == []
    assert response.hallucination_check_passed is True
    assert response.coverage_note and "doesn't cover" in response.coverage_note
    assert "Our database currently holds" in response.coverage_note
    assert len(response.action_plan) == 4
    assert all(step.statutory_basis is None for step in response.action_plan)


def test_graph_applicability_check_keeps_only_applicable_sources(in_memory_store):
    """Only candidates judged applicable reach the planner and can be cited."""
    settings = Settings(openai_api_key=None)
    fake_client = FakeJudgeClient(keep=lambda header: "Specific Relief Act" in header)
    graph = LegalAgentGraph(
        settings=settings,
        store=in_memory_store,
        applicability_judge=ApplicabilityJudge(settings=settings, client=fake_client),
    )
    scenario = "My landlord locked the flat while I was at work and threw my belongings out without any court order."

    response, trace = graph.run_with_trace(scenario)

    assert trace.nodes_visited == COVERED_PATH
    assert trace.applicability_mode == "llm"
    assert trace.coverage_gap is False
    assert trace.retrieved_contexts, "the fixture corpus has Specific Relief Act sections for dispossession"
    assert all("Specific Relief Act" in (ctx.act_name or "") for ctx in trace.retrieved_contexts)
    assert all("Specific Relief Act" in stat.act_name for stat in response.statutory_citations)
    assert response.precedent_citations == []
    assert response.coverage_note is None


def test_graph_applicability_error_fails_open(in_memory_store):
    """A failing applicability call passes the top reranked candidates through instead of abstaining."""
    settings = Settings(openai_api_key=None)
    fake_client = FakeJudgeClient(error=RuntimeError("rate limited"))
    graph = LegalAgentGraph(
        settings=settings,
        store=in_memory_store,
        applicability_judge=ApplicabilityJudge(settings=settings, client=fake_client),
    )

    _, trace = graph.run_with_trace("A speeding truck hit my scooter and the driver fled. I have fractures.")

    assert trace.nodes_visited == COVERED_PATH
    assert trace.applicability_mode == "error"
    assert trace.coverage_gap is False
    assert 0 < len(trace.retrieved_contexts) <= settings.rerank_top_k


def test_scope_checker_matches_whole_words_only():
    """Offline heuristics must not read "scared" as "car" or "parent" as "rent" (D3)."""
    checker = ScopeChecker(settings=Settings(openai_api_key=None))

    assert checker.check_scope("I am scared of my parents and my current school.").is_legal is False
    assert checker.check_scope("I filed the first complaint last year about my neighbour.").domain == (
        LegalDomain.GENERAL_DISPUTE
    )
    cat = checker.check_scope("The watchman of our society killed my pet cat.")
    assert cat.is_legal is True
    assert cat.domain == LegalDomain.GENERAL_DISPUTE
    assert checker.check_scope("My landlord refuses to return my rent deposit.").domain == LegalDomain.PROPERTY_CONFLICT


def _statute_ctx(chunk_id: str, act_name: str, section: str, text: str = "Statutory text.") -> RetrievedContext:
    return RetrievedContext(
        chunk_id=chunk_id,
        doc_type=DocumentType.STATUTE,
        title=f"{act_name} - Section {section}",
        citation_or_section=f"Section {section}",
        text=text,
        source_url=f"https://www.indiacode.nic.in/{chunk_id}",
        act_name=act_name,
        score=0.8,
    )


def _verification_state(contexts, actions=None, statutes=None, precedents=None) -> AgentState:
    return {
        "scenario": "Scenario",
        "is_legal": True,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "retrieved_contexts": contexts,
        "draft_action_plan": actions or [],
        "draft_statutory_citations": statutes or [],
        "draft_precedent_citations": precedents or [],
    }


def _step(basis: str) -> ProceduralActionStep:
    return ProceduralActionStep(
        step_number=1,
        phase=ProceduralPhase.FORUM_FILING,
        title="Step",
        description="Do something.",
        forum_or_authority="Forum",
        statutory_basis=basis,
    )


def test_verification_statutory_basis_is_act_aware():
    """BNSS s.173 being retrieved must not back a step that claims MVA s.173."""
    verifier = VerificationNode(settings=Settings(openai_api_key=None))
    contexts = [_statute_ctx("bnss_173", "Bharatiya Nagarik Suraksha Sanhita, 2023", "173")]
    actions = [
        _step("Section 173 Motor Vehicles Act, 1988"),
        _step("Section 173 BNSS, 2023"),
        _step("Section 173"),
        _step("Section 173 Bharatiya Nyaya Sanhita, 2023"),
    ]

    audit = verifier.verify(_verification_state(contexts, actions=actions))
    bases = [step.statutory_basis for step in audit["verified_action_plan"]]

    assert bases == [None, "Section 173 BNSS, 2023", "Section 173", None]


def test_verification_never_regrounds_or_pads_citations():
    """A citation must match a retrieved statute of the same Act; no precedent is added when none was cited."""
    verifier = VerificationNode(settings=Settings(openai_api_key=None))
    contexts = [
        _statute_ctx("bns_106", "Bharatiya Nyaya Sanhita, 2023", "106"),
        RetrievedContext(
            chunk_id="prec_1",
            doc_type=DocumentType.PRECEDENT,
            title="National Insurance Co. Ltd. v. Pranay Sethi (2017)",
            citation_or_section="(2017) 16 SCC 680",
            text="Future prospects in compensation.",
            source_url="https://indiankanoon.org/doc/1/",
            court="Supreme Court of India",
            score=0.7,
        ),
    ]
    statutes = [
        StatutoryCitation(
            act_name="Transfer of Property Act, 1882",
            section_number="106",
            summary_of_provision="Duration of leases",
            applicability_to_scenario="n/a",
            source_url="fake",
        ),
        StatutoryCitation(
            act_name="Bharatiya Nagarik Suraksha Sanhita, 2023",
            section_number="106",
            summary_of_provision="Wrong Act",
            applicability_to_scenario="n/a",
            source_url="fake",
        ),
    ]

    audit = verifier.verify(_verification_state(contexts, statutes=statutes))

    assert audit["verified_statutory_citations"] == []
    assert len(audit["unverified_citations_stripped"]) == 2
    assert audit["verified_precedent_citations"] == [], "no precedent is forced in when the planner cited none"


def test_verification_node_has_no_store_dependency():
    """Verification checks only the retrieved contexts; it takes no vector store (no corpus-wide re-grounding)."""
    with pytest.raises(TypeError):
        VerificationNode(settings=Settings(openai_api_key=None), store=object())


def test_verification_strips_bare_section_when_multiple_acts_share_number():
    """A bare 'Section 106' is ambiguous when both BNS s.106 and TPA s.106 were retrieved."""
    verifier = VerificationNode(settings=Settings(openai_api_key=None))
    contexts = [
        _statute_ctx("bns_106", "Bharatiya Nyaya Sanhita, 2023", "106"),
        _statute_ctx("tpa_106", "Transfer of Property Act, 1882", "106"),
    ]
    actions = [
        _step("Section 106"),
        _step("Section 106 BNS, 2023"),
        _step("Section 106 Transfer of Property Act, 1882"),
    ]
    audit = verifier.verify(_verification_state(contexts, actions=actions))
    bases = [step.statutory_basis for step in audit["verified_action_plan"]]
    assert bases == [None, "Section 106 BNS, 2023", "Section 106 Transfer of Property Act, 1882"]


def test_cat_scenario_retrieves_animal_cruelty_laws_and_withholds_unscreened_offline(in_memory_store):
    """Offline cat scenario retrieves BNS 325 & PCA 11 into candidate_contexts and withholds unscreened general_dispute citations."""
    settings = Settings(openai_api_key=None)
    graph = LegalAgentGraph(settings=settings, store=in_memory_store)
    scenario = "The watchman of our housing society killed my pet cat with a stick yesterday."

    # 1. Offline mode: general_dispute withholds unscreened citations with screening_unavailable reason
    offline_resp, offline_trace = graph.run_with_trace(scenario)
    assert offline_trace.nodes_visited == GAP_PATH
    assert offline_trace.coverage_gap is True
    assert offline_trace.coverage_gap_reason == "screening_unavailable"
    assert offline_resp.statutory_citations == []
    assert offline_resp.coverage_note and "matches laws to your facts" in offline_resp.coverage_note
    candidate_ids = {c.chunk_id for c in offline_trace.candidate_contexts}
    assert "central_bns_2023_sec_325" in candidate_ids or "central_pca_1960_sec_11" in candidate_ids

    # 2. With applicability judge keeping animal-cruelty laws, only animal laws are cited (never BNS 101/103/106)
    fake_client = FakeJudgeClient(
        keep=lambda header: ("325" in header or "Cruelty to Animals" in header or "Nagaraja" in header or "173" in header)
    )
    judged_graph = LegalAgentGraph(
        settings=settings,
        store=in_memory_store,
        applicability_judge=ApplicabilityJudge(settings=settings, client=fake_client),
    )
    resp, trace = judged_graph.run_with_trace(scenario)
    assert trace.nodes_visited == COVERED_PATH
    assert trace.coverage_gap is False
    cited_secs = {(s.act_name, s.section_number) for s in resp.statutory_citations}
    assert any(sec == "325" or sec == "11" for _, sec in cited_secs)
    assert not any(sec in {"101", "103", "106", "304A"} for _, sec in cited_secs)

