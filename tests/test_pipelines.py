"""Unit and integration tests for Approach A (Baseline) and Approach B (Proposed LangGraph)."""

import pytest
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.guardrails.scope_checker import ScopeChecker
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


@pytest.fixture
def in_memory_store():
    """Provides an in-memory vector store populated with test corpus."""
    from solvemycase.data.ingestion.normalizer import load_legal_precedents, load_legal_provisions_from_parquet
    from solvemycase.data.vectorstore.indexer import EmbeddingProvider
    from pathlib import Path

    store = QdrantLegalStore(vector_dim=1536, in_memory=True)
    embedder = EmbeddingProvider(dim=1536)

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
    checker = ScopeChecker()

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
    """Test Approach A baseline RAG pipeline execution."""
    baseline = VanillaRAGBaseline(store=in_memory_store)
    scenario = "Speeding truck collided with motorcyclist causing severe spinal injury. Driver ran away."

    response = baseline.run(scenario)
    assert response.domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT
    assert len(response.action_plan) > 0
    assert len(response.statutory_citations) > 0


def test_verification_node_strips_hallucinated_citations():
    """Verify that the verification node catches and strips ungrounded citations."""
    verifier = VerificationNode()

    # Create real context
    real_context = [
        RetrievedContext(
            chunk_id="bns_106",
            doc_type=DocumentType.STATUTE,
            title="Bharatiya Nyaya Sanhita, 2023 - Section 106",
            citation_or_section="Section 106",
            text="Causing death by negligence... rash and negligent driving of vehicle",
            source_url="https://www.indiacode.nic.in/handle/123456789/20062",
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
    assert verified_statutes[0].source_url == "https://www.indiacode.nic.in/handle/123456789/20062"

    # Assert that the action step with stripped citation basis had its basis sanitized
    assert audit["verified_action_plan"][0].statutory_basis is None


def test_proposed_langgraph_pipeline(in_memory_store):
    """Test Approach B LangGraph state graph execution end-to-end."""
    graph_runner = LegalAgentGraph(store=in_memory_store)

    # 1. Test rejection flow
    rejected_res = graph_runner.run("Tell me the history of ancient Rome.")
    assert len(rejected_res.action_plan) == 0
    assert "Query rejected" in rejected_res.scenario_summary

    # 2. Test successful dispute flow
    scenario = "My landlord locked the flat while I was at work and threw my belongings out without any court order."
    response = graph_runner.run(scenario)

    assert response.domain == LegalDomain.PROPERTY_CONFLICT
    assert len(response.action_plan) >= 3
    assert any(step.phase == ProceduralPhase.FORUM_FILING for step in response.action_plan)
    assert response.hallucination_check_passed is True

    # Check that all verified citations have legitimate URLs
    for stat in response.statutory_citations:
        assert stat.source_url.startswith("https://")
    for prec in response.precedent_citations:
        assert prec.source_url.startswith("https://")


def test_graph_stream_yields_every_stage_and_matches_run(in_memory_store):
    """stream() must report each node in order and end with the same response as run()."""
    graph = LegalAgentGraph(store=in_memory_store)
    scenario = "A speeding truck hit my scooter and the driver fled. I have fractures and hospital bills."

    updates = list(graph.stream(scenario))
    node_names = [name for name, _ in updates]

    assert node_names == [
        "guardrail",
        "decontextualize",
        "retrieve_and_rerank",
        "procedural_planner",
        "verification",
        "synthesis",
    ]
    streamed_response = updates[-1][1]["final_response"]
    assert streamed_response == graph.run(scenario)


def test_graph_stream_rejection_path(in_memory_store):
    """Out-of-scope queries stream guardrail -> handle_rejection with a final response."""
    graph = LegalAgentGraph(store=in_memory_store)
    updates = list(graph.stream("Please write a poem about the monsoon clouds over Mumbai."))

    assert [name for name, _ in updates] == ["guardrail", "handle_rejection"]
    assert updates[-1][1]["final_response"].action_plan == []
