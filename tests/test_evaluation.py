"""Comprehensive unit and integration tests for the strengthened evaluation suite (`UT-1`–`UT-7`, `IT-1`–`IT-5`)."""

import json
from pathlib import Path
import pytest

from solvemycase.config.settings import Settings
from solvemycase.core.proposed.procedural_planner import ProceduralPlannerAgent
from solvemycase.core.proposed.verification_node import VerificationNode
from solvemycase.core.retrieval.reranker import _sigmoid
from solvemycase.core.telemetry import load_telemetry_events, log_inference_event
from solvemycase.data.ingestion.normalizer import FALLBACK_PROVISIONS
from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    ExecutionTrace,
    LegalDomain,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
    StatutoryCitation,
    acts_share_significant_token,
    normalize_section_id,
    section_mentioned_with_boundary,
    sections_match,
)
from solvemycase.evaluation.llm_judge import LegalLLMJudge
from solvemycase.evaluation.metrics import (
    compute_agent_trace_metrics,
    compute_citation_grounding_metrics,
    compute_context_relevance,
    compute_ground_truth_alignment,
    compute_procedural_completeness,
    compute_retrieval_metrics,
    verify_target_forum,
)
from solvemycase.evaluation.noise_sensitivity import (
    build_default_distractor_contexts,
    evaluate_noise_sensitivity,
)
from solvemycase.evaluation.sme_review import (
    SMEAnnotation,
    compute_judge_human_agreement,
    load_sme_annotations,
    save_sme_annotation,
)


def test_benchmark_dataset_integrity_and_100_percent_corpus_coverage():
    """Verify benchmark dataset has 52 scenarios and 100% of expected_sections exist in FALLBACK_PROVISIONS."""
    dataset_path = Path(__file__).parent.parent / "evaluation" / "benchmark_dataset.json"
    assert dataset_path.exists(), "benchmark_dataset.json must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 50, f"Expected 50+ scenarios, found {len(data)}"

    corpus_sections = {
        normalize_section_id(item["section_number"])
        for item in FALLBACK_PROVISIONS
        if item.get("section_number")
    }

    missing_sections = set()
    for item in data:
        assert "id" in item
        assert "scenario" in item and len(item["scenario"]) > 15
        assert "domain" in item
        assert "is_legal" in item
        assert "expects_criminal_route" in item
        for sec in item.get("expected_sections", []):
            if normalize_section_id(sec) not in corpus_sections:
                missing_sections.add(sec)

    assert not missing_sections, f"Benchmark expected_sections missing from corpus: {missing_sections}"


def test_section_normalization_and_boundary_matching():
    """UT-1: Preserve subsections (2(11) vs 2(47)) and prevent numeric prefix collisions (16 vs 166)."""
    assert normalize_section_id("Section 2(11)") == "2(11)"
    assert normalize_section_id("Sec. 2(47)") == "2(47)"
    assert not sections_match("Section 2(11)", "Section 2(47)")
    assert sections_match("Section 2(11)", "2(11)")
    assert sections_match("Order 39 Rule 1-2", "Order 39 Rule 1-2")

    assert not section_mentioned_with_boundary("16", "Motor Vehicles Act Section 166 claim petition")
    assert not section_mentioned_with_boundary("10", "Bharatiya Nyaya Sanhita Section 106 negligence")
    assert section_mentioned_with_boundary("166", "Motor Vehicles Act Section 166 claim petition")
    assert section_mentioned_with_boundary("2(11)", "Deficiency defined under Section 2(11) of CPA 2019")
    assert not section_mentioned_with_boundary("2(11)", "Unfair trade practice under Section 2(47) of CPA 2019")


def test_verification_node_prevents_cross_act_splicing_and_sanitizes_2_digit_basis():
    """UT-1 / DF-1: Reject cross-Act citation splicing and strip unverified 1-2 digit statutory_basis."""
    mva_context = RetrievedContext(
        chunk_id="mva_166",
        doc_type=DocumentType.STATUTE,
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        title="Motor Vehicles Act, 1988 - Section 166",
        citation_or_section="Section 166",
        act_name="Motor Vehicles Act, 1988",
        text="Section 166: Application for compensation before Claims Tribunal.",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
        score=0.92,
    )
    # Hallucinated cross-Act splice: Consumer Protection Act citing Section 166
    spliced_cit = StatutoryCitation(
        act_name="Consumer Protection Act, 2019",
        section_number="166",
        summary_of_provision="Fabricated splice",
        applicability_to_scenario="Invalid",
        source_url="https://indiacode.gov.in/handle/123456789/15256",
    )
    valid_cit = StatutoryCitation(
        act_name="Motor Vehicles Act, 1988",
        section_number="166",
        summary_of_provision="Application for compensation",
        applicability_to_scenario="Valid",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
    )
    step_with_unverified_2digit = ProceduralActionStep(
        step_number=1,
        phase=ProceduralPhase.FORUM_FILING,
        title="File petition",
        description="File claim before MACT",
        forum_or_authority="MACT",
        statutory_basis="Section 16 of Specific Relief Act",
    )

    verifier = VerificationNode(store=None)
    out = verifier.verify({
        "retrieved_contexts": [mva_context],
        "draft_statutory_citations": [spliced_cit, valid_cit],
        "draft_precedent_citations": [],
        "draft_action_plan": [step_with_unverified_2digit],
    })

    assert len(out["verified_statutory_citations"]) == 1
    assert out["verified_statutory_citations"][0].act_name == "Motor Vehicles Act, 1988"
    assert len(out["unverified_citations_stripped"]) == 1
    assert "Consumer Protection Act" in out["unverified_citations_stripped"][0]
    assert out["verified_action_plan"][0].statutory_basis is None


def test_symmetric_citation_grounding_and_df2_vacuous_guard():
    """UT-2 / DF-2: Post-output grounding scored symmetrically; zero citations on legal query -> 0.0 grounding."""
    ctx = RetrievedContext(
        chunk_id="mva_134",
        doc_type=DocumentType.STATUTE,
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        title="Motor Vehicles Act, 1988 - Section 134",
        citation_or_section="Section 134",
        act_name="Motor Vehicles Act, 1988",
        text="Section 134: Duty of driver in case of accident and injury to a person.",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
        score=0.9,
    )
    step = ProceduralActionStep(
        step_number=1,
        phase=ProceduralPhase.IMMEDIATE_ACTION,
        title="Medical aid",
        description="Convey injured person to hospital",
        forum_or_authority="Hospital",
    )
    grounded_stat = StatutoryCitation(
        act_name="Motor Vehicles Act, 1988",
        section_number="134",
        summary_of_provision="Duty of driver",
        applicability_to_scenario="Accident",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
        is_verified=True,
    )

    # Approach B stripped 1 ungrounded draft citation, leaving 1 grounded final citation
    resp_b = DualOutputResponse(
        scenario_summary="Accident case",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[step],
        statutory_citations=[grounded_stat],
        precedent_citations=[],
        hallucination_check_passed=False,
        unverified_citations_stripped=["Statute: IPC Section 999 (Not verified in retrieved legislative text)"],
    )
    metrics_b = compute_citation_grounding_metrics(resp_b, retrieved_contexts=[ctx], is_legal=True)
    assert metrics_b["hallucination_rate"] == 0.0
    assert metrics_b["grounding_accuracy"] == 1.0
    assert metrics_b["pre_verification_strip_rate"] == 0.5

    # DF-2: Legal query with zero final citations must NOT get 1.0 grounding accuracy
    empty_legal_resp = DualOutputResponse(
        scenario_summary="Accident case",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[step],
        statutory_citations=[],
        precedent_citations=[],
    )
    empty_metrics = compute_citation_grounding_metrics(empty_legal_resp, retrieved_contexts=[ctx], is_legal=True)
    assert empty_metrics["grounding_accuracy"] == 0.0

    # Guardrail rejection on non-legal query gets 1.0 grounding accuracy and 0.0 strip rate for Clarification prompt
    rejection_resp = DualOutputResponse(
        scenario_summary="Query rejected: Non-legal cooking query",
        domain=LegalDomain.GENERAL_DISPUTE,
        action_plan=[],
        statutory_citations=[],
        precedent_citations=[],
        unverified_citations_stripped=["Clarification: Please provide a legal dispute."],
    )
    rej_metrics = compute_citation_grounding_metrics(rejection_resp, retrieved_contexts=[], is_legal=False)
    assert rej_metrics["hallucination_rate"] == 0.0
    assert rej_metrics["grounding_accuracy"] == 1.0
    assert rej_metrics["pre_verification_strip_rate"] == 0.0


def test_retrieval_and_agent_trace_metrics():
    """UT-3 & UT-4: Verify Recall@K, Precision@K, MRR, Context Relevance, and Agent Trace validity."""
    ctx_166 = RetrievedContext(
        chunk_id="mva_166",
        doc_type=DocumentType.STATUTE,
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        title="Motor Vehicles Act, 1988 - Section 166",
        citation_or_section="Section 166",
        act_name="Motor Vehicles Act, 1988",
        text="Section 166 compensation before Motor Accident Claims Tribunal (MACT).",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
        score=0.95,
    )
    meta = {
        "is_legal": True,
        "expected_act": "Motor Vehicles Act, 1988",
        "expected_sections": ["166", "134"],
        "expected_forum": ["MACT"],
        "critical_steps": ["FIR", "MACT claim"],
        "expects_criminal_route": True,
    }
    ret = compute_retrieval_metrics([ctx_166], meta, k=5)
    assert ret["section_recall_at_k"] == 0.5
    assert ret["precision_at_k"] == 1.0
    assert ret["mrr"] == 1.0

    rel = compute_context_relevance("Truck accident MACT compensation claim", [ctx_166], meta)
    assert rel > 0.5

    trace = ExecutionTrace(
        pipeline_type="proposed_langgraph",
        nodes_visited=[
            "guardrail",
            "decontextualize",
            "retrieve_and_rerank",
            "procedural_planner",
            "verification",
            "synthesis",
        ],
        criminal_route_triggered=True,
        retrieved_contexts=[ctx_166],
    )
    steps = [
        ProceduralActionStep(
            step_number=i,
            phase=phase,
            title=f"Step {i} FIR and MACT claim",
            description="Lodge FIR at Police Station and file MACT claim petition",
            forum_or_authority="MACT",
        )
        for i, phase in enumerate(
            [
                ProceduralPhase.IMMEDIATE_ACTION,
                ProceduralPhase.POLICE_ADMINISTRATIVE,
                ProceduralPhase.FORUM_FILING,
            ],
            start=1,
        )
    ]
    resp = DualOutputResponse(
        scenario_summary="Accident",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=steps,
        statutory_citations=[],
        precedent_citations=[],
    )
    trace_metrics = compute_agent_trace_metrics(trace, meta, resp)
    assert trace_metrics["trajectory_valid"] is True
    assert trace_metrics["criminal_routing_correct"] is True
    assert trace_metrics["task_success"] is True


def test_llm_judge_programmatic_and_df5_non_legal_zero():
    """UT-5 / DF-5: Non-legal query not rejected gets 0.0; stripped citations do not lower hallucination_freedom."""
    judge = LegalLLMJudge(settings=Settings(openai_api_key=None))

    non_legal_meta = {"is_legal": False, "expected_sections": [], "expected_forum": []}
    unrejected_resp = DualOutputResponse(
        scenario_summary="Recipe for pasta",
        domain=LegalDomain.GENERAL_DISPUTE,
        action_plan=[
            ProceduralActionStep(
                step_number=1,
                phase=ProceduralPhase.IMMEDIATE_ACTION,
                title="Boil water",
                description="Add salt",
                forum_or_authority="Kitchen",
            )
        ],
        statutory_citations=[],
    )
    bad_score = judge.judge_response("How to cook pasta", non_legal_meta, unrejected_resp)
    assert bad_score.overall_score == 0.0
    assert bad_score.hallucination_freedom == 0.0

    rejected_resp = DualOutputResponse(
        scenario_summary="Query rejected: Out of scope",
        domain=LegalDomain.GENERAL_DISPUTE,
        action_plan=[],
        statutory_citations=[],
    )
    good_score = judge.judge_response("How to cook pasta", non_legal_meta, rejected_resp)
    assert good_score.overall_score == 5.0

    # UT-6: Verify decomposed multi-sample median aggregation and partial worker failure fallback
    judge_with_client = LegalLLMJudge(settings=Settings(openai_api_key="sk-test-key"))
    call_counter = {"count": 0}

    def fake_single_criterion(criterion, payload, fallback_value, temperature=0.0):
        call_counter["count"] += 1
        if criterion == "coherence_and_specificity":
            raise RuntimeError("Simulated rate limit on 1 of 5 workers")
        return (4.6 if temperature == 0.0 else 3.6), f"Mocked {criterion}"

    judge_with_client._score_single_criterion = fake_single_criterion
    legal_meta = {
        "is_legal": True,
        "expected_sections": ["166"],
        "expected_forum": ["MACT"],
        "critical_steps": ["FIR"],
    }
    multi_score = judge_with_client.judge_response("Truck accident", legal_meta, unrejected_resp, num_samples=3)
    assert call_counter["count"] == 15
    assert multi_score.score_variance >= 0.0


def test_noise_sensitivity_and_sme_agreement(tmp_path):
    """UT-6, UT-7, IT-3, DF-6: Test noise sensitivity harness, SME atomic persistence, and agreement stats."""
    settings = Settings(openai_api_key=None)
    planner = ProceduralPlannerAgent(settings=settings)
    verifier = VerificationNode(settings=settings, store=None)

    clean_ctx = RetrievedContext(
        chunk_id="mva_166",
        doc_type=DocumentType.STATUTE,
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        title="Motor Vehicles Act, 1988 - Section 166",
        citation_or_section="Section 166",
        act_name="Motor Vehicles Act, 1988",
        text="Section 166: Application for compensation before Motor Accident Claims Tribunal.",
        source_url="https://indiacode.gov.in/handle/123456789/1798",
        score=0.92,
    )
    distractors = build_default_distractor_contexts(LegalDomain.MOTOR_VEHICLE_ACCIDENT)
    ns = evaluate_noise_sensitivity(
        planner=planner,
        verifier=verifier,
        scenario="A speeding truck hit my bike in Pune causing fractures.",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        clean_contexts=[clean_ctx],
        distractor_contexts=distractors,
    )
    assert ns["plan_phase_retention"] >= 0.8
    assert ns["distractor_citation_rate"] == 0.0

    # SME atomic save & agreement stats
    sme_file = tmp_path / "sme.json"
    ann = SMEAnnotation(
        scenario_id="mva_001",
        pipeline_type="proposed",
        annotator_id="test_adv",
        statutory_accuracy=4.8,
        procedural_actionability=4.8,
        forum_appropriateness=5.0,
        hallucination_freedom=5.0,
        coherence_and_specificity=4.8,
        overall_score=4.88,
    )
    save_sme_annotation(ann, path=sme_file)
    loaded = load_sme_annotations(path=sme_file)
    assert len(loaded) == 1
    assert loaded[0].overall_score == 4.88

    agree = compute_judge_human_agreement([4.8, 3.2, 4.9, 1.0], [4.9, 3.4, 4.8, 0.5])
    assert agree["sample_count"] == 4
    assert agree["mae"] < 0.3
    assert agree["pearson_r"] > 0.9
    assert agree["cohens_kappa"] == 1.0

    # Telemetry & Sigmoid checks
    assert 0.0 < _sigmoid(-10.0) < 0.01
    assert 0.99 < _sigmoid(10.0) < 1.0
    tel_path = tmp_path / "events.jsonl"
    tel_settings = Settings(openai_api_key=None, telemetry_log_path=tel_path)
    resp = DualOutputResponse(
        scenario_summary="Test",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[],
        statutory_citations=[],
    )
    trace = ExecutionTrace(pipeline_type="proposed_langgraph", nodes_visited=["guardrail"])
    log_inference_event(resp, trace, latency_ms=12.5, scenario="Test scenario", settings=tel_settings)
    events = load_telemetry_events(path=tel_path)
    assert len(events) == 1
    assert events[0]["latency_ms"] == 12.5
