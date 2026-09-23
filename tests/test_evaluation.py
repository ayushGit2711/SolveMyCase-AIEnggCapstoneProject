"""Tests for evaluation metrics and benchmark scenarios."""

from pathlib import Path
import json
import pytest

from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    LegalDomain,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
    StatutoryCitation,
)
from solvemycase.evaluation.metrics import (
    compute_citation_grounding_metrics,
    compute_procedural_completeness,
    verify_target_forum,
)
from solvemycase.evaluation.llm_judge import LegalLLMJudge


def test_benchmark_dataset_integrity():
    """Verify benchmark dataset has 50+ scenarios with valid fields."""
    dataset_path = Path(__file__).parent.parent / "evaluation" / "benchmark_dataset.json"
    assert dataset_path.exists(), "benchmark_dataset.json must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 50, f"Expected 50+ scenarios, found {len(data)}"

    for item in data:
        assert "id" in item
        assert "scenario" in item and len(item["scenario"]) > 15
        assert "domain" in item
        assert "is_legal" in item


def test_metrics_computation():
    """Verify metrics calculation logic."""
    step = ProceduralActionStep(
        step_number=1,
        phase=ProceduralPhase.IMMEDIATE_ACTION,
        title="Immediate medical aid",
        description="Convey injured person to nearest hospital",
        forum_or_authority="Hospital",
    )
    stat = StatutoryCitation(
        act_name="Motor Vehicles Act, 1988",
        section_number="134",
        summary_of_provision="Duty of driver",
        applicability_to_scenario="Accident",
        source_url="https://indiacode.nic.in",
        is_verified=True,
    )

    resp = DualOutputResponse(
        scenario_summary="Accident case",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[step],
        statutory_citations=[stat],
        precedent_citations=[],
        hallucination_check_passed=True,
        unverified_citations_stripped=[],
    )

    grounding = compute_citation_grounding_metrics(resp)
    assert grounding["hallucination_rate"] == 0.0
    assert grounding["grounding_accuracy"] == 1.0

    proc = compute_procedural_completeness(resp)
    assert proc["completeness_score"] == round(1.0 / 6.0, 4)
    assert proc["step_count"] == 1

    assert verify_target_forum(resp, ["Hospital"]) is True
    assert verify_target_forum(resp, ["DCDRC"]) is False


def test_programmatic_llm_judge():
    """Verify fallback programmatic evaluation scoring."""
    judge = LegalLLMJudge()
    meta = {
        "is_legal": True,
        "expected_sections": ["134", "166"],
        "expected_forum": ["Hospital", "MACT"],
    }
    step = ProceduralActionStep(
        step_number=1,
        phase=ProceduralPhase.IMMEDIATE_ACTION,
        title="Immediate medical aid",
        description="Convey injured person to hospital",
        forum_or_authority="Hospital",
    )
    stat = StatutoryCitation(
        act_name="Motor Vehicles Act, 1988",
        section_number="134",
        summary_of_provision="Duty of driver",
        applicability_to_scenario="Accident",
        source_url="https://indiacode.nic.in",
        is_verified=True,
    )
    resp = DualOutputResponse(
        scenario_summary="Accident case",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[step],
        statutory_citations=[stat],
        precedent_citations=[],
        hallucination_check_passed=True,
        unverified_citations_stripped=[],
    )

    score = judge.judge_response("Motor accident scenario", meta, resp)
    assert 0.0 <= score.overall_score <= 5.0
    assert score.statutory_accuracy > 0.0
