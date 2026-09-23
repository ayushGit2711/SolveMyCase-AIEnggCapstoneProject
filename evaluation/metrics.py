"""Quantitative evaluation metrics for legal RAG pipelines.

Measures:
1. Statutory & Precedent Grounding Rate (0% tolerance for hallucinations).
2. Procedural Completeness (coverage of chronological stages from FIR to forum filing).
3. Forum and Authority Accuracy.
4. Latency and Token Efficiency.
"""

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    ProceduralPhase,
    RetrievedContext,
)


class EvaluationMetricSummary(BaseModel):
    """Container for quantitative benchmark results of a single scenario run."""
    scenario_id: str
    pipeline_type: str = Field(..., description="'baseline' or 'proposed'")
    domain: str
    latency_seconds: float
    total_statutory_citations: int
    verified_statutory_citations: int
    unverified_citations_count: int
    statutory_hallucination_rate: float = Field(
        ..., description="Ratio of ungrounded citations to total citations (0.0 is ideal)."
    )
    grounding_accuracy: float = Field(
        ..., description="Ratio of verified citations to total citations (1.0 is ideal)."
    )
    procedural_step_count: int
    procedural_phases_covered: List[str]
    phase_completeness_score: float = Field(
        ..., description="Fraction of all 6 standard procedural phases addressed (0.0 to 1.0)."
    )
    correct_forum_identified: bool = Field(
        default=True, description="Whether the designated statutory forum was recognized."
    )


def compute_citation_grounding_metrics(
    response: DualOutputResponse,
    retrieved_contexts: Optional[List[RetrievedContext]] = None,
) -> Dict[str, float]:
    """Compute hallucination rate and grounding accuracy for statutory and case citations.

    Returns:
        Dict with 'hallucination_rate' (0.0 is perfect) and 'grounding_accuracy' (1.0 is perfect).
    """
    total_citations = len(response.statutory_citations) + len(response.precedent_citations)
    unverified_count = len(response.unverified_citations_stripped)

    # In baseline (Approach A), citations are directly in the response without stripped tracking
    if retrieved_contexts is not None and not response.unverified_citations_stripped:
        combined_text = " ".join(
            f"{c.title} {c.citation_or_section} {c.text}" for c in retrieved_contexts
        ).lower()

        unverified = 0
        for stat in response.statutory_citations:
            sec_clean = stat.section_number.strip().lower()
            if sec_clean and f"section {sec_clean}" not in combined_text and sec_clean not in combined_text:
                unverified += 1
        unverified_count = unverified

    all_proposed = total_citations + unverified_count
    if all_proposed == 0:
        return {"hallucination_rate": 0.0, "grounding_accuracy": 1.0}

    hallucination_rate = round(unverified_count / all_proposed, 4)
    grounding_accuracy = round(total_citations / all_proposed, 4)

    return {
        "hallucination_rate": hallucination_rate,
        "grounding_accuracy": grounding_accuracy,
    }


def compute_procedural_completeness(response: DualOutputResponse) -> Dict[str, any]:
    """Measure coverage across the 6 essential procedural phases in Indian dispute resolution."""
    all_phases = {
        ProceduralPhase.IMMEDIATE_ACTION,
        ProceduralPhase.POLICE_ADMINISTRATIVE,
        ProceduralPhase.EVIDENTIARY_DOCUMENTATION,
        ProceduralPhase.LEGAL_NOTICE,
        ProceduralPhase.FORUM_FILING,
        ProceduralPhase.LIMITATION_APPEAL,
    }

    covered_phases: Set[ProceduralPhase] = {step.phase for step in response.action_plan}
    completeness_score = round(len(covered_phases) / len(all_phases), 4)

    return {
        "phases_covered": [p.value for p in covered_phases],
        "completeness_score": completeness_score,
        "step_count": len(response.action_plan),
    }


def verify_target_forum(response: DualOutputResponse, expected_keywords: List[str]) -> bool:
    """Verify if the action plan targets the legally designated forum or tribunal."""
    if not expected_keywords:
        return True

    all_plan_text = " ".join(
        f"{step.title} {step.description} {step.forum_or_authority}"
        for step in response.action_plan
    ).lower()

    return any(kw.lower() in all_plan_text for kw in expected_keywords)
