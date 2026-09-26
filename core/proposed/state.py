"""LangGraph state definition for the solvemycase agentic pipeline."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from solvemycase.data.ingestion.schema import (
    DualOutputResponse,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    RetrievedContext,
    StatutoryCitation,
)


class AgentState(TypedDict):
    """Execution state passed through the LangGraph StateGraph nodes."""

    # User Input & Guardrails
    scenario: str
    is_legal: bool
    domain: LegalDomain
    rejection_reason: Optional[str]
    clarification_prompt: Optional[str]
    key_entities: Dict[str, Any]

    # Decontextualized Retrieval Queries & Routing Telemetry
    statute_queries: List[str]
    precedent_queries: List[str]
    criminal_route_triggered: bool
    criminal_queries: List[str]
    retrieval_gate_triggered: bool

    # Legal context. Each key has exactly one writer:
    # - candidate_contexts: reranked candidates, written only by retrieve_and_rerank;
    # - retrieved_contexts: the candidates that apply to the facts, written only by applicability_check
    #   (empty on a coverage gap). The planner and the verifier read only retrieved_contexts.
    candidate_contexts: List[RetrievedContext]
    retrieved_contexts: List[RetrievedContext]

    # Applicability check / coverage gap
    coverage_gap: bool
    coverage_gap_reason: Optional[str]
    coverage_note: Optional[str]
    applicability_mode: Optional[str]
    applicability_rejected: List[str]

    # Draft Plan (Pre-Verification)
    draft_action_plan: List[ProceduralActionStep]
    draft_statutory_citations: List[StatutoryCitation]
    draft_precedent_citations: List[PrecedentCitation]

    # Verified Plan (Post-Verification)
    verified_action_plan: List[ProceduralActionStep]
    verified_statutory_citations: List[StatutoryCitation]
    verified_precedent_citations: List[PrecedentCitation]
    hallucination_check_passed: bool
    unverified_citations_stripped: List[str]

    # State loop controls
    iteration_count: int
    final_response: Optional[DualOutputResponse]
