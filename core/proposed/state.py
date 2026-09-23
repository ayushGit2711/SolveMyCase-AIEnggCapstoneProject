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

    # Decontextualized Retrieval Queries
    statute_queries: List[str]
    precedent_queries: List[str]

    # Retrieved & Reranked Legal Context
    retrieved_contexts: List[RetrievedContext]

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
