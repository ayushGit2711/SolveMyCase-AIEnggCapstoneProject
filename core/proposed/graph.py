"""LangGraph StateGraph orchestration for solvemycase (Approach B).

Assembles the full multi-agent, deterministic legal control flow:
Guardrail -> Decontextualization -> Hybrid Retrieval & Reranker ->
Procedural Planner -> Verification Node -> Dual-Output Synthesis.
"""

from typing import Any, Dict, List, Optional
from langgraph.graph import END, StateGraph

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.guardrails.scope_checker import ScopeChecker
from solvemycase.core.proposed.procedural_planner import ProceduralPlannerAgent
from solvemycase.core.proposed.state import AgentState
from solvemycase.core.proposed.verification_node import VerificationNode
from solvemycase.core.retrieval.decontextualizer import LegalDecontextualizer
from solvemycase.core.retrieval.reranker import LegalCrossEncoderReranker
from solvemycase.data.ingestion.schema import (
    DualOutputResponse,
    LegalDomain,
    RetrievedContext,
)
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore


class LegalAgentGraph:
    """Orchestrates the LangGraph agent state graph for Indian legal problem solving."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantLegalStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or QdrantLegalStore(settings=self.settings)
        self.embedder = embedder or EmbeddingProvider(settings=self.settings)
        self.scope_checker = ScopeChecker(settings=self.settings)
        self.decontextualizer = LegalDecontextualizer(settings=self.settings)
        self.reranker = LegalCrossEncoderReranker(settings=self.settings)
        self.planner = ProceduralPlannerAgent(settings=self.settings)
        self.verifier = VerificationNode(settings=self.settings)

        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph workflow."""
        builder = StateGraph(AgentState)

        # Add Nodes
        builder.add_node("guardrail", self._guardrail_step)
        builder.add_node("decontextualize", self._decontextualize_step)
        builder.add_node("retrieve_and_rerank", self._retrieve_and_rerank_step)
        builder.add_node("procedural_planner", self._procedural_planner_step)
        builder.add_node("verification", self._verification_step)
        builder.add_node("synthesis", self._synthesis_step)
        builder.add_node("handle_rejection", self._rejection_step)

        # Entry Point
        builder.set_entry_point("guardrail")

        # Conditional Edge after Guardrail
        builder.add_conditional_edges(
            "guardrail",
            self._route_after_guardrail,
            {
                "proceed": "decontextualize",
                "reject": "handle_rejection",
            },
        )

        # Linear Pipeline
        builder.add_edge("decontextualize", "retrieve_and_rerank")
        builder.add_edge("retrieve_and_rerank", "procedural_planner")
        builder.add_edge("procedural_planner", "verification")
        builder.add_edge("verification", "synthesis")
        builder.add_edge("synthesis", END)
        builder.add_edge("handle_rejection", END)

        return builder.compile()

    # Node Step Functions
    def _guardrail_step(self, state: AgentState) -> Dict[str, Any]:
        result = self.scope_checker.check_scope(state["scenario"])
        return {
            "is_legal": result.is_legal,
            "domain": result.domain,
            "rejection_reason": result.rejection_reason,
            "clarification_prompt": result.clarification_prompt,
            "key_entities": result.key_entities,
        }

    def _route_after_guardrail(self, state: AgentState) -> str:
        if state.get("is_legal", False):
            return "proceed"
        return "reject"

    def _rejection_step(self, state: AgentState) -> Dict[str, Any]:
        reason = state.get("rejection_reason") or "Query out of scope."
        prompt = state.get("clarification_prompt") or "Please provide a valid Indian legal dispute scenario."
        rejection_response = DualOutputResponse(
            scenario_summary=f"Query rejected: {reason}",
            domain=state.get("domain", LegalDomain.GENERAL_DISPUTE),
            action_plan=[],
            statutory_citations=[],
            precedent_citations=[],
            hallucination_check_passed=True,
            unverified_citations_stripped=[f"Clarification: {prompt}"],
        )
        return {"final_response": rejection_response}

    def _decontextualize_step(self, state: AgentState) -> Dict[str, Any]:
        queries = self.decontextualizer.decontextualize(
            scenario=state["scenario"],
            domain=state["domain"],
            entities=state.get("key_entities"),
        )
        return {
            "statute_queries": queries.statute_queries,
            "precedent_queries": queries.precedent_queries,
        }

    def _retrieve_and_rerank_step(self, state: AgentState) -> Dict[str, Any]:
        all_queries = state.get("statute_queries", []) + state.get("precedent_queries", [])
        if not all_queries:
            all_queries = [state["scenario"]]

        # Hybrid Search across sub-queries
        candidates_map: Dict[str, RetrievedContext] = {}
        for q in all_queries:
            q_emb = self.embedder.get_embeddings([q])[0]
            hits = self.store.hybrid_search(
                query_text=q,
                query_embedding=q_emb,
                top_k=self.settings.max_retrieved_chunks,
                domain_filter=state.get("domain"),
            )
            for hit in hits:
                if hit.chunk_id not in candidates_map or hit.score > candidates_map[hit.chunk_id].score:
                    candidates_map[hit.chunk_id] = hit

        candidates = list(candidates_map.values())

        # Cross-Encoder Reranking
        reranked = self.reranker.rerank(
            query=state["scenario"],
            candidates=candidates,
            top_k=self.settings.rerank_top_k,
        )

        return {"retrieved_contexts": reranked}

    def _procedural_planner_step(self, state: AgentState) -> Dict[str, Any]:
        return self.planner.plan(state)

    def _verification_step(self, state: AgentState) -> Dict[str, Any]:
        return self.verifier.verify(state)

    def _synthesis_step(self, state: AgentState) -> Dict[str, Any]:
        response = DualOutputResponse(
            scenario_summary=state["scenario"][:240] + ("..." if len(state["scenario"]) > 240 else ""),
            domain=state["domain"],
            action_plan=state.get("verified_action_plan", []),
            statutory_citations=state.get("verified_statutory_citations", []),
            precedent_citations=state.get("verified_precedent_citations", []),
            hallucination_check_passed=state.get("hallucination_check_passed", True),
            unverified_citations_stripped=state.get("unverified_citations_stripped", []),
        )
        return {"final_response": response}

    def run(self, scenario: str) -> DualOutputResponse:
        """Execute the complete Approach B pipeline for a given scenario.

        Args:
            scenario: Factual legal scenario narrative.

        Returns:
            DualOutputResponse with verified action plan and citations.
        """
        initial_state: AgentState = {
            "scenario": scenario,
            "is_legal": False,
            "domain": LegalDomain.GENERAL_DISPUTE,
            "rejection_reason": None,
            "clarification_prompt": None,
            "key_entities": {},
            "statute_queries": [],
            "precedent_queries": [],
            "retrieved_contexts": [],
            "draft_action_plan": [],
            "draft_statutory_citations": [],
            "draft_precedent_citations": [],
            "verified_action_plan": [],
            "verified_statutory_citations": [],
            "verified_precedent_citations": [],
            "hallucination_check_passed": False,
            "unverified_citations_stripped": [],
            "iteration_count": 0,
            "final_response": None,
        }

        output_state = self.graph.invoke(initial_state)
        return output_state["final_response"]
