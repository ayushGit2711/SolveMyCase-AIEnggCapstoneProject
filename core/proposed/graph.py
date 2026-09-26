"""LangGraph StateGraph orchestration for solvemycase (Approach B).

Assembles the full multi-agent, deterministic legal control flow:
Guardrail -> Decontextualization -> Hybrid Retrieval & Reranker -> Applicability Check ->
Procedural Planner -> Verification Node -> Dual-Output Synthesis.

When the applicability check finds no provision or judgment in the corpus that applies to the facts, the
graph skips planning and verification and synthesizes an honest coverage-gap answer instead of citing the
nearest unrelated law.
"""

import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langgraph.graph import END, StateGraph

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.guardrails.scope_checker import ScopeChecker
from solvemycase.core.proposed.applicability_judge import ApplicabilityJudge, withhold_unscreened
from solvemycase.core.proposed.coverage import (
    build_coverage_gap_response,
    no_verified_citations_note,
    summarize_corpus_coverage,
)
from solvemycase.core.proposed.procedural_planner import ProceduralPlannerAgent
from solvemycase.core.proposed.state import AgentState
from solvemycase.core.proposed.verification_node import VerificationNode
from solvemycase.core.retrieval.decontextualizer import LegalDecontextualizer
from solvemycase.core.retrieval.reranker import LegalCrossEncoderReranker
from solvemycase.core.telemetry import log_inference_event
from solvemycase.data.ingestion.schema import (
    CLARIFICATION_PREFIX,
    ApplicabilityMode,
    CoverageGapReason,
    DualOutputResponse,
    ExecutionTrace,
    LegalDomain,
    REJECTION_SUMMARY_PREFIX,
    RetrievedContext,
    compile_keyword_pattern,
    should_trigger_criminal_route,
)
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

logger = logging.getLogger(__name__)

# Route labels of the conditional edges.
ROUTE_PROCEED = "proceed"
ROUTE_REJECT = "reject"
ROUTE_COVERED = "covered"
ROUTE_GAP = "gap"

# Statute sub-queries that look criminal are also sent to the criminal-code sub-layer.
_CRIMINAL_QUERY_HINT = compile_keyword_pattern([
    r"bns", r"bnss", r"ipc", r"crpc", r"nyaya", r"nagarik", r"penal", r"criminal", r"fir", r"police",
    r"negligen\w*", r"offen[cs]e\w*", r"rash", r"cruelty",
])


def _dedupe_texts(texts: List[str]) -> List[str]:
    """Drop empty and case-insensitively repeated query strings, keeping the first occurrence."""
    seen = set()
    unique = []
    for text in texts:
        cleaned = (text or "").strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            unique.append(cleaned)
    return unique


def _merge_hits(candidates_map: Dict[str, RetrievedContext], hits: List[RetrievedContext]) -> None:
    """Add hits to the candidate pool, keeping each chunk's best hybrid score."""
    for hit in hits:
        if hit.chunk_id not in candidates_map or hit.score > candidates_map[hit.chunk_id].score:
            candidates_map[hit.chunk_id] = hit


class LegalAgentGraph:
    """Orchestrates the LangGraph agent state graph for Indian legal problem solving."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantLegalStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
        applicability_judge: Optional[ApplicabilityJudge] = None,
        reranker: Optional[LegalCrossEncoderReranker] = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or QdrantLegalStore(settings=self.settings)
        self.embedder = embedder or EmbeddingProvider(settings=self.settings)
        self.scope_checker = ScopeChecker(settings=self.settings)
        self.decontextualizer = LegalDecontextualizer(settings=self.settings)
        self.reranker = reranker or LegalCrossEncoderReranker(settings=self.settings)
        self.applicability_judge = applicability_judge or ApplicabilityJudge(settings=self.settings)
        self.planner = ProceduralPlannerAgent(settings=self.settings)
        self.verifier = VerificationNode(settings=self.settings)
        self._coverage_cache: Optional[Tuple[Tuple[int, int], str]] = None

        self.graph = self._build_graph()

    def coverage_summary(self) -> str:
        """Plain-language summary of the laws in the loaded corpus (cached until the corpus changes)."""
        docs = self.store.corpus_documents
        key = (id(docs), len(docs))
        if self._coverage_cache is None or self._coverage_cache[0] != key:
            self._coverage_cache = (key, summarize_corpus_coverage(docs))
        return self._coverage_cache[1]

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph workflow."""
        builder = StateGraph(AgentState)

        # Add Nodes
        builder.add_node("guardrail", self._guardrail_step)
        builder.add_node("decontextualize", self._decontextualize_step)
        builder.add_node("retrieve_and_rerank", self._retrieve_and_rerank_step)
        builder.add_node("applicability_check", self._applicability_step)
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
                ROUTE_PROCEED: "decontextualize",
                ROUTE_REJECT: "handle_rejection",
            },
        )

        # Linear Pipeline
        builder.add_edge("decontextualize", "retrieve_and_rerank")
        builder.add_edge("retrieve_and_rerank", "applicability_check")

        # Conditional Edge after the applicability check: plan only when some law applies to the facts.
        builder.add_conditional_edges(
            "applicability_check",
            self._route_after_applicability,
            {
                ROUTE_COVERED: "procedural_planner",
                ROUTE_GAP: "synthesis",
            },
        )
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
            return ROUTE_PROCEED
        return ROUTE_REJECT

    def _rejection_step(self, state: AgentState) -> Dict[str, Any]:
        reason = state.get("rejection_reason") or "Query out of scope."
        prompt = state.get("clarification_prompt") or "Please provide a valid Indian legal dispute scenario."
        rejection_response = DualOutputResponse(
            scenario_summary=f"{REJECTION_SUMMARY_PREFIX}{reason}",
            domain=state.get("domain", LegalDomain.GENERAL_DISPUTE),
            action_plan=[],
            statutory_citations=[],
            precedent_citations=[],
            hallucination_check_passed=True,
            unverified_citations_stripped=[f"{CLARIFICATION_PREFIX}{prompt}"],
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
        scenario = state["scenario"]
        domain = state.get("domain")
        statute_queries = list(state.get("statute_queries", []) or [])
        precedent_queries = list(state.get("precedent_queries", []) or [])
        sub_queries = statute_queries + precedent_queries

        # 2a. Hybrid search over every sub-query plus the raw scenario.
        search_queries = _dedupe_texts(sub_queries + [scenario])

        # 2b. Criminal Code Deep RAG (Parallel Sub-Layer for BNS/BNSS/IPC)
        has_criminal_indicators = should_trigger_criminal_route(domain, scenario)
        crim_queries: List[str] = []
        if has_criminal_indicators:
            crim_queries = [q for q in statute_queries if _CRIMINAL_QUERY_HINT.search(q or "")]
            if not crim_queries:
                crim_queries = [f"{scenario} criminal offence fir police bns"]
            crim_queries = crim_queries[:2]

        embedding_for = self._query_embeddings(_dedupe_texts(search_queries + crim_queries))

        candidates_map: Dict[str, RetrievedContext] = {}
        for q in search_queries:
            hits = self.store.hybrid_search(
                query_text=q,
                query_embedding=embedding_for(q),
                top_k=self.settings.max_retrieved_chunks,
                domain_filter=domain,
            )
            _merge_hits(candidates_map, hits)

        for cq in crim_queries:
            crim_hits = self.store.criminal_code_search(
                query_text=cq,
                query_embedding=embedding_for(cq),
                top_k=5,
            )
            _merge_hits(candidates_map, crim_hits)

        # Cross-Encoder Reranking: score = best match against the scenario or any legal sub-query. Every
        # candidate is scored once; the applicability check then trims the best ones to rerank_top_k.
        pool = list(candidates_map.values())
        scored = self.reranker.rerank(query=scenario, candidates=pool, top_k=len(pool), extra_queries=sub_queries)

        # 2c. Recall aid: widen the pool with an unfiltered search when even the best candidate scores low.
        # Whether any of it applies is decided by the applicability check, not by this threshold.
        retrieval_gate_triggered = False
        if not scored or scored[0].score < self.settings.retrieval_min_confidence:
            retrieval_gate_triggered = True
            broad_hits = self.store.hybrid_search(
                query_text=scenario,
                query_embedding=embedding_for(scenario),
                top_k=self.settings.max_retrieved_chunks,
                domain_filter=None,
            )
            new_hits = [hit for hit in broad_hits if hit.chunk_id not in candidates_map]
            if new_hits:
                scored = scored + self.reranker.rerank(
                    query=scenario, candidates=new_hits, top_k=len(new_hits), extra_queries=sub_queries
                )
                scored.sort(key=lambda ctx: ctx.score, reverse=True)

        candidate_k = max(self.settings.applicability_candidate_k, self.settings.rerank_top_k)
        return {
            "candidate_contexts": scored[:candidate_k],
            "criminal_route_triggered": bool(has_criminal_indicators),
            "criminal_queries": crim_queries if has_criminal_indicators else [],
            "retrieval_gate_triggered": retrieval_gate_triggered,
        }

    def _query_embeddings(self, texts: List[str]):
        """Embed all query texts in one batched call; return a lookup text -> vector (None = skip dense search).

        When the embedder had to fall back to another embedding space than the index was built with (e.g. an
        OpenAI outage turns query vectors into offline hash vectors), dense hits would be noise, so the searches
        run on BM25 only.
        """
        vectors = self.embedder.get_embeddings(texts) if texts else []
        embedder_id = getattr(self.embedder, "last_embedder_id", None)
        dense_ok = self.store.embedding_space_matches(embedder_id)
        if not dense_ok:
            logger.warning(
                "Query embeddings (%s) don't match the index (%s); using keyword search only.",
                embedder_id,
                sorted(self.store.loaded_embedder_ids),
            )
        by_text = {text.lower(): vec for text, vec in zip(texts, vectors)}

        def embedding_for(text: str) -> Optional[List[float]]:
            if not dense_ok:
                return None
            key = (text or "").strip().lower()
            if key not in by_text:
                by_text[key] = self.embedder.get_embeddings([text or ""])[0]
            return by_text[key]

        return embedding_for

    def _applicability_step(self, state: AgentState) -> Dict[str, Any]:
        domain = state.get("domain")
        assessment = self.applicability_judge.assess(
            scenario=state["scenario"],
            domain=domain,
            candidates=state.get("candidate_contexts", []) or [],
            top_k=self.settings.rerank_top_k,
        )
        mode = ApplicabilityMode(assessment.mode)
        applicable = list(assessment.applicable)
        gap_reason: Optional[CoverageGapReason] = None
        if assessment.coverage_gap:
            gap_reason = CoverageGapReason.NO_APPLICABLE_LAW
        elif withhold_unscreened(domain, mode):
            # No LLM verdict for a question without a dedicated corpus: don't cite unscreened sources.
            applicable = []
            gap_reason = CoverageGapReason.SCREENING_UNAVAILABLE
        return {
            "retrieved_contexts": applicable,
            "coverage_gap": gap_reason is not None,
            "coverage_gap_reason": gap_reason.value if gap_reason else None,
            "applicability_mode": mode.value,
            "applicability_rejected": [cand.title for cand in assessment.rejected],
        }

    def _route_after_applicability(self, state: AgentState) -> str:
        return ROUTE_GAP if state.get("coverage_gap", False) else ROUTE_COVERED

    def _procedural_planner_step(self, state: AgentState) -> Dict[str, Any]:
        return self.planner.plan(state)

    def _verification_step(self, state: AgentState) -> Dict[str, Any]:
        return self.verifier.verify(state)

    def _synthesis_step(self, state: AgentState) -> Dict[str, Any]:
        domain = state.get("domain") or LegalDomain.GENERAL_DISPUTE
        if state.get("coverage_gap", False):
            reason = CoverageGapReason(state.get("coverage_gap_reason") or CoverageGapReason.NO_APPLICABLE_LAW)
            gap_response = build_coverage_gap_response(state["scenario"], domain, self.coverage_summary(), reason)
            return {"final_response": gap_response, "coverage_note": gap_response.coverage_note}

        statutory = state.get("verified_statutory_citations", [])
        precedents = state.get("verified_precedent_citations", [])
        coverage_note = None
        if not statutory and not precedents:
            coverage_note = no_verified_citations_note(self.coverage_summary())

        response = DualOutputResponse(
            scenario_summary=state["scenario"][:240] + ("..." if len(state["scenario"]) > 240 else ""),
            domain=domain,
            action_plan=state.get("verified_action_plan", []),
            statutory_citations=statutory,
            precedent_citations=precedents,
            hallucination_check_passed=state.get("hallucination_check_passed", True),
            unverified_citations_stripped=state.get("unverified_citations_stripped", []),
            coverage_note=coverage_note,
        )
        return {"final_response": response, "coverage_note": coverage_note}

    @staticmethod
    def _initial_state(scenario: str) -> AgentState:
        """Build the empty AgentState that seeds a graph execution."""
        return {
            "scenario": scenario,
            "is_legal": False,
            "domain": LegalDomain.GENERAL_DISPUTE,
            "rejection_reason": None,
            "clarification_prompt": None,
            "key_entities": {},
            "statute_queries": [],
            "precedent_queries": [],
            "criminal_route_triggered": False,
            "criminal_queries": [],
            "retrieval_gate_triggered": False,
            "candidate_contexts": [],
            "retrieved_contexts": [],
            "coverage_gap": False,
            "coverage_gap_reason": None,
            "coverage_note": None,
            "applicability_mode": None,
            "applicability_rejected": [],
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

    def run_with_trace(self, scenario: str) -> Tuple[DualOutputResponse, ExecutionTrace]:
        """Execute the complete Approach B pipeline and return both the response and ExecutionTrace.

        The node list comes from the actual execution (graph.stream updates), so the trace shows which
        branch ran (full plan, coverage gap, or rejection).
        """
        t0 = time.perf_counter()
        output_state: Dict[str, Any] = dict(self._initial_state(scenario))
        nodes_visited: List[str] = []
        for node_name, update in self.stream(scenario):
            nodes_visited.append(node_name)
            output_state.update(update)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        final_response: DualOutputResponse = output_state["final_response"]

        trace = ExecutionTrace(
            pipeline_type="proposed_langgraph",
            nodes_visited=nodes_visited,
            statute_queries=output_state.get("statute_queries", []),
            precedent_queries=output_state.get("precedent_queries", []),
            criminal_route_triggered=bool(output_state.get("criminal_route_triggered", False)),
            criminal_queries=output_state.get("criminal_queries", []),
            retrieval_gate_triggered=bool(output_state.get("retrieval_gate_triggered", False)),
            candidate_contexts=output_state.get("candidate_contexts", []),
            retrieved_contexts=output_state.get("retrieved_contexts", []),
            coverage_gap=bool(output_state.get("coverage_gap", False)),
            coverage_gap_reason=output_state.get("coverage_gap_reason"),
            applicability_mode=output_state.get("applicability_mode"),
            applicability_rejected=list(output_state.get("applicability_rejected", []) or []),
        )
        log_inference_event(
            scenario=scenario,
            response=final_response,
            trace=trace,
            latency_ms=latency_ms,
            settings=self.settings,
        )
        return final_response, trace

    def run(self, scenario: str) -> DualOutputResponse:
        """Execute the complete Approach B pipeline for a given scenario.

        Args:
            scenario: Factual legal scenario narrative.

        Returns:
            DualOutputResponse with verified action plan and citations.
        """
        response, _ = self.run_with_trace(scenario)
        return response

    def stream(self, scenario: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Execute the pipeline node by node, yielding each node's state update as it completes.

        Lets callers (e.g. the UI) report live progress. The last yielded update comes from
        either the "synthesis" or "handle_rejection" node and contains "final_response".

        Args:
            scenario: Factual legal scenario narrative.

        Yields:
            Tuples of (node_name, state_update_dict).
        """
        for chunk in self.graph.stream(self._initial_state(scenario), stream_mode="updates"):
            for node_name, update in chunk.items():
                yield node_name, update or {}
