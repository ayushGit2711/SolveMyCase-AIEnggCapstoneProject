"""Quantitative evaluation metrics for legal RAG and agentic pipelines.

Measures:
1. Post-Output Statutory & Precedent Grounding Rate (symmetric across Approach A & B) + Pre-Verification Strip Rate.
2. Retrieval IR Metrics (Section Recall@K, Precision@K, Mean Reciprocal Rank, Context Relevance).
3. Ground-Truth Alignment (Expected Section Recall, Forum Accuracy, Critical Steps Recall).
4. Agentic Control-Flow & Guardrail Metrics (Trajectory Validity, Criminal Routing Accuracy, Guardrail TNR/FPR, Task Success).
"""

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    ExecutionTrace,
    ProceduralPhase,
    RetrievedContext,
    acts_share_significant_token,
    normalize_section_id,
    section_mentioned_with_boundary,
    sections_match,
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
        ..., description="Ratio of ungrounded final citations to total final citations (0.0 is ideal)."
    )
    grounding_accuracy: float = Field(
        ..., description="Ratio of verified final citations to total final citations (1.0 is ideal)."
    )
    pre_verification_strip_rate: float = Field(
        default=0.0,
        description="Fraction of draft citations caught and stripped by VerificationNode prior to output.",
    )
    procedural_step_count: int
    procedural_phases_covered: List[str]
    phase_completeness_score: float = Field(
        ..., description="Fraction of all 6 standard procedural phases addressed (0.0 to 1.0)."
    )
    correct_forum_identified: bool = Field(
        default=True, description="Whether the designated statutory forum was recognized."
    )
    section_recall_at_k: float = Field(
        default=0.0, description="Fraction of gold expected_sections present in top-K retrieved contexts."
    )
    precision_at_k: float = Field(
        default=0.0, description="Fraction of top-K retrieved chunks matching gold expected_sections/act."
    )
    mrr: float = Field(
        default=0.0, description="Mean Reciprocal Rank of the first relevant statutory chunk in retrieved contexts."
    )
    context_relevance: float = Field(
        default=0.0, description="Mean relevance score (0.0 to 1.0) of retrieved chunks to the scenario."
    )
    expected_section_recall: float = Field(
        default=0.0, description="Fraction of benchmark expected_sections cited in the final response."
    )
    critical_steps_recall: float = Field(
        default=0.0, description="Fraction of benchmark critical_steps addressed in the action plan."
    )
    trajectory_valid: bool = Field(
        default=True, description="Whether the pipeline visited the exact expected sequence of graph nodes."
    )
    criminal_routing_correct: bool = Field(
        default=True, description="Whether criminal code deep RAG routing matched scenario expectation."
    )
    task_success: bool = Field(
        default=True, description="End-to-end task completion indicator."
    )


def is_guardrail_rejection(response: DualOutputResponse) -> bool:
    """Return True if the response represents an out-of-scope guardrail rejection."""
    if response.scenario_summary.startswith("Query rejected:"):
        return True
    if not response.action_plan and any(
        s.startswith("Clarification:") for s in response.unverified_citations_stripped
    ):
        return True
    return False


def _is_statutory_citation_grounded(
    act_name: str,
    section_number: str,
    retrieved_contexts: Optional[List[RetrievedContext]],
    store: Optional[Any] = None,
    fallback_verified_flag: bool = False,
) -> bool:
    """Check whether a single statutory citation is grounded in retrieved contexts or Qdrant store."""
    sec_norm = normalize_section_id(section_number)
    if not sec_norm:
        return False

    if retrieved_contexts is not None:
        for ctx in retrieved_contexts:
            ctx_act = ctx.act_name or ctx.title
            if not acts_share_significant_token(act_name, ctx_act):
                continue
            if sections_match(sec_norm, ctx.citation_or_section):
                return True
            ctx_blob = f"{ctx.title} {ctx.citation_or_section} {ctx.text}"
            if section_mentioned_with_boundary(sec_norm, ctx_blob):
                return True

    if store is not None and hasattr(store, "exact_section_search"):
        try:
            hits = store.exact_section_search(sec_norm, act_name)
            if hits:
                return True
        except Exception:
            pass

    if retrieved_contexts is None and store is None:
        return bool(fallback_verified_flag)

    return False


def _is_precedent_citation_grounded(
    case_title: str,
    retrieved_contexts: Optional[List[RetrievedContext]],
    fallback_verified_flag: bool = False,
) -> bool:
    """Check whether a single precedent citation is grounded in retrieved precedent contexts."""
    if not case_title or not case_title.strip():
        return False

    if retrieved_contexts is None:
        return bool(fallback_verified_flag)

    precedent_contexts = [c for c in retrieved_contexts if c.doc_type == DocumentType.PRECEDENT]
    if not precedent_contexts:
        return False

    title_words = [
        w
        for w in re.split(r"\W+", case_title.lower())
        if len(w) > 3 and w not in {"state", "union", "india", "others", "versus", "anr", "ors"}
    ]
    if not title_words:
        return False

    for p_ctx in precedent_contexts:
        ctx_text_lower = f"{p_ctx.title} {p_ctx.citation_or_section} {p_ctx.text}".lower()
        if any(w in ctx_text_lower for w in title_words):
            return True

    return False


def compute_citation_grounding_metrics(
    response: DualOutputResponse,
    retrieved_contexts: Optional[List[RetrievedContext]] = None,
    store: Optional[Any] = None,
    is_legal: bool = True,
) -> Dict[str, float]:
    """Compute post-output hallucination rate, grounding accuracy, and pre-verification strip rate.

    Both Approach A (Vanilla RAG) and Approach B (LangGraph) are scored identically on the
    citations actually delivered in the final `DualOutputResponse`. Citations stripped prior
    to output by `VerificationNode` are tracked separately in `pre_verification_strip_rate`.
    """
    real_stripped = [
        s for s in response.unverified_citations_stripped if not s.startswith("Clarification:")
    ]
    stripped_count = len(real_stripped)

    total_final = len(response.statutory_citations) + len(response.precedent_citations)
    total_proposed = total_final + stripped_count
    pre_strip_rate = round(stripped_count / total_proposed, 4) if total_proposed > 0 else 0.0

    rejected = is_guardrail_rejection(response)

    # Guardrail rejection or out-of-scope query handling (DF-2)
    if not is_legal:
        if rejected and total_final == 0:
            return {
                "hallucination_rate": 0.0,
                "grounding_accuracy": 1.0,
                "pre_verification_strip_rate": pre_strip_rate,
                "ungrounded_final_count": 0,
            }
        # Non-legal query where pipeline failed to reject and emitted citations/advice
        if total_final == 0:
            return {
                "hallucination_rate": 0.0,
                "grounding_accuracy": 0.0,
                "pre_verification_strip_rate": pre_strip_rate,
                "ungrounded_final_count": 0,
            }

    if total_final == 0:
        # DF-2: For a legal query (is_legal=True), emitting zero citations cannot earn 1.0 grounding accuracy
        return {
            "hallucination_rate": 0.0,
            "grounding_accuracy": 0.0 if is_legal else 1.0,
            "pre_verification_strip_rate": pre_strip_rate,
            "ungrounded_final_count": 0,
        }

    verified_final = 0
    for stat in response.statutory_citations:
        if _is_statutory_citation_grounded(
            act_name=stat.act_name,
            section_number=stat.section_number,
            retrieved_contexts=retrieved_contexts,
            store=store,
            fallback_verified_flag=stat.is_verified,
        ):
            verified_final += 1

    for prec in response.precedent_citations:
        if _is_precedent_citation_grounded(
            case_title=prec.case_title,
            retrieved_contexts=retrieved_contexts,
            fallback_verified_flag=prec.is_verified,
        ):
            verified_final += 1

    ungrounded_final = total_final - verified_final
    hallucination_rate = round(ungrounded_final / total_final, 4)
    grounding_accuracy = round(verified_final / total_final, 4)

    return {
        "hallucination_rate": hallucination_rate,
        "grounding_accuracy": grounding_accuracy,
        "pre_verification_strip_rate": pre_strip_rate,
        "ungrounded_final_count": ungrounded_final,
    }


def compute_procedural_completeness(response: DualOutputResponse) -> Dict[str, Any]:
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


_FORUM_SYNONYMS: Dict[str, List[str]] = {
    "mact": ["mact", "motor accident claims tribunal", "claims tribunal"],
    "consumer commission": ["consumer commission", "district commission", "dcdrc", "scdrc", "ncdrc", "consumer disputes redressal"],
    "dcdrc": ["dcdrc", "district commission", "district consumer", "consumer commission"],
    "scdrc": ["scdrc", "state commission", "consumer commission"],
    "civil court": ["civil court", "civil judge", "district court", "competent court"],
    "police station": ["police station", "fir", "sho", "magistrate"],
}


def verify_target_forum(
    response: DualOutputResponse,
    expected_keywords: List[str],
    is_legal: bool = True,
) -> bool:
    """Verify if the action plan targets the legally designated forum or tribunal.

    For non-legal queries (is_legal=False), returns True iff the pipeline properly rejected the query (DF-5).
    """
    if not is_legal:
        return is_guardrail_rejection(response)

    if not expected_keywords:
        return not is_guardrail_rejection(response) and len(response.action_plan) > 0

    all_plan_text = " ".join(
        f"{step.title} {step.description} {step.forum_or_authority}"
        for step in response.action_plan
    ).lower()

    for kw in expected_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in all_plan_text:
            return True
        for syn in _FORUM_SYNONYMS.get(kw_lower, []):
            if syn in all_plan_text:
                return True
    return False


def compute_ground_truth_alignment(
    expected_meta: Dict[str, Any],
    response: DualOutputResponse,
) -> Dict[str, Any]:
    """Compare a pipeline response against gold benchmark expectations (`expected_sections`, `expected_forum`, `critical_steps`)."""
    is_legal = bool(expected_meta.get("is_legal", True))
    rejected = is_guardrail_rejection(response)

    if not is_legal:
        score = 1.0 if rejected else 0.0
        return {
            "expected_section_recall": score,
            "expected_forum_matched": rejected,
            "critical_steps_recall": score,
        }

    if rejected:
        return {
            "expected_section_recall": 0.0,
            "expected_forum_matched": False,
            "critical_steps_recall": 0.0,
        }

    # 1. Expected statutory sections recall
    expected_sections: List[str] = expected_meta.get("expected_sections", [])
    cited_sections = [s.section_number for s in response.statutory_citations]
    plan_statutory_blob = " ".join(
        f"{step.statutory_basis or ''} {step.description}" for step in response.action_plan
    )

    matched_sections = 0
    for exp_sec in expected_sections:
        if any(sections_match(exp_sec, c_sec) for c_sec in cited_sections) or section_mentioned_with_boundary(
            exp_sec, plan_statutory_blob
        ):
            matched_sections += 1

    section_recall = round(matched_sections / len(expected_sections), 4) if expected_sections else 1.0

    # 2. Expected forum match
    forum_matched = verify_target_forum(
        response,
        expected_meta.get("expected_forum", []),
        is_legal=is_legal,
    )

    # 3. Critical steps recall
    critical_steps: List[str] = expected_meta.get("critical_steps", [])
    full_plan_blob = " ".join(
        f"{s.title} {s.description} {s.forum_or_authority} {s.statutory_basis or ''}"
        for s in response.action_plan
    ).lower()

    matched_steps = 0
    for crit in critical_steps:
        tokens = [
            t for t in re.split(r"\W+", crit.lower()) if len(t) >= 3 and t not in {"the", "and", "for", "under", "with"}
        ]
        if not tokens:
            continue
        if any(t in full_plan_blob for t in tokens):
            matched_steps += 1

    steps_recall = round(matched_steps / len(critical_steps), 4) if critical_steps else 1.0

    return {
        "expected_section_recall": section_recall,
        "expected_forum_matched": forum_matched,
        "critical_steps_recall": steps_recall,
    }


def compute_retrieval_metrics(
    retrieved_contexts: List[RetrievedContext],
    expected_meta: Dict[str, Any],
    k: int = 5,
) -> Dict[str, float]:
    """Compute Retrieval IR metrics: Section Recall@K, Precision@K, and Mean Reciprocal Rank (MRR)."""
    is_legal = bool(expected_meta.get("is_legal", True))
    if not is_legal:
        # Non-legal queries should ideally trigger 0 retrievals when blocked by guardrail
        clean_block = 1.0 if len(retrieved_contexts) == 0 else 0.0
        return {
            "section_recall_at_k": clean_block,
            "precision_at_k": clean_block,
            "mrr": clean_block,
        }

    expected_sections: List[str] = expected_meta.get("expected_sections", [])
    expected_act: str = expected_meta.get("expected_act") or ""
    top_k_chunks = retrieved_contexts[: max(1, k)]

    if not top_k_chunks:
        return {"section_recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    # Section Recall@K
    matched_expected = 0
    for exp_sec in expected_sections:
        found = False
        for ctx in top_k_chunks:
            if sections_match(exp_sec, ctx.citation_or_section) or section_mentioned_with_boundary(
                exp_sec, f"{ctx.title} {ctx.citation_or_section} {ctx.text}"
            ):
                found = True
                break
        if found:
            matched_expected += 1

    recall_at_k = round(matched_expected / len(expected_sections), 4) if expected_sections else 1.0

    # Precision@K and MRR
    relevant_chunks = 0
    first_relevant_rank: Optional[int] = None
    for idx, ctx in enumerate(top_k_chunks, start=1):
        is_rel = False
        for exp_sec in expected_sections:
            if sections_match(exp_sec, ctx.citation_or_section) or section_mentioned_with_boundary(
                exp_sec, f"{ctx.title} {ctx.citation_or_section}"
            ):
                is_rel = True
                break
        if not is_rel and expected_act and acts_share_significant_token(expected_act, ctx.act_name or ctx.title):
            is_rel = True
        if not is_rel and ctx.doc_type == DocumentType.PRECEDENT:
            is_rel = True

        if is_rel:
            relevant_chunks += 1
            if first_relevant_rank is None:
                first_relevant_rank = idx

    precision_at_k = round(relevant_chunks / len(top_k_chunks), 4)
    mrr = round(1.0 / first_relevant_rank, 4) if first_relevant_rank is not None else 0.0

    return {
        "section_recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
    }


def compute_context_relevance(
    scenario: str,
    retrieved_contexts: List[RetrievedContext],
    expected_meta: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute mean context relevance score in [0.0, 1.0] for retrieved chunks."""
    if expected_meta is not None and not expected_meta.get("is_legal", True):
        return 1.0 if len(retrieved_contexts) == 0 else 0.2

    if not retrieved_contexts:
        return 0.0

    scenario_tokens = {
        w for w in re.split(r"\W+", scenario.lower()) if len(w) > 3
    }
    expected_sections = (expected_meta or {}).get("expected_sections", [])
    expected_act = (expected_meta or {}).get("expected_act") or ""

    chunk_scores: List[float] = []
    for ctx in retrieved_contexts:
        score = 0.0
        ctx_blob = f"{ctx.title} {ctx.citation_or_section} {ctx.text}".lower()
        ctx_tokens = {w for w in re.split(r"\W+", ctx_blob) if len(w) > 3}
        if scenario_tokens:
            overlap = len(scenario_tokens & ctx_tokens) / max(1, min(len(scenario_tokens), 12))
            score += min(0.5, overlap * 0.6)
        if expected_act and acts_share_significant_token(expected_act, ctx.act_name or ctx.title):
            score += 0.25
        if any(
            sections_match(sec, ctx.citation_or_section) or section_mentioned_with_boundary(sec, ctx_blob)
            for sec in expected_sections
        ):
            score += 0.35
        elif ctx.doc_type == DocumentType.PRECEDENT:
            score += 0.25
        chunk_scores.append(min(1.0, score))

    return round(sum(chunk_scores) / len(chunk_scores), 4)


def compute_agent_trace_metrics(
    trace: ExecutionTrace,
    expected_meta: Dict[str, Any],
    response: DualOutputResponse,
) -> Dict[str, Any]:
    """Evaluate agentic control-flow trajectory, criminal routing accuracy, and guardrail TNR/FPR."""
    is_legal = bool(expected_meta.get("is_legal", True))
    rejected = is_guardrail_rejection(response)
    expects_crim = bool(expected_meta.get("expects_criminal_route", False))

    if trace.pipeline_type == "proposed_langgraph":
        expected_nodes = (
            [
                "guardrail",
                "decontextualize",
                "retrieve_and_rerank",
                "procedural_planner",
                "verification",
                "synthesis",
            ]
            if is_legal
            else ["guardrail", "handle_rejection"]
        )
        trajectory_valid = trace.nodes_visited == expected_nodes
    else:
        # Vanilla RAG has no guardrail node; valid trajectory only on in-scope queries
        trajectory_valid = is_legal and trace.nodes_visited == [
            "embed_query",
            "hybrid_search",
            "monolithic_generate",
        ]

    criminal_routing_correct = bool(trace.criminal_route_triggered) == expects_crim

    guardrail_tnr = 1.0 if (not is_legal and rejected) else (0.0 if not is_legal else 1.0)
    guardrail_fpr = 1.0 if (is_legal and rejected) else 0.0

    if not is_legal:
        task_success = rejected
    else:
        forum_ok = verify_target_forum(response, expected_meta.get("expected_forum", []), is_legal=True)
        task_success = (not rejected) and forum_ok and len(response.action_plan) >= 3

    return {
        "trajectory_valid": trajectory_valid,
        "criminal_routing_correct": criminal_routing_correct,
        "guardrail_tnr": guardrail_tnr,
        "guardrail_fpr": guardrail_fpr,
        "task_success": task_success,
    }
