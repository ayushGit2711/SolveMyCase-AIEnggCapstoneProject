"""Comparative benchmark runner for Approach A (Vanilla RAG) vs Approach B (Proposed LangGraph).

Executes the 52 benchmark scenarios across both pipelines using `run_with_trace()`, collecting:
- Post-output citation grounding accuracy & hallucination rate (symmetric across A & B) + pre-verification strip rate.
- Retrieval IR metrics: Section Recall@K, Precision@K, MRR, Context Relevance.
- Ground-truth alignment: Expected Section Recall, Forum Accuracy, Critical Steps Recall.
- Agentic control-flow & guardrail metrics: Trajectory Validity, Criminal Routing Accuracy, Guardrail TNR/FPR, Task Success.
- Decomposed 5-dimension LLM-as-a-Judge ratings.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore
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


class ComparativeRunner:
    """Executes comparative evaluation across Baseline and Proposed legal pipelines."""

    def __init__(self, settings: Optional[Settings] = None, judge_samples: Optional[int] = None):
        self.settings = settings or get_settings()
        self.judge_samples = judge_samples or self.settings.eval_judge_samples
        self.store = QdrantLegalStore(settings=self.settings)
        self.embedder = EmbeddingProvider(settings=self.settings)
        self.baseline = VanillaRAGBaseline(settings=self.settings, store=self.store, embedder=self.embedder)
        self.proposed = LegalAgentGraph(settings=self.settings, store=self.store, embedder=self.embedder)
        self.judge = LegalLLMJudge(settings=self.settings)

    def load_scenarios(self, dataset_path: Optional[Path] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load benchmark scenarios from JSON file."""
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "benchmark_dataset.json"

        with open(dataset_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        if limit:
            scenarios = scenarios[:limit]

        return scenarios

    def evaluate_scenario(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run single scenario across both pipelines with full execution traces and evaluate results."""
        scenario_text = item["scenario"]
        expected_meta = item
        is_legal = bool(item.get("is_legal", True))

        # 1. Run Baseline (Approach A)
        t0 = time.perf_counter()
        baseline_resp, b_trace = self.baseline.run_with_trace(scenario_text)
        t_baseline = time.perf_counter() - t0

        b_grounding = compute_citation_grounding_metrics(
            baseline_resp,
            retrieved_contexts=b_trace.retrieved_contexts,
            store=self.store,
            is_legal=is_legal,
        )
        b_procedural = compute_procedural_completeness(baseline_resp)
        b_forum = verify_target_forum(baseline_resp, expected_meta.get("expected_forum", []), is_legal=is_legal)
        b_retrieval = compute_retrieval_metrics(
            b_trace.retrieved_contexts, expected_meta, k=self.settings.rerank_top_k
        )
        b_ctx_rel = compute_context_relevance(scenario_text, b_trace.retrieved_contexts, expected_meta)
        b_align = compute_ground_truth_alignment(expected_meta, baseline_resp)
        b_agent = compute_agent_trace_metrics(b_trace, expected_meta, baseline_resp)
        b_judge = self.judge.judge_response(
            scenario_text,
            expected_meta,
            baseline_resp,
            retrieved_contexts=b_trace.retrieved_contexts,
            store=self.store,
            num_samples=self.judge_samples,
        )

        # 2. Run Proposed (Approach B)
        t0 = time.perf_counter()
        proposed_resp, p_trace = self.proposed.run_with_trace(scenario_text)
        t_proposed = time.perf_counter() - t0

        p_grounding = compute_citation_grounding_metrics(
            proposed_resp,
            retrieved_contexts=p_trace.retrieved_contexts,
            store=self.store,
            is_legal=is_legal,
        )
        p_procedural = compute_procedural_completeness(proposed_resp)
        p_forum = verify_target_forum(proposed_resp, expected_meta.get("expected_forum", []), is_legal=is_legal)
        p_retrieval = compute_retrieval_metrics(
            p_trace.retrieved_contexts, expected_meta, k=self.settings.rerank_top_k
        )
        p_ctx_rel = compute_context_relevance(scenario_text, p_trace.retrieved_contexts, expected_meta)
        p_align = compute_ground_truth_alignment(expected_meta, proposed_resp)
        p_agent = compute_agent_trace_metrics(p_trace, expected_meta, proposed_resp)
        p_judge = self.judge.judge_response(
            scenario_text,
            expected_meta,
            proposed_resp,
            retrieved_contexts=p_trace.retrieved_contexts,
            store=self.store,
            num_samples=self.judge_samples,
        )

        real_stripped = [
            s for s in proposed_resp.unverified_citations_stripped if not s.startswith("Clarification:")
        ]

        return {
            "scenario_id": item["id"],
            "domain": item["domain"],
            "is_legal": is_legal,
            "baseline": {
                "latency_seconds": round(t_baseline, 4),
                "hallucination_rate": b_grounding["hallucination_rate"],
                "grounding_accuracy": b_grounding["grounding_accuracy"],
                "pre_verification_strip_rate": b_grounding["pre_verification_strip_rate"],
                "completeness_score": b_procedural["completeness_score"],
                "step_count": b_procedural["step_count"],
                "forum_matched": b_forum,
                "section_recall_at_k": b_retrieval["section_recall_at_k"],
                "precision_at_k": b_retrieval["precision_at_k"],
                "mrr": b_retrieval["mrr"],
                "context_relevance": b_ctx_rel,
                "expected_section_recall": b_align["expected_section_recall"],
                "critical_steps_recall": b_align["critical_steps_recall"],
                "trajectory_valid": b_agent["trajectory_valid"],
                "criminal_routing_correct": b_agent["criminal_routing_correct"],
                "guardrail_tnr": b_agent["guardrail_tnr"],
                "guardrail_fpr": b_agent["guardrail_fpr"],
                "task_success": b_agent["task_success"],
                "judge_score": b_judge.overall_score,
                "statutory_accuracy": b_judge.statutory_accuracy,
                "procedural_actionability": b_judge.procedural_actionability,
                "forum_appropriateness": b_judge.forum_appropriateness,
                "hallucination_freedom": b_judge.hallucination_freedom,
                "coherence_and_specificity": b_judge.coherence_and_specificity,
                "flagged_for_human_review": b_judge.flagged_for_human_review,
            },
            "proposed": {
                "latency_seconds": round(t_proposed, 4),
                "hallucination_rate": p_grounding["hallucination_rate"],
                "grounding_accuracy": p_grounding["grounding_accuracy"],
                "pre_verification_strip_rate": p_grounding["pre_verification_strip_rate"],
                "completeness_score": p_procedural["completeness_score"],
                "step_count": p_procedural["step_count"],
                "forum_matched": p_forum,
                "section_recall_at_k": p_retrieval["section_recall_at_k"],
                "precision_at_k": p_retrieval["precision_at_k"],
                "mrr": p_retrieval["mrr"],
                "context_relevance": p_ctx_rel,
                "expected_section_recall": p_align["expected_section_recall"],
                "critical_steps_recall": p_align["critical_steps_recall"],
                "trajectory_valid": p_agent["trajectory_valid"],
                "criminal_routing_correct": p_agent["criminal_routing_correct"],
                "guardrail_tnr": p_agent["guardrail_tnr"],
                "guardrail_fpr": p_agent["guardrail_fpr"],
                "task_success": p_agent["task_success"],
                "judge_score": p_judge.overall_score,
                "statutory_accuracy": p_judge.statutory_accuracy,
                "procedural_actionability": p_judge.procedural_actionability,
                "forum_appropriateness": p_judge.forum_appropriateness,
                "hallucination_freedom": p_judge.hallucination_freedom,
                "coherence_and_specificity": p_judge.coherence_and_specificity,
                "flagged_for_human_review": p_judge.flagged_for_human_review,
                "unverified_stripped_count": len(real_stripped),
            },
        }

    def run_benchmark(
        self,
        limit: Optional[int] = None,
        output_file: Optional[Path] = None,
        workers: int = 4,
    ) -> Dict[str, Any]:
        """Execute complete benchmark suite and generate summary metrics."""
        from concurrent.futures import ThreadPoolExecutor

        scenarios = self.load_scenarios(limit=limit)
        print("\n==================================================")
        print(f"  RUNNING BENCHMARK EVALUATION ({len(scenarios)} SCENARIOS, {workers} WORKERS)")
        print("==================================================\n")

        # Warm up CrossEncoder once before spawning threads
        self.proposed.reranker._get_encoder()

        if workers <= 1:
            results: List[Dict[str, Any]] = []
            for i, item in enumerate(scenarios, 1):
                print(f"[{i:02d}/{len(scenarios):02d}] Evaluating {item['id']} ({item['domain']})... ", end="", flush=True)
                res = self.evaluate_scenario(item)
                results.append(res)
                print(f"Done (Base: {res['baseline']['latency_seconds']:.2f}s | Prop: {res['proposed']['latency_seconds']:.2f}s)")
        else:
            ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(scenarios)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(self.evaluate_scenario, item): (idx, item)
                    for idx, item in enumerate(scenarios)
                }
                completed_count = 0
                for fut, (idx, item) in future_map.items():
                    res = fut.result()
                    ordered_results[idx] = res
                    completed_count += 1
                    print(
                        f"[{completed_count:02d}/{len(scenarios):02d}] {item['id']} ({item['domain']}) -> "
                        f"Base Judge={res['baseline']['judge_score']:.2f} | Prop Judge={res['proposed']['judge_score']:.2f}"
                    )
            results = [r for r in ordered_results if r is not None]

        summary = self._compute_aggregate_summary(results)

        if output_file is None:
            output_file = Path(__file__).parent / "benchmark_results.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

        print(f"\n[Benchmark] Complete results saved to: {output_file}")
        self._print_markdown_report(summary)

        return {"summary": summary, "results": results}

    def _compute_side_summary(self, results: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        """Compute aggregate statistics for either 'baseline' or 'proposed'."""
        legal_rows = [r[key] for r in results if r.get("is_legal", True)]
        non_legal_rows = [r[key] for r in results if not r.get("is_legal", True)]
        all_rows = [r[key] for r in results]

        def _mean(vals: List[float], scale: float = 1.0, digits: int = 2) -> float:
            if not vals:
                return 0.0
            return round(float(np.mean(vals)) * scale, digits)

        return {
            "avg_latency_seconds": _mean([r["latency_seconds"] for r in all_rows], 1.0, 3),
            "avg_hallucination_rate": _mean([r["hallucination_rate"] for r in all_rows], 100.0, 2),
            "avg_grounding_accuracy": _mean([r["grounding_accuracy"] for r in all_rows], 100.0, 2),
            "avg_pre_verification_strip_rate": _mean([r.get("pre_verification_strip_rate", 0.0) for r in all_rows], 100.0, 2),
            "avg_completeness_score": _mean([r["completeness_score"] for r in legal_rows or all_rows], 1.0, 3),
            "forum_accuracy": _mean([1.0 if r["forum_matched"] else 0.0 for r in all_rows], 100.0, 2),
            "avg_section_recall_at_k": _mean([r.get("section_recall_at_k", 0.0) for r in legal_rows or all_rows], 100.0, 2),
            "avg_precision_at_k": _mean([r.get("precision_at_k", 0.0) for r in legal_rows or all_rows], 100.0, 2),
            "avg_mrr": _mean([r.get("mrr", 0.0) for r in legal_rows or all_rows], 1.0, 3),
            "avg_context_relevance": _mean([r.get("context_relevance", 0.0) for r in legal_rows or all_rows], 100.0, 2),
            "avg_expected_section_recall": _mean([r.get("expected_section_recall", 0.0) for r in all_rows], 100.0, 2),
            "avg_critical_steps_recall": _mean([r.get("critical_steps_recall", 0.0) for r in all_rows], 100.0, 2),
            "trajectory_accuracy": _mean([1.0 if r.get("trajectory_valid", False) else 0.0 for r in all_rows], 100.0, 2),
            "criminal_routing_accuracy": _mean([1.0 if r.get("criminal_routing_correct", False) else 0.0 for r in all_rows], 100.0, 2),
            "guardrail_tnr": _mean([r.get("guardrail_tnr", 0.0) for r in non_legal_rows], 100.0, 2) if non_legal_rows else 100.0,
            "guardrail_fpr": _mean([r.get("guardrail_fpr", 0.0) for r in legal_rows], 100.0, 2) if legal_rows else 0.0,
            "task_success_rate": _mean([1.0 if r.get("task_success", False) else 0.0 for r in all_rows], 100.0, 2),
            "avg_judge_score": _mean([r["judge_score"] for r in all_rows], 1.0, 2),
            "avg_statutory_accuracy": _mean([r.get("statutory_accuracy", 0.0) for r in all_rows], 1.0, 2),
            "avg_procedural_actionability": _mean([r.get("procedural_actionability", 0.0) for r in all_rows], 1.0, 2),
            "avg_forum_appropriateness": _mean([r.get("forum_appropriateness", 0.0) for r in all_rows], 1.0, 2),
            "avg_hallucination_freedom": _mean([r.get("hallucination_freedom", 0.0) for r in all_rows], 1.0, 2),
            "avg_coherence_and_specificity": _mean([r.get("coherence_and_specificity", 0.0) for r in all_rows], 1.0, 2),
        }

    def _compute_aggregate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mean statistics across all scenario runs."""
        return {
            "total_scenarios": len(results),
            "baseline": self._compute_side_summary(results, "baseline"),
            "proposed": self._compute_side_summary(results, "proposed"),
        }

    def _print_markdown_report(self, summary: Dict[str, Any]) -> None:
        """Display formatted comparison table."""
        b = summary["baseline"]
        p = summary["proposed"]

        print("\n" + "=" * 82)
        print("                   BENCHMARK PERFORMANCE COMPARISON TABLE                   ")
        print("=" * 82)
        print(f"Scenarios Evaluated: {summary['total_scenarios']}")
        print("-" * 82)
        print(f"{'Evaluation Metric':<38} | {'Approach A (Baseline)':<20} | {'Approach B (Proposed)'}")
        print("-" * 82)
        print(f"{'Post-Output Hallucination Rate':<38} | {b['avg_hallucination_rate']:>18.1f}% | {p['avg_hallucination_rate']:>18.1f}%")
        print(f"{'Citation Grounding Accuracy':<38} | {b['avg_grounding_accuracy']:>18.1f}% | {p['avg_grounding_accuracy']:>18.1f}%")
        print(f"{'Pre-Verification Strip Rate':<38} | {b['avg_pre_verification_strip_rate']:>18.1f}% | {p['avg_pre_verification_strip_rate']:>18.1f}%")
        print(f"{'Retrieval Section Recall@K':<38} | {b['avg_section_recall_at_k']:>18.1f}% | {p['avg_section_recall_at_k']:>18.1f}%")
        print(f"{'Retrieval Mean Reciprocal Rank (MRR)':<38} | {b['avg_mrr']:>19.2f} | {p['avg_mrr']:>19.2f}")
        print(f"{'Expected Section Recall (Final)':<38} | {b['avg_expected_section_recall']:>18.1f}% | {p['avg_expected_section_recall']:>18.1f}%")
        print(f"{'Procedural Phase Completeness':<38} | {b['avg_completeness_score']:>19.2f} | {p['avg_completeness_score']:>19.2f}")
        print(f"{'Designated Forum Accuracy':<38} | {b['forum_accuracy']:>18.1f}% | {p['forum_accuracy']:>18.1f}%")
        print(f"{'Agent Trajectory Validity':<38} | {b['trajectory_accuracy']:>18.1f}% | {p['trajectory_accuracy']:>18.1f}%")
        print(f"{'Guardrail Out-of-Scope TNR':<38} | {b['guardrail_tnr']:>18.1f}% | {p['guardrail_tnr']:>18.1f}%")
        print(f"{'End-to-End Task Success Rate':<38} | {b['task_success_rate']:>18.1f}% | {p['task_success_rate']:>18.1f}%")
        print(f"{'LLM Judge Overall Rating (0-5)':<38} | {b['avg_judge_score']:>19.2f} | {p['avg_judge_score']:>19.2f}")
        print(f"{'Mean End-to-End Latency':<38} | {b['avg_latency_seconds']:>17.2f}s | {p['avg_latency_seconds']:>17.2f}s")
        print("=" * 82 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run solvemycase benchmark evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios to run.")
    parser.add_argument("--judge-samples", type=int, default=None, help="Number of LLM judge passes per scenario.")
    args = parser.parse_args()

    runner = ComparativeRunner(judge_samples=args.judge_samples)
    runner.run_benchmark(limit=args.limit)
