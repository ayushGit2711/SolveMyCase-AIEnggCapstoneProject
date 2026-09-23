"""Comparative benchmark runner for Approach A (Vanilla RAG) vs Approach B (Proposed LangGraph).

Executes the 50+ benchmark scenarios across both pipelines, collecting latency,
citation grounding, procedural completeness, and judge evaluation scores.
Outputs structured JSON and a formatted comparative Markdown report.
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
from solvemycase.data.ingestion.schema import DualOutputResponse
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore
from solvemycase.evaluation.llm_judge import LegalLLMJudge
from solvemycase.evaluation.metrics import (
    compute_citation_grounding_metrics,
    compute_procedural_completeness,
    verify_target_forum,
)


class ComparativeRunner:
    """Executes comparative evaluation across Baseline and Proposed legal pipelines."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
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
        """Run single scenario across both pipelines and evaluate results."""
        scenario_text = item["scenario"]
        expected_meta = item

        # 1. Run Baseline (Approach A)
        t0 = time.perf_counter()
        baseline_resp = self.baseline.run(scenario_text)
        t_baseline = time.perf_counter() - t0

        # Baseline metrics
        b_grounding = compute_citation_grounding_metrics(baseline_resp)
        b_procedural = compute_procedural_completeness(baseline_resp)
        b_forum = verify_target_forum(baseline_resp, expected_meta.get("expected_forum", []))
        b_judge = self.judge.judge_response(scenario_text, expected_meta, baseline_resp)

        # 2. Run Proposed (Approach B)
        t0 = time.perf_counter()
        proposed_resp = self.proposed.run(scenario_text)
        t_proposed = time.perf_counter() - t0

        # Proposed metrics
        p_grounding = compute_citation_grounding_metrics(proposed_resp)
        p_procedural = compute_procedural_completeness(proposed_resp)
        p_forum = verify_target_forum(proposed_resp, expected_meta.get("expected_forum", []))
        p_judge = self.judge.judge_response(scenario_text, expected_meta, proposed_resp)

        return {
            "scenario_id": item["id"],
            "domain": item["domain"],
            "is_legal": item["is_legal"],
            "baseline": {
                "latency_seconds": round(t_baseline, 4),
                "hallucination_rate": b_grounding["hallucination_rate"],
                "grounding_accuracy": b_grounding["grounding_accuracy"],
                "completeness_score": b_procedural["completeness_score"],
                "step_count": b_procedural["step_count"],
                "forum_matched": b_forum,
                "judge_score": b_judge.overall_score,
                "statutory_accuracy": b_judge.statutory_accuracy,
                "hallucination_freedom": b_judge.hallucination_freedom,
            },
            "proposed": {
                "latency_seconds": round(t_proposed, 4),
                "hallucination_rate": p_grounding["hallucination_rate"],
                "grounding_accuracy": p_grounding["grounding_accuracy"],
                "completeness_score": p_procedural["completeness_score"],
                "step_count": p_procedural["step_count"],
                "forum_matched": p_forum,
                "judge_score": p_judge.overall_score,
                "statutory_accuracy": p_judge.statutory_accuracy,
                "hallucination_freedom": p_judge.hallucination_freedom,
                "unverified_stripped_count": len(proposed_resp.unverified_citations_stripped),
            },
        }

    def run_benchmark(
        self,
        limit: Optional[int] = None,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute complete benchmark suite and generate summary metrics."""
        scenarios = self.load_scenarios(limit=limit)
        print(f"\n==================================================")
        print(f"  RUNNING BENCHMARK EVALUATION ({len(scenarios)} SCENARIOS)")
        print(f"==================================================\n")

        results: List[Dict[str, Any]] = []
        for i, item in enumerate(scenarios, 1):
            print(f"[{i:02d}/{len(scenarios):02d}] Evaluating {item['id']} ({item['domain']})... ", end="", flush=True)
            res = self.evaluate_scenario(item)
            results.append(res)
            print(f"Done (Base: {res['baseline']['latency_seconds']:.2f}s | Prop: {res['proposed']['latency_seconds']:.2f}s)")

        # Aggregate Statistics
        summary = self._compute_aggregate_summary(results)

        if output_file is None:
            output_file = Path(__file__).parent / "benchmark_results.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)

        print(f"\n[Benchmark] Complete results saved to: {output_file}")
        self._print_markdown_report(summary)

        return {"summary": summary, "results": results}

    def _compute_aggregate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate mean statistics across all scenario runs."""
        b_latencies = [r["baseline"]["latency_seconds"] for r in results]
        p_latencies = [r["proposed"]["latency_seconds"] for r in results]

        b_hallucinations = [r["baseline"]["hallucination_rate"] for r in results]
        p_hallucinations = [r["proposed"]["hallucination_rate"] for r in results]

        b_groundings = [r["baseline"]["grounding_accuracy"] for r in results]
        p_groundings = [r["proposed"]["grounding_accuracy"] for r in results]

        b_completeness = [r["baseline"]["completeness_score"] for r in results]
        p_completeness = [r["proposed"]["completeness_score"] for r in results]

        b_judges = [r["baseline"]["judge_score"] for r in results]
        p_judges = [r["proposed"]["judge_score"] for r in results]

        b_forums = [1 if r["baseline"]["forum_matched"] else 0 for r in results]
        p_forums = [1 if r["proposed"]["forum_matched"] else 0 for r in results]

        return {
            "total_scenarios": len(results),
            "baseline": {
                "avg_latency_seconds": round(float(np.mean(b_latencies)), 3),
                "avg_hallucination_rate": round(float(np.mean(b_hallucinations)) * 100, 2),
                "avg_grounding_accuracy": round(float(np.mean(b_groundings)) * 100, 2),
                "avg_completeness_score": round(float(np.mean(b_completeness)), 3),
                "forum_accuracy": round(float(np.mean(b_forums)) * 100, 2),
                "avg_judge_score": round(float(np.mean(b_judges)), 2),
            },
            "proposed": {
                "avg_latency_seconds": round(float(np.mean(p_latencies)), 3),
                "avg_hallucination_rate": round(float(np.mean(p_hallucinations)) * 100, 2),
                "avg_grounding_accuracy": round(float(np.mean(p_groundings)) * 100, 2),
                "avg_completeness_score": round(float(np.mean(p_completeness)), 3),
                "forum_accuracy": round(float(np.mean(p_forums)) * 100, 2),
                "avg_judge_score": round(float(np.mean(p_judges)), 2),
            },
        }

    def _print_markdown_report(self, summary: Dict[str, Any]) -> None:
        """Display formatted comparison table."""
        b = summary["baseline"]
        p = summary["proposed"]

        print("\n" + "=" * 78)
        print("                 BENCHMARK PERFORMANCE COMPARISON TABLE                 ")
        print("=" * 78)
        print(f"Scenarios Evaluated: {summary['total_scenarios']}")
        print("-" * 78)
        print(f"{'Evaluation Metric':<35} | {'Approach A (Baseline)':<20} | {'Approach B (Proposed)'}")
        print("-" * 78)
        print(f"{'Statutory Hallucination Rate':<35} | {b['avg_hallucination_rate']:>18.1f}% | {p['avg_hallucination_rate']:>18.1f}%")
        print(f"{'Citation Grounding Accuracy':<35} | {b['avg_grounding_accuracy']:>18.1f}% | {p['avg_grounding_accuracy']:>18.1f}%")
        print(f"{'Procedural Phase Completeness':<35} | {b['avg_completeness_score']:>19.2f} | {p['avg_completeness_score']:>19.2f}")
        print(f"{'Designated Forum Accuracy':<35} | {b['forum_accuracy']:>18.1f}% | {p['forum_accuracy']:>18.1f}%")
        print(f"{'LLM Judge Overall Rating (0-5)':<35} | {b['avg_judge_score']:>19.2f} | {p['avg_judge_score']:>19.2f}")
        print(f"{'Mean End-to-End Latency':<35} | {b['avg_latency_seconds']:>17.2f}s | {p['avg_latency_seconds']:>17.2f}s")
        print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run solvemycase benchmark evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios to run.")
    args = parser.parse_args()

    runner = ComparativeRunner()
    runner.run_benchmark(limit=args.limit)
