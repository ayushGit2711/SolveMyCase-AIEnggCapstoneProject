"""📊 Benchmark: dataset composition and the latest comparative results, reported honestly."""

from collections import Counter
from typing import Any, Dict, List, Optional

import streamlit as st

from solvemycase.ui.components.formatting import domain_label_from_value
from solvemycase.ui.state import load_benchmark_dataset, load_benchmark_results

GUARDRAIL_DOMAIN = "general_dispute"  # Benchmark cases in this domain are out-of-scope guardrail tests.
GUARDRAIL_LABEL = "Out of scope (guardrail)"


def _category(domain: Optional[str]) -> str:
    """Display label for a benchmark domain value."""
    return GUARDRAIL_LABEL if domain == GUARDRAIL_DOMAIN else domain_label_from_value(domain)


SUMMARY_ROWS = [
    ("Hallucination rate (%)", "avg_hallucination_rate", "{:.1f}"),
    ("Citation grounding accuracy (%)", "avg_grounding_accuracy", "{:.1f}"),
    ("Procedural completeness (0–1)", "avg_completeness_score", "{:.2f}"),
    ("Forum accuracy (%)", "forum_accuracy", "{:.1f}"),
    ("LLM-judge score (0–5)", "avg_judge_score", "{:.2f}"),
    ("Mean latency (s)", "avg_latency_seconds", "{:.2f}"),
]


def _per_scenario_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-scenario benchmark results into table rows."""
    rows = []
    for r in results:
        rows.append({
            "Scenario": r["scenario_id"],
            "Domain": _category(r.get("domain")),
            "A judge": r["baseline"]["judge_score"],
            "B judge": r["proposed"]["judge_score"],
            "A phases": r["baseline"]["completeness_score"],
            "B phases": r["proposed"]["completeness_score"],
            "B removed": r["proposed"].get("unverified_stripped_count", 0),
            "A latency (s)": r["baseline"]["latency_seconds"],
            "B latency (s)": r["proposed"]["latency_seconds"],
        })
    return rows


def render() -> None:
    """Render the benchmark page."""
    st.title("Benchmark")
    dataset = load_benchmark_dataset()
    results = load_benchmark_results()

    st.subheader("Evaluation dataset")
    counts = Counter(_category(s.get("domain")) for s in dataset)
    cols = st.columns(len(counts) + 1)
    cols[0].metric("Total scenarios", len(dataset))
    for col, (name, count) in zip(cols[1:], sorted(counts.items())):
        col.metric(name, count)

    st.subheader("Latest results")
    if not results:
        st.info("No benchmark has been run yet. Run `python -m solvemycase.evaluation.comparative_runner`.")
        return

    summary = results["summary"]
    evaluated = summary.get("total_scenarios", 0)
    if evaluated < len(dataset):
        st.warning(
            f"These results cover only **{evaluated} of {len(dataset)}** scenarios, so they are not representative. "
            "Run the full benchmark (with an OpenAI key) before quoting numbers."
        )

    b, p = summary["baseline"], summary["proposed"]
    st.dataframe(
        [{"Metric": label, "A · Baseline": fmt.format(b[key]), "B · Proposed": fmt.format(p[key])} for label, key, fmt in SUMMARY_ROWS],
        hide_index=True,
        width="stretch",
    )

    st.markdown("##### Judge score by approach")
    st.bar_chart({"A · Baseline": [b["avg_judge_score"]], "B · Proposed": [p["avg_judge_score"]]}, stack=False, horizontal=True)

    per_scenario = results.get("results", [])
    if per_scenario:
        st.markdown("##### Per-scenario results")
        rows = _per_scenario_rows(per_scenario)
        domains = sorted({row["Domain"] for row in rows})
        chosen = st.multiselect("Filter by domain", domains, default=domains)
        st.dataframe([row for row in rows if row["Domain"] in chosen], hide_index=True, width="stretch")
