"""📊 Benchmark: dataset composition, multi-layer RAG/Agent/Judge comparative results, and SME calibration."""

from collections import Counter
from typing import Any, Dict, List, Optional

import streamlit as st

from solvemycase.evaluation.sme_review import (
    SMEAnnotation,
    compute_judge_human_agreement,
    load_sme_annotations,
    save_sme_annotation,
)
from solvemycase.ui.components.formatting import domain_label_from_value
from solvemycase.ui.state import load_benchmark_dataset, load_benchmark_results

GUARDRAIL_DOMAIN = "general_dispute"  # Legacy fallback when row lacks is_legal/coverage.
GUARDRAIL_LABEL = "Out of scope (guardrail)"
UNCOVERED_LABEL = "Outside corpus coverage"


def _category(item: Any) -> str:
    """Display label for a benchmark scenario or result row."""
    if isinstance(item, dict):
        if not item.get("is_legal", True):
            return GUARDRAIL_LABEL
        if item.get("coverage") == "uncovered":
            return UNCOVERED_LABEL
        domain = item.get("domain")
        if domain == GUARDRAIL_DOMAIN and "is_legal" not in item and "coverage" not in item:
            return GUARDRAIL_LABEL
        return domain_label_from_value(domain)
    return GUARDRAIL_LABEL if item == GUARDRAIL_DOMAIN else domain_label_from_value(item)


SUMMARY_ROWS = [
    ("Post-output hallucination rate (%)", "avg_hallucination_rate", "{:.1f}"),
    ("Irrelevant citation rate (%)", "avg_irrelevant_citation_rate", "{:.1f}"),
    ("Citation grounding accuracy (%)", "avg_grounding_accuracy", "{:.1f}"),
    ("Pre-verification strip rate (%)", "avg_pre_verification_strip_rate", "{:.1f}"),
    ("Uncovered-topic honesty rate (%)", "uncovered_honesty_rate", "{:.1f}"),
    ("False coverage-gap rate (%)", "false_gap_rate", "{:.1f}"),
    ("Retrieval Section Recall@K (%)", "avg_section_recall_at_k", "{:.1f}"),
    ("Retrieval Precision@K (%)", "avg_precision_at_k", "{:.1f}"),
    ("Retrieval Mean Reciprocal Rank (MRR)", "avg_mrr", "{:.2f}"),
    ("Context relevance (%)", "avg_context_relevance", "{:.1f}"),
    ("Expected section recall (%)", "avg_expected_section_recall", "{:.1f}"),
    ("Critical steps recall (%)", "avg_critical_steps_recall", "{:.1f}"),
    ("Procedural completeness (0–1)", "avg_completeness_score", "{:.2f}"),
    ("Forum accuracy (%)", "forum_accuracy", "{:.1f}"),
    ("Agent trajectory validity (%)", "trajectory_accuracy", "{:.1f}"),
    ("Criminal routing accuracy (%)", "criminal_routing_accuracy", "{:.1f}"),
    ("Guardrail out-of-scope TNR (%)", "guardrail_tnr", "{:.1f}"),
    ("Guardrail in-scope FPR (%)", "guardrail_fpr", "{:.1f}"),
    ("End-to-end task success (%)", "task_success_rate", "{:.1f}"),
    ("LLM-judge score (0–5)", "avg_judge_score", "{:.2f}"),
    ("Mean latency (s)", "avg_latency_seconds", "{:.2f}"),
]


def _fmt_val(side: Dict[str, Any], key: str, fmt: str) -> str:
    """Safely format a summary metric value using .get() fallback (DF-6)."""
    val = side.get(key)
    if val is None:
        return "—"
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return str(val)


def _per_scenario_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-scenario benchmark results into table rows."""
    rows = []
    for r in results:
        b = r.get("baseline", {})
        p = r.get("proposed", {})
        rows.append({
            "Scenario": r.get("scenario_id", ""),
            "Domain": _category(r),
            "A judge": b.get("judge_score", 0.0),
            "B judge": p.get("judge_score", 0.0),
            "A Recall@K": b.get("section_recall_at_k", 0.0),
            "B Recall@K": p.get("section_recall_at_k", 0.0),
            "A phases": b.get("completeness_score", 0.0),
            "B phases": p.get("completeness_score", 0.0),
            "B removed": p.get("unverified_stripped_count", 0),
            "A latency (s)": b.get("latency_seconds", 0.0),
            "B latency (s)": p.get("latency_seconds", 0.0),
        })
    return rows


def _render_sme_tab(results: Dict[str, Any]) -> None:
    """Render SME review annotation form and Judge-vs-Human calibration agreement metrics."""
    annotations = load_sme_annotations()
    per_scenario_map = {
        r.get("scenario_id"): r for r in results.get("results", []) if r.get("scenario_id")
    }

    judge_list: List[float] = []
    human_list: List[float] = []
    comparison_rows: List[Dict[str, Any]] = []
    for ann in annotations:
        scen_res = per_scenario_map.get(ann.scenario_id)
        if scen_res and ann.pipeline_type in ("baseline", "proposed"):
            j_score = float(scen_res.get(ann.pipeline_type, {}).get("judge_score", 0.0))
            judge_list.append(j_score)
            human_list.append(ann.overall_score)
            comparison_rows.append({
                "Scenario": ann.scenario_id,
                "Pipeline": "B · Proposed" if ann.pipeline_type == "proposed" else "A · Baseline",
                "Annotator": ann.annotator_id,
                "SME Overall (0–5)": round(ann.overall_score, 2),
                "LLM Judge (0–5)": round(j_score, 2),
                "|Δ|": round(abs(j_score - ann.overall_score), 2),
                "Notes": ann.notes,
            })

    agreement = compute_judge_human_agreement(judge_list, human_list)
    mcols = st.columns(4)
    mcols[0].metric("SME Annotations", agreement["sample_count"])
    mcols[1].metric("Judge–SME MAE", f"{agreement['mae']:.2f}")
    mcols[2].metric("Pearson r", f"{agreement['pearson_r']:.2f}")
    mcols[3].metric("Cohen's κ (≥3.5)", f"{agreement['cohens_kappa']:.2f}")

    if comparison_rows:
        st.dataframe(comparison_rows, hide_index=True, width="stretch")

    with st.expander("➕ Record / Update Legal Expert (SME) Annotation"):
        scenario_ids = sorted(per_scenario_map.keys()) or ["mva_001", "prop_001", "cons_001", "guardrail_001"]
        with st.form("sme_annotation_form"):
            fcols = st.columns(3)
            scen_id = fcols[0].selectbox("Scenario ID", scenario_ids)
            pipe_type = fcols[1].selectbox("Pipeline", ["proposed", "baseline"])
            annotator = fcols[2].text_input("Annotator ID", value="legal_sme_01")

            scols = st.columns(5)
            s_stat = scols[0].slider("Statutory (0–5)", 0.0, 5.0, 4.5, 0.1)
            s_proc = scols[1].slider("Procedural (0–5)", 0.0, 5.0, 4.5, 0.1)
            s_forum = scols[2].slider("Forum (0–5)", 0.0, 5.0, 5.0, 0.1)
            s_hall = scols[3].slider("Grounding (0–5)", 0.0, 5.0, 5.0, 0.1)
            s_coh = scols[4].slider("Specificity (0–5)", 0.0, 5.0, 4.5, 0.1)
            notes = st.text_input("Advocate review notes", value="")

            if st.form_submit_button("Save SME Rating"):
                overall = round((s_stat + s_proc + s_forum + s_hall + s_coh) / 5.0, 2)
                save_sme_annotation(
                    SMEAnnotation(
                        scenario_id=scen_id,
                        pipeline_type=pipe_type,
                        annotator_id=annotator,
                        statutory_accuracy=s_stat,
                        procedural_actionability=s_proc,
                        forum_appropriateness=s_forum,
                        hallucination_freedom=s_hall,
                        coherence_and_specificity=s_coh,
                        overall_score=overall,
                        notes=notes,
                    )
                )
                st.success(f"Saved SME annotation for `{scen_id}` ({pipe_type}) — overall {overall}/5.0.")


def render() -> None:
    """Render the benchmark page."""
    st.header("Benchmark")
    dataset = load_benchmark_dataset()
    results = load_benchmark_results()

    st.subheader("Evaluation dataset")
    counts = Counter(_category(s) for s in dataset)
    cols = st.columns(len(counts) + 1)
    cols[0].metric("Total scenarios", len(dataset))
    for col, (name, count) in zip(cols[1:], sorted(counts.items())):
        col.metric(name, count)

    st.subheader("Latest results")
    if not results:
        st.info("No benchmark has been run yet. Run `python -m solvemycase.evaluation.comparative_runner`.")
        return

    summary = results.get("summary", {})
    evaluated = summary.get("total_scenarios", 0)
    if evaluated < len(dataset):
        st.warning(
            f"These results cover only **{evaluated} of {len(dataset)}** scenarios, so they are not representative. "
            "Run the full benchmark (with an OpenAI key) before quoting numbers."
        )

    b, p = summary.get("baseline", {}), summary.get("proposed", {})
    tab_summary, tab_scenarios, tab_sme = st.tabs([
        "📊 Summary & RAG/Agent Metrics",
        "🔍 Per-Scenario Explorer",
        "👩‍⚖️ SME Review & Judge Calibration",
    ])

    with tab_summary:
        st.dataframe(
            [
                {
                    "Metric": label,
                    "A · Baseline": _fmt_val(b, key, fmt),
                    "B · Proposed": _fmt_val(p, key, fmt),
                }
                for label, key, fmt in SUMMARY_ROWS
            ],
            hide_index=True,
            width="stretch",
        )

        st.markdown("##### Judge score by approach")
        st.bar_chart(
            {
                "A · Baseline": [float(b.get("avg_judge_score", 0.0))],
                "B · Proposed": [float(p.get("avg_judge_score", 0.0))],
            },
            stack=False,
            horizontal=True,
        )

    with tab_scenarios:
        per_scenario = results.get("results", [])
        if per_scenario:
            rows = _per_scenario_rows(per_scenario)
            domains = sorted({row["Domain"] for row in rows})
            chosen = st.multiselect("Filter by domain", domains, default=domains)
            st.dataframe([row for row in rows if row["Domain"] in chosen], hide_index=True, width="stretch")

    with tab_sme:
        _render_sme_tab(results)
