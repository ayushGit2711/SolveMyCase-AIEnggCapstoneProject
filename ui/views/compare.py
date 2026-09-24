"""⚖️ Compare A vs B: side-by-side research view for the capstone evaluation."""

import logging
import time
from typing import Any, Dict, List

import streamlit as st

from solvemycase.data.ingestion.schema import DualOutputResponse
from solvemycase.evaluation.metrics import compute_procedural_completeness
from solvemycase.ui.components.action_plan import render_action_plan
from solvemycase.ui.components.citations import render_citations
from solvemycase.ui.components.formatting import (
    PHASE_LABELS,
    domain_label,
    domain_label_from_value,
    is_guardrail_rejection,
    removed_references,
    verified_citation_count,
)
from solvemycase.ui.components.progress import run_with_progress
from solvemycase.ui.components.result import render_rejection, render_scenario_echo, render_verification_details
from solvemycase.ui.components.scenario_input import scenario_input, set_scenario_text
from solvemycase.ui.state import get_engines, load_benchmark_dataset

logger = logging.getLogger(__name__)

KEY_PREFIX = "compare"
RESULT_KEY = "compare_result"
BENCHMARK_PICK_KEY = "compare_benchmark_pick"
BENCHMARK_LABEL_CHARS = 70


def _benchmark_label(case: Dict[str, Any]) -> str:
    """Short selectbox label: id, category (or 'Out of scope'), and the start of the scenario."""
    category = domain_label_from_value(case.get("domain")) if case.get("is_legal", True) else "Out of scope"
    text = " ".join(str(case.get("scenario", "")).split())
    if len(text) > BENCHMARK_LABEL_CHARS:
        text = text[:BENCHMARK_LABEL_CHARS].rstrip() + "…"
    return f"{case.get('id', '?')} · {category} · {text}"


def _apply_benchmark_pick(cases_by_label: Dict[str, str]) -> None:
    """on_change callback: load the chosen benchmark scenario into the text area."""
    label = st.session_state.get(BENCHMARK_PICK_KEY)
    if label in cases_by_label:
        set_scenario_text(KEY_PREFIX, cases_by_label[label])


def _benchmark_picker(cases: List[Dict[str, Any]]) -> None:
    """Selectbox over the annotated benchmark (incl. guardrail cases) that fills the scenario box."""
    if not cases:
        return
    cases_by_label = {_benchmark_label(c): str(c.get("scenario", "")) for c in cases}
    st.selectbox(
        "Or load a benchmark scenario",
        options=list(cases_by_label),
        index=None,
        placeholder=f"Choose one of {len(cases)} annotated scenarios…",
        key=BENCHMARK_PICK_KEY,
        on_change=_apply_benchmark_pick,
        args=(cases_by_label,),
    )


def _summarize(response: DualOutputResponse, latency: float) -> Dict[str, Any]:
    """Metrics shown for one approach."""
    procedural = compute_procedural_completeness(response)
    return {
        "response": response,
        "latency": latency,
        "phases": len(procedural["phases_covered"]),
        "citations": verified_citation_count(response),
        "removed": len(removed_references(response)),
    }


def _run_both(scenario: str) -> Dict[str, Any]:
    """Run A then B sequentially (fair latency; the embedded Qdrant client is shared)."""
    engines = get_engines()
    with st.spinner("Running Approach A (baseline)..."):
        t0 = time.perf_counter()
        baseline = engines.baseline.run(scenario)
        t_baseline = time.perf_counter() - t0
    t0 = time.perf_counter()
    proposed = run_with_progress(engines.proposed, scenario, label="Running Approach B (proposed)...")
    t_proposed = time.perf_counter() - t0
    return {"scenario": scenario, "baseline": _summarize(baseline, t_baseline), "proposed": _summarize(proposed, t_proposed)}


def _render_metrics(base: Dict[str, Any], prop: Dict[str, Any]) -> None:
    """Headline metric row: B's value with the B − A delta."""
    total_phases = len(PHASE_LABELS)
    st.caption("Values show Approach B; the delta is B minus A.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Citations shown", prop["citations"], f"{prop['citations'] - base['citations']:+d}",
        help="Laws and judgments included in the final answer. For B, every one passed verification.",
    )
    m2.metric(
        "Citations removed", prop["removed"], f"{prop['removed'] - base['removed']:+d}", delta_color="inverse",
        help="References B's verifier couldn't match to retrieved official text. A has no verifier, so it never removes any.",
    )
    m3.metric(
        "Phases covered", f"{prop['phases']}/{total_phases}", f"{prop['phases'] - base['phases']:+d}",
        help=f"How many of the {total_phases} procedural phases (immediate action → appeal) the plan covers.",
    )
    m4.metric(
        "Latency", f"{prop['latency']:.1f}s", f"{prop['latency'] - base['latency']:+.1f}s", delta_color="inverse",
        help="End-to-end time. B does more work (guardrail, multi-query retrieval, reranking, verification).",
    )


def _render_column(title: str, caption: str, summary: Dict[str, Any], key_prefix: str, show_checks: bool) -> None:
    """One approach's result. Uses bordered containers (not expanders) because citations use expanders."""
    response: DualOutputResponse = summary["response"]
    st.subheader(title)
    st.caption(caption)
    if is_guardrail_rejection(response):
        render_rejection(response)
        return
    st.caption(f"{domain_label(response.domain)} · {len(response.action_plan)} steps · {summary['latency']:.1f}s")
    with st.container(border=True):
        st.markdown("**📋 Action plan**")
        render_action_plan(response.action_plan, compact=True)
    with st.container(border=True):
        st.markdown("**📜 Citations**")
        render_citations(response.statutory_citations, response.precedent_citations, key_prefix=f"{key_prefix}_cit")
    if show_checks:
        with st.container(border=True):
            st.markdown("**🔍 Verification details**")
            render_verification_details(response)


def render() -> None:
    """Render the comparison page."""
    st.title("Compare Approach A vs Approach B")
    st.markdown(
        "**A: Baseline RAG**: single dense retrieval and one prompt, no verification.  \n"
        "**B: Proposed agent**: guardrail, query expansion, hybrid retrieval with reranking, planner, and citation verification."
    )

    _benchmark_picker(load_benchmark_dataset())
    scenario = scenario_input(key_prefix=KEY_PREFIX, submit_label="Run both approaches")
    if scenario:
        try:
            st.session_state[RESULT_KEY] = _run_both(scenario)
        except Exception:  # Keep the page usable; details go to the server log only.
            logger.exception("Comparison run failed")
            st.session_state.pop(RESULT_KEY, None)
            st.error("Sorry, the comparison failed. Please try again in a moment.")

    saved = st.session_state.get(RESULT_KEY)
    if not saved:
        return

    st.divider()
    render_scenario_echo(saved["scenario"])
    _render_metrics(saved["baseline"], saved["proposed"])
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        _render_column("A · Baseline RAG", "Single-pass generation, no verification", saved["baseline"], "cmp_a", show_checks=False)
    with col_b:
        _render_column("B · Proposed agent", "Verified citations only", saved["proposed"], "cmp_b", show_checks=True)
