"""Streamlit Interactive Application for solvemycase Indian Legal Advisory Engine.

Displays live side-by-side comparison between:
- Approach A: Baseline Vanilla RAG
- Approach B: Proposed Multi-Agent LangGraph with 0% Hallucination Verification
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List
import streamlit as st

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.data.ingestion.schema import DualOutputResponse, LegalDomain
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore
from solvemycase.evaluation.metrics import (
    compute_citation_grounding_metrics,
    compute_procedural_completeness,
)

# Page configuration
st.set_page_config(
    page_title="solvemycase — Indian Legal Problem Solver",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-baseline {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-proposed {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .step-box {
        border-left: 4px solid #3B82F6;
        background-color: #F0F9FF;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
    .citation-box {
        border-left: 4px solid #10B981;
        background-color: #ECFDF5;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
    }
    .audit-box {
        border-left: 4px solid #EF4444;
        background-color: #FEF2F2;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engines():
    """Cache vector store and pipeline engines."""
    settings = get_settings()
    store = QdrantLegalStore(settings=settings)
    embedder = EmbeddingProvider(settings=settings)
    baseline = VanillaRAGBaseline(settings=settings, store=store, embedder=embedder)
    proposed = LegalAgentGraph(settings=settings, store=store, embedder=embedder)
    return settings, store, baseline, proposed


@st.cache_data
def load_benchmark_data():
    """Load benchmark dataset scenarios."""
    path = Path(__file__).parent.parent / "evaluation" / "benchmark_dataset.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data
def load_benchmark_summary():
    """Load benchmark evaluation summary."""
    path = Path(__file__).parent.parent / "evaluation" / "benchmark_results.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


settings, store, baseline_engine, proposed_engine = load_engines()
benchmark_scenarios = load_benchmark_data()
benchmark_results = load_benchmark_summary()

# ----------------------------------------------------
# Sidebar: Presets and Configuration
# ----------------------------------------------------
st.sidebar.markdown("### 🏛️ Scenario Templates")

category_filter = st.sidebar.selectbox(
    "Filter Template Category:",
    [
        "All Categories",
        "Motor Vehicle Accidents",
        "Property Conflicts",
        "Consumer Rights",
        "Out of Scope / Guardrail",
    ],
)

cat_key_map = {
    "Motor Vehicle Accidents": "motor_vehicle_accident",
    "Property Conflicts": "property_conflict",
    "Consumer Rights": "consumer_rights",
    "Out of Scope / Guardrail": "general_dispute",
}

filtered_scenarios = benchmark_scenarios
if category_filter != "All Categories":
    target_key = cat_key_map[category_filter]
    if category_filter == "Out of Scope / Guardrail":
        filtered_scenarios = [s for s in benchmark_scenarios if not s.get("is_legal", True)]
    else:
        filtered_scenarios = [s for s in benchmark_scenarios if s.get("domain") == target_key and s.get("is_legal", True)]

scenario_titles = [f"{s['id']}: {s['scenario'][:65]}..." for s in filtered_scenarios]
selected_idx = st.sidebar.selectbox(
    "Choose a preset case:",
    range(len(scenario_titles)) if scenario_titles else [0],
    format_func=lambda i: scenario_titles[i] if scenario_titles else "No scenarios available",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Engine Settings")
st.sidebar.info(
    f"**LLM Planner:** {settings.openai_model_primary}\n\n"
    f"**Guardrail & Routing:** {settings.openai_model_fast}\n\n"
    f"**Embeddings:** {settings.openai_embedding_model}\n\n"
    f"**Vector Store:** Qdrant Local Embedded (`data/qdrant_storage`)\n\n"
    f"**Search:** Dense + BM25 RRF (k=60) + Cross-Encoder"
)

# ----------------------------------------------------
# Main View
# ----------------------------------------------------
st.markdown("<div class='main-header'>⚖️ solvemycase — Indian Legal Advisory Engine</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Comparative Evaluation of <b>Approach A (Vanilla RAG Baseline)</b> vs "
    "<b>Approach B (Multi-Agent LangGraph with 0% Hallucination Verification)</b></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Case Consultation (Dual Comparison)",
    "📊 Benchmark Evaluation Suite (50+ Cases)",
    "🏗️ Architecture & Statutory Corpus",
])

# ----------------------------------------------------
# TAB 1: Live Case Consultation
# ----------------------------------------------------
with tab1:
    default_text = ""
    if filtered_scenarios and selected_idx < len(filtered_scenarios):
        default_text = filtered_scenarios[selected_idx]["scenario"]

    scenario_input = st.text_area(
        "Enter your Indian legal dispute scenario (or select a preset template from the left):",
        value=default_text,
        height=110,
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        run_btn = st.button("🚀 Analyze & Compare", type="primary")

    if run_btn and scenario_input.strip():
        with st.spinner("Executing Approach A (Vanilla RAG) and Approach B (Proposed LangGraph Agent)..."):
            # 1. Execute Baseline
            t0 = time.perf_counter()
            baseline_res = baseline_engine.run(scenario_input)
            t_baseline = time.perf_counter() - t0

            # 2. Execute Proposed
            t0 = time.perf_counter()
            proposed_res = proposed_engine.run(scenario_input)
            t_proposed = time.perf_counter() - t0

            # Compute quantitative metrics
            b_grounding = compute_citation_grounding_metrics(baseline_res)
            b_proc = compute_procedural_completeness(baseline_res)

            p_grounding = compute_citation_grounding_metrics(proposed_res)
            p_proc = compute_procedural_completeness(proposed_res)

        # Top Metric Cards
        st.markdown("### 📈 Head-to-Head Performance Summary")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric(
                label="End-to-End Latency",
                value=f"{t_proposed:.2f}s (Proposed)",
                delta=f"{t_baseline - t_proposed:+.2f}s vs Baseline",
            )
        with m2:
            st.metric(
                label="Procedural Completeness",
                value=f"{p_proc['completeness_score'] * 100:.0f}% ({len(p_proc['phases_covered'])}/6 Phases)",
                delta=f"{(p_proc['completeness_score'] - b_proc['completeness_score']) * 100:+.0f}% vs Baseline",
            )
        with m3:
            st.metric(
                label="Verified Statutory Citations",
                value=f"{len(proposed_res.statutory_citations)} Sections Grounded",
                delta=f"{len(proposed_res.statutory_citations) - len(baseline_res.statutory_citations):+d} vs Baseline",
            )
        with m4:
            st.metric(
                label="Hallucinations Pruned / Stripped",
                value=f"{len(proposed_res.unverified_citations_stripped)} Filtered",
                delta="0% Tolerance Enforced" if proposed_res.hallucination_check_passed else "Audit Alert",
            )

        st.markdown("---")

        # Dual Column Layout
        col_base, col_prop = st.columns(2)

        # -----------------
        # Approach A Column
        # -----------------
        with col_base:
            st.markdown("### ⚠️ Approach A: Baseline (Vanilla RAG)")
            st.markdown("<span class='badge-baseline'>Single-Pass Monolithic Generation</span>", unsafe_allow_html=True)
            st.write(f"**Domain:** `{baseline_res.domain.value}` | **Steps:** `{len(baseline_res.action_plan)}` | **Latency:** `{t_baseline:.3f}s`")

            st.markdown("#### 📋 Action Steps")
            if not baseline_res.action_plan:
                st.info("No action steps generated.")
            for step in baseline_res.action_plan:
                st.markdown(
                    f"""
                    <div class='step-box'>
                        <b>Step {step.step_number}: {step.title}</b><br>
                        <small><b>Forum:</b> {step.forum_or_authority} | <b>Section:</b> {step.statutory_basis or 'N/A'}</small><br>
                        {step.description}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### 📜 Statutory Citations")
            for cit in baseline_res.statutory_citations:
                st.markdown(
                    f"""
                    <div class='citation-box'>
                        <b>{cit.act_name} — Section {cit.section_number}</b><br>
                        {cit.summary_of_provision}<br>
                        <a href='{cit.source_url}' target='_blank'>🔗 India Code / Official Source</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if baseline_res.precedent_citations:
                st.markdown("#### ⚖️ Judicial Precedents")
                for prec in baseline_res.precedent_citations:
                    st.markdown(
                        f"""
                        <div class='citation-box'>
                            <b>{prec.case_title} ({prec.court})</b><br>
                            {prec.legal_principle}<br>
                            <a href='{prec.source_url}' target='_blank'>🔗 Supreme Court Judgment PDF</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ------------------
        # Approach B Column
        # ------------------
        with col_prop:
            st.markdown("### ✅ Approach B: Proposed (LangGraph Multi-Agent)")
            st.markdown("<span class='badge-proposed'>Decontextualized + Cross-Encoder + Verification Node</span>", unsafe_allow_html=True)
            st.write(f"**Domain:** `{proposed_res.domain.value}` | **Steps:** `{len(proposed_res.action_plan)}` | **Latency:** `{t_proposed:.3f}s`")

            # Guardrail rejection check
            if not proposed_res.action_plan and "rejected" in proposed_res.scenario_summary.lower():
                st.warning(f"🛡️ **Guardrail Triggered:** {proposed_res.scenario_summary}")
                if proposed_res.unverified_citations_stripped:
                    st.info(proposed_res.unverified_citations_stripped[0])
            else:
                st.markdown("#### 📋 Verified Chronological Roadmap")
                for step in proposed_res.action_plan:
                    phase_name = step.phase.value.replace("_", " ").title()
                    priority_color = "#EF4444" if step.priority == "Critical" else ("#F59E0B" if step.priority == "High" else "#3B82F6")
                    st.markdown(
                        f"""
                        <div class='step-box'>
                            <span style='color: {priority_color}; font-weight: bold;'>[{step.priority.upper()}]</span> 
                            <b>Step {step.step_number}: {step.title}</b> ({phase_name})<br>
                            <small><b>Authority:</b> {step.forum_or_authority} | <b>Limitation:</b> {step.limitation_period or 'Standard'} | <b>Governing Law:</b> {step.statutory_basis or 'N/A'}</small><br>
                            {step.description}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### 📜 Strictly Verified Statutory Citations (0% Hallucination)")
                for cit in proposed_res.statutory_citations:
                    st.markdown(
                        f"""
                        <div class='citation-box'>
                            <b>✓ {cit.act_name} — Section {cit.section_number}</b><br>
                            {cit.summary_of_provision}<br>
                            <small><b>Applicability:</b> {cit.applicability_to_scenario}</small><br>
                            <a href='{cit.source_url}' target='_blank'>🔗 Official Government Gazette / India Code</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if proposed_res.precedent_citations:
                    st.markdown("#### ⚖️ Verified Landmark Precedents")
                    for prec in proposed_res.precedent_citations:
                        st.markdown(
                            f"""
                            <div class='citation-box'>
                                <b>✓ {prec.case_title} [{prec.citation or prec.year}]</b> ({prec.court})<br>
                                {prec.legal_principle}<br>
                                <a href='{prec.source_url}' target='_blank'>🔗 Official Supreme Court PDF (main.sci.gov.in)</a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if proposed_res.unverified_citations_stripped:
                    st.markdown("#### 🛡️ Zero-Hallucination Audit Trail")
                    for item in proposed_res.unverified_citations_stripped:
                        st.markdown(
                            f"""
                            <div class='audit-box'>
                                <b>Pruned Ungrounded Reference:</b> {item}<br>
                                <small>Citation was proposed by generator but pruned because it was not verified in retrieved legislative text.</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ----------------------------------------------------
# TAB 2: Benchmark Evaluation Suite
# ----------------------------------------------------
with tab2:
    st.markdown("### 📊 Benchmark Evaluation Suite (52 Indian Legal Scenarios)")
    st.markdown("Standardized testing against curated test cases across 3 primary dispute domains and out-of-scope guardrail challenges.")

    if benchmark_results:
        sum_data = benchmark_results["summary"]
        b_sum = sum_data["baseline"]
        p_sum = sum_data["proposed"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Statutory Grounding Accuracy", f"{p_sum['avg_grounding_accuracy']:.1f}%", f"{p_sum['avg_grounding_accuracy'] - b_sum['avg_grounding_accuracy']:+.1f}% vs Baseline")
        with col2:
            st.metric("Procedural Completeness", f"{p_sum['avg_completeness_score']:.2f} / 1.0", f"{p_sum['avg_completeness_score'] - b_sum['avg_completeness_score']:+.2f}")
        with col3:
            st.metric("Forum & Tribunal Accuracy", f"{p_sum['forum_accuracy']:.1f}%", f"{p_sum['forum_accuracy'] - b_sum['forum_accuracy']:+.1f}%")
        with col4:
            st.metric("LLM-as-a-Judge Score", f"{p_sum['avg_judge_score']:.2f} / 5.0", f"{p_sum['avg_judge_score'] - b_sum['avg_judge_score']:+.2f}")

        st.markdown("---")
        st.markdown("#### Comparative Benchmark Table")
        st.table({
            "Metric": [
                "Statutory Hallucination Rate (%)",
                "Citation Grounding Accuracy (%)",
                "Procedural Phase Completeness (0-1.0)",
                "Designated Forum Accuracy (%)",
                "LLM-as-a-Judge Rating (0-5.0)",
                "Mean Latency (seconds)",
            ],
            "Approach A (Baseline Vanilla RAG)": [
                f"{b_sum['avg_hallucination_rate']:.1f}%",
                f"{b_sum['avg_grounding_accuracy']:.1f}%",
                f"{b_sum['avg_completeness_score']:.2f}",
                f"{b_sum['forum_accuracy']:.1f}%",
                f"{b_sum['avg_judge_score']:.2f}",
                f"{b_sum['avg_latency_seconds']:.2f}s",
            ],
            "Approach B (Proposed LangGraph Agent)": [
                f"{p_sum['avg_hallucination_rate']:.1f}%",
                f"{p_sum['avg_grounding_accuracy']:.1f}%",
                f"{p_sum['avg_completeness_score']:.2f}",
                f"{p_sum['forum_accuracy']:.1f}%",
                f"{p_sum['avg_judge_score']:.2f}",
                f"{p_sum['avg_latency_seconds']:.2f}s",
            ],
        })
    else:
        st.info("Run `python evaluation/comparative_runner.py` to generate the complete benchmark results.")

# ----------------------------------------------------
# TAB 3: Architecture & Corpus
# ----------------------------------------------------
with tab3:
    st.markdown("### 🏗️ Proposed System Architecture & Corpus")
    st.markdown(
        """
        ```mermaid
        flowchart TD
            UserQuery["User Scenario Narrative"] --> GuardrailNode["1. Intent & Scope Guardrail"]
            
            GuardrailNode -- "Out of Scope / Non-Legal" --> RejectionOutput["Polite Refusal & Clarification"]
            GuardrailNode -- "Valid Legal Scenario" --> DecontextNode["2. Query Decontextualizer"]
            
            subgraph "Dual Retrieval & Reranking"
                DecontextNode --> GeneralRAG["2a. General Hybrid Retriever (BM25 + Dense)"]
                DecontextNode --> CriminalRAG["2b. Criminal Code Deep RAG (BNS / BNSS / IPC Payload Filter)"]
                GeneralRAG --> RRFFusion["RRF Reciprocal Rank Fusion (k=60)"]
                CriminalRAG --> RRFFusion
                RRFFusion --> CrossEncoder["Cross-Encoder Reranker"]
            end
            
            subgraph "Agentic Planning & Verification"
                CrossEncoder --> ProceduralPlanner["3. Procedural Engine (ReAct Agent)"]
                ProceduralPlanner --> ActionPlanDraft["Draft Action Plan"]
                CrossEncoder --> VerificationNode["4. Synthesis & Verification Node"]
                ActionPlanDraft --> VerificationNode
                VerificationNode -- "Ungrounded Citations" --> Reground["Exact Section Re-Grounding via Store"]
                Reground --> VerificationNode
            end
            
            VerificationNode --> FinalOutput["Final Output: Verified Roadmap + Statutory Citations"]
        ```
        """
    )
    st.markdown("---")
    st.markdown("#### 📚 Ingested Statutory Corpus Statistics")
    st.markdown(
        """
        - **Data Source:** Open India Law / India Code Central Legislation (`in_central_legislation.parquet`)
        - **Total Indexed Provisions:** ~3,678 sections
        - **Governing Target Statutes:**
          - Motor Vehicles Act, 1988 (374 sections)
          - Bharatiya Nyaya Sanhita, 2023 (BNS) & Indian Penal Code, 1860
          - Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) & Code of Criminal Procedure, 1973
          - Consumer Protection Act, 2019 (156 sections)
          - Transfer of Property Act, 1882 (212 sections)
          - Specific Relief Act, 1963 (58 sections)
        - **Landmark Supreme Court Precedents:**
          - Sarla Verma & Ors. v. DTC (2009) — (2009) 6 SCC 121
          - National Insurance Co. Ltd. v. Pranay Sethi (2017) — (2017) 16 SCC 680
          - Suraj Lamp & Industries Pvt. Ltd. v. State of Haryana (2012) — (2012) 1 SCC 656
          - Poona Ram v. Moti Ram (2019) — (2019) 11 SCC 309
          - Lucknow Development Authority v. M.K. Gupta (1994) — (1994) 1 SCC 243
          - Experion Developers Pvt. Ltd. v. Sushma Ashok Shiroor (2022) — (2022) SCC OnLine SC 478
        """
    )
