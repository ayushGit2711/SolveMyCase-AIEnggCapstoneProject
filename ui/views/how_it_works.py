"""ℹ️ How it works: architecture diagram, live corpus statistics, and engine settings."""

import streamlit as st

from solvemycase.ui.components.formatting import escape_markdown
from solvemycase.ui.state import compute_corpus_stats, get_engines, is_llm_mode

ARCHITECTURE_DOT = """
digraph G {
    rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#EFF6FF", color="#3B82F6", fontname="Helvetica"];
    edge [color="#64748B", fontname="Helvetica", fontsize=10];
    q [label="Your situation", fillcolor="#F1F5F9"];
    g [label="1. Scope guardrail\\n(is it a legal problem? which domain?)"];
    r [label="Friendly refusal +\\nclarification", fillcolor="#FEF3C7", color="#D97706"];
    d [label="2. Query expansion\\n(statute + precedent sub-queries)"];
    h [label="3a. Hybrid search\\n(dense + BM25, RRF)"];
    c [label="3b. Criminal-code search\\n(BNS / BNSS / IPC / CrPC)"];
    x [label="3c. Cross-encoder reranking\\n(best match over all sub-queries)"];
    a [label="4. Applicability check\\n(keep only laws that apply to the facts)"];
    n [label="Honest coverage note +\\nsafe general steps (no citations)", fillcolor="#FEF3C7", color="#D97706"];
    p [label="5. Procedural planner\\n(6-phase action plan)"];
    v [label="6. Citation verification\\n(match retrieved text, else remove)"];
    o [label="Verified action plan +\\nlaws & judgments", fillcolor="#DCFCE7", color="#16A34A"];
    q -> g; g -> r [label="out of scope"]; g -> d [label="legal"];
    d -> h; d -> c; h -> x; c -> x; x -> a;
    a -> n [label="nothing applies"]; a -> p [label="law applies"]; p -> v; v -> o;
}
"""


def render() -> None:
    """Render the architecture and corpus page."""
    engines = get_engines()
    settings = engines.settings

    st.header("How it works")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")

    st.subheader("Legal corpus (live)")
    stats = compute_corpus_stats(engines.store)
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents indexed", stats["total_documents"])
    c2.metric("Statutory sections", stats["statute_count"])
    c3.metric("Court judgments", stats["precedent_count"])
    if stats["total_documents"] == 0:
        st.warning("The index is empty. Run `python -m solvemycase.data.vectorstore.indexer`.")
    else:
        st.caption(escape_markdown(engines.proposed.coverage_summary()))
        left, right = st.columns(2)
        with left:
            st.markdown("##### Sections per Act")
            st.dataframe(
                [{"Act": act, "Sections": n} for act, n in stats["sections_per_act"].items()],
                hide_index=True,
                width="stretch",
            )
        with right:
            st.markdown("##### Judgments")
            for title in stats["precedent_titles"]:
                st.markdown(f"- {escape_markdown(title)}")

    st.subheader("Engine settings")
    qdrant_mode = "Remote Qdrant" if settings.qdrant_url else f"Embedded Qdrant ({settings.qdrant_path})"
    if not settings.applicability_check_enabled:
        applicability = "Disabled"
    elif is_llm_mode(settings):
        applicability = f"{settings.openai_model_fast}, top {settings.applicability_candidate_k} candidates"
    else:
        applicability = "Skipped offline (top reranked candidates pass through)"
    st.dataframe(
        [
            {"Setting": "Mode", "Value": "OpenAI" if is_llm_mode(settings) else "Offline fallbacks (no API key)"},
            {"Setting": "Planner model", "Value": settings.openai_model_primary},
            {"Setting": "Guardrail / routing model", "Value": settings.openai_model_fast},
            {"Setting": "Embedding model", "Value": settings.openai_embedding_model},
            {"Setting": "Vector store", "Value": qdrant_mode},
            {"Setting": "Candidates retrieved / kept after rerank", "Value": f"{settings.max_retrieved_chunks} / {settings.rerank_top_k}"},
            {"Setting": "Cross-encoder", "Value": settings.cross_encoder_model},
            {"Setting": "Applicability check", "Value": applicability},
        ],
        hide_index=True,
        width="stretch",
    )
