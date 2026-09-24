"""Verified statute and precedent renderers."""

from typing import List

import streamlit as st

from solvemycase.data.ingestion.schema import PrecedentCitation, StatutoryCitation
from solvemycase.ui.components.formatting import escape_markdown, safe_http_url


def _source_link(url: str, label: str, key: str) -> None:
    """Render a link button only for safe absolute http(s) URLs (e.g. never javascript:); otherwise render nothing."""
    safe_url = safe_http_url(url)
    if safe_url:
        st.link_button(label, safe_url, key=key)


def render_citations(
    statutes: List[StatutoryCitation],
    precedents: List[PrecedentCitation],
    key_prefix: str,
) -> None:
    """Render statutory citations and court judgments as expandable cards.

    Args:
        statutes: Verified (or baseline-proposed) statutory citations.
        precedents: Verified (or baseline-proposed) precedents.
        key_prefix: Unique prefix for widget keys (several columns may render citations).
    """
    if not statutes and not precedents:
        st.info("No citations to show for this answer.")
        return

    if statutes:
        st.markdown("##### Laws that apply")
        for i, cit in enumerate(statutes):
            with st.expander(escape_markdown(f"{cit.act_name} · Section {cit.section_number}")):
                st.markdown(escape_markdown(cit.summary_of_provision))
                if cit.applicability_to_scenario:
                    st.caption(f"**Why it applies:** {escape_markdown(cit.applicability_to_scenario)}")
                _source_link(cit.source_url, "Open official source ↗", key=f"{key_prefix}_stat_{i}")

    if precedents:
        st.markdown("##### Court judgments")
        for i, prec in enumerate(precedents):
            ref = prec.citation or (str(prec.year) if prec.year else "")
            with st.expander(escape_markdown(f"{prec.case_title} {ref}".strip())):
                st.caption(escape_markdown(prec.court))
                st.markdown(escape_markdown(prec.legal_principle))
                _source_link(prec.source_url, "Open judgment ↗", key=f"{key_prefix}_prec_{i}")
