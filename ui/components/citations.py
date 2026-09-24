"""Verified statute and precedent renderers."""

from typing import List, Optional

import streamlit as st

from solvemycase.data.ingestion.schema import PrecedentCitation, StatutoryCitation
from solvemycase.ui.components.formatting import (
    escape_markdown,
    precedent_read_url,
    safe_http_url,
    statute_search_url,
)


def _link(url: Optional[str], label: str, key: str, container=st) -> None:
    """Render a link button only for safe absolute http(s) URLs (e.g. never javascript:)."""
    safe_url = safe_http_url(url)
    if safe_url:
        container.link_button(label, safe_url, key=key)


def _statute_links(cit: StatutoryCitation, key: str) -> None:
    """Primary Indian Kanoon link plus the official India Code page when one is stored."""
    primary = statute_search_url(cit.act_name, cit.section_number)
    official = safe_http_url(cit.source_url)
    col_primary, col_official = st.columns(2)
    _link(primary, "Read on Indian Kanoon ↗", key=f"{key}_ik", container=col_primary)
    if official and official != primary:
        _link(official, "India Code (official) ↗", key=f"{key}_official", container=col_official)


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
                _statute_links(cit, key=f"{key_prefix}_stat_{i}")

    if precedents:
        st.markdown("##### Court judgments")
        for i, prec in enumerate(precedents):
            ref = prec.citation or (str(prec.year) if prec.year else "")
            with st.expander(escape_markdown(f"{prec.case_title} {ref}".strip())):
                st.caption(escape_markdown(prec.court))
                st.markdown(escape_markdown(prec.legal_principle))
                _link(precedent_read_url(prec.source_url, prec.case_title), "Read judgment ↗", key=f"{key_prefix}_prec_{i}")
