"""Result views: full citizen result, guardrail rejection, and verification details."""

import streamlit as st

from solvemycase.data.ingestion.schema import DualOutputResponse
from solvemycase.ui.components.action_plan import render_action_plan, render_deadlines
from solvemycase.ui.components.citations import render_citations
from solvemycase.ui.components.export import response_to_markdown
from solvemycase.ui.components.formatting import (
    DISCLAIMER,
    domain_label,
    escape_markdown,
    extract_deadlines,
    is_guardrail_rejection,
    rejection_details,
    removed_references,
    verified_citation_count,
)


def render_scenario_echo(scenario: str) -> None:
    """Remind the user which scenario the results below answer."""
    st.caption(f"**Answering:** {escape_markdown(scenario)}")


def render_rejection(response: DualOutputResponse) -> None:
    """Friendly message when the guardrail decides the query is out of scope."""
    details = rejection_details(response)
    st.info(
        f"**We couldn't treat this as a legal problem.** {escape_markdown(details['reason'])}\n\n"
        f"{escape_markdown(details['clarification'])}"
    )


def render_verification_details(response: DualOutputResponse) -> None:
    """Explain what verification did, in neutral language."""
    removed = removed_references(response)
    verified_count = verified_citation_count(response)
    st.markdown(
        f"Every law and judgment in this answer was checked against official legal text we retrieved. "
        f"**{verified_count}** passed the check."
    )
    if removed:
        st.markdown(
            f"We removed **{len(removed)}** reference(s) that couldn't be matched to official text, "
            "so they aren't shown in your plan:"
        )
        for item in removed:
            st.markdown(f"- {escape_markdown(item)}")
    else:
        st.success("No references needed to be removed.")


def render_full_result(scenario: str, response: DualOutputResponse, key_prefix: str) -> None:
    """Citizen-facing result: summary strip, tabbed details, and a Markdown download."""
    if is_guardrail_rejection(response):
        render_rejection(response)
        return

    verified_count = verified_citation_count(response)
    deadlines = extract_deadlines(response.action_plan)

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Type of case", domain_label(response.domain))
        c2.metric("Steps to take", len(response.action_plan))
        c3.metric("Verified laws & judgments", verified_count)
        if deadlines:
            st.caption(f"⏰ Most urgent deadline: **{escape_markdown(deadlines[0]['Deadline'])}**: {escape_markdown(deadlines[0]['Action'])}")

    tab_plan, tab_laws, tab_deadlines, tab_checks = st.tabs(
        ["📋 Action plan", "📜 Laws & cases", "⏰ Deadlines", "🔍 How we checked"]
    )
    with tab_plan:
        render_action_plan(response.action_plan)
    with tab_laws:
        render_citations(response.statutory_citations, response.precedent_citations, key_prefix=f"{key_prefix}_cit")
    with tab_deadlines:
        render_deadlines(response.action_plan)
    with tab_checks:
        render_verification_details(response)

    st.download_button(
        "⬇️ Download plan (Markdown)",
        data=response_to_markdown(scenario, response),
        file_name="solvemycase_action_plan.md",
        mime="text/markdown",
        key=f"{key_prefix}_download",
    )
    st.caption(DISCLAIMER)
