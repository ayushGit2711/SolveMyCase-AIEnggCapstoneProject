"""Action-plan timeline and deadline table renderers (native Streamlit components only)."""

from typing import List

import streamlit as st

from solvemycase.data.ingestion.schema import ProceduralActionStep
from solvemycase.ui.components.formatting import (
    PHASE_LABELS,
    escape_markdown,
    extract_deadlines,
    group_steps_by_phase,
    has_real_deadline,
    priority_badge_color,
)


def render_action_plan(steps: List[ProceduralActionStep], compact: bool = False) -> None:
    """Render steps grouped under chronological phase headings.

    Args:
        steps: Procedural steps from a DualOutputResponse.
        compact: When True (side-by-side comparison), omit phase headings.
    """
    if not steps:
        st.info("No action steps were generated.")
        return

    for phase, phase_steps in group_steps_by_phase(steps).items():
        if not compact:
            st.markdown(f"##### {PHASE_LABELS[phase]}")
        for step in phase_steps:
            with st.container(border=True):
                st.badge(escape_markdown((step.priority or "Step").upper()), color=priority_badge_color(step.priority))
                st.markdown(f"**Step {step.step_number}: {escape_markdown(step.title)}**")
                st.markdown(escape_markdown(step.description))
                details = [f"📍 **Where:** {escape_markdown(step.forum_or_authority)}"]
                if has_real_deadline(step):
                    details.append(f"⏰ **Deadline:** {escape_markdown(step.limitation_period.strip())}")
                if step.statutory_basis:
                    details.append(f"📜 **Legal basis:** {escape_markdown(step.statutory_basis)}")
                st.caption("  \n".join(details))


def render_deadlines(steps: List[ProceduralActionStep]) -> None:
    """Render limitation periods as a table, most urgent first."""
    deadlines = extract_deadlines(steps)
    if not deadlines:
        st.info("No specific deadlines were identified for this situation.")
        return
    st.dataframe(deadlines, hide_index=True, width="stretch")
    st.caption("Missing a limitation deadline can bar your claim. Confirm exact dates with an advocate.")
