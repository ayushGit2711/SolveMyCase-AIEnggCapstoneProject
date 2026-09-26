"""🧭 Get Help: the default, citizen-facing page (Approach B only)."""

import logging

import streamlit as st

from solvemycase.ui.components.formatting import DISCLAIMER, escape_markdown
from solvemycase.ui.components.progress import run_with_progress
from solvemycase.ui.components.result import render_full_result, render_scenario_echo
from solvemycase.ui.components.scenario_input import scenario_input
from solvemycase.ui.state import compute_domain_coverage_cards, get_engines

logger = logging.getLogger(__name__)

RESULT_KEY = "help_result"


def render_coverage_cards(store) -> None:
    """Display the four problem categories handled by our live legal database in a spacious 2x2 layout."""
    cards = compute_domain_coverage_cards(store)
    st.subheader("Problems we handle right now (from our live legal database)", anchor=False)
    st.caption(
        f"**What we cover:** Our database currently holds verified Indian statutes and Supreme Court judgments "
        f"across the 4 categories below. {DISCLAIMER}"
    )

    for idx in range(0, len(cards), 2):
        if idx > 0:
            st.write("")
        row_cards = cards[idx : idx + 2]
        cols = st.columns(2, gap="large")
        for col, card in zip(cols, row_cards):
            with col:
                with st.container(border=True):
                    bullets = "\n".join(f"- {escape_markdown(item)}" for item in card["problems_list"])
                    st.markdown(f"**{escape_markdown(card['title'])}**\n\n{bullets}")
                    pills = " · ".join(f"`{b}`" for b in card["act_badges"])
                    st.caption(
                        f"📚 **{card['statute_count']}** sections · "
                        f"⚖️ **{card['precedent_count']}** SC judgments  |  {pills}"
                    )


def render() -> None:
    """Render the Get Help page."""
    engines = get_engines()
    st.header("Get a step-by-step legal action plan")
    st.markdown(
        "Describe your situation in plain language. We'll tell you **what to do, in what order, where to go, "
        "and by when**, citing only Indian laws and judgments from our database that apply to your facts."
    )

    scenario = scenario_input(key_prefix="help", submit_label="Get my action plan")

    if scenario:
        try:
            response = run_with_progress(engines.proposed, scenario)
        except Exception:  # Keep the page usable; details go to the server log only.
            logger.exception("Get Help pipeline run failed")
            st.session_state.pop(RESULT_KEY, None)
            st.error("Sorry, something went wrong while analysing your situation. Please try again in a moment.")
        else:
            # Persist so the result survives reruns triggered by other widgets.
            st.session_state[RESULT_KEY] = {"scenario": scenario, "response": response}

    saved = st.session_state.get(RESULT_KEY)
    if saved:
        st.divider()
        render_scenario_echo(saved["scenario"])
        render_full_result(saved["scenario"], saved["response"], key_prefix="help")

    st.divider()
    render_coverage_cards(engines.store)

