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
    """Display the four problem categories handled by our live legal database in a balanced 2x2 grid."""
    cards = compute_domain_coverage_cards(store)
    st.subheader("Problems we handle right now (from our live legal database)", anchor=False)
    for idx in range(0, len(cards), 2):
        row_cards = cards[idx : idx + 2]
        cols = st.columns(2, gap="medium")
        for col, card in zip(cols, row_cards):
            with col:
                with st.container(border=True):
                    st.markdown(f"##### {escape_markdown(card['title'])}")
                    st.caption(
                        f"📚 **{card['statute_count']}** statutory sections · "
                        f"⚖️ **{card['precedent_count']}** SC judgment(s)"
                    )
                    bullets = "\n".join(f"- {escape_markdown(item)}" for item in card["problems_list"])
                    st.markdown(bullets)
                    if card["act_badges"]:
                        pills = " · ".join(f"`{b}`" for b in card["act_badges"])
                        st.caption(f"**Indexed Acts:** {pills}")


def render() -> None:
    """Render the Get Help page."""
    engines = get_engines()
    st.header("Get a step-by-step legal action plan")
    st.markdown(
        "Describe your situation in plain language. We'll tell you **what to do, in what order, where to go, "
        "and by when**, citing only Indian laws and judgments from our database that apply to your facts."
    )
    render_coverage_cards(engines.store)
    with st.expander("📋 View full list of indexed Acts & out-of-coverage policy", expanded=False):
        st.caption(f"**What we cover:** {escape_markdown(engines.proposed.coverage_summary())} {DISCLAIMER}")

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
