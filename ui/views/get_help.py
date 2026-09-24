"""🧭 Get Help: the default, citizen-facing page (Approach B only)."""

import logging

import streamlit as st

from solvemycase.ui.components.formatting import DISCLAIMER
from solvemycase.ui.components.progress import run_with_progress
from solvemycase.ui.components.result import render_full_result, render_scenario_echo
from solvemycase.ui.components.scenario_input import scenario_input
from solvemycase.ui.state import get_engines

logger = logging.getLogger(__name__)

RESULT_KEY = "help_result"


def render() -> None:
    """Render the Get Help page."""
    st.title("Get a step-by-step legal action plan")
    st.markdown(
        "Describe your situation in plain language. We'll tell you **what to do, in what order, where to go, "
        "and by when**, citing only Indian laws and judgments we could verify against official text."
    )
    st.caption(f"Covers motor accidents, property disputes and consumer complaints. {DISCLAIMER}")

    scenario = scenario_input(key_prefix="help", submit_label="Get my action plan")

    if scenario:
        engines = get_engines()
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
