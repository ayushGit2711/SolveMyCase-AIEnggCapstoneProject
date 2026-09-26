"""👩‍⚖️ Connect to a Lawyer: browse specialist advocates by complaint type and request a consultation."""

import streamlit as st

from solvemycase.ui.components.formatting import DISCLAIMER
from solvemycase.ui.components.lawyers import (
    CITY_OPTIONS,
    COMPLAINT_ALL,
    COMPLAINT_CATEGORIES,
    LAWYER_DIRECTORY,
    filter_lawyers,
    infer_complaint_category,
    render_lawyer_card,
)


def render() -> None:
    """Render the Connect to a Lawyer directory page."""
    st.header("Connect to a lawyer")
    st.markdown(
        "Filter advocates by the **type of complaint** and your **city / consultation mode**, then open any profile "
        "to request a callback or video consultation."
    )
    st.caption(
        "ℹ️ **Prototype directory:** Advocate profiles and Bar Council IDs shown below are dummy demonstration data "
        f"for the SOLVE MY CASE capstone project. {DISCLAIMER}"
    )

    saved_help = st.session_state.get("help_result")
    default_scenario = ""
    default_cat = COMPLAINT_ALL
    if isinstance(saved_help, dict) and saved_help.get("response") is not None:
        resp = saved_help["response"]
        default_scenario = str(saved_help.get("scenario") or "")
        default_cat = infer_complaint_category(
            scenario=default_scenario,
            domain=getattr(resp, "domain", None),
            coverage_gap=bool(getattr(resp, "coverage_note", None)),
        )

    if "lawyer_filter_category" not in st.session_state:
        st.session_state["lawyer_filter_category"] = (
            default_cat if default_cat in COMPLAINT_CATEGORIES else COMPLAINT_ALL
        )

    f_col1, f_col2 = st.columns([2, 1])
    selected_category = f_col1.selectbox(
        "Type of complaint",
        options=COMPLAINT_CATEGORIES,
        key="lawyer_filter_category",
    )
    selected_city = f_col2.selectbox(
        "City / Mode",
        options=CITY_OPTIONS,
        key="lawyer_filter_city",
    )

    matched = filter_lawyers(category=selected_category, city=selected_city)
    if not matched:
        st.info(
            "No advocates matched both filters. Showing all advocates who offer pan-India online consultations "
            "for this complaint type."
        )
        matched = filter_lawyers(category=selected_category, city="All cities / Online") or list(LAWYER_DIRECTORY)

    st.caption(f"Showing **{len(matched)}** advocate(s) for **{selected_category}**.")

    for idx in range(0, len(matched), 2):
        pair = matched[idx : idx + 2]
        cols = st.columns(len(pair))
        for col, lawyer in zip(cols, pair):
            with col:
                render_lawyer_card(
                    lawyer,
                    key_prefix="dir",
                    default_scenario=default_scenario,
                    default_category=(
                        selected_category if selected_category != COMPLAINT_ALL else lawyer["categories"][0]
                    ),
                )
