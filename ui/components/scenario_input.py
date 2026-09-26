"""Scenario input: text area, one-click example chips, and client-side validation."""

from typing import Dict, List, Optional

import streamlit as st

MIN_SCENARIO_LENGTH = 10  # Mirrors the backend API validation (LegalQueryRequest.min_length).
MAX_SCENARIO_LENGTH = 4000

EXAMPLE_SCENARIOS: Dict[str, str] = {
    "🚗 Hit-and-run": (
        "A speeding truck hit my father's scooter at an intersection in Pune and the driver fled. "
        "He is in hospital with multiple fractures. What should we do?"
    ),
    "🚗 Insurer refusing claim": (
        "My car was hit from behind by a bus in Delhi last month. The bus company's insurer is refusing "
        "to pay for repairs and my medical bills. How do I claim compensation?"
    ),
    "🏠 Landlord locked me out": (
        "My landlord in Bengaluru changed the locks of my rented flat while I was away, even though my "
        "rent is fully paid and the lease runs for 8 more months."
    ),
    "🏠 Neighbour encroaching": (
        "My neighbour has started building a wall that extends two feet into my registered plot. "
        "He ignored my verbal objections. What legal steps can I take?"
    ),
    "🛒 Defective phone, no refund": (
        "I bought a phone online that stopped working in 10 days. The seller and the e-commerce "
        "platform both refuse a refund or replacement despite the warranty."
    ),
    "🛒 Builder delayed flat": (
        "The builder promised possession of my flat in 2022 but it is still incomplete. I have paid "
        "90% of the price. Can I get a refund with interest?"
    ),
    "🐾 Pet harmed by watchman": (
        "Our society watchman beat and killed my pet cat with a stick last night. We have CCTV footage. "
        "What criminal complaint can we file?"
    ),
}

PLACEHOLDER = (
    "Describe what happened: where and when, who is involved, any police report or documents you have, "
    "and what outcome you want."
)


def text_key(key_prefix: str) -> str:
    """Widget key of the scenario text area for a page."""
    return f"{key_prefix}_scenario"


def _persist_key(key_prefix: str) -> str:
    """Non-widget session key mirroring the text so it survives page switches.

    Streamlit discards widget state when a page stops rendering the widget, so
    the text is mirrored here and restored on the next render.
    """
    return f"_{key_prefix}_scenario_saved"


def set_scenario_text(key_prefix: str, text: str) -> None:
    """Programmatically fill a page's scenario box (safe inside widget callbacks)."""
    st.session_state[text_key(key_prefix)] = text
    st.session_state[_persist_key(key_prefix)] = text


def _apply_example(key_prefix: str, pills_key: str) -> None:
    """on_change callback: copy the selected example into the text area."""
    choice = st.session_state.get(pills_key)
    if choice:
        set_scenario_text(key_prefix, EXAMPLE_SCENARIOS[choice])


def _save_text(key_prefix: str) -> None:
    """on_change callback: mirror the typed text into the persistent key."""
    st.session_state[_persist_key(key_prefix)] = st.session_state.get(text_key(key_prefix), "")


def scenario_input(key_prefix: str, submit_label: str, examples: Optional[List[str]] = None) -> Optional[str]:
    """Render the scenario form and return the scenario text when the user submits.

    Args:
        key_prefix: Unique prefix so multiple pages keep independent widget state.
        submit_label: Primary button label.
        examples: Subset of EXAMPLE_SCENARIOS keys to offer (defaults to all).

    Returns:
        The stripped scenario text on a valid submit, else None.
    """
    area_key, pills_key = text_key(key_prefix), f"{key_prefix}_example"
    if area_key not in st.session_state:
        st.session_state[area_key] = st.session_state.get(_persist_key(key_prefix), "")

    st.pills(
        "Try an example",
        options=examples or list(EXAMPLE_SCENARIOS),
        key=pills_key,
        on_change=_apply_example,
        args=(key_prefix, pills_key),
    )
    scenario = st.text_area(
        "Your situation",
        key=area_key,
        placeholder=PLACEHOLDER,
        height=130,
        max_chars=MAX_SCENARIO_LENGTH,
        on_change=_save_text,
        args=(key_prefix,),
    )

    # The button stays enabled: a disabled button gives no feedback and ignores
    # text typed but not yet committed (Streamlit commits on blur / Ctrl+Enter).
    if not st.button(submit_label, type="primary", key=f"{key_prefix}_submit"):
        return None
    text = (scenario or "").strip()
    st.session_state[_persist_key(key_prefix)] = scenario or ""
    if len(text) < MIN_SCENARIO_LENGTH:
        st.warning(f"Please describe your situation in at least {MIN_SCENARIO_LENGTH} characters.")
        return None
    return text
