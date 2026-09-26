"""Live pipeline progress for Approach B, driven by LegalAgentGraph.stream()."""

from typing import Dict

import streamlit as st

from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.data.ingestion.schema import DualOutputResponse

# User-facing descriptions of each LangGraph node, shown as they complete.
NODE_MESSAGES: Dict[str, str] = {
    "guardrail": "Understood your question and identified the type of case",
    "decontextualize": "Worked out which laws and judgments to look for",
    "retrieve_and_rerank": "Searched our legal database",
    "applicability_check": "Checked which laws apply to your facts",
    "procedural_planner": "Drafted your step-by-step plan",
    "verification": "Matched every citation to official legal text in our database",
    "synthesis": "Prepared your results",
    "handle_rejection": "This doesn't look like a legal problem we can help with",
}


def run_with_progress(engine: LegalAgentGraph, scenario: str, label: str = "Analysing your situation...") -> DualOutputResponse:
    """Execute the agent graph while showing each completed stage in an st.status box.

    Raises:
        RuntimeError: If the graph finishes without producing a final response.
    """
    final_response = None
    with st.status(label, expanded=True) as status:
        for node_name, update in engine.stream(scenario):
            st.write(f"✓ {NODE_MESSAGES.get(node_name, node_name.replace('_', ' ').capitalize())}")
            if update.get("final_response") is not None:
                final_response = update["final_response"]
        if final_response is None:
            status.update(label="Something went wrong", state="error")
            raise RuntimeError("The pipeline finished without producing a response.")
        status.update(label="Done", state="complete", expanded=False)
    return final_response
