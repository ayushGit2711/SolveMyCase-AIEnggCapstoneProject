"""Streamlit AppTest coverage for every page, run fully offline against an isolated Qdrant index."""

import json
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from solvemycase.config.settings import get_settings
from solvemycase.data.ingestion.schema import DocumentType

APP_FILE = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")
TIMEOUT_SECONDS = 180  # First run loads the cross-encoder model.

FIXTURE_DATASET = [
    {"id": "mva_t1", "scenario": "A truck hit my scooter in Pune and the driver fled the spot.", "domain": "motor_vehicle_accident", "is_legal": True},
    {"id": "con_t1", "scenario": "The seller refuses to replace my defective phone under warranty.", "domain": "consumer_rights", "is_legal": True},
    {"id": "grd_t1", "scenario": "Please write a poem about the monsoon over Mumbai.", "domain": "general_dispute", "is_legal": False},
]

_APPROACH_SUMMARY = {
    "avg_hallucination_rate": 0.0,
    "avg_grounding_accuracy": 100.0,
    "avg_completeness_score": 0.5,
    "forum_accuracy": 100.0,
    "avg_judge_score": 3.0,
    "avg_latency_seconds": 1.0,
}
_APPROACH_RESULT = {"judge_score": 3.0, "completeness_score": 0.5, "latency_seconds": 1.0, "unverified_stripped_count": 0}
FIXTURE_RESULTS = {
    "summary": {"total_scenarios": 1, "baseline": _APPROACH_SUMMARY, "proposed": _APPROACH_SUMMARY},
    "results": [{"scenario_id": "mva_t1", "domain": "motor_vehicle_accident", "baseline": _APPROACH_RESULT, "proposed": _APPROACH_RESULT}],
}


@pytest.fixture(scope="module", autouse=True)
def offline_isolated_index(tmp_path_factory):
    """Point the app at a freshly indexed temp Qdrant dir, fixture benchmark files, and no API key.

    Yields:
        Expected corpus counts computed from the index actually built here.
    """
    from solvemycase.data.vectorstore.indexer import run_indexing_pipeline
    from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

    eval_dir = tmp_path_factory.mktemp("evaluation")
    (eval_dir / "benchmark_dataset.json").write_text(json.dumps(FIXTURE_DATASET), encoding="utf-8")
    (eval_dir / "benchmark_results.json").write_text(json.dumps(FIXTURE_RESULTS), encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path_factory.mktemp("qdrant")))
    monkeypatch.setenv("OPENAI_API_KEY", "")  # Env vars override any local .env file.
    monkeypatch.setattr("solvemycase.ui.state.EVALUATION_DIR", eval_dir)
    get_settings.cache_clear()
    st.cache_resource.clear()
    st.cache_data.clear()

    store = QdrantLegalStore(settings=get_settings())
    run_indexing_pipeline(store=store)
    expected = {
        "documents": len(store.corpus_documents),
        "precedents": sum(1 for d in store.corpus_documents if d.doc_type == DocumentType.PRECEDENT),
    }
    store.client.close()  # Embedded Qdrant: release the folder lock for the app's own client.

    yield expected

    st.cache_resource.clear()
    st.cache_data.clear()
    get_settings.cache_clear()
    monkeypatch.undo()


def _run_view(module_name: str) -> AppTest:
    """Render a single view module in isolation.

    AppTest.from_function serializes the function's source, so the module name is passed
    via kwargs rather than captured from this scope.
    """
    def page(module_name):
        import importlib

        importlib.import_module(f"solvemycase.ui.views.{module_name}").render()

    return AppTest.from_function(page, kwargs={"module_name": module_name}, default_timeout=TIMEOUT_SECONDS).run()


def _all_text(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.info]
    return "\n".join(str(p) for p in parts)


def test_entrypoint_renders_default_get_help_page():
    at = AppTest.from_file(APP_FILE, default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception
    assert at.title[0].value == "Get a step-by-step legal action plan"
    assert any("Offline mode" in w.value for w in at.sidebar.warning)


def test_get_help_submit_shows_plan_and_persists_across_reruns():
    at = _run_view("get_help")
    scenario = "A speeding truck hit my father's scooter in Pune and fled. He has fractures."
    at.text_area(key="help_scenario").input(scenario).run()
    at.button(key="help_submit").click().run()

    assert not at.exception
    assert [t.label for t in at.tabs] == ["📋 Action plan", "📜 Laws & cases", "⏰ Deadlines", "🔍 How we checked"]
    assert "Step 1:" in _all_text(at)
    assert any(c.value.startswith("**Answering:**") for c in at.caption)

    # Editing the input triggers a rerun; the previous result must still be shown.
    at.text_area(key="help_scenario").input("Something else entirely, not yet submitted.").run()
    assert "Step 1:" in _all_text(at)


def test_get_help_out_of_scope_shows_friendly_rejection():
    at = _run_view("get_help")
    at.text_area(key="help_scenario").input("Please write a poem about the monsoon over Mumbai.").run()
    at.button(key="help_submit").click().run()

    assert not at.exception
    assert any("couldn't treat this as a legal problem" in i.value for i in at.info)
    assert not at.tabs


def test_short_input_submit_warns_instead_of_running():
    at = _run_view("get_help")
    assert not at.button(key="help_submit").disabled
    at.text_area(key="help_scenario").input("too short").run()
    at.button(key="help_submit").click().run()

    assert not at.exception
    assert any("at least" in w.value for w in at.warning)
    assert not at.tabs


def test_example_chip_fills_scenario():
    from solvemycase.ui.components.scenario_input import EXAMPLE_SCENARIOS

    at = _run_view("get_help")
    label = next(iter(EXAMPLE_SCENARIOS))
    at.button_group(key="help_example").set_value(label).run()

    assert not at.exception
    assert at.text_area(key="help_scenario").value == EXAMPLE_SCENARIOS[label]


def test_compare_page_runs_both_approaches():
    at = _run_view("compare")
    at.text_area(key="compare_scenario").input(
        "My landlord changed the locks of my rented flat although rent is fully paid."
    ).run()
    at.button(key="compare_submit").click().run()

    assert not at.exception
    assert [m.label for m in at.metric] == ["Citations shown", "Citations removed", "Phases covered", "Latency"]
    assert {"A · Baseline RAG", "B · Proposed agent"} <= {s.value for s in at.subheader}
    # Sections are bordered containers, so the citation expanders inside are never nested.
    assert not {"📋 Action plan", "📜 Citations"} & {e.label for e in at.expander}


def test_compare_benchmark_picker_fills_scenario_including_guardrail_cases():
    at = _run_view("compare")
    picker = at.selectbox(key="compare_benchmark_pick")
    assert len(picker.options) == len(FIXTURE_DATASET)
    guardrail_label = next(o for o in picker.options if o.startswith("grd_t1"))
    assert "Out of scope" in guardrail_label

    picker.set_value(guardrail_label).run()
    assert not at.exception
    assert at.text_area(key="compare_scenario").value == FIXTURE_DATASET[2]["scenario"]


def test_benchmark_page_warns_on_partial_results():
    at = _run_view("benchmark")
    assert not at.exception
    assert at.metric[0].label == "Total scenarios"
    assert at.metric[0].value == str(len(FIXTURE_DATASET))
    assert any("1 of 3" in w.value and "not representative" in w.value for w in at.warning)


def test_how_it_works_shows_live_corpus_stats(offline_isolated_index):
    at = _run_view("how_it_works")
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Documents indexed"] == str(offline_isolated_index["documents"])
    assert metrics["Court judgments"] == str(offline_isolated_index["precedents"])


def test_malicious_llm_output_is_rendered_inert():
    def page():
        from solvemycase.data.ingestion.schema import (
            DualOutputResponse,
            LegalDomain,
            PrecedentCitation,
            ProceduralActionStep,
            ProceduralPhase,
            StatutoryCitation,
        )
        from solvemycase.ui.components.result import render_full_result

        payload = '<img src=x onerror="alert(1)"> [click](javascript:alert(1))'
        response = DualOutputResponse(
            scenario_summary=payload,
            domain=LegalDomain.CONSUMER_RIGHTS,
            action_plan=[
                ProceduralActionStep(
                    step_number=1, phase=ProceduralPhase.IMMEDIATE_ACTION, title=payload, description=payload,
                    forum_or_authority=payload, limitation_period=payload, priority="<b>high</b>",
                )
            ],
            statutory_citations=[
                StatutoryCitation(
                    act_name=payload, section_number="1", summary_of_provision=payload,
                    applicability_to_scenario=payload, source_url="javascript:alert(1)",
                )
            ],
            precedent_citations=[
                PrecedentCitation(
                    case_title=payload, court=payload, year=2020, legal_principle=payload,
                    source_url="https://main.sci.gov.in/ok",
                )
            ],
            unverified_citations_stripped=[payload],
        )
        render_full_result(payload, response, key_prefix="xss")

    at = AppTest.from_function(page, default_timeout=TIMEOUT_SECONDS).run()
    assert not at.exception

    rendered = [str(e.value) for e in list(at.markdown) + list(at.caption)]
    assert any("onerror" in text for text in rendered)  # The payload is shown ...
    for text in rendered:  # ... but only in escaped form, so it cannot become HTML or a link.
        assert not re.search(r"(?<!\\)<img", text)
        assert not re.search(r"(?<!\\)\]\(javascript:", text)

    link_urls = [e.proto.url for e in at.get("link_button")]
    assert link_urls == ["https://main.sci.gov.in/ok"]


def test_every_graph_node_has_a_progress_message():
    from solvemycase.ui.components.progress import NODE_MESSAGES
    from solvemycase.ui.state import get_engines

    nodes = set(get_engines().proposed.graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes
    assert nodes <= set(NODE_MESSAGES)
