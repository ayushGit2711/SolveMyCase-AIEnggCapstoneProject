"""Streamlit entrypoint for solvemycase: page config, navigation, and sidebar.

Run from the parent directory of the package:
    PYTHONPATH=. streamlit run solvemycase/ui/app.py

Page content lives in ``ui/views``; reusable widgets live in ``ui/components``.
"""

import importlib.util
from pathlib import Path
import sys

if importlib.util.find_spec("solvemycase") is None:
    _repo_root = Path(__file__).resolve().parent.parent
    _spec = importlib.util.spec_from_file_location(
        "solvemycase",
        _repo_root / "__init__.py",
        submodule_search_locations=[str(_repo_root)],
    )
    if _spec and _spec.loader:
        _pkg = importlib.util.module_from_spec(_spec)
        sys.modules["solvemycase"] = _pkg
        _spec.loader.exec_module(_pkg)

import streamlit as st

from solvemycase.ui.components.formatting import DISCLAIMER
from solvemycase.ui.state import compute_corpus_stats, get_engines, is_llm_mode
from solvemycase.ui.views import benchmark, compare, get_help, how_it_works

st.set_page_config(
    page_title="solvemycase: Indian legal action plans",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    """Show runtime mode, corpus size, and the disclaimer."""
    engines = get_engines()
    with st.sidebar:
        st.markdown("### ⚖️ solvemycase")
        if is_llm_mode(engines.settings):
            st.success("OpenAI connected", icon="🟢")
        else:
            st.warning("Offline mode: answers use simple fallbacks. Add OPENAI_API_KEY for full quality.", icon="🟡")
        stats = compute_corpus_stats(engines.store)
        st.caption(f"Corpus: {stats['statute_count']} statutory sections · {stats['precedent_count']} judgments")
        st.divider()
        st.caption(DISCLAIMER)


navigation = st.navigation(
    [
        st.Page(get_help.render, title="Get help", icon="🧭", url_path="get-help", default=True),
        st.Page(compare.render, title="Compare A vs B", icon="⚖️", url_path="compare"),
        st.Page(benchmark.render, title="Benchmark", icon="📊", url_path="benchmark"),
        st.Page(how_it_works.render, title="How it works", icon="ℹ️", url_path="how-it-works"),
    ]
)
render_sidebar()
navigation.run()
