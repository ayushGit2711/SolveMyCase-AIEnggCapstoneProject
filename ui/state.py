"""Shared Streamlit state: cached engines, runtime mode detection, and live corpus statistics.

Engines are created once per server process via ``st.cache_resource``. Embedded Qdrant allows
only one client per storage folder, so every page must obtain the store through this module
rather than constructing its own.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import streamlit as st

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.data.ingestion.schema import DocumentType, LegalDomain
from solvemycase.data.vectorstore.indexer import EmbeddingProvider, ensure_index_current
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"

_DOMAIN_CARD_SPEC = (
    (
        LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "🚗 Motor Vehicle Accidents",
        "Hit-and-run crashes, rash/negligent driving injuries, MACT compensation claims & insurer claim refusals.",
    ),
    (
        LegalDomain.PROPERTY_CONFLICT,
        "🏠 Property & Tenancy",
        "Unlawful eviction/lockout, tenant overstay after lease expiry, plot/driveway encroachment & injunctions.",
    ),
    (
        LegalDomain.CONSUMER_RIGHTS,
        "🛒 Consumer & Builder Delays",
        "Defective goods/electronics, e-commerce refund refusals, courier/service deficiency & RERA flat delays.",
    ),
    (
        LegalDomain.GENERAL_DISPUTE,
        "🐾 Animal Cruelty & Criminal",
        "Killing, poisoning or maiming pets/street animals (BNS s.325, PCA s.11), cheating, breach of trust & FIRs.",
    ),
)


class Engines(NamedTuple):
    """Container for the long-lived pipeline objects shared by all pages."""

    settings: Settings
    store: QdrantLegalStore
    baseline: VanillaRAGBaseline
    proposed: LegalAgentGraph


@st.cache_resource(show_spinner="Loading legal corpus and models...")
def get_engines() -> Engines:
    """Create (once per process) the vector store, embedder, and both pipelines.

    The store is re-indexed when it is empty or when its corpus fingerprint / embedding model no longer
    matches the corpus shipped with the code, so a redeployed app never serves stale law text or links.
    """
    settings = get_settings()
    store = QdrantLegalStore(settings=settings)
    embedder = EmbeddingProvider(settings=settings)
    ensure_index_current(store, settings=settings, embedder=embedder)
    return Engines(
        settings=settings,
        store=store,
        baseline=VanillaRAGBaseline(settings=settings, store=store, embedder=embedder),
        proposed=LegalAgentGraph(settings=settings, store=store, embedder=embedder),
    )


def is_llm_mode(settings: Settings) -> bool:
    """Return True when an OpenAI key is configured (otherwise offline heuristics are used)."""
    return bool(settings.openai_api_key and settings.openai_api_key.get_secret_value())


def compute_corpus_stats(store: QdrantLegalStore) -> Dict[str, Any]:
    """Summarize the corpus actually loaded in the store (never hard-coded numbers).

    Returns:
        Dict with total counts, per-Act section counts, and precedent titles.
    """
    statutes = [d for d in store.corpus_documents if d.doc_type == DocumentType.STATUTE]
    precedents = [d for d in store.corpus_documents if d.doc_type == DocumentType.PRECEDENT]
    sections_per_act = Counter(d.act_name or "Unknown Act" for d in statutes)
    return {
        "total_documents": len(store.corpus_documents),
        "statute_count": len(statutes),
        "precedent_count": len(precedents),
        "sections_per_act": dict(sorted(sections_per_act.items(), key=lambda kv: (-kv[1], kv[0]))),
        "precedent_titles": sorted(d.title for d in precedents),
    }


def compute_domain_coverage_cards(store: QdrantLegalStore) -> List[Dict[str, Any]]:
    """Build per-domain problem & law coverage summaries from the live Qdrant store."""
    cards: List[Dict[str, Any]] = []
    docs = list(store.corpus_documents)
    for domain, title, problems in _DOMAIN_CARD_SPEC:
        dom_docs = [d for d in docs if d.domain == domain]
        dom_statutes = [d for d in dom_docs if d.doc_type == DocumentType.STATUTE]
        dom_precedents = [d for d in dom_docs if d.doc_type == DocumentType.PRECEDENT]
        act_counts = Counter(d.act_name or "Unknown Act" for d in dom_statutes)
        acts_list = [f"{act} ({cnt})" for act, cnt in sorted(act_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        cards.append(
            {
                "domain": domain.value,
                "title": title,
                "problems": problems,
                "statute_count": len(dom_statutes),
                "precedent_count": len(dom_precedents),
                "acts_summary": ", ".join(acts_list),
            }
        )
    return cards


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_benchmark_dataset() -> List[Dict[str, Any]]:
    """Load the annotated benchmark scenarios (empty list if missing)."""
    return _load_json(EVALUATION_DIR / "benchmark_dataset.json") or []


@st.cache_data
def load_benchmark_results() -> Optional[Dict[str, Any]]:
    """Load the latest comparative benchmark output, if one has been generated."""
    return _load_json(EVALUATION_DIR / "benchmark_results.json")
