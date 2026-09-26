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

_ACT_SHORT_NAMES = {
    "Bharatiya Nyaya Sanhita, 2023": "BNS, 2023",
    "Bharatiya Nagarik Suraksha Sanhita, 2023": "BNSS, 2023",
    "Code of Criminal Procedure, 1973": "CrPC, 1973",
    "Indian Penal Code, 1860": "IPC, 1860",
    "Motor Vehicles Act, 1988": "MV Act, 1988",
    "Transfer of Property Act, 1882": "TPA, 1882",
    "Specific Relief Act, 1963": "SRA, 1963",
    "Code of Civil Procedure, 1908": "CPC, 1908",
    "Consumer Protection Act, 2019": "CPA, 2019",
    "Real Estate (Regulation and Development) Act, 2016": "RERA, 2016",
    "Prevention of Cruelty to Animals Act, 1960": "PCA Act, 1960",
}

_DOMAIN_CARD_SPEC = (
    (
        LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "🚗 Motor Vehicle Accidents",
        (
            "Hit-and-run crashes, rash or drunken driving injuries & FIRs",
            "MACT compensation claims for injury, disability or fatal accidents",
            "Third-party & own-damage motor insurance claim refusals",
        ),
    ),
    (
        LegalDomain.PROPERTY_CONFLICT,
        "🏠 Property & Tenancy",
        (
            "Unlawful eviction, landlord lockouts & summary recovery of possession",
            "Tenant overstay after lease expiry, rent default & eviction notices",
            "Neighbour encroachment, driveway/easement obstruction & civil injunctions",
        ),
    ),
    (
        LegalDomain.CONSUMER_RIGHTS,
        "🛒 Consumer & Builder Delays",
        (
            "Defective goods, electronics or vehicles & warranty/refund refusals",
            "E-commerce, courier, banking & service deficiency complaints",
            "RERA flat possession delays & builder refund with interest",
        ),
    ),
    (
        LegalDomain.GENERAL_DISPUTE,
        "🐾 Animal Cruelty & Criminal",
        (
            "Killing, poisoning or maiming pets & street animals (BNS s.325, PCA s.11)",
            "Police FIR / Zero-FIR registration & Magistrate directions (BNSS s.173/175)",
            "Cheating (BNS s.318), breach of trust (BNS s.316) & mob violence (BNS s.190/191)",
        ),
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
    for domain, title, problems_list in _DOMAIN_CARD_SPEC:
        dom_docs = [d for d in docs if d.domain == domain]
        dom_statutes = [d for d in dom_docs if d.doc_type == DocumentType.STATUTE]
        dom_precedents = [d for d in dom_docs if d.doc_type == DocumentType.PRECEDENT]
        act_counts = Counter(d.act_name or "Unknown Act" for d in dom_statutes)
        sorted_acts = sorted(act_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        acts_list = [f"{act} ({cnt})" for act, cnt in sorted_acts]
        act_badges = [f"{_ACT_SHORT_NAMES.get(act, act)} ({cnt})" for act, cnt in sorted_acts]
        cards.append(
            {
                "domain": domain.value,
                "title": title,
                "problems_list": list(problems_list),
                "problems": " · ".join(problems_list),
                "statute_count": len(dom_statutes),
                "precedent_count": len(dom_precedents),
                "act_badges": act_badges,
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
