"""Unit tests for QdrantLegalStore, BM25 indexing, and hybrid retrieval."""

import pytest
from solvemycase.config.settings import Settings
from solvemycase.data.ingestion.normalizer import load_legal_precedents, load_legal_provisions_from_parquet
from solvemycase.data.ingestion.schema import DocumentType, LegalDomain
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore
from pathlib import Path


def test_embedding_provider_deterministic():
    """Verify that EmbeddingProvider generates vectors with correct shape and normalized values."""
    provider = EmbeddingProvider(dim=1536)
    embeddings = provider.get_embeddings(["Section 106 BNS rash driving", "Tenant eviction notice"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1536
    assert len(embeddings[1]) == 1536

    # Test determinism
    repeat = provider.get_embeddings(["Section 106 BNS rash driving"])
    assert embeddings[0] == repeat[0]


def test_qdrant_hybrid_search_in_memory():
    """Test full in-memory Qdrant indexing and hybrid BM25 + dense search."""
    store = QdrantLegalStore(vector_dim=1536, in_memory=True)
    provider = EmbeddingProvider(dim=1536)

    provisions = load_legal_provisions_from_parquet(Path("/nonexistent"))
    precedents = load_legal_precedents()

    prov_texts = [f"{p.act_name} Section {p.section_number}: {p.text}" for p in provisions]
    prec_texts = [f"{pr.title} {pr.citation or ''}: {pr.text}" for pr in precedents]

    prov_embeddings = provider.get_embeddings(prov_texts)
    prec_embeddings = provider.get_embeddings(prec_texts)

    count = store.index_provisions_and_precedents(
        provisions=provisions,
        precedents=precedents,
        provision_embeddings=prov_embeddings,
        precedent_embeddings=prec_embeddings,
    )
    assert count == len(provisions) + len(precedents)

    # 1. Test Motor Vehicle Accident search
    mva_query = "compensation claim fatal car accident Section 166 Motor Vehicles Act Sarla Verma"
    mva_embedding = provider.get_embeddings([mva_query])[0]
    mva_results = store.hybrid_search(
        query_text=mva_query,
        query_embedding=mva_embedding,
        top_k=3,
        domain_filter=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    )

    assert len(mva_results) > 0
    top_result = mva_results[0]
    assert top_result.score > 0
    assert top_result.source_url.startswith("https://")
    # Verify top result is related to Motor Vehicles Act or Sarla Verma
    matched_text = f"{top_result.title} {top_result.text}".lower()
    assert "motor vehicles" in matched_text or "sarla verma" in matched_text or "accident" in matched_text

    # 2. Test Property Conflict search
    prop_query = "unauthorized dispossession suit under Section 6 Specific Relief Act injunction"
    prop_embedding = provider.get_embeddings([prop_query])[0]
    prop_results = store.hybrid_search(
        query_text=prop_query,
        query_embedding=prop_embedding,
        top_k=3,
        domain_filter=LegalDomain.PROPERTY_CONFLICT,
    )
    assert len(prop_results) > 0
    assert any("specific relief" in r.title.lower() or "dispossessed" in r.text.lower() for r in prop_results)

    # 3. Test Consumer Rights search
    cpa_query = "deficiency in service defective product complaint before District Commission Section 35"
    cpa_embedding = provider.get_embeddings([cpa_query])[0]
    cpa_results = store.hybrid_search(
        query_text=cpa_query,
        query_embedding=cpa_embedding,
        top_k=3,
        domain_filter=LegalDomain.CONSUMER_RIGHTS,
    )
    assert len(cpa_results) > 0
    assert any("consumer protection" in r.title.lower() or "deficiency" in r.text.lower() for r in cpa_results)


def test_reopened_store_rehydrates_corpus_from_disk(tmp_path):
    """A store opened by a different process than the indexer (e.g. UI/API) must still retrieve.

    Regression test: the BM25 index and chunk registry used to exist only in the indexing
    instance, so a reopened store returned zero hits and every citation was stripped.
    """
    # Isolated settings: temp Qdrant dir, no .env, no API key -> deterministic offline embeddings.
    settings = Settings(_env_file=None, QDRANT_PATH=str(tmp_path / "qdrant"), OPENAI_API_KEY=None)
    provider = EmbeddingProvider(settings=settings, dim=1536)

    provisions = load_legal_provisions_from_parquet(Path("/nonexistent"))
    precedents = load_legal_precedents()

    indexing_store = QdrantLegalStore(settings=settings, vector_dim=1536)
    indexing_store.index_provisions_and_precedents(
        provisions=provisions,
        precedents=precedents,
        provision_embeddings=provider.get_embeddings([f"{p.act_name} Section {p.section_number}: {p.text}" for p in provisions]),
        precedent_embeddings=provider.get_embeddings([f"{pr.title}: {pr.text}" for pr in precedents]),
    )
    indexing_store.client.close()  # Embedded Qdrant allows a single client per storage folder.

    reopened_store = QdrantLegalStore(settings=settings, vector_dim=1536)

    assert len(reopened_store.corpus_documents) == len(provisions) + len(precedents)
    assert reopened_store.bm25_index is not None

    query = "compensation claim Section 166 Motor Vehicles Act"
    hits = reopened_store.hybrid_search(query_text=query, query_embedding=provider.get_embeddings([query])[0], top_k=5)
    assert len(hits) > 0

    regrounded = reopened_store.exact_section_search("166", "Motor Vehicles Act, 1988")
    assert len(regrounded) == 1
    assert regrounded[0].act_name == "Motor Vehicles Act, 1988"
