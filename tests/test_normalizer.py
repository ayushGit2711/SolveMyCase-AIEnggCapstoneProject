"""Unit tests for corpus normalization and fallback mechanisms."""

from pathlib import Path
from solvemycase.data.ingestion.normalizer import (
    load_legal_provisions_from_parquet,
    load_legal_precedents,
    load_unified_corpus,
)
from solvemycase.data.ingestion.schema import LegalDomain


def test_load_fallback_provisions():
    """Verify that fallback provisions load correctly when no parquet is available."""
    provisions = load_legal_provisions_from_parquet(Path("/nonexistent/file.parquet"))
    assert len(provisions) >= 10

    # Ensure key Acts are represented
    acts = {p.act_name for p in provisions}
    assert any("Motor Vehicles Act" in a for a in acts)
    assert any("Bharatiya Nyaya Sanhita" in a for a in acts)
    assert any("Consumer Protection Act" in a for a in acts)
    assert any("Transfer of Property Act" in a for a in acts)

    # Ensure all provisions have traceable source URLs
    for p in provisions:
        assert p.source_url.startswith("https://")
        assert len(p.text) > 30


def test_load_landmark_precedents():
    """Verify landmark precedents structure and official court citations."""
    precedents = load_legal_precedents()
    assert len(precedents) >= 6

    # Verify domains coverage
    domains = {p.domain for p in precedents}
    assert LegalDomain.MOTOR_VEHICLE_ACCIDENT in domains
    assert LegalDomain.PROPERTY_CONFLICT in domains
    assert LegalDomain.CONSUMER_RIGHTS in domains

    for p in precedents:
        assert p.court == "Supreme Court of India"
        assert p.citation is not None
        assert p.source_url.startswith("https://")


def test_load_unified_corpus():
    """Verify unified corpus loads both statutes and precedents."""
    corpus = load_unified_corpus(Path("/nonexistent/file.parquet"))
    assert "provisions" in corpus
    assert "precedents" in corpus
    assert len(corpus["provisions"]) > 0
    assert len(corpus["precedents"]) > 0
