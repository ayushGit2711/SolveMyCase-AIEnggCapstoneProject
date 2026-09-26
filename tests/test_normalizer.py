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
    assert len(provisions) >= 45

    # Ensure key Acts are represented, including the Prevention of Cruelty to Animals Act, 1960
    acts = {p.act_name for p in provisions}
    assert any("Motor Vehicles Act" in a for a in acts)
    assert any("Bharatiya Nyaya Sanhita" in a for a in acts)
    assert any("Consumer Protection Act" in a for a in acts)
    assert any("Transfer of Property Act" in a for a in acts)
    assert "Prevention of Cruelty to Animals Act, 1960" in acts

    by_id = {p.doc_id: p for p in provisions}
    # BNS 147/149 mislabelled entries were replaced by BNS 190/191
    assert "central_bns_2023_sec_147" not in by_id
    assert "central_bns_2023_sec_149" not in by_id
    assert by_id["central_bns_2023_sec_190"].section_number == "190"
    assert by_id["central_bns_2023_sec_191"].section_number == "191"
    assert by_id["central_bns_2023_sec_325"].section_number == "325"
    assert by_id["central_pca_1960_sec_11"].act_category == "criminal"
    assert by_id["central_mva_1988_sec_147"].section_number == "147"
    assert by_id["central_mva_1988_sec_150"].section_number == "150"
    assert by_id["central_sra_1963_sec_10"].section_number == "10"

    # Repealed IPC/CrPC entries carry explicit repealed markers
    for repealed_id in ("central_ipc_1860_sec_279", "central_ipc_1860_sec_304a", "central_crpc_1973_sec_154"):
        assert "repealed" in (by_id[repealed_id].title or "").lower()

    # Criminal provisions have domain=GENERAL_DISPUTE so every domain's search sees them
    for p in provisions:
        if p.act_category == "criminal":
            assert p.domain == LegalDomain.GENERAL_DISPUTE

    # Ensure all provisions have traceable source URLs
    for p in provisions:
        assert p.source_url.startswith("https://")
        assert len(p.text) > 30


def test_load_landmark_precedents():
    """Verify landmark precedents structure and official court citations."""
    precedents = load_legal_precedents()
    assert len(precedents) >= 9

    # Verify domains coverage, including general_dispute (AWBI v. A. Nagaraja)
    domains = {p.domain for p in precedents}
    assert LegalDomain.MOTOR_VEHICLE_ACCIDENT in domains
    assert LegalDomain.PROPERTY_CONFLICT in domains
    assert LegalDomain.CONSUMER_RIGHTS in domains
    assert LegalDomain.GENERAL_DISPUTE in domains
    assert any("Nagaraja" in p.title for p in precedents)

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
