"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

import solvemycase.api.main as api_main
from solvemycase.api.main import app
from solvemycase.config.settings import Settings, get_settings

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _isolated_api_engine(tmp_path_factory):
    """Ensure API tests use an isolated temporary Qdrant directory instead of mutating data/qdrant_storage."""
    tmp_qdrant = tmp_path_factory.mktemp("api_qdrant")
    test_settings = Settings(openai_api_key=None, qdrant_path=tmp_qdrant)
    orig_get_settings = api_main.get_settings
    api_main.get_settings = lambda: test_settings
    api_main._settings = None
    api_main._store = None
    api_main._embedder = None
    api_main._baseline = None
    api_main._proposed = None
    try:
        yield
    finally:
        api_main.get_settings = orig_get_settings
        api_main._settings = None
        api_main._store = None
        api_main._embedder = None
        api_main._baseline = None
        api_main._proposed = None
        get_settings.cache_clear()


def test_health_endpoint():
    """Verify health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "solvemycase"


def test_domains_endpoint():
    """Verify supported domains endpoint."""
    response = client.get("/domains")
    assert response.status_code == 200
    domains = response.json()
    assert len(domains) >= 3
    domain_keys = [d["domain"] for d in domains]
    assert "motor_vehicle_accident" in domain_keys
    assert "property_conflict" in domain_keys
    assert "consumer_rights" in domain_keys


def test_query_baseline_endpoint():
    """Verify baseline query endpoint."""
    payload = {
        "scenario": "A speeding commercial truck hit my two-wheeler causing severe injuries and the driver fled without stopping."
    }
    response = client.post("/query/baseline", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "baseline_vanilla_rag"
    assert "response" in data
    assert len(data["response"]["action_plan"]) > 0


def test_query_proposed_endpoint():
    """Verify proposed agent query endpoint."""
    payload = {
        "scenario": "A builder delayed delivery of my flat by four years despite taking 95 percent payment. Can I get a full refund?"
    }
    response = client.post("/query/proposed", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline"] == "proposed_langgraph_agent"
    assert "response" in data
    assert len(data["response"]["action_plan"]) > 0
    assert data["response"]["hallucination_check_passed"] is True
    assert "coverage_note" in data["response"]  # Optional; set only when no law in the corpus applies.
