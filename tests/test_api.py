"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

from solvemycase.api.main import app

client = TestClient(app)


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
