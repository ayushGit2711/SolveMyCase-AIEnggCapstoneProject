"""FastAPI application for solvemycase Indian Legal Advisory Engine.

Exposes REST endpoints for:
- Health and status (/health)
- Baseline Vanilla RAG query execution (/query/baseline)
- Proposed LangGraph agentic query execution (/query/proposed)
- Side-by-side comparative execution (/compare)
- Supported legal domains & metadata (/domains)
- Latest benchmark summary (/benchmark/summary)
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.baseline.vanilla_rag import VanillaRAGBaseline
from solvemycase.core.proposed.graph import LegalAgentGraph
from solvemycase.data.ingestion.schema import DualOutputResponse, LegalDomain
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore
from solvemycase.evaluation.metrics import (
    compute_citation_grounding_metrics,
    compute_procedural_completeness,
)

app = FastAPI(
    title="solvemycase — Indian Legal Advisory API",
    description="Deterministic Indian statutory mapping and procedural dispute roadmap engine.",
    version="1.0.0",
)

# Enable CORS for Streamlit / external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singletons
_settings: Optional[Settings] = None
_store: Optional[QdrantLegalStore] = None
_embedder: Optional[EmbeddingProvider] = None
_baseline: Optional[VanillaRAGBaseline] = None
_proposed: Optional[LegalAgentGraph] = None


def get_engine():
    """Lazy initialize engine instances."""
    global _settings, _store, _embedder, _baseline, _proposed
    if _settings is None:
        _settings = get_settings()
        _store = QdrantLegalStore(settings=_settings)
        _embedder = EmbeddingProvider(settings=_settings)
        _baseline = VanillaRAGBaseline(settings=_settings, store=_store, embedder=_embedder)
        _proposed = LegalAgentGraph(settings=_settings, store=_store, embedder=_embedder)
    return _baseline, _proposed


class LegalQueryRequest(BaseModel):
    """User input scenario for legal consultation."""
    scenario: str = Field(..., min_length=10, description="Factual description of the legal problem.")


class QueryExecutionResult(BaseModel):
    """Pipeline execution result with execution latency and audit metadata."""
    pipeline: str
    latency_seconds: float
    response: DualOutputResponse
    grounding_metrics: Dict[str, float]
    procedural_metrics: Dict[str, Any]


class ComparativeResponse(BaseModel):
    """Side-by-side comparison of Baseline vs Proposed approaches."""
    scenario: str
    baseline: QueryExecutionResult
    proposed: QueryExecutionResult


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "service": "solvemycase",
        "version": "1.0.0",
        "vector_store": "qdrant_embedded",
    }


@app.get("/domains")
def get_supported_domains() -> List[Dict[str, str]]:
    """List of supported dispute domains."""
    return [
        {"domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT.value, "label": "Motor Vehicle Accidents & Negligence"},
        {"domain": LegalDomain.PROPERTY_CONFLICT.value, "label": "Property Conflicts & Injunctions"},
        {"domain": LegalDomain.CONSUMER_RIGHTS.value, "label": "Consumer Protection & Deficiency in Service"},
        {"domain": LegalDomain.GENERAL_DISPUTE.value, "label": "General Civil & Criminal Disputes"},
    ]


@app.post("/query/baseline", response_model=QueryExecutionResult)
def query_baseline(req: LegalQueryRequest) -> QueryExecutionResult:
    """Execute Approach A: Baseline Vanilla RAG."""
    baseline_engine, _ = get_engine()
    t0 = time.perf_counter()
    try:
        response = baseline_engine.run(req.scenario)
        latency = round(time.perf_counter() - t0, 4)
        grounding = compute_citation_grounding_metrics(response)
        procedural = compute_procedural_completeness(response)

        return QueryExecutionResult(
            pipeline="baseline_vanilla_rag",
            latency_seconds=latency,
            response=response,
            grounding_metrics=grounding,
            procedural_metrics=procedural,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Baseline execution error: {str(err)}")


@app.post("/query/proposed", response_model=QueryExecutionResult)
def query_proposed(req: LegalQueryRequest) -> QueryExecutionResult:
    """Execute Approach B: Proposed LangGraph Multi-Agent RAG."""
    _, proposed_engine = get_engine()
    t0 = time.perf_counter()
    try:
        response = proposed_engine.run(req.scenario)
        latency = round(time.perf_counter() - t0, 4)
        grounding = compute_citation_grounding_metrics(response)
        procedural = compute_procedural_completeness(response)

        return QueryExecutionResult(
            pipeline="proposed_langgraph_agent",
            latency_seconds=latency,
            response=response,
            grounding_metrics=grounding,
            procedural_metrics=procedural,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Proposed agent execution error: {str(err)}")


@app.post("/compare", response_model=ComparativeResponse)
def compare_pipelines(req: LegalQueryRequest) -> ComparativeResponse:
    """Execute both pipelines concurrently on the same scenario and return side-by-side comparison."""
    baseline_res = query_baseline(req)
    proposed_res = query_proposed(req)

    return ComparativeResponse(
        scenario=req.scenario,
        baseline=baseline_res,
        proposed=proposed_res,
    )


@app.get("/benchmark/summary")
def get_benchmark_summary() -> Dict[str, Any]:
    """Retrieve latest benchmark performance summary metrics."""
    res_path = Path(__file__).parent.parent / "evaluation" / "benchmark_results.json"
    if not res_path.exists():
        return {"status": "no_benchmark_run_yet", "message": "Run evaluation/comparative_runner.py to generate."}

    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("summary", {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("solvemycase.api.main:app", host="0.0.0.0", port=8000, reload=True)
