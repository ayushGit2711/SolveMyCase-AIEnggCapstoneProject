"""Structured JSONL runtime telemetry logger for SolveMyCase inference monitoring.

Records every live pipeline execution (domain, guardrail outcome, RetrievalQualityGate activation,
verified vs. stripped citation counts, and latency) to a local JSONL file so drift in
unverified-citation strip rates can be monitored over time. All filesystem writes are wrapped in
try/except OSError so read-only environments never fail inference.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import (
    DualOutputResponse,
    ExecutionTrace,
    stripped_unverified_citations,
)


def log_inference_event(
    response: DualOutputResponse,
    trace: ExecutionTrace,
    latency_seconds: Optional[float] = None,
    *,
    latency_ms: Optional[float] = None,
    scenario: Optional[str] = None,
    settings: Optional[Settings] = None,
    log_path: Optional[Path] = None,
    path: Optional[Path] = None,
) -> None:
    """Append a single inference telemetry event to the JSONL log file.

    Never raises OSError on unwritable directories.
    """
    cfg = settings or get_settings()
    target = log_path or path or cfg.resolve_path(cfg.telemetry_log_path)
    stripped_real = stripped_unverified_citations(response)
    ranked_contexts = getattr(trace, "candidate_contexts", None) or trace.retrieved_contexts
    top_score = round(float(ranked_contexts[0].score), 4) if ranked_contexts else 0.0
    if latency_ms is not None and latency_seconds is None:
        latency_seconds = float(latency_ms) / 1000.0
    elif latency_seconds is not None and latency_ms is None:
        latency_ms = float(latency_seconds) * 1000.0
    else:
        latency_seconds = float(latency_seconds or 0.0)
        latency_ms = float(latency_ms or 0.0)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_type": trace.pipeline_type,
        "domain": response.domain.value,
        "scenario_preview": (scenario or response.scenario_summary)[:120],
        "is_legal": "handle_rejection" not in trace.nodes_visited,
        "criminal_route_triggered": trace.criminal_route_triggered,
        "retrieval_gate_triggered": trace.retrieval_gate_triggered,
        "coverage_gap": bool(getattr(trace, "coverage_gap", False)),
        "coverage_gap_reason": getattr(trace, "coverage_gap_reason", None),
        "applicability_mode": getattr(trace, "applicability_mode", None),
        "top_retrieval_score": top_score,
        "verified_citations": len(response.statutory_citations) + len(response.precedent_citations),
        "stripped_citations": len(stripped_real),
        "latency_s": round(float(latency_seconds), 4),
        "latency_ms": round(float(latency_ms), 2),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        # Read-only or sandboxed filesystems must never interrupt user-facing inference.
        return


def load_telemetry_events(
    settings: Optional[Settings] = None,
    log_path: Optional[Path] = None,
    path: Optional[Path] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Read recent telemetry events from the JSONL log file (empty list if missing/unreadable)."""
    cfg = settings or get_settings()
    target = log_path or path or cfg.resolve_path(cfg.telemetry_log_path)
    try:
        if not target.exists():
            return []
        with open(target, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        events = [json.loads(line) for line in lines[-limit:]]
        return events
    except (OSError, json.JSONDecodeError):
        return []
