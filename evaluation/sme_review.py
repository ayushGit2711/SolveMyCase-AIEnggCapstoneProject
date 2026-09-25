"""Subject-Matter Expert (SME) Review & Judge Calibration Store (`P3 Slice 2` & `DF-6`).

Supports:
- Structured human legal expert annotations (`SMEAnnotation`) stored in `evaluation/sme_annotations.json`.
- Atomic file persistence (`os.replace`) so concurrent UI writes never corrupt JSON (`DF-6`).
- Inter-annotator / Judge-vs-Human agreement metrics: Mean Absolute Error (MAE), Pearson r, Cohen's kappa, and within-1pt agreement rate.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


DEFAULT_SME_ANNOTATIONS_PATH = Path(__file__).parent / "sme_annotations.json"


class SMEAnnotation(BaseModel):
    """Single human legal expert rating of a pipeline output on a benchmark scenario."""
    scenario_id: str
    pipeline_type: str = Field(..., description="'baseline' or 'proposed'")
    annotator_id: str = Field(default="legal_sme_01")
    statutory_accuracy: float = Field(..., ge=0.0, le=5.0)
    procedural_actionability: float = Field(..., ge=0.0, le=5.0)
    forum_appropriateness: float = Field(..., ge=0.0, le=5.0)
    hallucination_freedom: float = Field(..., ge=0.0, le=5.0)
    coherence_and_specificity: float = Field(default=4.0, ge=0.0, le=5.0)
    overall_score: float = Field(..., ge=0.0, le=5.0)
    notes: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_sme_annotations(path: Optional[Path] = None) -> List[SMEAnnotation]:
    """Load SME annotations from disk."""
    target = path or DEFAULT_SME_ANNOTATIONS_PATH
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        return [SMEAnnotation(**item) for item in raw if isinstance(item, dict)]
    except Exception:
        return []


def save_sme_annotation(
    annotation: SMEAnnotation,
    path: Optional[Path] = None,
) -> List[SMEAnnotation]:
    """Upsert an SMEAnnotation by (scenario_id, pipeline_type, annotator_id) using atomic os.replace."""
    target = path or DEFAULT_SME_ANNOTATIONS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = load_sme_annotations(target)

    updated: List[SMEAnnotation] = []
    replaced = False
    for item in existing:
        if (
            item.scenario_id == annotation.scenario_id
            and item.pipeline_type == annotation.pipeline_type
            and item.annotator_id == annotation.annotator_id
        ):
            updated.append(annotation)
            replaced = True
        else:
            updated.append(item)
    if not replaced:
        updated.append(annotation)

    payload = [a.model_dump() for a in updated]
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=".sme_tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(target))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return updated


def compute_judge_human_agreement(
    judge_scores: List[float],
    human_scores: List[float],
    pass_threshold: float = 3.5,
) -> Dict[str, Any]:
    """Compute MAE, Pearson r, binary pass/fail Cohen's kappa, and within-1.0pt agreement rate."""
    n = min(len(judge_scores), len(human_scores))
    if n == 0:
        return {
            "sample_count": 0,
            "mae": 0.0,
            "pearson_r": 0.0,
            "cohens_kappa": 0.0,
            "within_1pt_rate": 0.0,
        }

    j_vals = [float(x) for x in judge_scores[:n]]
    h_vals = [float(y) for y in human_scores[:n]]

    abs_diffs = [abs(j - h) for j, h in zip(j_vals, h_vals)]
    mae = round(sum(abs_diffs) / n, 4)
    within_1pt_rate = round(sum(1 for d in abs_diffs if d <= 1.0) / n, 4)

    # Pearson correlation r
    mean_j = sum(j_vals) / n
    mean_h = sum(h_vals) / n
    num = sum((j - mean_j) * (h - mean_h) for j, h in zip(j_vals, h_vals))
    den_j = math.sqrt(sum((j - mean_j) ** 2 for j in j_vals))
    den_h = math.sqrt(sum((h - mean_h) ** 2 for h in h_vals))
    if den_j > 1e-9 and den_h > 1e-9:
        pearson_r = round(max(-1.0, min(1.0, num / (den_j * den_h))), 4)
    else:
        pearson_r = 1.0 if mae < 0.5 else 0.0

    # Binary pass/fail Cohen's kappa at `pass_threshold`
    tp = sum(1 for j, h in zip(j_vals, h_vals) if j >= pass_threshold and h >= pass_threshold)
    tn = sum(1 for j, h in zip(j_vals, h_vals) if j < pass_threshold and h < pass_threshold)
    fp = sum(1 for j, h in zip(j_vals, h_vals) if j >= pass_threshold and h < pass_threshold)
    fn = sum(1 for j, h in zip(j_vals, h_vals) if j < pass_threshold and h >= pass_threshold)

    p_o = (tp + tn) / n
    p_j_pos = (tp + fp) / n
    p_h_pos = (tp + fn) / n
    p_e = (p_j_pos * p_h_pos) + ((1.0 - p_j_pos) * (1.0 - p_h_pos))

    if abs(1.0 - p_e) > 1e-9:
        kappa = round((p_o - p_e) / (1.0 - p_e), 4)
    else:
        kappa = 1.0 if p_o == 1.0 else 0.0

    return {
        "sample_count": n,
        "mae": mae,
        "pearson_r": pearson_r,
        "cohens_kappa": kappa,
        "within_1pt_rate": within_1pt_rate,
    }
