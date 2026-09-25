"""Decomposed, rubric-anchored LLM-as-a-Judge and programmatic evaluation engine.

Scores dual-output responses across five criteria (0.0 to 5.0):
1. statutory_accuracy: Precision & completeness of cited Indian Acts/sections against benchmark expectations.
2. procedural_actionability: Chronological clarity, concrete authorities, and coverage of critical steps.
3. forum_appropriateness: Proper jurisdiction and forum (MACT, DCDRC/SCDRC, Civil Court, Magistrate, etc.).
4. hallucination_freedom: Post-output citation grounding (5.0 for 0% ungrounded final citations; 0.0 for fabricated laws).
5. coherence_and_specificity: Tailoring to scenario facts, limitation periods, and actionable clarity.

Features:
- Reasoning-before-score JSON schema with explicit 0-5 rubric anchors.
- Decomposed parallel per-criterion evaluation via ThreadPoolExecutor(max_workers=5) with per-criterion fallback isolation.
- Multi-sample variance tracking (`num_samples`) and automatic `flagged_for_human_review` when variance > 0.5.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import statistics
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import DualOutputResponse, RetrievedContext
from solvemycase.evaluation.metrics import (
    compute_citation_grounding_metrics,
    compute_ground_truth_alignment,
    is_guardrail_rejection,
)


class JudgeScore(BaseModel):
    """Evaluation score breakdown produced by the decomposed LLM-as-a-Judge."""
    statutory_accuracy: float = Field(..., ge=0.0, le=5.0)
    procedural_actionability: float = Field(..., ge=0.0, le=5.0)
    forum_appropriateness: float = Field(..., ge=0.0, le=5.0)
    hallucination_freedom: float = Field(..., ge=0.0, le=5.0)
    coherence_and_specificity: float = Field(default=4.0, ge=0.0, le=5.0)
    overall_score: float = Field(..., ge=0.0, le=5.0)
    score_variance: float = Field(default=0.0, ge=0.0)
    flagged_for_human_review: bool = Field(default=False)
    rubric_checks: Dict[str, bool] = Field(default_factory=dict)
    reasoning: str


CRITERION_RUBRICS: Dict[str, str] = {
    "statutory_accuracy": (
        "Evaluate STATUTORY ACCURACY (0.0 to 5.0).\n"
        "Anchor 5.0: Cites the exact applicable Indian Act(s) and gold expected sections with accurate statutory summaries.\n"
        "Anchor 3.0: Cites the right Act and at least one relevant section, but misses key companion sections.\n"
        "Anchor 1.0: Cites tangential or generic provisions only.\n"
        "Anchor 0.0: Cites zero statutes on a valid legal dispute or cites completely wrong domain statutes."
    ),
    "procedural_actionability": (
        "Evaluate PROCEDURAL ACTIONABILITY (0.0 to 5.0).\n"
        "Anchor 5.0: Covers 5-6 chronological procedural phases (immediate action, police/admin, evidence, notice, forum filing, limitation) and addresses all benchmark critical steps.\n"
        "Anchor 3.0: Covers 3-4 phases with basic steps but omits key evidentiary or limitation timelines.\n"
        "Anchor 1.0: Vague or unsequenced advice lacking concrete steps.\n"
        "Anchor 0.0: Empty action plan on a valid legal query."
    ),
    "forum_appropriateness": (
        "Evaluate FORUM APPROPRIATENESS (0.0 to 5.0).\n"
        "Anchor 5.0: Explicitly designates the exact statutory tribunal/court (e.g., MACT, DCDRC/SCDRC, Civil Judge/Court, Judicial Magistrate) matching benchmark expected_forum.\n"
        "Anchor 3.0: Mentions a generic court/authority without specifying the specialized statutory forum.\n"
        "Anchor 0.0: Directs the user to an improper forum or omits forum guidance entirely."
    ),
    "hallucination_freedom": (
        "Evaluate HALLUCINATION FREEDOM (0.0 to 5.0) strictly on the FINAL delivered citations (do NOT penalize draft citations that were already stripped prior to output).\n"
        "Anchor 5.0: 100% of final delivered statutory and precedent citations are grounded in retrieved legal text (0% post-output hallucination rate).\n"
        "Anchor 2.5: Mostly grounded citations, with 1 minor unverified section reference.\n"
        "Anchor 0.0: Contains fabricated/ungrounded sections or emits zero citations on a valid legal scenario."
    ),
    "coherence_and_specificity": (
        "Evaluate COHERENCE AND SPECIFICITY (0.0 to 5.0).\n"
        "Anchor 5.0: Tailors every step to the user's specific facts, parties, injuries/property/defect details, and statutory limitation periods.\n"
        "Anchor 3.0: Moderately tailored but relies partly on boilerplate templates.\n"
        "Anchor 1.0: Generic one-size-fits-all response."
    ),
}


class LegalLLMJudge:
    """Evaluates legal advisory responses using decomposed rubric-anchored LLM judging and deterministic audits."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_key = self.settings.openai_api_key.get_secret_value() if self.settings.openai_api_key else None
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def judge_response(
        self,
        scenario: str,
        expected_meta: Dict[str, Any],
        response: DualOutputResponse,
        retrieved_contexts: Optional[List[RetrievedContext]] = None,
        store: Optional[Any] = None,
        num_samples: Optional[int] = None,
    ) -> JudgeScore:
        """Score a DualOutputResponse against scenario requirements and deterministic rubric anchors."""
        is_legal = bool(expected_meta.get("is_legal", True))

        # 1. Out-of-scope / Guardrail rejection evaluation (DF-5: 0.0 when non-legal query is not rejected)
        if not is_legal:
            is_rejected = is_guardrail_rejection(response)
            score = 5.0 if is_rejected else 0.0
            return JudgeScore(
                statutory_accuracy=score,
                procedural_actionability=score,
                forum_appropriateness=score,
                hallucination_freedom=score,
                coherence_and_specificity=score,
                overall_score=score,
                score_variance=0.0,
                flagged_for_human_review=not is_rejected,
                rubric_checks={"guardrail_rejected_out_of_scope": is_rejected},
                reasoning=(
                    "Out-of-scope query was accurately intercepted and rejected by the guardrail."
                    if is_rejected
                    else "Pipeline failed to reject an out-of-scope non-legal query (0.0 across all dimensions)."
                ),
            )

        # 2. Deterministic audit baselines (always computed to anchor rubric_checks and isolate fallbacks)
        prog_score = self._judge_programmatic(
            expected_meta=expected_meta,
            response=response,
            retrieved_contexts=retrieved_contexts,
            store=store,
        )

        if not self.client:
            return prog_score

        samples = max(1, num_samples if num_samples is not None else self.settings.eval_judge_samples)
        try:
            sample_scores: List[JudgeScore] = []
            for sample_idx in range(samples):
                temp = 0.0 if sample_idx == 0 else 0.2
                s_score = self._judge_with_openai_decomposed(
                    scenario=scenario,
                    expected_meta=expected_meta,
                    response=response,
                    programmatic_baseline=prog_score,
                    temperature=temp,
                )
                sample_scores.append(s_score)

            if len(sample_scores) == 1:
                return sample_scores[0]

            # Aggregate multi-sample judge runs via median and compute variance
            med_stat = round(statistics.median([s.statutory_accuracy for s in sample_scores]), 2)
            med_proc = round(statistics.median([s.procedural_actionability for s in sample_scores]), 2)
            med_forum = round(statistics.median([s.forum_appropriateness for s in sample_scores]), 2)
            med_hall = round(statistics.median([s.hallucination_freedom for s in sample_scores]), 2)
            med_coh = round(statistics.median([s.coherence_and_specificity for s in sample_scores]), 2)
            overall_list = [s.overall_score for s in sample_scores]
            variance = round(statistics.pvariance(overall_list), 4) if len(overall_list) > 1 else 0.0
            overall = round((med_stat + med_proc + med_forum + med_hall + med_coh) / 5.0, 2)

            return JudgeScore(
                statutory_accuracy=med_stat,
                procedural_actionability=med_proc,
                forum_appropriateness=med_forum,
                hallucination_freedom=med_hall,
                coherence_and_specificity=med_coh,
                overall_score=overall,
                score_variance=variance,
                flagged_for_human_review=bool(variance > 0.5 or overall < 3.0),
                rubric_checks=prog_score.rubric_checks,
                reasoning=sample_scores[0].reasoning,
            )
        except Exception as err:
            print(f"[LLMJudge] OpenAI decomposed judge failed ({err}). Returning deterministic rubric evaluation.")
            return prog_score

    def _score_single_criterion(
        self,
        criterion: str,
        payload: Dict[str, Any],
        fallback_value: float,
        temperature: float = 0.0,
    ) -> Tuple[float, str]:
        """Evaluate a single criterion with reasoning-before-score JSON output."""
        rubric = CRITERION_RUBRICS[criterion]
        system_prompt = (
            "You are a Senior High Court Advocate evaluating one specific dimension of an Indian legal AI response.\n"
            f"{rubric}\n\n"
            "You MUST think step-by-step and output a JSON object with keys in this exact order:\n"
            '{"reasoning": "<1-2 sentence justification citing concrete evidence>", '
            '"checklist_passed": true, '
            '"score": <float between 0.0 and 5.0>}'
        )
        completion = self.client.chat.completions.create(
            model=self.settings.openai_model_fast,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(completion.choices[0].message.content)
        raw_score = float(data.get("score", fallback_value))
        clamped = round(max(0.0, min(5.0, raw_score)), 2)
        reason = str(data.get("reasoning", f"{criterion}={clamped}"))
        return clamped, reason

    def _judge_with_openai_decomposed(
        self,
        scenario: str,
        expected_meta: Dict[str, Any],
        response: DualOutputResponse,
        programmatic_baseline: JudgeScore,
        temperature: float = 0.0,
    ) -> JudgeScore:
        """Run 5 parallel single-criterion LLM judge evaluations with deterministic anchor blending."""
        payload = {
            "scenario": scenario,
            "expected_ground_truth": {
                "expected_act": expected_meta.get("expected_act"),
                "expected_sections": expected_meta.get("expected_sections", []),
                "expected_forum": expected_meta.get("expected_forum", []),
                "critical_steps": expected_meta.get("critical_steps", []),
            },
            "deterministic_audit": programmatic_baseline.rubric_checks,
            "model_final_response": {
                "summary": response.scenario_summary,
                "domain": response.domain.value,
                "action_steps": [
                    {
                        "step": s.step_number,
                        "phase": s.phase.value,
                        "title": s.title,
                        "description": s.description,
                        "forum": s.forum_or_authority,
                        "statutory_basis": s.statutory_basis,
                        "limitation_period": s.limitation_period,
                    }
                    for s in response.action_plan
                ],
                "final_statutory_citations": [
                    {"act": sc.act_name, "section": sc.section_number, "summary": sc.summary_of_provision}
                    for sc in response.statutory_citations
                ],
                "final_precedent_citations": [
                    {"case_title": pc.case_title, "court": pc.court, "principle": pc.legal_principle}
                    for pc in response.precedent_citations
                ],
            },
        }

        fallback_map = {
            "statutory_accuracy": programmatic_baseline.statutory_accuracy,
            "procedural_actionability": programmatic_baseline.procedural_actionability,
            "forum_appropriateness": programmatic_baseline.forum_appropriateness,
            "hallucination_freedom": programmatic_baseline.hallucination_freedom,
            "coherence_and_specificity": programmatic_baseline.coherence_and_specificity,
        }

        results: Dict[str, float] = {}
        reasons: List[str] = []

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                crit: pool.submit(
                    self._score_single_criterion,
                    crit,
                    payload,
                    fallback_map[crit],
                    temperature,
                )
                for crit in CRITERION_RUBRICS
            }
            for crit, fut in futures.items():
                try:
                    sc, rsn = fut.result(timeout=25.0)
                    results[crit] = sc
                    reasons.append(f"[{crit}] {rsn}")
                except Exception as exc:
                    print(f"[LLMJudge] Worker '{crit}' failed ({exc}); using deterministic anchor.")
                    results[crit] = fallback_map[crit]
                    reasons.append(f"[{crit}] Fallback deterministic score={fallback_map[crit]}")

        # Anchor hallucination_freedom to deterministic post-output citation grounding so LLM cannot hallucinate a penalty
        results["hallucination_freedom"] = round(
            0.6 * programmatic_baseline.hallucination_freedom + 0.4 * results["hallucination_freedom"], 2
        )

        criterion_values = [
            results["statutory_accuracy"],
            results["procedural_actionability"],
            results["forum_appropriateness"],
            results["hallucination_freedom"],
            results["coherence_and_specificity"],
        ]
        overall = round(sum(criterion_values) / 5.0, 2)
        spread = round(statistics.pvariance(criterion_values), 4)

        return JudgeScore(
            statutory_accuracy=results["statutory_accuracy"],
            procedural_actionability=results["procedural_actionability"],
            forum_appropriateness=results["forum_appropriateness"],
            hallucination_freedom=results["hallucination_freedom"],
            coherence_and_specificity=results["coherence_and_specificity"],
            overall_score=overall,
            score_variance=spread,
            flagged_for_human_review=bool(overall < 3.0 or not programmatic_baseline.rubric_checks.get("zero_post_output_hallucinations", True)),
            rubric_checks=programmatic_baseline.rubric_checks,
            reasoning=" | ".join(reasons[:3]),
        )

    def _judge_programmatic(
        self,
        expected_meta: Dict[str, Any],
        response: DualOutputResponse,
        retrieved_contexts: Optional[List[RetrievedContext]] = None,
        store: Optional[Any] = None,
    ) -> JudgeScore:
        """Deterministic rubric-anchored evaluation scoring (used directly offline and as anchor online)."""
        alignment = compute_ground_truth_alignment(expected_meta, response)
        grounding = compute_citation_grounding_metrics(
            response=response,
            retrieved_contexts=retrieved_contexts,
            store=store,
            is_legal=bool(expected_meta.get("is_legal", True)),
        )

        sec_recall = alignment["expected_section_recall"]
        forum_hit = bool(alignment["expected_forum_matched"])
        crit_recall = alignment["critical_steps_recall"]
        grounding_acc = grounding["grounding_accuracy"]
        hall_rate = grounding["hallucination_rate"]

        # 1. Statutory Accuracy (0-5)
        if expected_meta.get("expected_sections"):
            statutory_score = round(min(5.0, 1.5 + 3.5 * sec_recall), 2) if response.statutory_citations else 0.5
        else:
            statutory_score = 4.5 if response.statutory_citations else 3.0

        # 2. Procedural Actionability (0-5)
        step_count = len(response.action_plan)
        unique_phases = len({s.phase for s in response.action_plan})
        if step_count == 0:
            procedural_score = 0.0
        else:
            phase_factor = min(1.0, unique_phases / 5.0)
            procedural_score = round(min(5.0, 1.5 + 2.0 * phase_factor + 1.5 * crit_recall), 2)

        # 3. Forum Appropriateness (0-5)
        forum_score = 5.0 if forum_hit else (2.0 if step_count > 0 else 0.0)

        # 4. Hallucination Freedom (0-5) — strictly based on FINAL post-output grounding (DF-2, DF-5)
        if len(response.statutory_citations) + len(response.precedent_citations) == 0:
            hallucination_score = 0.0
        else:
            hallucination_score = round(max(0.0, 5.0 * (1.0 - hall_rate) * grounding_acc), 2)

        # 5. Coherence & Specificity (0-5)
        has_limitations = sum(1 for s in response.action_plan if s.limitation_period) >= 2
        has_statutory_basis = sum(1 for s in response.action_plan if s.statutory_basis) >= 2
        coherence_score = round(
            min(
                5.0,
                2.0
                + (1.2 if has_limitations else 0.4)
                + (1.2 if has_statutory_basis else 0.4)
                + (0.6 * crit_recall),
            ),
            2,
        )

        rubric_checks = {
            "cites_expected_statute": bool(sec_recall > 0.0),
            "identifies_expected_forum": forum_hit,
            "covers_critical_steps": bool(crit_recall >= 0.5),
            "zero_post_output_hallucinations": bool(
                hall_rate == 0.0 and (len(response.statutory_citations) + len(response.precedent_citations)) > 0
            ),
            "multi_phase_chronology": bool(unique_phases >= 3),
        }

        overall = round(
            (statutory_score + procedural_score + forum_score + hallucination_score + coherence_score) / 5.0,
            2,
        )
        reasoning = (
            f"Rubric audit: expected_section_recall={sec_recall:.0%}, critical_steps_recall={crit_recall:.0%}, "
            f"forum_matched={forum_hit}, post_output_grounding={grounding_acc:.0%} "
            f"(pre-verification stripped={grounding['pre_verification_strip_rate']:.0%})."
        )

        return JudgeScore(
            statutory_accuracy=statutory_score,
            procedural_actionability=procedural_score,
            forum_appropriateness=forum_score,
            hallucination_freedom=hallucination_score,
            coherence_and_specificity=coherence_score,
            overall_score=overall,
            score_variance=0.0,
            flagged_for_human_review=bool(overall < 3.0 or hall_rate > 0.0),
            rubric_checks=rubric_checks,
            reasoning=reasoning,
        )
