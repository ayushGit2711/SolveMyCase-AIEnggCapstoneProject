"""Coverage-gap handling: say honestly what the corpus covers instead of citing unrelated law.

The corpus holds selected sections of a few Acts plus some judgments. When a legal question falls outside it
(e.g. divorce, unpaid salary), the pipeline must not cite the nearest unrelated provision. It answers with safe,
generic next steps that have no statutory basis, and a note describing what the database does cover. The
summary is derived from the loaded corpus, so it never goes stale when documents are added.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from solvemycase.data.ingestion.schema import (
    CoverageGapReason,
    DocumentType,
    DualOutputResponse,
    LegalDomain,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
)

CRIMINAL_GROUP = "criminal"

# (group key, plain-language label) in display order.
_COVERAGE_GROUPS: List[Tuple[object, str]] = [
    (CRIMINAL_GROUP, "crimes and police complaints"),
    (LegalDomain.MOTOR_VEHICLE_ACCIDENT, "road accidents and motor insurance claims"),
    (LegalDomain.PROPERTY_CONFLICT, "property, tenancy and possession"),
    (LegalDomain.CONSUMER_RIGHTS, "consumer complaints"),
    (LegalDomain.GENERAL_DISPUTE, "other matters"),
]

GENERIC_HELP_LINE = (
    "For free legal advice, call the NALSA legal aid helpline 15100 or contact your District Legal Services "
    "Authority (DLSA)."
)


def _is_repealed(doc: RetrievedContext) -> bool:
    return "repealed" in (doc.title or "").lower()


def _group_key(doc: RetrievedContext) -> object:
    if (doc.act_category or "").strip().lower() == CRIMINAL_GROUP:
        return CRIMINAL_GROUP
    return doc.domain


def summarize_corpus_coverage(corpus_documents: Optional[Iterable[RetrievedContext]]) -> str:
    """Describe which laws the loaded corpus covers, grouped by subject, e.g. for a coverage-gap note.

    Repealed Acts (IPC, CrPC) are left out because the app does not rely on them for current events.
    """
    docs = list(corpus_documents or [])
    acts_by_group: Dict[object, List[str]] = {}
    precedent_titles = set()
    for doc in docs:
        if doc.doc_type == DocumentType.PRECEDENT:
            precedent_titles.add((doc.title or "").strip().lower())
            continue
        if doc.doc_type != DocumentType.STATUTE or _is_repealed(doc) or not (doc.act_name or "").strip():
            continue
        acts = acts_by_group.setdefault(_group_key(doc), [])
        act_name = doc.act_name.strip()
        if act_name not in acts:
            acts.append(act_name)

    parts = []
    for key, label in _COVERAGE_GROUPS:
        acts = acts_by_group.get(key)
        if acts:
            parts.append(f"{label} ({'; '.join(acts)})")
    if not parts:
        return "Our legal database is empty right now, so no law can be cited."

    summary = "Our database currently holds selected sections of these laws: " + "; ".join(parts)
    if precedent_titles:
        count = len(precedent_titles)
        noun = "judgment" if count == 1 else "judgments"
        summary += f", plus {count} Supreme Court and High Court {noun} in these areas"
    return summary + "."


def coverage_gap_note(coverage_summary: str) -> str:
    """Note shown when no law in the corpus applies to the facts."""
    return (
        "Our database doesn't cover the law for this situation yet, so we are not citing any law or court "
        "judgment rather than risk pointing you to the wrong one. "
        f"{coverage_summary} The general steps below are safe starting points; a lawyer or your District Legal "
        "Services Authority can tell you which law applies."
    )


def screening_unavailable_note(coverage_summary: str) -> str:
    """Note shown when the check that matches laws to the facts could not run (no LLM, or it failed)."""
    return (
        "The check that matches laws to your facts isn't available right now, so we are not citing any law or "
        "court judgment rather than risk pointing you to the wrong one. Please try again later. "
        f"{coverage_summary} The general steps below are safe starting points; a lawyer or your District Legal "
        "Services Authority can tell you which law applies."
    )


def no_verified_citations_note(coverage_summary: str) -> str:
    """Softer note shown when a plan was produced but no citation survived verification."""
    return (
        "We could not confirm any law or judgment in our database that applies to these facts, so this plan "
        f"cites none. {coverage_summary} Laws outside our database aren't covered. {GENERIC_HELP_LINE}"
    )


def _scenario_summary(scenario: str) -> str:
    scenario = scenario or ""
    return scenario[:240] + ("..." if len(scenario) > 240 else "")


def generic_next_steps() -> List[ProceduralActionStep]:
    """Four safe steps that hold for any legal problem and cite no law."""
    return [
        ProceduralActionStep(
            step_number=1,
            phase=ProceduralPhase.EVIDENTIARY_DOCUMENTATION,
            title="Write down what happened and keep your evidence",
            description=(
                "Note the date, time, place and the people involved while you still remember them. Keep photos, "
                "videos, messages, receipts, medical papers and the names and phone numbers of witnesses."
            ),
            forum_or_authority="You (personal record)",
            statutory_basis=None,
            limitation_period=None,
            priority="High",
        ),
        ProceduralActionStep(
            step_number=2,
            phase=ProceduralPhase.POLICE_ADMINISTRATIVE,
            title="If a crime may be involved, report it to the police",
            description=(
                "Go to the nearest police station, describe what happened in writing and ask for a copy of your "
                "complaint or FIR."
            ),
            forum_or_authority="Nearest police station",
            statutory_basis=None,
            limitation_period=None,
            priority="High",
        ),
        ProceduralActionStep(
            step_number=3,
            phase=ProceduralPhase.LEGAL_NOTICE,
            title="Get free legal advice",
            description=(
                "Call the NALSA legal aid helpline 15100 or visit your District Legal Services Authority (DLSA). "
                "They can tell you which law applies and help you for free if you are eligible."
            ),
            forum_or_authority="NALSA helpline 15100 / District Legal Services Authority",
            statutory_basis=None,
            limitation_period=None,
            priority="High",
        ),
        ProceduralActionStep(
            step_number=4,
            phase=ProceduralPhase.LIMITATION_APPEAL,
            title="Act promptly",
            description=(
                "Legal remedies have time limits, and some are short. Take advice and act as soon as you can so "
                "you do not lose your remedy."
            ),
            forum_or_authority="The court or authority that handles your matter",
            statutory_basis=None,
            limitation_period=None,
            priority="Medium",
        ),
    ]


def build_coverage_gap_response(
    scenario: str,
    domain: LegalDomain,
    coverage_summary: str,
    reason: CoverageGapReason = CoverageGapReason.NO_APPLICABLE_LAW,
) -> DualOutputResponse:
    """Response for a legal question we can't ground in the corpus: no citations, safe generic steps, honest note.

    Args:
        scenario: The user's facts.
        domain: Classified legal domain.
        coverage_summary: What the corpus covers (see ``summarize_corpus_coverage``).
        reason: NO_APPLICABLE_LAW when nothing in the corpus applies; SCREENING_UNAVAILABLE when the
            applicability check could not run, in which case the note does not claim the law is missing.
    """
    if reason == CoverageGapReason.SCREENING_UNAVAILABLE:
        note = screening_unavailable_note(coverage_summary)
    else:
        note = coverage_gap_note(coverage_summary)
    return DualOutputResponse(
        scenario_summary=_scenario_summary(scenario),
        domain=domain or LegalDomain.GENERAL_DISPUTE,
        action_plan=generic_next_steps(),
        statutory_citations=[],
        precedent_citations=[],
        hallucination_check_passed=True,
        unverified_citations_stripped=[],
        coverage_note=note,
    )
