"""Presentation helpers shared across UI components.

Pure functions only (no Streamlit calls) so they can be unit-tested directly.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from solvemycase.core.proposed.graph import CLARIFICATION_PREFIX, REJECTION_SUMMARY_PREFIX
from solvemycase.data.ingestion.schema import (
    DualOutputResponse,
    LegalDomain,
    ProceduralActionStep,
    ProceduralPhase,
)

DISCLAIMER = (
    "solvemycase provides procedural information, not legal advice. "
    "Consult a qualified advocate for your specific situation."
)

# Indian Kanoon search reliably resolves "Section N in <Act>" to the section page and supports the
# title: operator for judgments. Official India Code deep links are often unreachable.
INDIAN_KANOON_SEARCH_URL = "https://indiankanoon.org/search/?formInput="

_MARKDOWN_SPECIAL_CHARS = re.compile(r"([\\`*_\[\]$~|<>#])")
_PLACEHOLDER_DEADLINES = {"", "N/A", "NA", "NONE", "-"}

DOMAIN_LABELS: Dict[LegalDomain, str] = {
    LegalDomain.MOTOR_VEHICLE_ACCIDENT: "🚗 Motor accident",
    LegalDomain.PROPERTY_CONFLICT: "🏠 Property dispute",
    LegalDomain.CONSUMER_RIGHTS: "🛒 Consumer complaint",
    LegalDomain.GENERAL_DISPUTE: "⚖️ General dispute",
}

# Ordered chronologically; drives grouping in the action-plan timeline.
PHASE_LABELS: Dict[ProceduralPhase, str] = {
    ProceduralPhase.IMMEDIATE_ACTION: "1 · Immediate action",
    ProceduralPhase.POLICE_ADMINISTRATIVE: "2 · Police & administrative",
    ProceduralPhase.EVIDENTIARY_DOCUMENTATION: "3 · Evidence & documents",
    ProceduralPhase.LEGAL_NOTICE: "4 · Legal notice",
    ProceduralPhase.FORUM_FILING: "5 · File your case",
    ProceduralPhase.LIMITATION_APPEAL: "6 · Deadlines & appeals",
}

PRIORITY_ORDER: Dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Colors accepted by st.badge.
PRIORITY_BADGE_COLORS: Dict[str, str] = {"critical": "red", "high": "orange", "medium": "blue", "low": "gray"}


def escape_markdown(text: Optional[str]) -> str:
    """Escape Markdown/LaTeX control characters so LLM or corpus text renders literally."""
    return _MARKDOWN_SPECIAL_CHARS.sub(r"\\\1", text or "")


def markdown_table_cell(text: Optional[str]) -> str:
    """Make text safe for a single Markdown table cell (no pipes or line breaks)."""
    return " ".join((text or "").split()).replace("|", "\\|")


def domain_label(domain: LegalDomain) -> str:
    """Human-friendly label for a legal domain."""
    return DOMAIN_LABELS.get(domain, domain.value.replace("_", " ").title())


def domain_label_from_value(value: Optional[str]) -> str:
    """Label for a raw domain string (e.g. from JSON); unknown or missing values are handled."""
    if not value:
        return "Unknown"
    try:
        return domain_label(LegalDomain(value))
    except ValueError:
        return value.replace("_", " ").title()


def priority_rank(priority: Optional[str]) -> int:
    """Sort key for step priority (unknown priorities sort last)."""
    return PRIORITY_ORDER.get((priority or "").strip().lower(), len(PRIORITY_ORDER))


def priority_badge_color(priority: Optional[str]) -> str:
    """st.badge color for a priority level."""
    return PRIORITY_BADGE_COLORS.get((priority or "").strip().lower(), "gray")


def has_real_deadline(step: ProceduralActionStep) -> bool:
    """True when the step's limitation period is an actual value, not a placeholder like 'N/A'."""
    return (step.limitation_period or "").strip().upper() not in _PLACEHOLDER_DEADLINES


def verified_citation_count(response: DualOutputResponse) -> int:
    """Number of statutory plus precedent citations included in a response."""
    return len(response.statutory_citations) + len(response.precedent_citations)


def group_steps_by_phase(steps: List[ProceduralActionStep]) -> Dict[ProceduralPhase, List[ProceduralActionStep]]:
    """Group steps under their phase, preserving chronological phase order and step numbers."""
    grouped: Dict[ProceduralPhase, List[ProceduralActionStep]] = {}
    for phase in PHASE_LABELS:
        phase_steps = sorted((s for s in steps if s.phase == phase), key=lambda s: s.step_number)
        if phase_steps:
            grouped[phase] = phase_steps
    return grouped


def extract_deadlines(steps: List[ProceduralActionStep]) -> List[Dict[str, str]]:
    """List steps that carry a limitation period, most urgent (priority, then order) first."""
    with_deadline = sorted((s for s in steps if has_real_deadline(s)), key=lambda s: (priority_rank(s.priority), s.step_number))
    return [
        {
            "Deadline": s.limitation_period.strip(),
            "Action": s.title,
            "Where": s.forum_or_authority,
            "Priority": (s.priority or "").title(),
        }
        for s in with_deadline
    ]


def is_guardrail_rejection(response: DualOutputResponse) -> bool:
    """True when the guardrail refused the query (no plan, rejection summary)."""
    return not response.action_plan and response.scenario_summary.startswith(REJECTION_SUMMARY_PREFIX)


def rejection_details(response: DualOutputResponse) -> Dict[str, str]:
    """Extract the rejection reason and clarification prompt from a guardrail response."""
    reason = response.scenario_summary[len(REJECTION_SUMMARY_PREFIX):].strip()
    clarification = ""
    for item in response.unverified_citations_stripped:
        if item.startswith(CLARIFICATION_PREFIX):
            clarification = item[len(CLARIFICATION_PREFIX):].strip()
            break
    return {"reason": reason, "clarification": clarification}


def removed_references(response: DualOutputResponse) -> List[str]:
    """Citations stripped by verification (excludes guardrail clarification messages)."""
    return [item for item in response.unverified_citations_stripped if not item.startswith(CLARIFICATION_PREFIX)]


def safe_http_url(url: Optional[str]) -> Optional[str]:
    """Return the URL only if it is an absolute http(s) link, else None (avoids javascript: etc.)."""
    if not url:
        return None
    parsed = urlparse(url.strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url.strip()
    return None


def statute_search_url(act_name: Optional[str], section_number: Optional[str]) -> Optional[str]:
    """Indian Kanoon search link whose top hit is the cited section (None if the citation is incomplete)."""
    act, section = (act_name or "").strip(), (section_number or "").strip()
    if not act or not section:
        return None
    return INDIAN_KANOON_SEARCH_URL + quote_plus(f"Section {section} in {act}")


def precedent_search_url(case_title: Optional[str]) -> Optional[str]:
    """Indian Kanoon title search for a judgment (None if no title)."""
    title = " ".join((case_title or "").split())
    if not title:
        return None
    return INDIAN_KANOON_SEARCH_URL + quote_plus(f"title: {title}")


def precedent_read_url(source_url: Optional[str], case_title: Optional[str]) -> Optional[str]:
    """Best link to read a judgment: its stored http(s) source, else an Indian Kanoon title search."""
    return safe_http_url(source_url) or precedent_search_url(case_title)
