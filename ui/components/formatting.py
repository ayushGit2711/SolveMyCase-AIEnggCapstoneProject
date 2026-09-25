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


_LEGACY_INDIACODE_HOSTS = {"www.indiacode.nic.in", "indiacode.nic.in", "www.indiacode.gov.in"}

# Maps legacy DSpace 6 India Code handles (from indiacode.nic.in) to live DSpace 9.1 handles on indiacode.gov.in
_LEGACY_INDIACODE_HANDLE_MAP = {
    "123456789/1798": "123456789/523268",
    "123456789/20062": "123456789/496548",
    "123456789/20063": "123456789/496550",
    "123456789/15256": "123456789/496115",
    "123456789/2262": "123456789/496421",
    "123456789/1579": "123456789/524827",
    "123456789/2114": "123456789/549142",
    "123456789/1978": "123456789/620185",
}

# Verified direct Indian Kanoon statutory document URLs for all corpus provisions
_DIRECT_INDIAN_KANOON_SECTION_URLS = {
    ("motor", "134"): "https://indiankanoon.org/doc/734786/",
    ("motor", "161"): "https://indiankanoon.org/doc/94760817/",
    ("motor", "165"): "https://indiankanoon.org/doc/69683245/",
    ("motor", "166"): "https://indiankanoon.org/doc/136948773/",
    ("motor", "185"): "https://indiankanoon.org/doc/139481594/",
    ("motor", "196"): "https://indiankanoon.org/doc/125316407/",
    ("motor", "199A"): "https://indiankanoon.org/doc/785258/",
    ("nyaya", "106"): "https://indiankanoon.org/doc/158521871/",
    ("nyaya", "147"): "https://indiankanoon.org/doc/89706772/",
    ("nyaya", "149"): "https://indiankanoon.org/doc/62022797/",
    ("nyaya", "281"): "https://indiankanoon.org/doc/2500944/",
    ("nyaya", "316"): "https://indiankanoon.org/doc/90373646/",
    ("nyaya", "318"): "https://indiankanoon.org/doc/149679501/",
    ("nyaya", "324"): "https://indiankanoon.org/doc/107231460/",
    ("nyaya", "351"): "https://indiankanoon.org/doc/196234891/",
    ("nyaya", "352"): "https://indiankanoon.org/doc/57413416/",
    ("suraksha", "173"): "https://indiankanoon.org/doc/165794322/",
    ("suraksha", "193"): "https://indiankanoon.org/doc/166275157/",
    ("suraksha", "480"): "https://indiankanoon.org/doc/172895630/",
    ("consumer", "2"): "https://indiankanoon.org/doc/47873513/",
    ("consumer", "35"): "https://indiankanoon.org/doc/128899101/",
    ("consumer", "84"): "https://indiankanoon.org/doc/197620408/",
    ("consumer", "85"): "https://indiankanoon.org/doc/61565718/",
    ("consumer", "89"): "https://indiankanoon.org/doc/112210594/",
    ("property", "10"): "https://indiankanoon.org/doc/576368/",
    ("property", "31"): "https://indiankanoon.org/doc/1901550/",
    ("property", "44"): "https://indiankanoon.org/doc/513068/",
    ("property", "53A"): "https://indiankanoon.org/doc/221518/",
    ("property", "54"): "https://indiankanoon.org/doc/613871/",
    ("property", "55"): "https://indiankanoon.org/doc/1484775/",
    ("property", "106"): "https://indiankanoon.org/doc/80042/",
    ("property", "107"): "https://indiankanoon.org/doc/515323/",
    ("property", "111"): "https://indiankanoon.org/doc/1134872/",
    ("property", "122"): "https://indiankanoon.org/doc/881325/",
    ("property", "123"): "https://indiankanoon.org/doc/515323/",
    ("relief", "6"): "https://indiankanoon.org/doc/1773715/",
    ("relief", "16"): "https://indiankanoon.org/doc/1779540/",
    ("relief", "38"): "https://indiankanoon.org/doc/1474155/",
    ("relief", "39"): "https://indiankanoon.org/doc/1558146/",
    ("penal", "279"): "https://indiankanoon.org/doc/1270101/",
    ("penal", "304A"): "https://indiankanoon.org/doc/1371604/",
    ("criminal", "154"): "https://indiankanoon.org/doc/1980578/",
}

# Verified live DSpace 9.1 section handles on indiacode.gov.in
_DIRECT_INDIACODE_SECTION_URLS = {
    ("motor", "134"): "https://indiacode.gov.in/handle/123456789/523232",
    ("motor", "161"): "https://indiacode.gov.in/handle/123456789/523262",
    ("motor", "165"): "https://indiacode.gov.in/handle/123456789/523267",
    ("motor", "166"): "https://indiacode.gov.in/handle/123456789/523268",
    ("motor", "185"): "https://indiacode.gov.in/handle/123456789/523290",
    ("motor", "196"): "https://indiacode.gov.in/handle/123456789/523303",
    ("motor", "199A"): "https://indiacode.gov.in/handle/123456789/538138",
    ("nyaya", "106"): "https://indiacode.gov.in/handle/123456789/545604",
    ("nyaya", "147"): "https://indiacode.gov.in/handle/123456789/545857",
    ("nyaya", "149"): "https://indiacode.gov.in/handle/123456789/545649",
    ("nyaya", "281"): "https://indiacode.gov.in/handle/123456789/545772",
    ("nyaya", "316"): "https://indiacode.gov.in/handle/123456789/545808",
    ("nyaya", "318"): "https://indiacode.gov.in/handle/123456789/545810",
    ("nyaya", "324"): "https://indiacode.gov.in/handle/123456789/545814",
    ("nyaya", "351"): "https://indiacode.gov.in/handle/123456789/545840",
    ("nyaya", "352"): "https://indiacode.gov.in/handle/123456789/545841",
    ("suraksha", "173"): "https://indiacode.gov.in/handle/123456789/546648",
    ("suraksha", "193"): "https://indiacode.gov.in/handle/123456789/546293",
    ("suraksha", "480"): "https://indiacode.gov.in/handle/123456789/546565",
    ("consumer", "2"): "https://indiacode.gov.in/handle/123456789/538576",
    ("consumer", "35"): "https://indiacode.gov.in/handle/123456789/538611",
    ("consumer", "84"): "https://indiacode.gov.in/handle/123456789/538685",
    ("consumer", "85"): "https://indiacode.gov.in/handle/123456789/538658",
    ("consumer", "89"): "https://indiacode.gov.in/handle/123456789/538662",
    ("property", "10"): "https://indiacode.gov.in/handle/123456789/535176",
    ("property", "31"): "https://indiacode.gov.in/handle/123456789/535324",
    ("property", "44"): "https://indiacode.gov.in/handle/123456789/535210",
    ("property", "53A"): "https://indiacode.gov.in/handle/123456789/535221",
    ("property", "54"): "https://indiacode.gov.in/handle/123456789/535222",
    ("property", "55"): "https://indiacode.gov.in/handle/123456789/535223",
    ("property", "106"): "https://indiacode.gov.in/handle/123456789/535285",
    ("property", "107"): "https://indiacode.gov.in/handle/123456789/535286",
    ("property", "111"): "https://indiacode.gov.in/handle/123456789/535290",
    ("property", "122"): "https://indiacode.gov.in/handle/123456789/535303",
    ("property", "123"): "https://indiacode.gov.in/handle/123456789/535304",
    ("relief", "6"): "https://indiacode.gov.in/handle/123456789/524827",
    ("relief", "16"): "https://indiacode.gov.in/handle/123456789/524838",
    ("relief", "38"): "https://indiacode.gov.in/handle/123456789/524862",
    ("relief", "39"): "https://indiacode.gov.in/handle/123456789/524863",
    ("penal", "279"): "https://indiacode.gov.in/handle/123456789/549142",
    ("penal", "304A"): "https://indiacode.gov.in/handle/123456789/549142",
    ("criminal", "154"): "https://indiacode.gov.in/handle/123456789/620185",
}

_CURATED_PRECEDENT_URLS = {
    "patel roadways": "https://indiankanoon.org/doc/1907957/",
    "birla yamaha": "https://indiankanoon.org/doc/1907957/",
    "bharathi knitting": "https://indiankanoon.org/doc/126678/",
    "dhl worldwide": "https://indiankanoon.org/doc/126678/",
    "sarla verma": "https://indiankanoon.org/doc/837924/",
    "pranay sethi": "https://indiankanoon.org/doc/139996215/",
    "suraj lamp": "https://indiankanoon.org/doc/1565619/",
    "poona ram": "https://indiankanoon.org/doc/56103097/",
    "lucknow development": "https://indiankanoon.org/doc/1375046/",
    "m.k. gupta": "https://indiankanoon.org/doc/1375046/",
    "experion developers": "https://indiankanoon.org/doc/71246029/",
    "sushma ashok shiroor": "https://indiankanoon.org/doc/71246029/",
}


def clean_section_label(section_number: Optional[str]) -> str:
    """Strip redundant 'Section' / 'Sec.' prefixes so display labels and queries do not repeat 'Section'."""
    raw = (section_number or "").strip()
    return re.sub(r"^(?:sections?|secs?\.?|s\.)\s*", "", raw, flags=re.IGNORECASE).strip()


def _resolve_act_section_key(act_name: Optional[str], section_number: Optional[str]):
    act_lower = (act_name or "").lower()
    sec_clean = clean_section_label(section_number)
    m = re.match(r"^(\d+[A-Za-z]?)", sec_clean)
    base_sec = m.group(1).upper() if m else sec_clean.upper()

    act_token = None
    for candidate in ("motor", "nyaya", "suraksha", "consumer", "property", "relief", "penal", "criminal"):
        if candidate in act_lower:
            act_token = candidate
            break
    return act_token, base_sec


def safe_http_url(url: Optional[str]) -> Optional[str]:
    """Return the URL only if it is an absolute http(s) link, else None (avoids javascript: etc.).

    Also rewrites legacy India Code hosts (www.indiacode.nic.in / www.indiacode.gov.in) and
    migrates legacy DSpace 6 handle paths to live DSpace 9.1 handles on https://indiacode.gov.in.
    """
    if not url:
        return None
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        host = parsed.netloc.lower()
        if host in _LEGACY_INDIACODE_HOSTS or host == "indiacode.gov.in":
            path = parsed.path
            for old_h, new_h in _LEGACY_INDIACODE_HANDLE_MAP.items():
                if old_h in path:
                    return f"https://indiacode.gov.in/handle/{new_h}"
            return parsed._replace(scheme="https", netloc="indiacode.gov.in", query="").geturl()
        return cleaned
    return None


def statute_search_url(act_name: Optional[str], section_number: Optional[str]) -> Optional[str]:
    """Direct Indian Kanoon section URL when known, else a law-scoped (`doctypes:laws`) search link."""
    act, section = (act_name or "").strip(), clean_section_label(section_number)
    if not act or not section:
        return None
    act_token, base_sec = _resolve_act_section_key(act, section)
    if act_token and (act_token, base_sec) in _DIRECT_INDIAN_KANOON_SECTION_URLS:
        return _DIRECT_INDIAN_KANOON_SECTION_URLS[(act_token, base_sec)]
    return INDIAN_KANOON_SEARCH_URL + quote_plus(f"Section {section} in {act} doctypes:laws")


def official_statute_url(
    act_name: Optional[str],
    section_number: Optional[str],
    source_url: Optional[str] = None,
) -> Optional[str]:
    """Live DSpace 9.1 India Code section handle when known, else sanitized official source_url."""
    act_token, base_sec = _resolve_act_section_key(act_name, section_number)
    if act_token and (act_token, base_sec) in _DIRECT_INDIACODE_SECTION_URLS:
        return _DIRECT_INDIACODE_SECTION_URLS[(act_token, base_sec)]
    return safe_http_url(source_url)


def precedent_search_url(case_title: Optional[str]) -> Optional[str]:
    """Indian Kanoon title search for a judgment (None if no title)."""
    title = " ".join((case_title or "").split())
    if not title:
        return None
    return INDIAN_KANOON_SEARCH_URL + quote_plus(f"title: {title}")


def precedent_read_url(
    source_url: Optional[str],
    case_title: Optional[str],
    is_verified: bool = True,
) -> Optional[str]:
    """Best link to read a judgment: curated landmark URL, verified source_url, or Indian Kanoon title search."""
    title_lower = " ".join((case_title or "").lower().split())
    for key_phrase, curated_url in _CURATED_PRECEDENT_URLS.items():
        if key_phrase in title_lower:
            return curated_url
    safe_src = safe_http_url(source_url)
    if is_verified and safe_src:
        return safe_src
    return precedent_search_url(case_title)
