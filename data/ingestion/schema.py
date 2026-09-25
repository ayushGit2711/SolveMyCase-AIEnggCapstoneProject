"""Data schemas for legal corpus ingestion, indexing, and dual-output generation.

Adheres strictly to the open-india-law unified chunking specification and Pydantic v2 validation.
"""

from enum import Enum
import re
from typing import Any, FrozenSet, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class LegalDomain(str, Enum):
    """Target Indian legal domains covered by solvemycase."""
    MOTOR_VEHICLE_ACCIDENT = "motor_vehicle_accident"
    PROPERTY_CONFLICT = "property_conflict"
    CONSUMER_RIGHTS = "consumer_rights"
    GENERAL_DISPUTE = "general_dispute"


class DocumentType(str, Enum):
    """Type of legal authority."""
    STATUTE = "statute"
    PRECEDENT = "precedent"


class ProceduralPhase(str, Enum):
    """Chronological stages of dispute resolution in India."""
    IMMEDIATE_ACTION = "immediate_action"
    POLICE_ADMINISTRATIVE = "police_administrative"
    EVIDENTIARY_DOCUMENTATION = "evidentiary_documentation"
    LEGAL_NOTICE = "legal_notice"
    FORUM_FILING = "forum_filing"
    LIMITATION_APPEAL = "limitation_appeal"


class LegalProvision(BaseModel):
    """Normalized legislative provision from India Code / open-india-law corpus.

    Matches section-level granularity of Central/State Acts.
    """
    doc_id: str = Field(..., description="Unique provision identifier (e.g., central_bns_2023_sec_106).")
    act_name: str = Field(..., description="Official title of the Act (e.g., Bharatiya Nyaya Sanhita, 2023).")
    section_number: str = Field(..., description="Section or rule number (e.g., 106, 166, 35).")
    title: Optional[str] = Field(None, description="Section heading or title.")
    text: str = Field(..., description="Verbatim statutory text of the provision.")
    source_url: str = Field(..., description="Traceable government source URL (India Code or official gazette).")
    jurisdiction: str = Field(default="Central", description="Central or State jurisdiction.")
    year: Optional[int] = Field(None, description="Year of enactment.")
    domain: LegalDomain = Field(default=LegalDomain.GENERAL_DISPUTE, description="Primary legal dispute domain.")
    act_category: Optional[str] = Field(default=None, description="Broad category: criminal, civil, consumer, traffic.")


class LegalPrecedent(BaseModel):
    """Case law judgment chunk from Supreme Court or High Courts.

    Adheres to the unified chunking schema from open-india-law.
    """
    doc_id: str = Field(..., description="Case judgment identifier.")
    chunk_id: str = Field(..., description="Chunk identifier: {doc_id}_{index:03d}.")
    court: str = Field(..., description="Name of court (e.g., Supreme Court of India, Delhi High Court).")
    court_type: str = Field(default="supreme_court", description="'supreme_court' or 'high_court'.")
    title: str = Field(..., description="Case title (Petitioner vs. Respondent).")
    citation: Optional[str] = Field(None, description="Standard Indian legal citation (e.g., (2023) 4 SCC 12).")
    year: Optional[int] = Field(None, description="Year of decision.")
    text: str = Field(..., description="Reasoned ratio decidendi or legal principle chunk.")
    source_url: str = Field(..., description="Official court PDF or Open Data repository link.")
    disposition: Optional[str] = Field(None, description="Disposal outcome (e.g., Allowed, Dismissed).")
    domain: LegalDomain = Field(default=LegalDomain.GENERAL_DISPUTE, description="Dispute domain.")


class RetrievedContext(BaseModel):
    """Standardized representation of a candidate chunk returned by hybrid search."""
    chunk_id: str
    doc_type: DocumentType
    title: str
    citation_or_section: str
    text: str
    source_url: str
    score: float = Field(default=0.0, description="Hybrid or reranker relevance score.")
    act_name: Optional[str] = None
    court: Optional[str] = None
    domain: LegalDomain = LegalDomain.GENERAL_DISPUTE
    act_category: Optional[str] = None


class ProceduralActionStep(BaseModel):
    """An individual chronological step in the dispute action plan."""
    step_number: int = Field(..., description="Chronological sequence number.")
    phase: ProceduralPhase = Field(..., description="Resolution stage category.")
    title: str = Field(..., description="Action title (e.g., File FIR at nearest Police Station).")
    description: str = Field(..., description="Detailed practical instructions.")
    forum_or_authority: str = Field(..., description="Competent authority (e.g., Police Station, MACT, DCDRC).")
    statutory_basis: Optional[str] = Field(None, description="Underlying governing section(s) if applicable.")
    limitation_period: Optional[str] = Field(None, description="Prescribed statutory deadline or timing rule.")
    priority: str = Field(default="High", description="Priority level: Critical, High, Medium, or Low.")


class StatutoryCitation(BaseModel):
    """Verified statutory section mapped to the scenario."""
    act_name: str
    section_number: str
    summary_of_provision: str
    applicability_to_scenario: str
    source_url: str
    is_verified: bool = Field(default=True, description="Strictly verified against retrieved corpus context.")


class PrecedentCitation(BaseModel):
    """Verified judicial precedent supporting the action plan."""
    case_title: str
    court: str
    year: Optional[int] = None
    citation: Optional[str] = None
    legal_principle: str
    source_url: str
    is_verified: bool = Field(default=True, description="Strictly verified against retrieved corpus context.")


class DualOutputResponse(BaseModel):
    """Dual-output synthesis produced by solvemycase."""
    scenario_summary: str = Field(..., description="Deconstructed factual situation summary.")
    domain: LegalDomain = Field(..., description="Classified dispute category.")
    action_plan: List[ProceduralActionStep] = Field(..., description="Chronological procedural roadmap.")
    statutory_citations: List[StatutoryCitation] = Field(..., description="Grounded Indian Acts & sections.")
    precedent_citations: List[PrecedentCitation] = Field(default_factory=list, description="Verified court judgments.")
    hallucination_check_passed: bool = Field(default=True, description="True if 0 ungrounded citations were found.")
    unverified_citations_stripped: List[str] = Field(
        default_factory=list,
        description="Any proposed citations discarded due to lack of strict corpus grounding."
    )


class ExecutionTrace(BaseModel):
    """Structured telemetry and routing trace captured during pipeline execution."""

    pipeline_type: str = Field(..., description="'baseline' or 'proposed'")
    nodes_visited: List[str] = Field(default_factory=list, description="Ordered list of graph nodes executed.")
    statute_queries: List[str] = Field(default_factory=list, description="Decontextualized statutory sub-queries.")
    precedent_queries: List[str] = Field(default_factory=list, description="Decontextualized precedent sub-queries.")
    criminal_route_triggered: bool = Field(default=False, description="Whether the criminal-code sub-layer ran.")
    criminal_queries: List[str] = Field(default_factory=list, description="Queries sent to criminal_code_search.")
    retrieval_gate_triggered: bool = Field(
        default=False, description="Whether RetrievalQualityGate fired unfiltered fallback retrieval."
    )
    retrieved_contexts: List[RetrievedContext] = Field(
        default_factory=list, description="Final retrieved and reranked context chunks passed to the planner."
    )


REJECTION_SUMMARY_PREFIX = "Query rejected: "
CLARIFICATION_PREFIX = "Clarification: "
CRIMINAL_ROUTE_KEYWORDS = (
    "fir",
    "police",
    "arrest",
    "bail",
    "accident",
    "death",
    "negligen",
    "hit and run",
    "cheat",
    "fraud",
    "abus",
    "insult",
    "threat",
    "intimidat",
    "harass",
    "theft",
    "stole",
    "stolen",
    "misappropriat",
    "assault",
)


def should_trigger_criminal_route(domain: Any, scenario: str) -> bool:
    """Return True if the domain or scenario text contains criminal-code indicators."""
    domain_str = domain.value if isinstance(domain, LegalDomain) else str(domain or "")
    scenario_lower = (scenario or "").lower()
    return domain_str in (LegalDomain.MOTOR_VEHICLE_ACCIDENT.value, LegalDomain.GENERAL_DISPUTE.value) or any(
        kw in scenario_lower for kw in CRIMINAL_ROUTE_KEYWORDS
    )


_ACT_STOPWORDS = frozenset({"the", "act", "of", "in", "and", "india", "indian", "code", "law", "laws", "section", "sec"})
_ACT_ABBREVIATIONS = {
    "mva": ("motor", "vehicles"),
    "mv": ("motor", "vehicles"),
    "bns": ("bharatiya", "nyaya", "sanhita"),
    "bnss": ("bharatiya", "nagarik", "suraksha", "sanhita"),
    "ipc": ("penal",),
    "crpc": ("criminal", "procedure"),
    "tpa": ("transfer", "property"),
    "sra": ("specific", "relief"),
    "cpa": ("consumer", "protection"),
    "cpc": ("civil", "procedure"),
    "rera": ("real", "estate", "regulation"),
}


def normalize_section_id(raw_section: Optional[str]) -> str:
    """Canonicalize a statutory section identifier while preserving parenthesized subsections.

    Examples:
        'Section 2(11)' -> '2(11)'
        'Sec. 53-A'     -> '53a'
        'Section 166'   -> '166'
        'Order 39 Rule 1' -> '39(1)'
    """
    if not raw_section:
        return ""
    s = raw_section.strip().lower()
    order_rule = re.search(r"order\s*(\d+)\s*rule\s*(\d+)", s)
    if order_rule:
        return f"{order_rule.group(1)}({order_rule.group(2)})"
    s = re.sub(r"^(?:sections?|secs?\.?|s\.|rule|art\.?|article)\s*", "", s).strip()
    # Extract primary section token with optional letter suffix and parenthesized clause
    m = re.search(r"(\d+\s*-?\s*[a-z]?(?:\(\s*[0-9a-z]+\s*\))?)", s)
    if not m:
        return re.sub(r"[^0-9a-z()]", "", s)
    token = m.group(1)
    return re.sub(r"[\s\-]", "", token)


def sections_match(expected_or_cited: Optional[str], candidate: Optional[str]) -> bool:
    """Return True if two section strings refer to the same statutory section without prefix collisions.

    Prevents '16' from matching '166' and '2(11)' from matching '2(47)', while allowing
    a parent section like '166' to match '166(2)' when no subsection was restricted.
    """
    a = normalize_section_id(expected_or_cited)
    b = normalize_section_id(candidate)
    if not a or not b:
        return False
    if a == b:
        return True
    # Allow parent section '166' to match '166(2)' only when one has no parenthesized clause
    if "(" not in a and b.startswith(f"{a}("):
        return True
    if "(" not in b and a.startswith(f"{b}("):
        return True
    return False


def extract_significant_act_tokens(act_name: Optional[str]) -> FrozenSet[str]:
    """Extract normalized, non-trivial statutory title tokens (expanding common legal acronyms)."""
    if not act_name:
        return frozenset()
    words = re.findall(r"[a-z]+", act_name.lower())
    tokens = set()
    for w in words:
        if w in _ACT_ABBREVIATIONS:
            tokens.update(_ACT_ABBREVIATIONS[w])
        elif w not in _ACT_STOPWORDS and len(w) > 2:
            tokens.add(w)
    return frozenset(tokens)


def acts_share_significant_token(act_a: Optional[str], act_b: Optional[str]) -> bool:
    """Return True when two Act names share at least one meaningful domain token (e.g. 'Motor'/'Vehicles')."""
    tokens_a = extract_significant_act_tokens(act_a)
    tokens_b = extract_significant_act_tokens(act_b)
    if not tokens_a or not tokens_b:
        return False
    return bool(tokens_a & tokens_b)


def section_mentioned_with_boundary(section_number: Optional[str], text: Optional[str]) -> bool:
    """Verify if 'Section <N>' appears in text with strict word boundaries (so Section 16 never matches Section 166)."""
    norm = normalize_section_id(section_number)
    if not norm or not text:
        return False
    escaped = re.escape(norm)
    # Require negative lookahead for digits, letters, or parenthesized clauses when norm has no parenthesis
    if "(" in norm:
        pattern = rf"\b(?:sections?|secs?\.?|s\.)\s*{escaped}(?![0-9a-z])"
    else:
        # Also allow optional hyphen before trailing letter (e.g. 53A or 53-A)
        m = re.match(r"^(\d+)([a-z])$", norm)
        sec_pat = rf"{m.group(1)}-?{m.group(2)}" if m else escaped
        pattern = rf"\b(?:sections?|secs?\.?|s\.)\s*{sec_pat}(?![0-9a-z]|\([0-9a-z]+\))"
    return bool(re.search(pattern, text.lower()))

