"""Quantitative evaluation metrics for legal RAG and agentic pipelines.

Measures:
1. Post-Output Statutory & Precedent Grounding Rate (symmetric across Approach A & B) + Pre-Verification Strip Rate.
2. Retrieval IR Metrics (Section Recall@K, Precision@K, Mean Reciprocal Rank, Context Relevance).
3. Ground-Truth Alignment (Expected Section Recall, Forum Accuracy, Critical Steps Recall).
4. Agentic Control-Flow & Guardrail Metrics (Trajectory Validity, Criminal Routing Accuracy, Guardrail TNR/FPR, Task Success).
5. Citation Relevance & Honest Abstention (Irrelevant-Citation Rate; legal questions the corpus does not cover).

Matching is Act-aware. A benchmark entry such as "BNS 106" only matches Section 106 of the Bharatiya Nyaya Sanhita,
never Section 106 of the Transfer of Property Act, and an entry without a qualifier (e.g. "166") belongs to the
scenario's ``expected_act``. Acts are identified through the explicit ``ACT_QUALIFIERS`` map, so BNS and BNSS, and
CPC and CrPC, are always told apart.

Benchmark items may carry two optional fields:
- ``coverage``: "covered" (default) or "uncovered". An uncovered item is a legal question whose governing law is
  deliberately absent from the corpus; the right answer cites nothing irrelevant, ideally an honest coverage gap.
- ``acceptable_acts``: entries ("MVA", "BNSS", "BNS 318") naming the law that may be cited without counting as
  irrelevant. Missing lists default from the domain (see ``default_acceptable_acts``).
"""

import functools
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    ExecutionTrace,
    ProceduralPhase,
    RetrievedContext,
    acts_share_significant_token,
    is_guardrail_rejection,
    normalize_section_id,
    section_mentioned_with_boundary,
    sections_match,
    stripped_unverified_citations,
)


class EvaluationMetricSummary(BaseModel):
    """Container for quantitative benchmark results of a single scenario run."""
    scenario_id: str
    pipeline_type: str = Field(..., description="'baseline' or 'proposed'")
    domain: str
    latency_seconds: float
    total_statutory_citations: int
    verified_statutory_citations: int
    unverified_citations_count: int
    statutory_hallucination_rate: float = Field(
        ..., description="Ratio of ungrounded final citations to total final citations (0.0 is ideal)."
    )
    grounding_accuracy: float = Field(
        ..., description="Ratio of verified final citations to total final citations (1.0 is ideal)."
    )
    pre_verification_strip_rate: float = Field(
        default=0.0,
        description="Fraction of draft citations caught and stripped by VerificationNode prior to output.",
    )
    procedural_step_count: int
    procedural_phases_covered: List[str]
    phase_completeness_score: float = Field(
        ..., description="Fraction of all 6 standard procedural phases addressed (0.0 to 1.0)."
    )
    correct_forum_identified: bool = Field(
        default=True, description="Whether the designated statutory forum was recognized."
    )
    section_recall_at_k: float = Field(
        default=0.0, description="Fraction of gold expected_sections present in top-K retrieved contexts."
    )
    precision_at_k: float = Field(
        default=0.0, description="Fraction of top-K retrieved chunks matching gold expected_sections/act."
    )
    mrr: float = Field(
        default=0.0, description="Mean Reciprocal Rank of the first relevant statutory chunk in retrieved contexts."
    )
    context_relevance: float = Field(
        default=0.0, description="Mean relevance score (0.0 to 1.0) of retrieved chunks to the scenario."
    )
    expected_section_recall: float = Field(
        default=0.0, description="Fraction of benchmark expected_sections cited in the final response."
    )
    critical_steps_recall: float = Field(
        default=0.0, description="Fraction of benchmark critical_steps addressed in the action plan."
    )
    trajectory_valid: bool = Field(
        default=True, description="Whether the pipeline visited the exact expected sequence of graph nodes."
    )
    criminal_routing_correct: bool = Field(
        default=True, description="Whether criminal code deep RAG routing matched scenario expectation."
    )
    task_success: bool = Field(
        default=True, description="End-to-end task completion indicator."
    )
    irrelevant_citation_rate: float = Field(
        default=0.0,
        description="Fraction of final citations whose Act is not acceptable for the facts, or precedents from "
        "another domain or outside the corpus (0.0 is ideal).",
    )
    coverage_gap: bool = Field(
        default=False, description="Whether the pipeline answered with a coverage-gap response."
    )


# ---------------------------------------------------------------------------
# Act identification: explicit qualifier -> Act map
# ---------------------------------------------------------------------------

#: Qualifier used in benchmark entries ("BNS 106", "PCA 11", "BNSS") -> the Act it names. The first block lists the
#: Acts held in the corpus; the rest let baseline answers that cite related law from memory be judged fairly.
ACT_QUALIFIERS: Dict[str, str] = {
    # Acts held in the corpus.
    "BNS": "Bharatiya Nyaya Sanhita, 2023",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "IPC": "Indian Penal Code, 1860",
    "CrPC": "Code of Criminal Procedure, 1973",
    "MVA": "Motor Vehicles Act, 1988",
    "TPA": "Transfer of Property Act, 1882",
    "SRA": "Specific Relief Act, 1963",
    "CPA": "Consumer Protection Act, 2019",
    "PCA": "Prevention of Cruelty to Animals Act, 1960",
    # Procedure, evidence, limitation, legal aid and the Constitution.
    "CPC": "Code of Civil Procedure, 1908",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2023",
    "IEA": "Indian Evidence Act, 1872",
    "LIM": "Limitation Act, 1963",
    "LSAA": "Legal Services Authorities Act, 1987",
    "CONST": "Constitution of India",
    # Property, tenancy and contract.
    "RERA": "Real Estate (Regulation and Development) Act, 2016",
    "REG": "Registration Act, 1908",
    "STAMP": "Indian Stamp Act, 1899",
    "ICA": "Indian Contract Act, 1872",
    "EASE": "Indian Easements Act, 1882",
    "HSA": "Hindu Succession Act, 1956 / Indian Succession Act, 1925",
    "PARTITION": "Partition Act, 1893",
    "SARFAESI": "SARFAESI Act, 2002",
    "RENT": "State rent control / tenancy Acts",
    "COOP": "State co-operative societies Acts",
    "LARR": "Right to Fair Compensation and Transparency in Land Acquisition, Rehabilitation and Resettlement Act, 2013",
    "REVENUE": "State land revenue codes",
    "ELEC": "Electricity Act, 2003",
    # Consumer sectors.
    "SOGA": "Sale of Goods Act, 1930",
    "FSSA": "Food Safety and Standards Act, 2006",
    "DCA": "Drugs and Cosmetics Act, 1940",
    "LMA": "Legal Metrology Act, 2009",
    "INSA": "Insurance Act, 1938 / IRDAI regulations",
    "AVIATION": "Carriage by Air Act, 1972 / Aircraft Act, 1934 / DGCA requirements",
    "TELECOM": "Telecom Regulatory Authority of India Act, 1997 / telecom regulations",
    "BANKING": "Banking Regulation Act, 1949 / Payment and Settlement Systems Act, 2007 / RBI directions",
    "ITA": "Information Technology Act, 2000",
    "MED": "Medical council / clinical establishments laws",
    "JJA": "Juvenile Justice (Care and Protection of Children) Act, 2015",
    # Family.
    "HMA": "Hindu Marriage Act, 1955",
    "HAMA": "Hindu Adoptions and Maintenance Act, 1956",
    "SMA": "Special Marriage Act, 1954",
    "PLAW": "Other personal-law marriage Acts (Divorce Act, 1869; Muslim and Parsi marriage laws)",
    "FCA": "Family Courts Act, 1984",
    "GWA": "Guardians and Wards Act, 1890",
    "DPA": "Dowry Prohibition Act, 1961",
    "PWDVA": "Protection of Women from Domestic Violence Act, 2005",
    # Labour.
    "EPFA": "Employees' Provident Funds and Miscellaneous Provisions Act, 1952",
    "CSS": "Code on Social Security, 2020",
    "PWA": "Payment of Wages Act, 1936",
    "COW": "Code on Wages, 2019",
    "IDA": "Industrial Disputes Act, 1947",
    "IRC": "Industrial Relations Code, 2020",
    "SEA": "State shops and establishments Acts",
    # Other.
    "NIA": "Negotiable Instruments Act, 1881",
}

# Phrase / abbreviation patterns (never generic single words such as "property" or "rent"). Abbreviations allow
# dots but no spaces between letters, so "BNS s.325" is BNS and only "BNSS" / "B.N.S.S." is BNSS.
_ACT_PATTERN_SOURCES: Dict[str, str] = {
    "BNSS": r"\bb\.?n\.?s\.?s\b|\b(?:bharatiya\s+)?naga?rik(?:\s+suraksha)?(?:\s+sanhita)?\b",
    "BNS": r"\bb\.?n\.?s\b|\b(?:bharatiya\s+)?nyaya(?:\s+sanhita)?\b",
    "BSA": r"\bb\.?s\.?a\b|\b(?:bharatiya\s+)?sakshya(?:\s+adhiniyam)?\b",
    "IPC": r"\bi\.?p\.?c\b|\b(?:indian\s+)?penal\s+code\b",
    "CrPC": r"\bcr\.?\s?p\.?\s?c\b|\b(?:code\s+of\s+)?criminal\s+procedure(?:\s+code)?\b",
    "CPC": r"\bc\.?p\.?c\b|\b(?:code\s+of\s+)?civil\s+procedure(?:\s+code)?\b",
    "IEA": r"\bi\.?e\.?a\b|\b(?:indian\s+)?evidence\s+act\b",
    "MVA": r"\bmva\b|\bm\.?v\.?\s?act\b|\bmotor\s+vehicles?\b|\bmotor\s+accidents?\s+claims?\b",
    "TPA": r"\bt\.?p\.?a\b|\btransfer\s+of\s+property\b",
    "SRA": r"\bs\.?r\.?a\b|\bspecific\s+relief\b",
    "CPA": r"\bc\.?p\.?a\b|\bconsumer\s+protection\b",
    "PCA": r"\bp\.?c\.?a\b|\b(?:prevention\s+of\s+)?cruelty\s+to\s+animals\b",
    "LIM": r"\blimitation\s+act\b",
    "LSAA": r"\blegal\s+services\s+authorit(?:y|ies)\b|\bnalsa\b|\bdlsa\b|\bslsa\b",
    "CONST": r"\bconstitution\b",
    "RERA": r"\brera\b|\breal\s+estate\s*\(\s*regulation|\breal\s+estate\s+regulation\b|\breal\s+estate\s+act\b",
    "REG": r"\bregistration\s+act\b",
    "STAMP": r"\bstamp\s+act\b",
    "ICA": r"\bi\.?c\.?a\b|\b(?:indian\s+)?contract\s+act\b",
    "EASE": r"\b(?:indian\s+)?easements?\s+act\b",
    "HSA": r"\b(?:hindu|indian)\s+succession\b",
    "PARTITION": r"\bpartition\s+act\b",
    "SARFAESI": r"\bsarfaesi\b|\bsecuriti[sz]ation\s+and\s+reconstruction\b",
    "RENT": r"\brent\s+(?:control\s+)?act\b|\brent\s+control\b|\btenancy\s+act\b|\bland\s+tenures?\b",
    "COOP": r"\bco-?\s?operative\s+societ(?:y|ies)\b",
    "LARR": r"\bland\s+acquisition\b|\bright\s+to\s+fair\s+compensation\b",
    "REVENUE": r"\bland\s+revenue\b|\brevenue\s+code\b",
    "ELEC": r"\belectricity\s+act\b",
    "SOGA": r"\bsale\s+of\s+goods\b",
    "FSSA": r"\bfood\s+safety\b|\bfssai?\b",
    "DCA": r"\bdrugs\s+(?:and|&)\s+cosmetics\b",
    "LMA": r"\blegal\s+metrology\b",
    "INSA": r"\binsurance\s+(?:act|ombudsman|regulatory)\b|\birdai?\b",
    "AVIATION": r"\bcarriage\s+by\s+air\b|\baircraft\s+(?:act|rules)\b|\bdgca\b|\bcivil\s+aviation\b",
    "TELECOM": r"\btrai\b|\btelegraph\s+act\b|\btelecom(?:munications?)?\s+(?:act|regulatory|consumers?|tariff)\b",
    "BANKING": (
        r"\bbanking\s+(?:regulation|ombudsman)\b|\bpayment\s+and\s+settlement\b|\breserve\s+bank\b|\brbi\b"
        r"|\bintegrated\s+ombudsman\b"
    ),
    "ITA": r"\binformation\s+technology\b|\bi\.?t\.?\s?act\b",
    "MED": r"\bmedical\s+council\b|\bnational\s+medical\s+commission\b|\bclinical\s+establishments?\b",
    "JJA": r"\bjuvenile\s+justice\b",
    "HMA": r"\bhindu\s+marriage\b|\bh\.?m\.?a\b",
    "HAMA": r"\bhindu\s+adoptions?\s+and\s+maintenance\b|\bhama\b",
    "SMA": r"\bspecial\s+marriage\b",
    "PLAW": (
        r"\bdivorce\s+act\b|\bdissolution\s+of\s+muslim\s+marriages\b|\bmuslim\s+women\b"
        r"|\bmuslim\s+personal\s+law\b|\bparsi\s+marriage\b|\bchristian\s+marriage\b"
    ),
    "FCA": r"\bfamily\s+courts?\s+act\b",
    "GWA": r"\bguardians?\s+and\s+wards\b",
    "DPA": r"\bdowry\s+prohibition\b",
    "PWDVA": r"\b(?:protection\s+of\s+women\s+from\s+)?domestic\s+violence\b|\bpwdva\b|\bd\.?v\.?\s?act\b",
    "EPFA": r"\b(?:employees'?\s+)?provident\s+funds?\b|\bepfo?\b",
    "CSS": r"\bcode\s+on\s+social\s+security\b|\bsocial\s+security\s+code\b",
    "PWA": r"\bpayment\s+of\s+wages\b",
    "COW": r"\bcode\s+on\s+wages\b|\bwage\s+code\b|\bminimum\s+wages\b",
    "IDA": r"\bindustrial\s+disputes?\b",
    "IRC": r"\bindustrial\s+relations\s+code\b",
    "SEA": r"\bshops\s+and\s+(?:commercial\s+)?establishments?\b",
    "NIA": r"\bnegotiable\s+instruments?\b|\bn\.?i\.?\s?act\b",
}
_ACT_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    key: re.compile(source, re.IGNORECASE) for key, source in _ACT_PATTERN_SOURCES.items()
}
_SQUASHED_QUALIFIERS: Dict[str, str] = {key.lower(): key for key in ACT_QUALIFIERS}


def _find_act_mentions(text: str) -> List[Tuple[int, int, str]]:
    """Return non-overlapping (start, end, act_key) Act mentions in text, leftmost first (longer wins ties)."""
    found: List[Tuple[int, int, str]] = []
    for key, pattern in _ACT_PATTERNS.items():
        found.extend((m.start(), m.end(), key) for m in pattern.finditer(text))
    found.sort(key=lambda item: (item[0], item[0] - item[1]))
    mentions: List[Tuple[int, int, str]] = []
    last_end = -1
    for start, end, key in found:
        if start >= last_end:
            mentions.append((start, end, key))
            last_end = end
    return mentions


@functools.lru_cache(maxsize=4096)
def _canonical_act_key_cached(name: str) -> Optional[str]:
    squashed = re.sub(r"[^a-z0-9]", "", name.lower())
    if squashed in _SQUASHED_QUALIFIERS:
        return _SQUASHED_QUALIFIERS[squashed]
    mentions = _find_act_mentions(name.lower())
    return mentions[0][2] if mentions else None


def canonical_act_key(name: Optional[str]) -> Optional[str]:
    """Map an Act name, abbreviation or statute title to its ``ACT_QUALIFIERS`` key, or None if unknown.

    The leftmost Act named wins, so "Indian Penal Code (now BNS)" is IPC, and a statute title such as
    "Indian Penal Code, 1860 - Section 279 [Repealed; see BNS s.281]" stays IPC.

    Args:
        name: Act name ("Bharatiya Nyaya Sanhita, 2023"), abbreviation ("Cr.P.C.") or title.

    Returns:
        The qualifier key (e.g. "BNS", "CrPC") or None when no known Act is named.
    """
    text = str(name or "").strip()
    if not text:
        return None
    return _canonical_act_key_cached(text)


def acts_equivalent(act_a: Optional[str], act_b: Optional[str]) -> bool:
    """Return True when two Act names denote the same Act (so BNS != BNSS and CPC != CrPC).

    Known Acts are compared by qualifier key; a known and an unknown Act never match. When neither Act is known,
    fall back to a shared significant title token.
    """
    key_a, key_b = canonical_act_key(act_a), canonical_act_key(act_b)
    if key_a or key_b:
        return key_a == key_b
    return acts_share_significant_token(act_a, act_b)


class ActSectionRef(NamedTuple):
    """A parsed benchmark reference: an Act and, optionally, one of its sections (None = the whole Act)."""

    act_key: Optional[str]
    act_label: str
    section: Optional[str]


_SECTION_WORDS = frozenset({
    "section", "sections", "sec", "secs", "s", "ss", "rule", "rules", "order", "article", "art", "clause", "u/s",
})
_YEAR_RE = re.compile(r"(?:18|19|20)\d\d")
_SECTION_TOKEN = r"\d+[a-z]{0,3}(?:\s*\(\s*[0-9a-z]+\s*\))*"
_SECTION_WORD = r"\b(?:sections?|secs?|ss?)\b\s*\.?"
# "BNS 106", "PCA 11", "BNS s.325", "Bharatiya Nagarik Suraksha Sanhita, 2023" (a trailing year = whole Act).
_REF_QUALIFIED = re.compile(
    rf"^\s*(?P<act>[a-z][a-z.&'()\- ]*?[a-z.)])\s*(?:,\s*|\s+)(?:{_SECTION_WORD}\s*)?(?P<sec>{_SECTION_TOKEN})\s*$",
    re.IGNORECASE,
)
# "Motor Vehicles Act, 1988 Section 166".
_REF_ACT_THEN_SECTION = re.compile(
    rf"^\s*(?P<act>.*?[a-z].*?)\s*[,:\-–]?\s*{_SECTION_WORD}\s*(?P<sec>{_SECTION_TOKEN})\s*$", re.IGNORECASE
)
# "Section 166 of the Motor Vehicles Act, 1988".
_REF_SECTION_THEN_ACT = re.compile(
    rf"^\s*{_SECTION_WORD}\s*(?P<sec>{_SECTION_TOKEN})\s*(?:[,\-–]|\bof\b|\bunder\b)?\s*(?:the\s+)?(?P<act>[a-z].*?)\s*$",
    re.IGNORECASE,
)


def _is_year(token: str) -> bool:
    return bool(_YEAR_RE.fullmatch((token or "").strip()))


def _act_label(key: Optional[str], raw: str) -> str:
    return ACT_QUALIFIERS.get(key, raw) if key else raw


def _unqualified_ref(section_text: str, default_act: Optional[str]) -> ActSectionRef:
    key = canonical_act_key(default_act)
    return ActSectionRef(key, _act_label(key, default_act or ""), normalize_section_id(section_text) or None)


def parse_act_section(entry: Optional[str], default_act: Optional[str] = None) -> Optional[ActSectionRef]:
    """Parse a benchmark entry into an Act (+ optional section) reference.

    Examples:
        "BNS 106" -> BNS s.106; "PCA 11" -> PCA s.11; "166" -> s.166 of ``default_act``;
        "2(11)" -> s.2(11) of ``default_act``; "BNSS" or "Bharatiya Nagarik Suraksha Sanhita, 2023" -> all of BNSS.

    Args:
        entry: The benchmark entry (an ``expected_sections`` or ``acceptable_acts`` item).
        default_act: Act that unqualified section numbers belong to (the scenario's ``expected_act``).

    Returns:
        The parsed reference, or None for an empty entry.
    """
    text = " ".join(str(entry or "").split())
    if not text:
        return None
    for pattern in (_REF_QUALIFIED, _REF_ACT_THEN_SECTION, _REF_SECTION_THEN_ACT):
        match = pattern.match(text)
        if not match:
            continue
        act_text = match.group("act").strip(" ,.-–:")
        sec_text = match.group("sec")
        if act_text.lower().replace(".", "").strip() in _SECTION_WORDS:
            return _unqualified_ref(sec_text, default_act)
        if _is_year(sec_text):
            key = canonical_act_key(text)
            return ActSectionRef(key, _act_label(key, text), None)
        key = canonical_act_key(act_text)
        return ActSectionRef(key, _act_label(key, act_text), normalize_section_id(sec_text) or None)
    if not re.search(r"\d", text):
        key = canonical_act_key(text)
        return ActSectionRef(key, _act_label(key, text), None)
    return _unqualified_ref(text, default_act)


def _ref_identity(ref: ActSectionRef) -> Tuple[str, Optional[str]]:
    return (ref.act_key or ref.act_label.strip().lower(), ref.section)


def _dedupe_refs(refs: Iterable[Optional[ActSectionRef]]) -> List[ActSectionRef]:
    seen: Set[Tuple[str, Optional[str]]] = set()
    unique: List[ActSectionRef] = []
    for ref in refs:
        if ref is None or _ref_identity(ref) in seen:
            continue
        seen.add(_ref_identity(ref))
        unique.append(ref)
    return unique


def expected_section_refs(expected_meta: Dict[str, Any]) -> List[ActSectionRef]:
    """Parse ``expected_sections``; unqualified entries belong to ``expected_act``."""
    default_act = expected_meta.get("expected_act")
    refs = [parse_act_section(str(entry), default_act=default_act) for entry in expected_meta.get("expected_sections") or []]
    return [ref for ref in refs if ref is not None]


def _expected_act_refs(expected_meta: Dict[str, Any]) -> List[ActSectionRef]:
    """Whole-Act refs for ``expected_act`` plus every Act named by a qualified expected section."""
    refs: List[ActSectionRef] = []
    expected_act = expected_meta.get("expected_act")
    if expected_act:
        key = canonical_act_key(expected_act)
        refs.append(ActSectionRef(key, _act_label(key, str(expected_act)), None))
    refs.extend(ActSectionRef(ref.act_key, ref.act_label, None) for ref in expected_section_refs(expected_meta))
    return _dedupe_refs(refs)


# ---------------------------------------------------------------------------
# Scenario coverage and acceptable law
# ---------------------------------------------------------------------------

#: Acts citable without counting as irrelevant, per domain, when an item has no explicit ``acceptable_acts``.
#: The plan's corpus-level lists (MVA -> MVA/BNS/BNSS/IPC/CrPC; property -> TPA/SRA/BNS/BNSS/CPC/RERA;
#: consumer -> CPA/BNS/BNSS) are extended with closely related non-corpus Acts so that a baseline answer citing, say,
#: the Registration Act for a property dispute is not counted as an irrelevant citation. General disputes need an
#: explicit list.
DOMAIN_ACCEPTABLE_ACTS: Dict[str, List[str]] = {
    "motor_vehicle_accident": ["MVA", "BNS", "BNSS", "IPC", "CrPC", "INSA", "JJA"],
    "property_conflict": [
        "TPA", "SRA", "BNS", "BNSS", "CPC", "RERA", "IPC", "CrPC", "REG", "STAMP", "ICA", "EASE", "HSA",
        "PARTITION", "SARFAESI", "RENT", "COOP", "LARR", "REVENUE", "ELEC",
    ],
    "consumer_rights": [
        "CPA", "BNS", "BNSS", "IPC", "CrPC", "SOGA", "FSSA", "DCA", "LMA", "INSA", "AVIATION", "TELECOM", "BANKING",
        "ITA", "RERA", "MED",
    ],
    "general_dispute": [],
}
#: Evidence, limitation, legal-aid and constitutional law may be cited for any legal question.
UNIVERSAL_ACCEPTABLE_ACTS: List[str] = ["BSA", "IEA", "LIM", "LSAA", "CONST"]


def _domain_value(domain: Any) -> Optional[str]:
    value = getattr(domain, "value", domain)
    return str(value) if value is not None else None


def is_uncovered_scenario(expected_meta: Optional[Dict[str, Any]]) -> bool:
    """True for a legal benchmark question whose governing law is deliberately absent from the corpus."""
    meta = expected_meta or {}
    return bool(meta.get("is_legal", True)) and str(meta.get("coverage") or "covered").strip().lower() == "uncovered"


def default_acceptable_acts(expected_meta: Dict[str, Any]) -> List[str]:
    """Default ``acceptable_acts`` for an item: its domain's Acts, its expected Acts, and the universal Acts."""
    if not bool(expected_meta.get("is_legal", True)):
        return []
    entries = list(DOMAIN_ACCEPTABLE_ACTS.get(_domain_value(expected_meta.get("domain")) or "", []))
    expected_key = canonical_act_key(expected_meta.get("expected_act"))
    if expected_key:
        entries.append(expected_key)
    entries.extend(ref.act_key for ref in expected_section_refs(expected_meta) if ref.act_key)
    entries.extend(UNIVERSAL_ACCEPTABLE_ACTS)
    return list(dict.fromkeys(entries))


def acceptable_act_refs(expected_meta: Dict[str, Any]) -> List[ActSectionRef]:
    """Law that may be cited for an item: ``acceptable_acts`` (or the default), expected sections, universal Acts.

    An explicit list is not widened to whole expected Acts, so "BNS 325" keeps the rest of the BNS unacceptable.
    """
    if not bool(expected_meta.get("is_legal", True)):
        return []
    entries = expected_meta.get("acceptable_acts")
    if entries is None:
        entries = default_acceptable_acts(expected_meta)
    refs: List[Optional[ActSectionRef]] = [parse_act_section(str(entry)) for entry in entries]
    refs.extend(expected_section_refs(expected_meta))
    refs.extend(parse_act_section(key) for key in UNIVERSAL_ACCEPTABLE_ACTS)
    return _dedupe_refs(refs)


def _precedent_domain_relevant(doc_domain: Any, expected_meta: Dict[str, Any]) -> bool:
    """A corpus judgment is relevant only to a covered legal scenario of the same domain."""
    if not bool(expected_meta.get("is_legal", True)) or is_uncovered_scenario(expected_meta):
        return False
    scenario_domain = _domain_value(expected_meta.get("domain"))
    return scenario_domain is not None and _domain_value(doc_domain) == scenario_domain


_CASE_STOPWORDS = frozenset({
    "state", "union", "india", "indian", "others", "other", "versus", "anr", "ors", "ltd", "limited", "pvt",
    "private", "company", "through", "the", "and", "with", "from", "for",
})
# Words shared by many unrelated parties (insurers, authorities, boards); an overlap made only of these is no match.
_INSTITUTIONAL_TOKENS = frozenset({
    "national", "insurance", "development", "authority", "transport", "corporation", "board", "welfare", "animal",
    "animals", "industries", "developers", "roadways", "express", "courier", "division", "worldwide", "bank",
    "municipal", "commissioner", "government", "delhi", "assurance", "united", "oriental", "general", "life",
    "railway", "railways", "police", "director", "secretary", "council", "society", "trust", "hospital",
    "university", "district", "collector",
})


def _case_tokens(text: Optional[str]) -> Set[str]:
    return {t for t in re.findall(r"[a-z]+", (text or "").lower()) if len(t) >= 4 and t not in _CASE_STOPWORDS}


def match_corpus_precedent(
    case_title: Optional[str], corpus_documents: Optional[Iterable[RetrievedContext]]
) -> Optional[RetrievedContext]:
    """Find the corpus judgment a cited case title refers to, or None if it is not in the corpus.

    A match needs at least half of the cited title's significant words, including one party-specific word, so
    "National Insurance v. Swaran Singh" does not match "National Insurance v. Pranay Sethi".
    """
    cited = _case_tokens(case_title)
    if not cited:
        return None
    best: Optional[RetrievedContext] = None
    best_key: Tuple[float, int] = (0.0, 0)
    for doc in corpus_documents or []:
        if doc.doc_type != DocumentType.PRECEDENT:
            continue
        overlap = cited & _case_tokens(doc.title)
        if not overlap - _INSTITUTIONAL_TOKENS:
            continue
        score = len(overlap) / len(cited)
        if score >= 0.5 and (score, len(overlap)) > best_key:
            best, best_key = doc, (score, len(overlap))
    return best


def _ctx_act_text(ctx: RetrievedContext) -> str:
    return ctx.act_name or ctx.title or ""


def _is_statute_ctx(ctx: RetrievedContext) -> bool:
    return ctx.doc_type != DocumentType.PRECEDENT


def _ref_matches_act(ref: ActSectionRef, act_text: Optional[str]) -> bool:
    other = canonical_act_key(act_text)
    if ref.act_key or other:
        return ref.act_key == other
    return acts_share_significant_token(ref.act_label, act_text)


def _matches_ref(act_text: Optional[str], section: Optional[str], ref: ActSectionRef) -> bool:
    """Act matches and, when the ref names a section, the section matches (parent/child allowed)."""
    if not _ref_matches_act(ref, act_text):
        return False
    return ref.section is None or sections_match(ref.section, section)


def _ctx_matches_ref(ctx: RetrievedContext, ref: ActSectionRef) -> bool:
    return _is_statute_ctx(ctx) and _matches_ref(_ctx_act_text(ctx), ctx.citation_or_section, ref)


def _context_acceptable(ctx: RetrievedContext, expected_meta: Dict[str, Any], acceptable: Sequence[ActSectionRef]) -> bool:
    if _is_statute_ctx(ctx):
        return any(_ctx_matches_ref(ctx, ref) for ref in acceptable)
    return _precedent_domain_relevant(ctx.domain, expected_meta)


@functools.lru_cache(maxsize=1)
def _default_corpus_precedents() -> Tuple[RetrievedContext, ...]:
    """Load curated precedents as RetrievedContext objects when no store corpus is passed."""
    from solvemycase.data.ingestion.normalizer import load_legal_precedents

    return tuple(
        RetrievedContext(
            chunk_id=p.chunk_id,
            doc_type=DocumentType.PRECEDENT,
            domain=p.domain,
            title=p.title,
            citation_or_section=p.citation or "",
            court=p.court,
            text=p.text,
            source_url=p.source_url,
            score=1.0,
        )
        for p in load_legal_precedents()
    )


def find_irrelevant_citations(
    response: DualOutputResponse,
    expected_meta: Dict[str, Any],
    corpus_documents: Optional[Iterable[RetrievedContext]] = None,
) -> List[str]:
    """List the final citations that do not apply to the scenario.

    A statute is irrelevant unless its Act (and section, where the acceptable entry names one) is acceptable for
    the item. A judgment is irrelevant unless it is a corpus judgment from the scenario's own domain; judgments are
    never relevant to an uncovered question. Every citation on a non-legal query is irrelevant.

    Args:
        response: The pipeline's final response.
        expected_meta: The benchmark item.
        corpus_documents: Loaded corpus (``store.corpus_documents``), used to look up judgment domains. Defaults to
            the curated precedent corpus when omitted.

    Returns:
        Labels such as "Statute: Bharatiya Nyaya Sanhita, 2023 s.106" or "Precedent: <case title>".
    """
    corpus = list(_default_corpus_precedents() if corpus_documents is None else corpus_documents)
    is_legal = bool(expected_meta.get("is_legal", True))
    acceptable = acceptable_act_refs(expected_meta) if is_legal else []
    irrelevant: List[str] = []
    for stat in response.statutory_citations:
        if not is_legal or not any(_matches_ref(stat.act_name, stat.section_number, ref) for ref in acceptable):
            irrelevant.append(f"Statute: {stat.act_name} s.{stat.section_number}")
    for prec in response.precedent_citations:
        doc = match_corpus_precedent(prec.case_title, corpus) if is_legal else None
        if doc is None or not _precedent_domain_relevant(doc.domain, expected_meta):
            irrelevant.append(f"Precedent: {prec.case_title}")
    return irrelevant


def compute_irrelevant_citation_rate(
    response: DualOutputResponse,
    expected_meta: Dict[str, Any],
    corpus_documents: Optional[Iterable[RetrievedContext]] = None,
) -> float:
    """Fraction of final citations that do not apply to the scenario (0.0 when nothing is cited)."""
    total = len(response.statutory_citations) + len(response.precedent_citations)
    if total == 0:
        return 0.0
    return round(len(find_irrelevant_citations(response, expected_meta, corpus_documents)) / total, 4)


# ---------------------------------------------------------------------------
# Statutory-basis parsing (which Act each section number in a free-text basis belongs to)
# ---------------------------------------------------------------------------

_BASIS_SECTION_NUMBER = re.compile(r"(?<![\w(])(\d+[a-z]?(?:\s*\(\s*[0-9a-z]{1,4}\s*\))*)(?!\w)")
_BASIS_SEGMENT = re.compile(r"[^;|\n/]+")
# Text allowed between an Act and a section it governs: "MVA s.166", "Motor Vehicles Act, 1988 - Section 166".
_ACT_FIRST_GAP = re.compile(
    r"^\s*(?:act\b\s*)?(?:,?\s*(?:18|19|20)\d\d\s*)?[,:\-–(]?\s*(?:(?:sections?|secs?|ss?|s)\b\s*\.?)?\s*$"
)
# Text allowed between a section and its Act: "Section 166 of the MVA", "Section 173 BNSS", "Section 35, CPA".
_SECTION_FIRST_GAP = re.compile(r"^\s*(?:[,(\-–:]\s*)?(?:(?:of|under|in)\s+)?(?:the\s+)?$")
# Sections listed together share an Act: "Sections 134 and 166 of the MVA", "Section 6 & Section 38 SRA".
_SECTION_CHAIN_GAP = re.compile(
    r"^\s*(?:,|&|and|or|to|read\s+with)?\s*(?:(?:sections?|secs?|ss?|s)\b\s*\.?)?\s*$"
)


def _nearest_act(start: int, end: int, acts: Sequence[Tuple[int, int, str]]) -> Optional[str]:
    best: Optional[Tuple[int, int, str]] = None
    for act_start, act_end, key in acts:
        distance = start - act_end if act_end <= start else act_start - end
        rank = (max(0, distance), 0 if act_end <= start else 1)
        if best is None or rank < best[:2]:
            best = (rank[0], rank[1], key)
    return best[2] if best else None


def _basis_section_acts(basis: Optional[str]) -> List[Tuple[str, Optional[str]]]:
    """Pair every section number in a statutory_basis string with the Act it belongs to (None if no Act named).

    Handles "Section 166, Motor Vehicles Act, 1988", "MVA s.166; BNS s.281", "Section 173 BNSS & Section 106 BNS"
    and "Sections 134 and 166 of the Motor Vehicles Act".
    """
    text = re.sub(r"\bu\s*/\s*s\b", "section", (basis or "").lower())
    mentions = _find_act_mentions(text)
    pairs: List[Tuple[str, Optional[str]]] = []
    for segment in _BASIS_SEGMENT.finditer(text):
        seg_start, seg_end = segment.span()
        items: List[Tuple[int, int, str, str]] = [
            (m.start(1), m.end(1), "S", m.group(1))
            for m in _BASIS_SECTION_NUMBER.finditer(text, seg_start, seg_end)
            if not _is_year(m.group(1))
        ]
        items.extend((s, e, "A", key) for s, e, key in mentions if s >= seg_start and e <= seg_end)
        items.sort(key=lambda item: item[0])
        if not any(kind == "S" for _, _, kind, _ in items):
            continue
        act_first = items[0][2] == "A"
        owners: List[Optional[str]] = [None] * len(items)
        for i, (start, end, kind, _) in enumerate(items):
            if kind != "S":
                continue
            prev_i = next((j for j in range(i - 1, -1, -1) if items[j][2] == "A"), None)
            next_i = next((j for j in range(i + 1, len(items)) if items[j][2] == "A"), None)
            prev_ok = prev_i is not None and bool(_ACT_FIRST_GAP.match(text[items[prev_i][1]:start]))
            next_ok = next_i is not None and bool(_SECTION_FIRST_GAP.match(text[end:items[next_i][0]]))
            if prev_ok and next_ok:
                owners[i] = items[prev_i if act_first else next_i][3]
            elif prev_ok:
                owners[i] = items[prev_i][3]
            elif next_ok:
                owners[i] = items[next_i][3]
        changed = True
        while changed:
            changed = False
            for i, (_, _, kind, _) in enumerate(items):
                if kind != "S" or owners[i] is not None:
                    continue
                for j in (i + 1, i - 1):
                    if 0 <= j < len(items) and items[j][2] == "S" and owners[j] is not None:
                        lo, hi = min(i, j), max(i, j)
                        if _SECTION_CHAIN_GAP.match(text[items[lo][1]:items[hi][0]]):
                            owners[i] = owners[j]
                            changed = True
                            break
        segment_acts = [(s, e, value) for s, e, kind, value in items if kind == "A"]
        for i, (start, end, kind, value) in enumerate(items):
            if kind != "S":
                continue
            owner = owners[i] or _nearest_act(start, end, segment_acts) or _nearest_act(start, end, mentions)
            pairs.append((normalize_section_id(value), owner))
    return pairs


def _statutory_basis_cites(basis: Optional[str], ref: ActSectionRef) -> bool:
    """True when a step's statutory_basis names both the ref's section and its Act (for that section)."""
    if not basis or not basis.strip():
        return False
    if ref.section is None:
        if ref.act_key:
            return any(key == ref.act_key for _, _, key in _find_act_mentions(basis.lower()))
        return acts_share_significant_token(ref.act_label, basis)
    for section, act_key in _basis_section_acts(basis):
        if not sections_match(ref.section, section):
            continue
        if ref.act_key is not None and act_key == ref.act_key:
            return True
        if ref.act_key is None and act_key is None and acts_share_significant_token(ref.act_label, basis):
            return True
    return False


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _is_statutory_citation_grounded(
    act_name: str,
    section_number: str,
    retrieved_contexts: Optional[List[RetrievedContext]],
    store: Optional[Any] = None,
    fallback_verified_flag: bool = False,
) -> bool:
    """Check whether a single statutory citation is grounded in retrieved contexts or Qdrant store."""
    sec_norm = normalize_section_id(section_number)
    if not sec_norm:
        return False

    if retrieved_contexts is not None:
        for ctx in retrieved_contexts:
            ctx_act = ctx.act_name or ctx.title
            if not acts_equivalent(act_name, ctx_act):
                continue
            if sections_match(sec_norm, ctx.citation_or_section):
                return True
            ctx_blob = f"{ctx.title} {ctx.citation_or_section} {ctx.text}"
            if section_mentioned_with_boundary(sec_norm, ctx_blob):
                return True

    if store is not None and hasattr(store, "exact_section_search"):
        try:
            hits = store.exact_section_search(sec_norm, act_name)
            if hits:
                return True
        except Exception:
            pass

    if retrieved_contexts is None and store is None:
        return bool(fallback_verified_flag)

    return False


def _is_precedent_citation_grounded(
    case_title: str,
    retrieved_contexts: Optional[List[RetrievedContext]],
    fallback_verified_flag: bool = False,
) -> bool:
    """Check whether a single precedent citation is grounded in retrieved precedent contexts."""
    if not case_title or not case_title.strip():
        return False

    if retrieved_contexts is None:
        return bool(fallback_verified_flag)

    precedent_contexts = [c for c in retrieved_contexts if c.doc_type == DocumentType.PRECEDENT]
    if not precedent_contexts:
        return False

    if match_corpus_precedent(case_title, precedent_contexts) is not None:
        return True
    title_tokens = _case_tokens(case_title) - _INSTITUTIONAL_TOKENS
    if not title_tokens:
        return False
    return any(
        any(
            re.search(rf"\b{re.escape(w)}\b", f"{p_ctx.title} {p_ctx.citation_or_section} {p_ctx.text}".lower())
            for w in title_tokens
        )
        for p_ctx in precedent_contexts
    )


def compute_citation_grounding_metrics(
    response: DualOutputResponse,
    retrieved_contexts: Optional[List[RetrievedContext]] = None,
    store: Optional[Any] = None,
    is_legal: bool = True,
    abstention_ok: bool = False,
) -> Dict[str, float]:
    """Compute post-output hallucination rate, grounding accuracy, and pre-verification strip rate.

    Both Approach A (Vanilla RAG) and Approach B (LangGraph) are scored identically on the
    citations actually delivered in the final `DualOutputResponse`. Citations stripped prior
    to output by `VerificationNode` are tracked separately in `pre_verification_strip_rate`.

    Args:
        response: The final response.
        retrieved_contexts: Contexts the pipeline actually used.
        store: Optional vector store for exact-section fallback lookups.
        is_legal: Whether the benchmark item is a legal question.
        abstention_ok: True for a question outside the corpus, where answering without citations is the honest
            answer; zero citations then earn 1.0 grounding (unless the guardrail wrongly rejected the question).
    """
    real_stripped = stripped_unverified_citations(response)
    stripped_count = len(real_stripped)

    total_final = len(response.statutory_citations) + len(response.precedent_citations)
    total_proposed = total_final + stripped_count
    pre_strip_rate = round(stripped_count / total_proposed, 4) if total_proposed > 0 else 0.0

    rejected = is_guardrail_rejection(response)

    # Guardrail rejection or out-of-scope query handling (DF-2)
    if not is_legal:
        if rejected and total_final == 0:
            return {
                "hallucination_rate": 0.0,
                "grounding_accuracy": 1.0,
                "pre_verification_strip_rate": pre_strip_rate,
                "ungrounded_final_count": 0,
            }
        # Non-legal query where pipeline failed to reject and emitted citations/advice
        if total_final == 0:
            return {
                "hallucination_rate": 0.0,
                "grounding_accuracy": 0.0,
                "pre_verification_strip_rate": pre_strip_rate,
                "ungrounded_final_count": 0,
            }

    if total_final == 0:
        # DF-2: For a legal query (is_legal=True), emitting zero citations cannot earn 1.0 grounding accuracy,
        # except for an honest abstention on a question the corpus does not cover.
        honest_abstention = abstention_ok and not rejected
        return {
            "hallucination_rate": 0.0,
            "grounding_accuracy": 1.0 if (honest_abstention or not is_legal) else 0.0,
            "pre_verification_strip_rate": pre_strip_rate,
            "ungrounded_final_count": 0,
        }

    verified_final = 0
    for stat in response.statutory_citations:
        if _is_statutory_citation_grounded(
            act_name=stat.act_name,
            section_number=stat.section_number,
            retrieved_contexts=retrieved_contexts,
            store=store,
            fallback_verified_flag=stat.is_verified,
        ):
            verified_final += 1

    for prec in response.precedent_citations:
        if _is_precedent_citation_grounded(
            case_title=prec.case_title,
            retrieved_contexts=retrieved_contexts,
            fallback_verified_flag=prec.is_verified,
        ):
            verified_final += 1

    ungrounded_final = total_final - verified_final
    hallucination_rate = round(ungrounded_final / total_final, 4)
    grounding_accuracy = round(verified_final / total_final, 4)

    return {
        "hallucination_rate": hallucination_rate,
        "grounding_accuracy": grounding_accuracy,
        "pre_verification_strip_rate": pre_strip_rate,
        "ungrounded_final_count": ungrounded_final,
    }


def compute_procedural_completeness(response: DualOutputResponse) -> Dict[str, Any]:
    """Measure coverage across the 6 essential procedural phases in Indian dispute resolution."""
    all_phases = {
        ProceduralPhase.IMMEDIATE_ACTION,
        ProceduralPhase.POLICE_ADMINISTRATIVE,
        ProceduralPhase.EVIDENTIARY_DOCUMENTATION,
        ProceduralPhase.LEGAL_NOTICE,
        ProceduralPhase.FORUM_FILING,
        ProceduralPhase.LIMITATION_APPEAL,
    }

    covered_phases: Set[ProceduralPhase] = {step.phase for step in response.action_plan}
    completeness_score = round(len(covered_phases) / len(all_phases), 4)

    return {
        "phases_covered": [p.value for p in covered_phases],
        "completeness_score": completeness_score,
        "step_count": len(response.action_plan),
    }


_FORUM_SYNONYMS: Dict[str, List[str]] = {
    "mact": ["mact", "motor accident claims tribunal", "claims tribunal"],
    "consumer commission": ["consumer commission", "district commission", "dcdrc", "scdrc", "ncdrc", "consumer disputes redressal"],
    "dcdrc": ["dcdrc", "district commission", "district consumer", "consumer commission"],
    "scdrc": ["scdrc", "state commission", "consumer commission"],
    "civil court": ["civil court", "civil judge", "district court", "competent court"],
    "police station": ["police station", "fir", "sho", "magistrate"],
}


def verify_target_forum(
    response: DualOutputResponse,
    expected_keywords: List[str],
    is_legal: bool = True,
) -> bool:
    """Verify if the action plan targets the legally designated forum or tribunal.

    For non-legal queries (is_legal=False), returns True iff the pipeline properly rejected the query (DF-5).
    """
    if not is_legal:
        return is_guardrail_rejection(response)

    if not expected_keywords:
        return not is_guardrail_rejection(response) and len(response.action_plan) > 0

    all_plan_text = " ".join(
        f"{step.title} {step.description} {step.forum_or_authority}"
        for step in response.action_plan
    ).lower()

    for kw in expected_keywords:
        kw_lower = kw.lower().strip()
        if kw_lower in all_plan_text:
            return True
        for syn in _FORUM_SYNONYMS.get(kw_lower, []):
            if syn in all_plan_text:
                return True
    return False


def compute_ground_truth_alignment(
    expected_meta: Dict[str, Any],
    response: DualOutputResponse,
) -> Dict[str, Any]:
    """Compare a pipeline response against gold benchmark expectations (`expected_sections`, `expected_forum`, `critical_steps`).

    Expected sections are matched Act-aware: by a final statutory citation of the same Act and section, or by a
    step whose statutory_basis names both the section and its Act. For a question outside the corpus there is no
    gold section, so the section score is the share of citations that apply (1.0 when nothing is cited).
    """
    is_legal = bool(expected_meta.get("is_legal", True))
    rejected = is_guardrail_rejection(response)

    if not is_legal:
        score = 1.0 if rejected else 0.0
        return {
            "expected_section_recall": score,
            "expected_forum_matched": rejected,
            "critical_steps_recall": score,
        }

    if rejected:
        return {
            "expected_section_recall": 0.0,
            "expected_forum_matched": False,
            "critical_steps_recall": 0.0,
        }

    # 1. Expected statutory sections recall
    if is_uncovered_scenario(expected_meta):
        section_recall = round(1.0 - compute_irrelevant_citation_rate(response, expected_meta), 4)
    else:
        expected_refs = expected_section_refs(expected_meta)
        bases = [step.statutory_basis for step in response.action_plan if step.statutory_basis]
        matched_sections = 0
        for ref in expected_refs:
            cited = any(_matches_ref(c.act_name, c.section_number, ref) for c in response.statutory_citations)
            if cited or any(_statutory_basis_cites(basis, ref) for basis in bases):
                matched_sections += 1
        section_recall = round(matched_sections / len(expected_refs), 4) if expected_refs else 1.0

    # 2. Expected forum match
    forum_matched = verify_target_forum(
        response,
        expected_meta.get("expected_forum", []),
        is_legal=is_legal,
    )

    # 3. Critical steps recall
    critical_steps: List[str] = expected_meta.get("critical_steps", [])
    full_plan_blob = " ".join(
        f"{s.title} {s.description} {s.forum_or_authority} {s.statutory_basis or ''}"
        for s in response.action_plan
    ).lower()

    matched_steps = 0
    for crit in critical_steps:
        tokens = [
            t for t in re.split(r"\W+", crit.lower()) if len(t) >= 3 and t not in {"the", "and", "for", "under", "with"}
        ]
        if not tokens:
            continue
        if any(t in full_plan_blob for t in tokens):
            matched_steps += 1

    steps_recall = round(matched_steps / len(critical_steps), 4) if critical_steps else 1.0

    return {
        "expected_section_recall": section_recall,
        "expected_forum_matched": forum_matched,
        "critical_steps_recall": steps_recall,
    }


def compute_retrieval_metrics(
    retrieved_contexts: List[RetrievedContext],
    expected_meta: Dict[str, Any],
    k: int = 5,
) -> Dict[str, float]:
    """Compute Retrieval IR metrics: Section Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).

    Covered legal items: recall counts an expected section only when a STATUTE chunk of the same Act and section is
    in the top K. A chunk is relevant for precision/MRR when it is a statute of an expected Act, or a judgment whose
    corpus domain equals the scenario domain. Uncovered items: recall is 1.0 (nothing to find) and precision/MRR
    measure how much of what was retrieved is acceptable law (1.0 when nothing was retrieved).
    """
    is_legal = bool(expected_meta.get("is_legal", True))
    if not is_legal:
        # Non-legal queries should ideally trigger 0 retrievals when blocked by guardrail
        clean_block = 1.0 if len(retrieved_contexts) == 0 else 0.0
        return {
            "section_recall_at_k": clean_block,
            "precision_at_k": clean_block,
            "mrr": clean_block,
        }

    top_k_chunks = list(retrieved_contexts or [])[: max(1, k)]

    if is_uncovered_scenario(expected_meta):
        if not top_k_chunks:
            return {"section_recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0}
        acceptable = acceptable_act_refs(expected_meta)
        flags = [_context_acceptable(ctx, expected_meta, acceptable) for ctx in top_k_chunks]
        first_ok = next((idx for idx, ok in enumerate(flags, start=1) if ok), None)
        return {
            "section_recall_at_k": 1.0,
            "precision_at_k": round(sum(flags) / len(flags), 4),
            "mrr": round(1.0 / first_ok, 4) if first_ok is not None else 0.0,
        }

    if not top_k_chunks:
        return {"section_recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}

    # Section Recall@K (Act-aware, statute chunks only)
    expected_refs = expected_section_refs(expected_meta)
    matched_expected = sum(
        1 for ref in expected_refs if any(_ctx_matches_ref(ctx, ref) for ctx in top_k_chunks)
    )
    recall_at_k = round(matched_expected / len(expected_refs), 4) if expected_refs else 1.0

    # Precision@K and MRR
    expected_acts = _expected_act_refs(expected_meta)
    relevant_chunks = 0
    first_relevant_rank: Optional[int] = None
    for idx, ctx in enumerate(top_k_chunks, start=1):
        if _is_statute_ctx(ctx):
            is_rel = any(_ref_matches_act(ref, _ctx_act_text(ctx)) for ref in expected_acts)
        else:
            is_rel = _precedent_domain_relevant(ctx.domain, expected_meta)
        if is_rel:
            relevant_chunks += 1
            if first_relevant_rank is None:
                first_relevant_rank = idx

    precision_at_k = round(relevant_chunks / len(top_k_chunks), 4)
    mrr = round(1.0 / first_relevant_rank, 4) if first_relevant_rank is not None else 0.0

    return {
        "section_recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
    }


def compute_context_relevance(
    scenario: str,
    retrieved_contexts: List[RetrievedContext],
    expected_meta: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute mean context relevance score in [0.0, 1.0] for retrieved chunks.

    Per chunk: word overlap with the scenario (up to 0.5), +0.25 for a statute of an expected Act, +0.35 for a
    statute matching an expected section (Act-aware), or +0.25 for a judgment from the scenario's domain. For a
    question outside the corpus the score is the share of retrieved chunks that are acceptable law (1.0 if none).
    """
    if expected_meta is not None and not expected_meta.get("is_legal", True):
        return 1.0 if len(retrieved_contexts) == 0 else 0.2

    meta = expected_meta or {}
    if is_uncovered_scenario(meta):
        if not retrieved_contexts:
            return 1.0
        acceptable = acceptable_act_refs(meta)
        ok = [1.0 if _context_acceptable(ctx, meta, acceptable) else 0.0 for ctx in retrieved_contexts]
        return round(sum(ok) / len(ok), 4)

    if not retrieved_contexts:
        return 0.0

    scenario_tokens = {
        w for w in re.split(r"\W+", scenario.lower()) if len(w) > 3
    }
    expected_refs = [ref for ref in expected_section_refs(meta) if ref.section]
    expected_acts = _expected_act_refs(meta)

    chunk_scores: List[float] = []
    for ctx in retrieved_contexts:
        score = 0.0
        ctx_blob = f"{ctx.title} {ctx.citation_or_section} {ctx.text}".lower()
        ctx_tokens = {w for w in re.split(r"\W+", ctx_blob) if len(w) > 3}
        if scenario_tokens:
            overlap = len(scenario_tokens & ctx_tokens) / max(1, min(len(scenario_tokens), 12))
            score += min(0.5, overlap * 0.6)
        if _is_statute_ctx(ctx):
            if any(_ref_matches_act(ref, _ctx_act_text(ctx)) for ref in expected_acts):
                score += 0.25
            if any(_ctx_matches_ref(ctx, ref) for ref in expected_refs):
                score += 0.35
        elif _precedent_domain_relevant(ctx.domain, meta):
            score += 0.25
        chunk_scores.append(min(1.0, score))

    return round(sum(chunk_scores) / len(chunk_scores), 4)


#: Node sequences of the proposed LangGraph pipeline and the baseline.
PROPOSED_FULL_PATH: Tuple[str, ...] = (
    "guardrail",
    "decontextualize",
    "retrieve_and_rerank",
    "applicability_check",
    "procedural_planner",
    "verification",
    "synthesis",
)
PROPOSED_GAP_PATH: Tuple[str, ...] = (
    "guardrail",
    "decontextualize",
    "retrieve_and_rerank",
    "applicability_check",
    "synthesis",
)
PROPOSED_REJECTION_PATH: Tuple[str, ...] = ("guardrail", "handle_rejection")
BASELINE_PATH: Tuple[str, ...] = ("embed_query", "hybrid_search", "monolithic_generate")


def compute_agent_trace_metrics(
    trace: ExecutionTrace,
    expected_meta: Dict[str, Any],
    response: DualOutputResponse,
    corpus_documents: Optional[Iterable[RetrievedContext]] = None,
) -> Dict[str, Any]:
    """Evaluate agentic control-flow trajectory, criminal routing accuracy, and guardrail TNR/FPR.

    Covered legal items expect the full proposed path (including ``applicability_check``); uncovered items accept
    the full path or the coverage-gap path. A covered item answered with a coverage gap is a task failure; an
    uncovered item succeeds when it is answered (not rejected) with at least 3 steps and no irrelevant citation.
    ``coverage_gap`` reports whether the pipeline answered with a coverage gap.
    """
    is_legal = bool(expected_meta.get("is_legal", True))
    uncovered = is_uncovered_scenario(expected_meta)
    rejected = is_guardrail_rejection(response)
    expects_crim = bool(expected_meta.get("expects_criminal_route", False))
    coverage_gap = bool(getattr(trace, "coverage_gap", False))
    visited = tuple(trace.nodes_visited)

    if trace.pipeline_type == "proposed_langgraph":
        if not is_legal:
            trajectory_valid = visited == PROPOSED_REJECTION_PATH
        elif uncovered:
            trajectory_valid = visited in (PROPOSED_FULL_PATH, PROPOSED_GAP_PATH)
        else:
            trajectory_valid = visited == PROPOSED_FULL_PATH
    else:
        # Vanilla RAG has no guardrail node; valid trajectory only on in-scope queries
        trajectory_valid = is_legal and visited == BASELINE_PATH

    criminal_routing_correct = bool(trace.criminal_route_triggered) == expects_crim

    guardrail_tnr = 1.0 if (not is_legal and rejected) else (0.0 if not is_legal else 1.0)
    guardrail_fpr = 1.0 if (is_legal and rejected) else 0.0

    if not is_legal:
        task_success = rejected
    elif uncovered:
        task_success = (
            (not rejected)
            and len(response.action_plan) >= 3
            and not find_irrelevant_citations(response, expected_meta, corpus_documents)
        )
    else:
        forum_ok = verify_target_forum(response, expected_meta.get("expected_forum", []), is_legal=True)
        task_success = (not rejected) and (not coverage_gap) and forum_ok and len(response.action_plan) >= 3

    return {
        "trajectory_valid": trajectory_valid,
        "criminal_routing_correct": criminal_routing_correct,
        "guardrail_tnr": guardrail_tnr,
        "guardrail_fpr": guardrail_fpr,
        "task_success": task_success,
        "coverage_gap": coverage_gap,
    }
