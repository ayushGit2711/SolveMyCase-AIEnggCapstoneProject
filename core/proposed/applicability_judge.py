"""Applicability check: keep only retrieved provisions and judgments that apply to the user's facts.

Hybrid search and cross-encoder reranking score word overlap, not legal applicability: for "a watchman
killed my pet cat" they happily return offences about the death of a human being. This node asks the fast
LLM, once per query, which of the top reranked candidates a lawyer would actually rely on for these facts.

- LLM mode: candidates the model marks applicable are kept in rerank order (capped at ``rerank_top_k``).
  Only a complete verdict that rejects every candidate counts as "nothing applies" (a coverage gap). A
  partial, truncated or malformed answer is treated as an error instead, so it can't produce a false gap.
- Offline / disabled / error: pass the top ``rerank_top_k`` candidates through unchanged and record the mode.
  The graph then withholds unscreened sources for general_dispute questions (see ``withhold_unscreened``).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from solvemycase.config.openai_client import build_openai_client
from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import ApplicabilityMode, DocumentType, LegalDomain, RetrievedContext

logger = logging.getLogger(__name__)

CANDIDATE_TEXT_CHARS = 700

# Output budget: JSON overhead plus one short verdict (id, flag, <= 20-word reason) per candidate.
_BASE_OUTPUT_TOKENS = 150
_TOKENS_PER_CANDIDATE = 60
_MAX_OUTPUT_TOKENS = 3000

# Modes in which no LLM verdict exists for the retrieved sources.
UNSCREENED_MODES = frozenset({ApplicabilityMode.OFFLINE, ApplicabilityMode.ERROR})

APPLICABILITY_SYSTEM_PROMPT = """You screen Indian legal sources for a legal-aid assistant.
You receive a person's facts and numbered candidate sources (statutory sections and court judgments) returned by a
search engine. Search engines also return sources that merely share words with the facts. Keep only the sources
that APPLY to these facts.

A source is applicable if a lawyer handling these facts would rely on it, because it is:
- the offence the facts disclose (e.g. killing or maiming an animal under BNS s.325 / PCA s.11, rash driving or death
  by negligence in a road crash, cheating to induce delivery of property under BNS s.318, criminal breach of trust
  under BNS s.316, criminal trespass/mischief to property);
- the civil remedy or liability for the facts (e.g. MACT compensation claim or insurer liability under the Motor
  Vehicles Act; recovery of possession, lease termination/eviction under Transfer of Property Act s.106 and s.111,
  specific performance, or perpetual/mandatory injunctions under Specific Relief Act s.38 and s.39 for property,
  tenancy, driveway, compound encroachment, or housing-society common-area/terrace access disputes; or consumer
  complaint under the Consumer Protection Act for defective goods or deficient commercial services);
- the forum or procedure for pursuing that remedy (e.g. registering an FIR under BNSS s.173 when a candidate
  substantive offence applies, filing before MACT, Civil Court, or Consumer Commission, temporary injunction under
  CPC Order 39); or
- a court judgment whose legal principle is on point for these facts.

A source is NOT applicable when its subject differs from the facts, even if words overlap. For example:
- an offence about the death of or hurt to a human being when only an animal was harmed, or an animal-cruelty law
  (BNS s.325, PCA s.11, AWBI v. Nagaraja) when an animal bit a human;
- a motor-vehicle provision when no motor vehicle is involved;
- a tenancy or property section for a consumer purchase, or a consumer judgment for a tenancy dispute;
- Consumer Protection Act provisions (including s.2(11) deficiency in service) for an employee claiming unpaid salary,
  wages, notice-period dues, or EPF from an employer (employment is a contract of personal service excluded by CPA
  s.2(42));
- Specific Relief Act s.6 (summary suit after actual dispossession of immovable property) when the person has not been
  dispossessed or the dispute is matrimonial/dowry harassment;
- BNS s.352 (insult to provoke breach of public peace) or BNS s.318 (cheating) when the complaint is solely
  social-media defamation of reputation (governed by BNS s.356), or BNS s.318 / BNS s.316 when a cheque for goods
  supplied is dishonoured by the bank for "Insufficient Funds" (governed by Negotiable Instruments Act 1881 s.138);
- a generic police-procedure section (BNSS s.173, s.176, s.193 or CrPC s.154 on FIR / investigation) when NONE of the
  candidate substantive offences in the list applies to the facts;
- a section of the repealed IPC / CrPC / Indian Evidence Act when the matching BNS / BNSS / BSA section is among the
  candidates, unless the facts clearly happened before 1 July 2024.

Judge every candidate independently, on its own text. If none of the candidates governs the user's legal issue (for
example: unpaid salary, divorce/maintenance, dowry harassment, defamation, cheque bounce, or dog bite), mark all
candidates `"applicable": false`.
Return JSON only, with one entry for EVERY candidate id:
{"assessments": [{"id": "S1", "applicable": true, "reason": "at most 20 words"}]}
"""


class ApplicabilityOutputError(ValueError):
    """The judge's answer can't be trusted: invalid JSON, truncated, incomplete or self-contradictory."""


class ApplicabilityAssessment(BaseModel):
    """Result of screening reranked candidates for applicability to the facts."""

    applicable: List[RetrievedContext] = Field(default_factory=list, description="Candidates kept, rerank order.")
    rejected: List[RetrievedContext] = Field(default_factory=list, description="Candidates judged not applicable.")
    coverage_gap: bool = Field(default=False, description="True when no candidate applies to the facts.")
    mode: ApplicabilityMode = Field(default=ApplicabilityMode.OFFLINE, description="How the check ran.")
    reasons: Dict[str, str] = Field(default_factory=dict, description="chunk_id -> short reason from the judge.")


def _clip(text: str, limit: int = CANDIDATE_TEXT_CHARS) -> str:
    """Trim text to about ``limit`` characters at a word boundary."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut} ..."


def _domain_value(domain: Any) -> str:
    if isinstance(domain, LegalDomain):
        return domain.value
    return str(domain or LegalDomain.GENERAL_DISPUTE.value)


def withhold_unscreened(domain: Any, mode: Any) -> bool:
    """Return True when unscreened sources must not be cited for this question.

    general_dispute has no dedicated corpus, so its nearest neighbours are often unrelated law (e.g. offences
    about a human death for a harmed pet). Without an LLM verdict (offline, or the check failed) such questions
    get a coverage-gap answer instead. Covered domains keep their domain-filtered sources, and an operator who
    disables the check entirely gets the unscreened behaviour.
    """
    try:
        mode = ApplicabilityMode(mode)
    except ValueError:
        return False
    return mode in UNSCREENED_MODES and _domain_value(domain) == LegalDomain.GENERAL_DISPUTE.value


def format_candidates(candidates: List[RetrievedContext]) -> str:
    """Render candidates as short-id blocks (S1..Sn) for the judge prompt."""
    blocks = []
    for idx, cand in enumerate(candidates, start=1):
        if cand.doc_type == DocumentType.STATUTE:
            header = f"[S{idx}] STATUTE | {cand.act_name or 'Unknown Act'} | {cand.citation_or_section} | {cand.title}"
        else:
            header = (
                f"[S{idx}] JUDGMENT | {cand.court or 'Court'} | {cand.citation_or_section} | {cand.title}"
            )
        blocks.append(f"{header}\nText: {_clip(cand.text)}")
    return "\n\n".join(blocks)


def _normalize_id(raw: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(raw or "")).upper()


_TRUE_WORDS = frozenset({"true", "yes", "y", "1", "applicable", "applies"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0", "not applicable", "inapplicable", "does not apply"})


def _as_verdict(value: Any) -> Optional[bool]:
    """Read an 'applicable' flag; None when the value is missing or not a recognisable yes/no."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return None


def parse_assessments(raw_content: Optional[str], n_candidates: int) -> Dict[str, Tuple[bool, str]]:
    """Parse the judge's JSON into {short_id: (applicable, reason)}.

    Unknown ids and unreadable flags are ignored, and an id given contradictory verdicts is dropped.

    Raises:
        ApplicabilityOutputError: the output is not the expected JSON or holds no usable verdict.
    """
    try:
        data = json.loads(raw_content or "")
    except json.JSONDecodeError as err:
        raise ApplicabilityOutputError(f"applicability output is not valid JSON ({err})") from err
    items = data.get("assessments") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ApplicabilityOutputError("applicability output has no 'assessments' list")

    valid_ids = {f"S{i}" for i in range(1, n_candidates + 1)}
    verdicts: Dict[str, Tuple[bool, str]] = {}
    conflicting = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        short_id = _normalize_id(item.get("id"))
        verdict = _as_verdict(item.get("applicable"))
        if short_id not in valid_ids or verdict is None:
            continue
        if short_id in verdicts and verdicts[short_id][0] != verdict:
            conflicting.add(short_id)
            continue
        verdicts.setdefault(short_id, (verdict, str(item.get("reason") or "")[:240]))
    for short_id in conflicting:
        verdicts.pop(short_id, None)
    if not verdicts:
        raise ApplicabilityOutputError("applicability output contains no usable verdict")
    return verdicts


def resolve_verdicts(
    candidates: List[RetrievedContext],
    verdicts: Dict[str, Tuple[bool, str]],
    top_k: int,
) -> ApplicabilityAssessment:
    """Turn per-candidate verdicts into an assessment (pure decision logic).

    - At least one candidate applies: keep the applicable ones in rerank order (at most ``top_k``).
    - Every candidate has an explicit "not applicable" verdict: coverage gap.
    - Otherwise some candidates were skipped and none was accepted. That is not evidence that nothing applies.

    Raises:
        ApplicabilityOutputError: for the incomplete case, so the caller fails open instead of reporting a gap.
    """
    applicable: List[RetrievedContext] = []
    rejected: List[RetrievedContext] = []
    reasons: Dict[str, str] = {}
    unassessed = 0
    for idx, cand in enumerate(candidates, start=1):
        verdict = verdicts.get(f"S{idx}")
        if verdict is None:
            unassessed += 1
            reasons[cand.chunk_id] = "not assessed by the applicability check"
            rejected.append(cand)
            continue
        is_applicable, reason = verdict
        reasons[cand.chunk_id] = reason
        (applicable if is_applicable else rejected).append(cand)

    if not applicable and unassessed:
        raise ApplicabilityOutputError(
            f"no candidate accepted and {unassessed} of {len(candidates)} left unassessed"
        )
    return ApplicabilityAssessment(
        applicable=applicable[:top_k],
        rejected=rejected,
        coverage_gap=not applicable,
        mode=ApplicabilityMode.LLM,
        reasons=reasons,
    )


class ApplicabilityJudge:
    """Screens reranked candidates with one fast-LLM call; fails open when no LLM verdict is available."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[Any] = None):
        self.settings = settings or get_settings()
        self._client = client
        if self._client is None:
            # The check sits on every query's critical path: keep its timeout short and retry at most once.
            self._client = build_openai_client(
                self.settings,
                timeout=self.settings.applicability_timeout_seconds,
                max_retries=min(1, self.settings.openai_max_retries),
            )

    def assess(
        self,
        scenario: str,
        domain: Any,
        candidates: List[RetrievedContext],
        top_k: Optional[int] = None,
    ) -> ApplicabilityAssessment:
        """Split candidates into applicable and rejected ones.

        Args:
            scenario: The user's facts.
            domain: Classified legal domain.
            candidates: Reranked candidates, best first.
            top_k: Maximum number of applicable candidates to keep (defaults to ``rerank_top_k``).

        Returns:
            ApplicabilityAssessment. ``coverage_gap`` is True when the LLM rejected every candidate or when
            there were no candidates at all.
        """
        top_k = top_k or self.settings.rerank_top_k
        candidates = list(candidates or [])

        if not self.settings.applicability_check_enabled:
            return self._fail_open(candidates, top_k, ApplicabilityMode.DISABLED)
        if self._client is None:
            return self._fail_open(candidates, top_k, ApplicabilityMode.OFFLINE)
        if not candidates:
            return ApplicabilityAssessment(coverage_gap=True, mode=ApplicabilityMode.LLM)

        try:
            verdicts = self._llm_verdicts(scenario, domain, candidates)
            return resolve_verdicts(candidates, verdicts, top_k)
        except Exception as err:  # API, timeout, JSON and incomplete-verdict errors all fail open
            logger.warning(
                "Applicability check failed (%s: %s); passing the top reranked candidates through.",
                type(err).__name__,
                err,
            )
            return self._fail_open(candidates, top_k, ApplicabilityMode.ERROR)

    @staticmethod
    def _fail_open(
        candidates: List[RetrievedContext], top_k: int, mode: ApplicabilityMode
    ) -> ApplicabilityAssessment:
        """Pass the top candidates through unchanged (no LLM verdict, so nothing is judged inapplicable)."""
        return ApplicabilityAssessment(
            applicable=candidates[:top_k],
            rejected=[],
            coverage_gap=not candidates,
            mode=mode,
        )

    def _llm_verdicts(
        self, scenario: str, domain: Any, candidates: List[RetrievedContext]
    ) -> Dict[str, Tuple[bool, str]]:
        user_msg = (
            f"Domain (from a classifier, may be approximate): {_domain_value(domain)}\n"
            f"Facts:\n{scenario}\n\n"
            f"Candidates:\n{format_candidates(candidates)}"
        )
        response = self._client.chat.completions.create(
            model=self.settings.openai_model_fast,
            messages=[
                {"role": "system", "content": APPLICABILITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=min(_MAX_OUTPUT_TOKENS, _BASE_OUTPUT_TOKENS + _TOKENS_PER_CANDIDATE * len(candidates)),
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason not in (None, "stop"):
            raise ApplicabilityOutputError(f"applicability output incomplete (finish_reason={finish_reason})")
        return parse_assessments(choice.message.content, len(candidates))
