"""Deterministic Citation Verification Node for Approach B.

Implements strict post-generation verification against retrieved legal contexts:
1. Audits every statutory citation (Act name + section number) against the retrieved STATUTE chunks that
   passed the applicability check, using exact section matching paired conjunctively with Act-token
   verification (preventing numeric prefix collisions like Section 16 vs 166 and cross-Act splicing such as
   BNS vs BNSS). Citations are never re-grounded against the wider corpus.
2. Audits every precedent citation (case title / citation) against retrieved judgments.
3. Strips any unverified or fabricated citation and sanitizes step-level statutory_basis fields
   (including 1- and 2-digit unverified section numbers, and sections attributed to the wrong Act).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.proposed.state import AgentState
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
    StatutoryCitation,
    acts_share_significant_token,
    canonical_act_keys,
    normalize_section_id,
    sections_match,
)

logger = logging.getLogger(__name__)

_EXPLICIT_SECTION_PATTERN = re.compile(
    r"\b(?:sections?|secs?\.?|s\.)\s*(\d+(?:\s*-?\s*[a-z](?![a-z]))?(?:\s*\(\s*[0-9a-z]+\s*\))?)",
    re.IGNORECASE,
)


class VerificationNode:
    """Enforces 0% statutory hallucination by verifying and pruning ungrounded citations.

    Citations are only verified against the retrieved contexts that passed the applicability check. A
    citation that is merely present somewhere in the corpus is not re-grounded, and no precedent is added
    when the planner cited none: a law is shown only if it was retrieved for, and applies to, the facts.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def verify(self, state: AgentState) -> Dict[str, Any]:
        """Audit draft action plan and citations against retrieved legal corpus.

        Args:
            state: Current AgentState containing retrieved_contexts and draft outputs.

        Returns:
            Dictionary updating verified_action_plan, verified_statutory_citations,
            verified_precedent_citations, hallucination_check_passed, and unverified_citations_stripped.
        """
        retrieved_contexts: List[RetrievedContext] = state.get("retrieved_contexts", [])
        draft_actions: List[ProceduralActionStep] = state.get("draft_action_plan", [])
        draft_statutes: List[StatutoryCitation] = state.get("draft_statutory_citations", [])
        draft_precedents: List[PrecedentCitation] = state.get("draft_precedent_citations", [])
        scenario_domain = state.get("domain")

        domain_compatible_contexts: List[RetrievedContext] = [
            ctx
            for ctx in retrieved_contexts
            if not scenario_domain
            or scenario_domain == LegalDomain.GENERAL_DISPUTE
            or ctx.domain is None
            or ctx.domain in (scenario_domain, LegalDomain.GENERAL_DISPUTE)
            or ctx.act_category == "criminal"
        ]

        verified_statutes_map: Dict[str, RetrievedContext] = {}
        verified_precedents_map: Dict[str, RetrievedContext] = {}

        for ctx in domain_compatible_contexts:
            if ctx.doc_type == DocumentType.STATUTE:
                norm_key = self._normalize_statute_key(ctx.act_name or ctx.title, ctx.citation_or_section)
                verified_statutes_map[norm_key] = ctx
            elif ctx.doc_type == DocumentType.PRECEDENT:
                norm_key = self._normalize_precedent_key(ctx.title)
                verified_precedents_map[norm_key] = ctx

        unverified_stripped: List[str] = []
        stripped_section_ids: Set[str] = set()

        # 1. Audit Statutory Citations (exact section + same Act, against retrieved STATUTE chunks only)
        verified_statutes: List[StatutoryCitation] = []
        for stat in draft_statutes:
            stat_key = self._normalize_statute_key(stat.act_name, stat.section_number)
            matched_ctx = verified_statutes_map.get(stat_key)

            if matched_ctx is None:
                matched_ctx = self._find_matching_statute_context(stat, domain_compatible_contexts)

            if matched_ctx is not None:
                if matched_ctx.source_url:
                    stat.source_url = matched_ctx.source_url
                stat.is_verified = True
                verified_statutes.append(stat)
            else:
                unverified_stripped.append(f"Statute: {stat.act_name} Section {stat.section_number}")
                norm_sec = normalize_section_id(stat.section_number)
                if norm_sec:
                    stripped_section_ids.add(norm_sec)
                logger.info(
                    "Stripped ungrounded statutory citation: %s Section %s",
                    stat.act_name,
                    stat.section_number,
                )

        # 2. Audit Case Precedent Citations strictly against retrieved PRECEDENT titles/citations
        _PRECEDENT_STOPWORDS = {
            "state", "union", "india", "others", "versus", "anr", "ors", "limited", "private",
            "company", "corporation", "authority", "development", "consumer", "service", "services",
            "delivery", "express", "transport", "insurance", "national", "industries", "developers",
            "builders", "division", "board", "trust", "kumar", "sharma", "singh", "ram", "pvt", "ltd",
            "animal", "animals", "welfare",
        }
        precedent_contexts = [c for c in retrieved_contexts if c.doc_type == DocumentType.PRECEDENT]
        verified_precedents: List[PrecedentCitation] = []
        for prec in draft_precedents:
            prec_key = self._normalize_precedent_key(prec.case_title)
            title_words = [
                w
                for w in re.split(r"\W+", prec.case_title.lower())
                if len(w) > 3 and w not in _PRECEDENT_STOPWORDS
            ]
            prec_cit_clean = re.sub(r"\s+", " ", (prec.citation or "").strip().lower())

            matched_ctx = verified_precedents_map.get(prec_key)
            if matched_ctx is None and precedent_contexts:
                for p_ctx in precedent_contexts:
                    p_title_blob = f"{p_ctx.title} {p_ctx.citation_or_section}".lower()
                    p_cit_clean = re.sub(r"\s+", " ", (p_ctx.citation_or_section or "").strip().lower())
                    citation_matched = bool(
                        prec_cit_clean and p_cit_clean and (prec_cit_clean in p_cit_clean or p_cit_clean in prec_cit_clean)
                    )
                    matched_words = [
                        w for w in title_words if re.search(rf"\b{re.escape(w)}\b", p_title_blob)
                    ]
                    if citation_matched or len(matched_words) >= 2 or (
                        len(title_words) == 1 and len(matched_words) == 1
                    ):
                        matched_ctx = p_ctx
                        break

            if matched_ctx is not None:
                # Canonicalize title, citation, court, and URL to prevent cross-judgment mismatch
                prec.case_title = re.sub(r"\s*\((?:\(\d{4}\)|\d{4}|AIR).*$", "", matched_ctx.title).strip()
                if matched_ctx.citation_or_section:
                    prec.citation = matched_ctx.citation_or_section
                if matched_ctx.court:
                    prec.court = matched_ctx.court
                if matched_ctx.source_url:
                    prec.source_url = matched_ctx.source_url

                prec.is_verified = True
                verified_precedents.append(prec)
            else:
                unverified_stripped.append(f"Precedent: {prec.case_title}")
                logger.info("Stripped ungrounded judicial precedent: %s", prec.case_title)

        # 3. Audit Action Plan Steps (including 1- and 2-digit unverified section references)
        allowed_pairs: List[Tuple[str, str]] = [
            (normalize_section_id(s.section_number), s.act_name or "")
            for s in verified_statutes
            if normalize_section_id(s.section_number)
        ]
        for ctx in retrieved_contexts:
            if ctx.doc_type == DocumentType.STATUTE:
                cid = normalize_section_id(ctx.citation_or_section)
                if cid:
                    allowed_pairs.append((cid, ctx.act_name or ctx.title or ""))

        stripped_tokens: Set[str] = set()
        for item in unverified_stripped:
            tokens = re.findall(r"\b\w+\b", item.lower())
            for t in tokens:
                if len(t) > 2 and t not in {
                    "statute", "precedent", "section", "act", "case", "india", "indian", "code", "1988", "2019", "2023", "1882", "1963"
                }:
                    stripped_tokens.add(t)

        verified_actions: List[ProceduralActionStep] = []
        for step in draft_actions:
            cloned_step = step.model_copy()
            if cloned_step.statutory_basis:
                basis = cloned_step.statutory_basis
                basis_tokens = set(re.findall(r"\b\w+\b", basis.lower()))
                explicit_secs = [
                    normalize_section_id(m.group(1))
                    for m in _EXPLICIT_SECTION_PATTERN.finditer(basis)
                ]
                named_acts = canonical_act_keys(basis)
                has_stripped_sec = any(sec in stripped_section_ids for sec in explicit_secs if sec)
                has_unverified_sec = any(
                    not self._section_allowed(sec, basis, named_acts, allowed_pairs)
                    for sec in explicit_secs
                    if sec
                )
                if has_stripped_sec or has_unverified_sec or (
                    not explicit_secs and basis_tokens.intersection(stripped_tokens)
                ):
                    cloned_step.statutory_basis = None

            verified_actions.append(cloned_step)

        hallucination_check_passed = len(unverified_stripped) == 0

        return {
            "verified_action_plan": verified_actions,
            "verified_statutory_citations": verified_statutes,
            "verified_precedent_citations": verified_precedents,
            "hallucination_check_passed": hallucination_check_passed,
            "unverified_citations_stripped": unverified_stripped,
        }

    @staticmethod
    def _section_allowed(
        section_id: str,
        basis: str,
        named_acts: frozenset,
        allowed_pairs: List[Tuple[str, str]],
    ) -> bool:
        """Return True if a section referenced in a step's statutory_basis is backed by a retrieved statute.

        When the basis names an Act ("Section 173, Motor Vehicles Act"), the section must belong to that Act:
        BNSS s.173 being retrieved does not back a claim about MVA s.173. Without an Act name, a section match
        is allowed only if the matching retrieved statutes belong to at most one canonical Act.
        """
        candidates = [act for sec, act in allowed_pairs if sections_match(section_id, sec)]
        if not candidates:
            return False
        if not named_acts:
            distinct_acts = {k for act in candidates for k in canonical_act_keys(act)}
            return len(distinct_acts) <= 1
        for act in candidates:
            act_keys = canonical_act_keys(act)
            if act_keys:
                if act_keys & named_acts:
                    return True
            elif acts_share_significant_token(act, basis):
                return True
        return False

    def _find_matching_statute_context(
        self,
        stat: StatutoryCitation,
        retrieved_contexts: List[RetrievedContext],
    ) -> Optional[RetrievedContext]:
        """Match a statutory citation to a retrieved STATUTE chunk with the same section and the same Act.

        Precedent chunks that merely mention the section are never used, so a statute's URL always comes from
        the statute itself.
        """
        for ctx in retrieved_contexts:
            if ctx.doc_type != DocumentType.STATUTE:
                continue
            if not sections_match(stat.section_number, ctx.citation_or_section):
                continue
            if acts_share_significant_token(stat.act_name, ctx.act_name or ctx.title):
                return ctx
        return None

    def _normalize_statute_key(self, act_name: Optional[str], section_str: Optional[str]) -> str:
        """Create a normalized key for statute matching, preserving parenthesized subsections."""
        act_clean = re.sub(r"[^\w]", "", (act_name or "").lower())
        sec_clean = normalize_section_id(section_str)
        return f"{act_clean}_{sec_clean}"

    def _normalize_precedent_key(self, title: str) -> str:
        """Create a normalized key for case title matching."""
        return re.sub(r"[^\w]", "", (title or "").lower())
