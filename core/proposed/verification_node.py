"""Deterministic Citation Verification Node for Approach B.

Implements strict post-generation verification against retrieved legal contexts:
1. Audits every statutory citation (Act name + section number) against retrieved corpus chunks
   using word-boundary matching paired conjunctively with Act-token verification (preventing
   numeric prefix collisions like Section 16 vs 166 and cross-Act splicing).
2. Audits every precedent citation (case title / citation) against retrieved judgments.
3. Strips any unverified or fabricated citation and sanitizes step-level statutory_basis fields
   (including 1- and 2-digit unverified section numbers).
"""

import re
from typing import Any, Dict, List, Optional, Set

from solvemycase.config.settings import Settings, get_settings
from solvemycase.core.proposed.state import AgentState
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    RetrievedContext,
    StatutoryCitation,
    acts_share_significant_token,
    normalize_section_id,
    section_mentioned_with_boundary,
    sections_match,
)
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore

_EXPLICIT_SECTION_PATTERN = re.compile(
    r"\b(?:sections?|secs?\.?|s\.)\s*(\d+\s*-?\s*[a-z]?(?:\(\s*[0-9a-z]+\s*\))?)",
    re.IGNORECASE,
)


class VerificationNode:
    """Enforces 0% statutory hallucination by verifying and pruning ungrounded citations."""

    def __init__(self, settings: Optional[Settings] = None, store: Optional[QdrantLegalStore] = None):
        self.settings = settings or get_settings()
        self.store = store

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

        # 1. Audit Statutory Citations (Conjunctive section boundary + Act-token check)
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
                regrounded = False
                if self.store:
                    reground_hits = self.store.exact_section_search(stat.section_number, stat.act_name)
                    if reground_hits:
                        stat.source_url = reground_hits[0].source_url
                        stat.is_verified = True
                        verified_statutes.append(stat)
                        regrounded = True
                        print(
                            f"[VerificationNode] RE-GROUNDED statutory citation: "
                            f"{stat.act_name} Section {stat.section_number}"
                        )

                if not regrounded:
                    unverified_stripped.append(f"Statute: {stat.act_name} Section {stat.section_number}")
                    norm_sec = normalize_section_id(stat.section_number)
                    if norm_sec:
                        stripped_section_ids.add(norm_sec)
                    print(
                        f"[VerificationNode] STRIPPED ungrounded statutory citation: "
                        f"{stat.act_name} Section {stat.section_number}"
                    )

        # 2. Audit Case Precedent Citations strictly against retrieved PRECEDENT titles/citations
        _PRECEDENT_STOPWORDS = {
            "state", "union", "india", "others", "versus", "anr", "ors", "limited", "private",
            "company", "corporation", "authority", "development", "consumer", "service", "services",
            "delivery", "express", "transport", "insurance", "national", "industries", "developers",
            "builders", "division", "board", "trust", "kumar", "sharma", "singh", "ram", "pvt", "ltd",
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

            matched_ctx = verified_precedents_map.get(prec_key)
            if matched_ctx is None and title_words and precedent_contexts:
                for p_ctx in precedent_contexts:
                    p_title_blob = f"{p_ctx.title} {p_ctx.citation_or_section}".lower()
                    if any(re.search(rf"\b{re.escape(w)}\b", p_title_blob) for w in title_words):
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
                print(f"[VerificationNode] STRIPPED ungrounded judicial precedent: {prec.case_title}")

        # 2b. If no precedent was verified (e.g., LLM omitted or cited unretrieved case), ground the top retrieved domain precedent
        if not verified_precedents and precedent_contexts:
            top_p = precedent_contexts[0]
            clean_title = re.sub(r"\s*\((?:\(\d{4}\)|\d{4}|AIR).*$", "", top_p.title).strip()
            year_match = re.search(r"\b(19\d\d|20\d\d)\b", f"{top_p.citation_or_section} {top_p.title}")
            verified_precedents.append(
                PrecedentCitation(
                    case_title=clean_title,
                    court=top_p.court or "Supreme Court of India",
                    year=int(year_match.group(1)) if year_match else None,
                    citation=top_p.citation_or_section,
                    legal_principle=top_p.text,
                    source_url=top_p.source_url,
                    is_verified=True,
                )
            )

        # 3. Audit Action Plan Steps (including 1- and 2-digit unverified section references)
        allowed_section_ids: Set[str] = {
            normalize_section_id(s.section_number) for s in verified_statutes if normalize_section_id(s.section_number)
        }
        for ctx in retrieved_contexts:
            if ctx.doc_type == DocumentType.STATUTE:
                cid = normalize_section_id(ctx.citation_or_section)
                if cid:
                    allowed_section_ids.add(cid)

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
                basis_lower = cloned_step.statutory_basis.lower()
                basis_tokens = set(re.findall(r"\b\w+\b", basis_lower))
                explicit_secs = [
                    normalize_section_id(m.group(1))
                    for m in _EXPLICIT_SECTION_PATTERN.finditer(cloned_step.statutory_basis)
                ]
                has_stripped_sec = any(sec in stripped_section_ids for sec in explicit_secs if sec)
                has_unverified_sec = any(
                    not any(sections_match(sec, allowed) for allowed in allowed_section_ids)
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

    def _find_matching_statute_context(
        self,
        stat: StatutoryCitation,
        retrieved_contexts: List[RetrievedContext],
    ) -> Optional[RetrievedContext]:
        """Match a statutory citation against retrieved chunks using strict section boundary AND Act token overlap."""
        for ctx in retrieved_contexts:
            ctx_act_source = f"{ctx.act_name or ''} {ctx.title}"
            if ctx.doc_type == DocumentType.STATUTE and sections_match(stat.section_number, ctx.citation_or_section):
                if acts_share_significant_token(stat.act_name, ctx_act_source):
                    return ctx
            chunk_blob = f"{ctx.title} {ctx.citation_or_section} {ctx.text}"
            if section_mentioned_with_boundary(stat.section_number, chunk_blob) and acts_share_significant_token(
                stat.act_name, f"{ctx_act_source} {ctx.text}"
            ):
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
