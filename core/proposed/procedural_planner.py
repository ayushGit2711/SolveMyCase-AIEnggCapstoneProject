"""Procedural Planning Agent for solvemycase.

Deconstructs legal scenarios into a chronological 6-phase dispute roadmap:
1. Immediate Actions
2. Police / Administrative Steps
3. Evidentiary Documentation
4. Legal Notice & Pre-Litigation
5. Forum & Jurisdictional Filing
6. Limitation Periods & Appeals
"""

import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI

from solvemycase.config.openai_client import build_openai_client
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
)

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert Indian Legal Procedural Engine.
Your task is to take a citizen's dispute scenario and the retrieved primary legal authorities,
and generate a comprehensive, strictly chronological step-by-step procedural roadmap.

The action plan must cover applicable chronological phases:
- immediate_action
- police_administrative (FIR under BNS/BNSS, GD entry, inspection)
- evidentiary_documentation (medical reports, receipts, title deeds, contracts)
- legal_notice (statutory notice, reply handling)
- forum_filing (MACT, District Consumer Commission, Civil Court)
- limitation_appeal (prescribed statutory deadlines)

For every step, specify:
- step_number (integer)
- phase (one of the 6 phases)
- title (action title)
- description (actionable practical details)
- forum_or_authority (e.g. Police Station, MACT, District Consumer Disputes Redressal Commission, Civil Court)
- statutory_basis (exact section and Act, e.g. "Section 166, Motor Vehicles Act, 1988"; leave empty if no context section applies)
- limitation_period (prescribed statutory time limit)
- priority (Critical, High, Medium, Low)

Citation rules (strict):
- Propose statutory citations and case precedents ONLY from the provided context documents, and ONLY when the
  document directly applies to the facts of this scenario. Never cite a document merely because it appears in the
  context; the context may contain loosely related material.
- A step's statutory_basis may only reference a section (with its Act) that appears in the context. Otherwise leave
  statutory_basis empty.
- For events on or after 1 July 2024 prefer the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita
  (BNSS) and Bharatiya Sakshya Adhiniyam (BSA) over the repealed IPC, CrPC and Indian Evidence Act.
- If the context has no law for some aspect of the problem, say so plainly in that step's description (for example
  "our database has no specific provision for this; consult a lawyer or the District Legal Services Authority")
  instead of citing an unrelated law.
- Precedent legal_principle must describe what the judgment actually held; do not embellish or extend it.

Format strictly as JSON:
{
    "action_plan": [
        {
            "step_number": 1,
            "phase": "immediate_action",
            "title": "...",
            "description": "...",
            "forum_or_authority": "...",
            "statutory_basis": "...",
            "limitation_period": "...",
            "priority": "Critical"
        }
    ],
    "statutory_citations": [
        {
            "act_name": "...",
            "section_number": "...",
            "summary_of_provision": "...",
            "applicability_to_scenario": "...",
            "source_url": "..."
        }
    ],
    "precedent_citations": [
        {
            "case_title": "...",
            "court": "...",
            "year": 2020,
            "citation": "...",
            "legal_principle": "...",
            "source_url": "..."
        }
    ]
}
"""


class ProceduralPlannerAgent:
    """Agent that creates structured, chronological legal procedural roadmaps."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._openai_client: Optional[OpenAI] = build_openai_client(self.settings)

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Generate draft procedural action plan and candidate citations.

        Args:
            state: Current AgentState.

        Returns:
            Dict containing draft_action_plan, draft_statutory_citations, draft_precedent_citations.
        """
        scenario = state["scenario"]
        domain = state["domain"]
        contexts = state.get("retrieved_contexts", [])

        context_text = "\n\n---\n\n".join(
            [f"[{c.doc_type.upper()}] {c.title} ({c.citation_or_section}):\n{c.text}\nSource URL: {c.source_url}" for c in contexts]
        )

        if self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model_primary,
                    messages=[
                        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Dispute Category: {domain.value}\n\nScenario:\n{scenario}\n\nVerified Legal Context:\n{context_text}",
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                data = json.loads(response.choices[0].message.content)
                actions = [ProceduralActionStep(**a) for a in data.get("action_plan", [])]
                statutes = [StatutoryCitation(**s) for s in data.get("statutory_citations", [])]
                precedents = [PrecedentCitation(**p) for p in data.get("precedent_citations", [])]

                return {
                    "draft_action_plan": actions,
                    "draft_statutory_citations": statutes,
                    "draft_precedent_citations": precedents,
                }
            except Exception as e:
                logger.warning("OpenAI planner generation failed (%s); using deterministic domain roadmap.", e)

        return self._generate_domain_roadmap(scenario, domain, contexts)

    def _generate_domain_roadmap(
        self, scenario: str, domain: LegalDomain, contexts: List[RetrievedContext]
    ) -> Dict[str, Any]:
        """Deterministic domain-specific roadmap generator for offline or test environments."""
        if domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT:
            actions = [
                ProceduralActionStep(
                    step_number=1,
                    phase=ProceduralPhase.IMMEDIATE_ACTION,
                    title="Emergency Medical Attention & Scene Documentation",
                    description="Ensure emergency medical care for injured persons (statutory duty under Section 134 Motor Vehicles Act). Photograph vehicle damage, skid marks, road conditions, and registration plates.",
                    forum_or_authority="Nearest Hospital & Scene of Incident",
                    statutory_basis="Section 134 Motor Vehicles Act, 1988",
                    limitation_period="Immediate (within 24 hours)",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=2,
                    phase=ProceduralPhase.POLICE_ADMINISTRATIVE,
                    title="Lodge First Information Report (FIR)",
                    description="Report accident details to the local police station having territorial jurisdiction. Ensure recording of FIR under Section 106/281 Bharatiya Nyaya Sanhita, 2023 (or IPC) and obtain a certified copy.",
                    forum_or_authority="Local Police Station / Jurisdictional SHO",
                    statutory_basis="Section 173 BNSS, 2023 & Section 106 BNS, 2023",
                    limitation_period="Promptly (ordinarily within 24-48 hours)",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=3,
                    phase=ProceduralPhase.EVIDENTIARY_DOCUMENTATION,
                    title="Procure Medico-Legal Certificate (MLC) & Police Report (FAR)",
                    description="Collect hospital discharge summaries, permanent disability certificates, and police First Accident Report (FAR/DAR) submitted to the Claims Tribunal.",
                    forum_or_authority="Government Hospital & Police Station",
                    statutory_basis="Rule 150A Central Motor Vehicles Rules",
                    limitation_period="Within 30-90 days of accident",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=4,
                    phase=ProceduralPhase.FORUM_FILING,
                    title="File Claim Petition before Motor Accidents Claims Tribunal (MACT)",
                    description="Lodge claim application under Section 166 (or Section 161 for hit-and-run) claiming compensation for medical expenses, loss of income, future prospects, and pain/suffering against offending driver, owner, and insurer.",
                    forum_or_authority="Motor Accidents Claims Tribunal (MACT)",
                    statutory_basis="Section 166 Motor Vehicles Act, 1988",
                    limitation_period="Within 6 months of occurrence of accident (as per 2019 Amendment)",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=5,
                    phase=ProceduralPhase.LIMITATION_APPEAL,
                    title="Statutory Appeal against MACT Award",
                    description="If aggrieved by the quantum of award or dismissal, file appeal under Section 173 of the Motor Vehicles Act before the jurisdictional High Court.",
                    forum_or_authority="High Court",
                    statutory_basis="Section 173 Motor Vehicles Act, 1988",
                    limitation_period="90 days from date of award",
                    priority="Medium",
                ),
            ]
        elif domain == LegalDomain.PROPERTY_CONFLICT:
            actions = [
                ProceduralActionStep(
                    step_number=1,
                    phase=ProceduralPhase.IMMEDIATE_ACTION,
                    title="Inspect Title Deeds & Current Possession Status",
                    description="Examine registered title deeds, chain of title, encumbrance certificates, and physical boundaries to verify ownership and settled possession.",
                    forum_or_authority="Sub-Registrar Office / Site Inspection",
                    statutory_basis="Section 54 Transfer of Property Act, 1882",
                    limitation_period="Immediate",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=2,
                    phase=ProceduralPhase.LEGAL_NOTICE,
                    title="Serve Statutory Notice to Quit / Cease Encroachment",
                    description="Issue a formal legal notice through registered post calling upon the opposing party to vacate, remove unauthorized structure, or cure breach within statutory notice period.",
                    forum_or_authority="Registered Post with A/D",
                    statutory_basis="Section 106 Transfer of Property Act, 1882",
                    limitation_period="15 days for monthly lease / immediate for trespass",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=3,
                    phase=ProceduralPhase.FORUM_FILING,
                    title="File Civil Suit for Recovery of Possession & Injunction",
                    description="If illegally dispossessed, institute a suit under Section 6 of the Specific Relief Act (summary possession) or regular title suit with prayer for temporary and perpetual injunction.",
                    forum_or_authority="Jurisdictional Civil Court (Senior/Junior Division)",
                    statutory_basis="Section 6 & Section 38 Specific Relief Act, 1963",
                    limitation_period="Strictly within 6 months of dispossession for Section 6 SRA",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=4,
                    phase=ProceduralPhase.LIMITATION_APPEAL,
                    title="Appeal / Revision against Decree or Order",
                    description="Lodge Regular First Appeal (RFA) under Section 96 CPC or Misc Appeal against refusal of temporary injunction under Order 43 CPC.",
                    forum_or_authority="District Court / High Court",
                    statutory_basis="Section 96 Code of Civil Procedure, 1908",
                    limitation_period="30 to 90 days depending on appellate forum",
                    priority="Medium",
                ),
            ]
        elif domain == LegalDomain.CONSUMER_RIGHTS:
            actions = [
                ProceduralActionStep(
                    step_number=1,
                    phase=ProceduralPhase.IMMEDIATE_ACTION,
                    title="Document Proof of Purchase & Defect Evidence",
                    description="Collate invoice, warranty card, transaction confirmation, delivery slip, service job sheets, and photographic/video evidence of product defect or service failure.",
                    forum_or_authority="Consumer / Merchant Records",
                    statutory_basis="Section 2(7) Consumer Protection Act, 2019",
                    limitation_period="Immediate",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=2,
                    phase=ProceduralPhase.LEGAL_NOTICE,
                    title="Serve Formal Consumer Legal Notice",
                    description="Dispatch formal notice to manufacturer and seller detailing deficiency in service or unfair trade practice, demanding refund, replacement, and compensation within 15 days.",
                    forum_or_authority="Registered Post / Email to Grievance Officer",
                    statutory_basis="Consumer Protection (E-Commerce) Rules, 2020",
                    limitation_period="15 days cure period",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=3,
                    phase=ProceduralPhase.FORUM_FILING,
                    title="File Consumer Complaint before District Commission (e-Daakhil)",
                    description="Lodge formal complaint under Section 35 of CPA 2019 via the e-Daakhil portal or physical registry claiming replacement, refund, and damages for mental agony and litigation costs.",
                    forum_or_authority="District Consumer Disputes Redressal Commission (DCDRC)",
                    statutory_basis="Section 35 Consumer Protection Act, 2019",
                    limitation_period="2 years from date of cause of action (Section 69 CPA 2019)",
                    priority="Critical",
                ),
                ProceduralActionStep(
                    step_number=4,
                    phase=ProceduralPhase.LIMITATION_APPEAL,
                    title="Statutory Appeal before State Commission",
                    description="If aggrieved by the order of the District Commission, file statutory appeal under Section 41 before the State Consumer Disputes Redressal Commission.",
                    forum_or_authority="State Consumer Disputes Redressal Commission (SCDRC)",
                    statutory_basis="Section 41 Consumer Protection Act, 2019",
                    limitation_period="45 days from date of District Commission order",
                    priority="Medium",
                ),
            ]
        else:
            actions = [
                ProceduralActionStep(
                    step_number=1,
                    phase=ProceduralPhase.IMMEDIATE_ACTION,
                    title="Collect Primary Records & Legal Timeline",
                    description="Compile all written communications, contracts, and relevant records chronologically.",
                    forum_or_authority="Private Records",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=2,
                    phase=ProceduralPhase.LEGAL_NOTICE,
                    title="Issue Formal Demand Notice",
                    description="Issue formal demand notice outlining obligations and breach.",
                    forum_or_authority="Registered Post",
                    priority="High",
                ),
                ProceduralActionStep(
                    step_number=3,
                    phase=ProceduralPhase.FORUM_FILING,
                    title="File Suit or Petition before Competent Court",
                    description="File proceeding before appropriate court having territorial and pecuniary jurisdiction.",
                    forum_or_authority="Competent Court",
                    priority="High",
                ),
            ]

        # Extract mapped citations from domain-compatible context documents
        statutes: List[StatutoryCitation] = []
        precedents: List[PrecedentCitation] = []

        for ctx in contexts:
            if (
                domain != LegalDomain.GENERAL_DISPUTE
                and ctx.domain is not None
                and ctx.domain not in (domain, LegalDomain.GENERAL_DISPUTE)
                and ctx.act_category != "criminal"
            ):
                continue
            if ctx.doc_type == DocumentType.STATUTE:
                statutes.append(
                    StatutoryCitation(
                        act_name=ctx.act_name or ctx.title,
                        section_number=ctx.citation_or_section.replace("Section ", ""),
                        summary_of_provision=ctx.text[:160] + "...",
                        applicability_to_scenario="Retrieved as potentially related to this domain; verify applicability with a lawyer.",
                        source_url=ctx.source_url,
                        is_verified=True,
                    )
                )
            elif ctx.doc_type == DocumentType.PRECEDENT:
                precedents.append(
                    PrecedentCitation(
                        case_title=ctx.title,
                        court=ctx.court or "Supreme Court of India",
                        legal_principle=ctx.text[:200] + "...",
                        source_url=ctx.source_url,
                        is_verified=True,
                    )
                )

        return {
            "draft_action_plan": actions,
            "draft_statutory_citations": statutes,
            "draft_precedent_citations": precedents,
        }
