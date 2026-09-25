"""RAG Noise-Sensitivity Stress Harness (`P1 Slice 2`).

Evaluates how resilient the ProceduralPlannerAgent + VerificationNode pipeline is when
irrelevant cross-domain statutory chunks (distractors) are injected into `retrieved_contexts`.
Measures:
- `distractor_citation_rate`: Fraction of final verified statutory citations originating from distractor chunks.
- `plan_phase_retention`: Fraction of procedural phases from the clean run retained under noise injection.
"""

from typing import Any, Dict, List
from solvemycase.core.proposed.procedural_planner import ProceduralPlannerAgent
from solvemycase.core.proposed.state import AgentState
from solvemycase.core.proposed.verification_node import VerificationNode
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    RetrievedContext,
    acts_share_significant_token,
    sections_match,
)


def build_default_distractor_contexts(target_domain: LegalDomain) -> List[RetrievedContext]:
    """Create cross-domain statutory distractor chunks unrelated to `target_domain`."""
    distractors: List[RetrievedContext] = []
    if target_domain != LegalDomain.PROPERTY_CONFLICT:
        distractors.append(
            RetrievedContext(
                chunk_id="distractor_tpa_122",
                doc_type=DocumentType.STATUTE,
                domain=LegalDomain.PROPERTY_CONFLICT,
                title="Transfer of Property Act, 1882 - Section 122",
                citation_or_section="Section 122",
                act_name="Transfer of Property Act, 1882",
                text="Section 122: Gift defined. Gift is the transfer of certain existing moveable or immoveable property made voluntarily and without consideration, by one person, called the donor, to another, called the donee.",
                source_url="https://indiacode.gov.in/handle/123456789/2338",
                score=0.31,
            )
        )
    if target_domain != LegalDomain.CONSUMER_RIGHTS:
        distractors.append(
            RetrievedContext(
                chunk_id="distractor_cpa_89",
                doc_type=DocumentType.STATUTE,
                domain=LegalDomain.CONSUMER_RIGHTS,
                title="Consumer Protection Act, 2019 - Section 89",
                citation_or_section="Section 89",
                act_name="Consumer Protection Act, 2019",
                text="Section 89: Punishment for false or misleading advertisement. Any manufacturer or service provider who causes a false or misleading advertisement to be made which is prejudicial to the interest of consumers shall be punished.",
                source_url="https://indiacode.gov.in/handle/123456789/15256",
                score=0.29,
            )
        )
    if target_domain != LegalDomain.MOTOR_VEHICLE_ACCIDENT:
        distractors.append(
            RetrievedContext(
                chunk_id="distractor_mva_185",
                doc_type=DocumentType.STATUTE,
                domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
                title="Motor Vehicles Act, 1988 - Section 185",
                citation_or_section="Section 185",
                act_name="Motor Vehicles Act, 1988",
                text="Section 185: Driving by a drunken person or by a person under the influence of drugs. Whoever, while driving, or attempting to drive, a motor vehicle, has in his blood, alcohol exceeding 30 mg per 100 ml of blood.",
                source_url="https://indiacode.gov.in/handle/123456789/1798",
                score=0.28,
            )
        )
    return distractors


def evaluate_noise_sensitivity(
    planner: ProceduralPlannerAgent,
    verifier: VerificationNode,
    scenario: str,
    domain: LegalDomain,
    clean_contexts: List[RetrievedContext],
    distractor_contexts: List[RetrievedContext],
) -> Dict[str, Any]:
    """Run clean vs. distractor-injected contexts through Planner + Verifier and compute noise sensitivity metrics."""
    clean_state: AgentState = {
        "scenario": scenario,
        "is_legal": True,
        "domain": domain,
        "retrieved_contexts": clean_contexts,
        "key_entities": {},
    }
    clean_plan_out = planner.plan(clean_state)
    clean_state.update(clean_plan_out)
    clean_ver_out = verifier.verify(clean_state)
    clean_phases = {step.phase.value for step in clean_ver_out.get("verified_action_plan", [])}

    # Interleave distractors with clean contexts
    noisy_contexts: List[RetrievedContext] = []
    max_len = max(len(clean_contexts), len(distractor_contexts))
    for idx in range(max_len):
        if idx < len(clean_contexts):
            noisy_contexts.append(clean_contexts[idx])
        if idx < len(distractor_contexts):
            noisy_contexts.append(distractor_contexts[idx])

    noisy_state: AgentState = {
        "scenario": scenario,
        "is_legal": True,
        "domain": domain,
        "retrieved_contexts": noisy_contexts,
        "key_entities": {},
    }
    noisy_plan_out = planner.plan(noisy_state)
    noisy_state.update(noisy_plan_out)
    noisy_ver_out = verifier.verify(noisy_state)

    noisy_citations = noisy_ver_out.get("verified_statutory_citations", [])
    noisy_phases = {step.phase.value for step in noisy_ver_out.get("verified_action_plan", [])}

    distractor_cited_count = 0
    for cit in noisy_citations:
        matches_clean = any(
            acts_share_significant_token(cit.act_name, c.act_name or c.title)
            and sections_match(cit.section_number, c.citation_or_section)
            for c in clean_contexts
        )
        matches_distractor = any(
            acts_share_significant_token(cit.act_name, d.act_name or d.title)
            and sections_match(cit.section_number, d.citation_or_section)
            for d in distractor_contexts
        )
        if matches_distractor and not matches_clean:
            distractor_cited_count += 1

    distractor_citation_rate = (
        round(distractor_cited_count / len(noisy_citations), 4) if noisy_citations else 0.0
    )
    plan_phase_retention = (
        round(len(clean_phases & noisy_phases) / len(clean_phases), 4) if clean_phases else 1.0
    )

    return {
        "clean_citation_count": len(clean_ver_out.get("verified_statutory_citations", [])),
        "noisy_citation_count": len(noisy_citations),
        "distractor_cited_count": distractor_cited_count,
        "distractor_citation_rate": distractor_citation_rate,
        "plan_phase_retention": plan_phase_retention,
    }
