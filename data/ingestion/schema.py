"""Data schemas for legal corpus ingestion, indexing, and dual-output generation.

Adheres strictly to the open-india-law unified chunking specification and Pydantic v2 validation.
"""

from enum import Enum
from typing import List, Optional
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
