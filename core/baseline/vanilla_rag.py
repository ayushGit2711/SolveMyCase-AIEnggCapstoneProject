"""Approach A: Baseline Vanilla RAG Pipeline.

Implements the single-prompt baseline architecture required for comparative evaluation:
1. Direct vector embedding of user query without decontextualization.
2. Single-pass dense retrieval from Qdrant vector store.
3. Monolithic prompt generating both legal advice and citations without agent verification.
"""

import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import (
    DocumentType,
    DualOutputResponse,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    ProceduralPhase,
    RetrievedContext,
    StatutoryCitation,
)
from solvemycase.data.vectorstore.indexer import EmbeddingProvider
from solvemycase.data.vectorstore.qdrant_store import QdrantLegalStore


VANILLA_RAG_SYSTEM_PROMPT = """You are an Indian legal assistant.
Given a user's legal scenario and retrieved legal context documents, generate:
1. A step-by-step procedural action plan.
2. A list of statutory provisions (sections of Indian Acts) and relevant court precedents.

Format your output strictly as a JSON object with this structure:
{
    "scenario_summary": "Summary of facts",
    "domain": "motor_vehicle_accident | property_conflict | consumer_rights | general_dispute",
    "action_plan": [
        {
            "step_number": 1,
            "phase": "immediate_action | police_administrative | evidentiary_documentation | legal_notice | forum_filing | limitation_appeal",
            "title": "Short title",
            "description": "Details",
            "forum_or_authority": "Relevant authority",
            "statutory_basis": "Section if known",
            "limitation_period": "Timeline if known",
            "priority": "Critical | High | Medium | Low"
        }
    ],
    "statutory_citations": [
        {
            "act_name": "Act name",
            "section_number": "Section number",
            "summary_of_provision": "Summary",
            "applicability_to_scenario": "Why applicable",
            "source_url": "URL from context or empty"
        }
    ],
    "precedent_citations": [
        {
            "case_title": "Case title",
            "court": "Court",
            "year": 2020,
            "citation": "Citation",
            "legal_principle": "Principle",
            "source_url": "URL from context or empty"
        }
    ]
}
"""


class VanillaRAGBaseline:
    """Baseline Vanilla RAG implementation without query expansion or verification nodes."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[QdrantLegalStore] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self.settings = settings or get_settings()
        self.store = store or QdrantLegalStore(settings=self.settings)
        self.embedder = embedder or EmbeddingProvider(settings=self.settings)
        self._openai_client: Optional[OpenAI] = None

        if self.settings.openai_api_key and self.settings.openai_api_key.get_secret_value():
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key.get_secret_value())

    def run(self, scenario: str, top_k: int = 4) -> DualOutputResponse:
        """Execute the baseline Vanilla RAG pipeline.

        Args:
            scenario: Natural language legal scenario narrative.
            top_k: Number of raw context chunks to retrieve.

        Returns:
            DualOutputResponse object.
        """
        # 1. Single direct vector embedding
        query_embedding = self.embedder.get_embeddings([scenario])[0]

        # 2. Direct dense retrieval from Qdrant
        retrieved_contexts: List[RetrievedContext] = self.store.hybrid_search(
            query_text=scenario,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        context_block = "\n\n---\n\n".join(
            [f"[{doc.doc_type.upper()}] {doc.title} ({doc.citation_or_section}):\n{doc.text}\nSource: {doc.source_url}" for doc in retrieved_contexts]
        )

        # 3. Monolithic single generation
        if self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model_primary,
                    messages=[
                        {"role": "system", "content": VANILLA_RAG_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"User Scenario:\n{scenario}\n\nRetrieved Legal Context:\n{context_block}",
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                raw_json = response.choices[0].message.content
                data = json.loads(raw_json)
                return DualOutputResponse(**data)
            except Exception as e:
                print(f"[VanillaRAG] OpenAI call failed ({e}). Returning structured fallback.")

        # Fallback offline generation based on retrieved contexts
        return self._generate_fallback_response(scenario, retrieved_contexts)

    def _generate_fallback_response(
        self, scenario: str, contexts: List[RetrievedContext]
    ) -> DualOutputResponse:
        """Deterministic response generation for offline test environments."""
        domain = LegalDomain.GENERAL_DISPUTE
        scenario_lower = scenario.lower()
        mva_words = ["accident", "vehicle", "car", "hit and run", "truck", "motorcycle", "motorcyclist", "collision", "collided", "injury", "injured"]
        prop_words = ["property", "tenant", "landlord", "land", "possession", "eviction", "encroach", "flat"]
        cpa_words = ["consumer", "defective", "product", "refund", "warranty", "seller"]

        if any(w in scenario_lower for w in mva_words):
            domain = LegalDomain.MOTOR_VEHICLE_ACCIDENT
        elif any(w in scenario_lower for w in prop_words):
            domain = LegalDomain.PROPERTY_CONFLICT
        elif any(w in scenario_lower for w in cpa_words):
            domain = LegalDomain.CONSUMER_RIGHTS

        actions = [
            ProceduralActionStep(
                step_number=1,
                phase=ProceduralPhase.IMMEDIATE_ACTION,
                title="Immediate Record & Incident Documentation",
                description="Secure physical evidence, photograph incident scene, and record timestamps and party details.",
                forum_or_authority="On-site / Local Authority",
                priority="Critical",
            ),
            ProceduralActionStep(
                step_number=2,
                phase=ProceduralPhase.POLICE_ADMINISTRATIVE if domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT else ProceduralPhase.LEGAL_NOTICE,
                title="Formal Reporting / Statutory Notice",
                description="Issue statutory legal notice or file formal police report under applicable provisions.",
                forum_or_authority="Police Station / Registered Post",
                priority="High",
            ),
            ProceduralActionStep(
                step_number=3,
                phase=ProceduralPhase.FORUM_FILING,
                title="File Claim before Competent Jurisdiction",
                description="Lodge petition before jurisdictional tribunal/commission along with supporting evidence.",
                forum_or_authority="Competent Court / Commission",
                priority="High",
            ),
        ]

        statutes: List[StatutoryCitation] = []
        precedents: List[PrecedentCitation] = []

        for ctx in contexts:
            if ctx.doc_type == DocumentType.STATUTE:
                statutes.append(
                    StatutoryCitation(
                        act_name=ctx.act_name or ctx.title,
                        section_number=ctx.citation_or_section.replace("Section ", ""),
                        summary_of_provision=ctx.text[:150] + "...",
                        applicability_to_scenario="Governs procedural requirements and substantive liability.",
                        source_url=ctx.source_url,
                        is_verified=True,
                    )
                )
            elif ctx.doc_type == DocumentType.PRECEDENT:
                precedents.append(
                    PrecedentCitation(
                        case_title=ctx.title,
                        court=ctx.court or "Supreme Court of India",
                        legal_principle=ctx.text[:180] + "...",
                        source_url=ctx.source_url,
                        is_verified=True,
                    )
                )

        return DualOutputResponse(
            scenario_summary=scenario[:200] + "...",
            domain=domain,
            action_plan=actions,
            statutory_citations=statutes,
            precedent_citations=precedents,
            hallucination_check_passed=True,
        )
