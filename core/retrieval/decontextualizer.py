"""Legal query decontextualization and targeted sub-query expansion.

Breaks down factual user narratives into dedicated statutory section searches
and judicial precedent searches to maximize retrieval recall and precision.
"""

import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import LegalDomain


class DecontextualizedQueries(BaseModel):
    """Decomposed queries targeting statutes and case precedents."""
    statute_queries: List[str] = Field(..., description="Targeted queries for Indian statutory provisions and sections.")
    precedent_queries: List[str] = Field(..., description="Semantic queries for relevant High Court and Supreme Court case precedents.")


DECONTEXT_SYSTEM_PROMPT = """You are an Indian legal research specialist.
Deconstruct the user's factual legal scenario into two sets of targeted search queries:
1. "statute_queries": 2-3 specific queries referencing relevant Indian Acts (e.g., Bharatiya Nyaya Sanhita 2023, Motor Vehicles Act 1988, Consumer Protection Act 2019, Transfer of Property Act 1882) and section numbers if relevant.
2. "precedent_queries": 2-3 conceptual queries for High Court and Supreme Court case law addressing the core legal principles.

Format strictly as JSON:
{
    "statute_queries": ["query 1", "query 2"],
    "precedent_queries": ["query 1", "query 2"]
}
"""


class LegalDecontextualizer:
    """Expands narrative legal queries into high-precision search strings."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._openai_client: Optional[OpenAI] = None

        if self.settings.openai_api_key and self.settings.openai_api_key.get_secret_value():
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key.get_secret_value())

    def decontextualize(
        self, scenario: str, domain: LegalDomain, entities: Optional[Dict] = None
    ) -> DecontextualizedQueries:
        """Deconstruct the scenario into statutory and precedent retrieval queries.

        Args:
            scenario: User's narrative scenario.
            domain: Classified legal domain.
            entities: Optional extracted entity dictionary.

        Returns:
            DecontextualizedQueries with statute and precedent query lists.
        """
        if self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.settings.openai_model_fast,
                    messages=[
                        {"role": "system", "content": DECONTEXT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Domain: {domain.value}\nScenario:\n{scenario}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(response.choices[0].message.content)
                return DecontextualizedQueries(**data)
            except Exception as e:
                print(f"[Decontextualizer] OpenAI call failed ({e}). Using rule-based decontextualization.")

        return self._heuristic_decontextualize(scenario, domain)

    def _heuristic_decontextualize(self, scenario: str, domain: LegalDomain) -> DecontextualizedQueries:
        """Rule-based legal query deconstruction tailored to the 3 target domains."""
        lower = scenario.lower()

        if domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT:
            statute_q = [
                "Motor Vehicles Act Section 166 application for compensation tribunal MACT",
                "Motor Vehicles Act Section 134 duty of driver medical attention accident",
                "Bharatiya Nyaya Sanhita Section 106 causing death by negligence hit and run Section 281 rash driving",
                "Bharatiya Nagarik Suraksha Sanhita Section 173 First Information Report FIR",
            ]
            if "hit and run" in lower:
                statute_q.append("Motor Vehicles Act Section 161 special compensation hit and run")

            precedent_q = [
                "Supreme Court motor accident compensation guidelines multiplier Sarla Verma Pranay Sethi",
                "Supreme Court hit and run road accident negligence liability insurance company",
            ]
            return DecontextualizedQueries(statute_queries=statute_q, precedent_queries=precedent_q)

        elif domain == LegalDomain.PROPERTY_CONFLICT:
            statute_q = [
                "Transfer of Property Act Section 106 duration of lease notice to quit 15 days",
                "Specific Relief Act Section 6 suit by person dispossessed of immovable property possession",
                "Specific Relief Act Section 38 perpetual temporary injunction property encroachment",
                "Transfer of Property Act Section 54 sale of immovable property registered instrument",
            ]
            precedent_q = [
                "Supreme Court Suraj Lamp property transfer General Power of Attorney registered sale deed",
                "Supreme Court Poona Ram settled possession injunction trespasser true owner",
            ]
            return DecontextualizedQueries(statute_queries=statute_q, precedent_queries=precedent_q)

        elif domain == LegalDomain.CONSUMER_RIGHTS:
            statute_q = [
                "Consumer Protection Act Section 35 complaint before District Commission limitation Section 69",
                "Consumer Protection Act Section 2(11) deficiency in service Section 2(47) unfair trade practice",
                "Consumer Protection Act Section 84 liability of product manufacturer defective goods",
            ]
            precedent_q = [
                "Supreme Court Lucknow Development Authority deficiency in service consumer commission",
                "Supreme Court Experion Developers builder delay flat possession full refund unfair trade practice",
            ]
            return DecontextualizedQueries(statute_queries=statute_q, precedent_queries=precedent_q)

        # General legal dispute
        return DecontextualizedQueries(
            statute_queries=[f"Indian legislation statutory provisions governing {scenario[:80]}"],
            precedent_queries=[f"Supreme Court landmark precedents on {scenario[:80]}"],
        )
