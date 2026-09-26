"""Legal query decontextualization and targeted sub-query expansion.

Breaks down factual user narratives into dedicated statutory section searches
and judicial precedent searches to maximize retrieval recall and precision.
"""

import json
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from solvemycase.config.openai_client import build_openai_client
from solvemycase.config.settings import Settings, get_settings
from solvemycase.data.ingestion.schema import LegalDomain

logger = logging.getLogger(__name__)


class DecontextualizedQueries(BaseModel):
    """Decomposed queries targeting statutes and case precedents."""
    statute_queries: List[str] = Field(..., description="Targeted queries for Indian statutory provisions and sections.")
    precedent_queries: List[str] = Field(..., description="Semantic queries for relevant High Court and Supreme Court case precedents.")


DECONTEXT_SYSTEM_PROMPT = """You are an Indian legal research specialist.
Deconstruct the user's factual legal scenario into two sets of targeted search queries:
1. "statute_queries": 2-3 specific queries naming the Indian Act(s) and section numbers that actually govern these facts.
   Examples of Acts: Bharatiya Nyaya Sanhita 2023, Motor Vehicles Act 1988, Consumer Protection Act 2019,
   Transfer of Property Act 1882 (Sections 53A, 54, 106, 108, 111 for sale, part performance, lease termination/notice
   to quit, and tenant eviction), Specific Relief Act 1963 (Sections 6, 10, 16, 38, 39 for dispossession, specific
   performance, and perpetual/mandatory injunctions against encroachment or blocking property/terrace/common-area
   access), Code of Civil Procedure 1908 (Order 39 for temporary injunctions). The examples are not a limit: if another
   Act governs the facts (e.g. Prevention of Cruelty to Animals Act 1960 for harm to an animal, or Negotiable
   Instruments Act 1881 Section 138 for cheque dishonour), name that actual Act—never relabel another Act's section as
   BNS or BNSS.
2. "precedent_queries": 2-3 conceptual queries for High Court and Supreme Court case law addressing the core legal principles.

Rules:
- For criminal offences and police procedure use Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita
  (BNSS) and Bharatiya Sakshya Adhiniyam (BSA). Use IPC / CrPC / Indian Evidence Act only if the events clearly
  happened before 1 July 2024.
- Put the legal subject of the facts into every query in plain words (e.g. "cruelty to animals killing a pet",
  "insurer liability to pay motor accident award", "deficiency in service by courier", "perpetual and mandatory
  injunction Specific Relief Act Section 38 Section 39"), so that a search does not drift to provisions that only
  share words with the facts.
- Do not add Acts or offences that the facts do not support (e.g. no human-death offence when an animal died).

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
        self._openai_client: Optional[OpenAI] = build_openai_client(self.settings)

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
                logger.warning("OpenAI decontextualization call failed (%s); using rule-based decontextualization.", e)

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

        # General legal dispute (including animal cruelty / pet injury offences)
        if any(tok in lower for tok in ("animal", "pet", "cat", "dog", "cattle", "livestock", "cruelty", "maim", "poison")):
            return DecontextualizedQueries(
                statute_queries=[
                    "Bharatiya Nyaya Sanhita Section 325 mischief by killing poisoning maiming animal",
                    "Prevention of Cruelty to Animals Act Section 11 treating animals cruelly beating kicking torturing",
                    "Bharatiya Nagarik Suraksha Sanhita Section 173 First Information Report FIR",
                ],
                precedent_queries=[
                    "Supreme Court Animal Welfare Board of India v. A. Nagaraja cruelty to animals PCA Act Section 11",
                ],
            )

        return DecontextualizedQueries(
            statute_queries=[f"Indian legislation statutory provisions governing {scenario[:80]}"],
            precedent_queries=[f"Supreme Court landmark precedents on {scenario[:80]}"],
        )
