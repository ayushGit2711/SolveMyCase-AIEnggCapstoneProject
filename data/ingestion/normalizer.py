"""Corpus normalizer for open-india-law legislation and judicial precedents.

Extracts, filters, and standardizes legislative provisions from India Code and landmark
Supreme Court / High Court precedents into validated Pydantic models.
"""

from pathlib import Path
from typing import Dict, List, Optional
import duckdb
import pyarrow.parquet as pq

from solvemycase.config.settings import get_settings
from solvemycase.data.ingestion.schema import (
    DocumentType,
    LegalDomain,
    LegalPrecedent,
    LegalProvision,
)

# Acts mapped to target dispute categories
TARGET_ACTS: Dict[str, LegalDomain] = {
    # Motor Vehicle Accidents & Negligence
    "motor vehicles act": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    "bharatiya nyaya sanhita": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    "indian penal code": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    "bharatiya nagarik suraksha sanhita": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    "code of criminal procedure": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    # Property Conflicts & Injunctions
    "transfer of property act": LegalDomain.PROPERTY_CONFLICT,
    "specific relief act": LegalDomain.PROPERTY_CONFLICT,
    "code of civil procedure": LegalDomain.PROPERTY_CONFLICT,
    # Consumer Protection & E-Commerce
    "consumer protection act": LegalDomain.CONSUMER_RIGHTS,
}

# Curated landmark precedents with authentic SC/HC citations and official government / court sources
CURATED_LANDMARK_PRECEDENTS: List[Dict[str, any]] = [
    # Motor Vehicle Accidents
    {
        "doc_id": "sc_mact_sarla_verma_2009",
        "chunk_id": "sc_mact_sarla_verma_2009_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Sarla Verma & Ors. v. Delhi Transport Corporation & Anr.",
        "citation": "(2009) 6 SCC 121",
        "year": 2009,
        "text": (
            "The Supreme Court formulated standard guidelines for calculating compensation under Section 166 of the "
            "Motor Vehicles Act, 1988 in fatal accident claims. Standardized the deduction for personal expenses of the deceased "
            "(1/3rd for 2-3 dependents, 1/4th for 4-6 dependents) and fixed age-multiplier tables (18 for age 15-25, 17 for age 26-30, "
            "16 for age 31-35, down to 5 for age above 65). Established that future prospects must be factored into income assessment."
        ),
        "source_url": "https://main.sci.gov.in/judgment/judis/34419.pdf",
        "disposition": "Allowed in part",
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    {
        "doc_id": "sc_mact_pranay_sethi_2017",
        "chunk_id": "sc_mact_pranay_sethi_2017_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "National Insurance Co. Ltd. v. Pranay Sethi & Ors.",
        "citation": "(2017) 16 SCC 680",
        "year": 2017,
        "text": (
            "A Constitution Bench settled the law on future prospects and conventional heads in motor accident claims under Section 166: "
            "Add 50% for permanent salaried individuals below 40 years, 30% for age 40-50, 15% for age 50-60; for self-employed/fixed wage: "
            "add 40% below 40 years, 25% for 40-50 years, 10% for 50-60 years. Conventional heads fixed at: Loss of Estate (Rs. 15,000), "
            "Loss of Consortium (Rs. 40,000), and Funeral Expenses (Rs. 15,000), subject to 10% enhancement every 3 years."
        ),
        "source_url": "https://main.sci.gov.in/supremecourt/2016/19875/19875_2016_Judgement_31-Oct-2017.pdf",
        "disposition": "Settled",
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    # Property Conflicts
    {
        "doc_id": "sc_prop_suraj_lamp_2012",
        "chunk_id": "sc_prop_suraj_lamp_2012_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Suraj Lamp & Industries Pvt. Ltd. v. State of Haryana & Anr.",
        "citation": "(2012) 1 SCC 656",
        "year": 2012,
        "text": (
            "The Supreme Court held that immovable property can be lawfully transferred only by a registered deed of conveyance. "
            "Transactions entered into through General Power of Attorney (GPA) sales, Sale Agreements (SA), and Wills do not confer "
            "ownership or legal title under Section 54 of the Transfer of Property Act, 1882, nor do they satisfy registration "
            "requirements under the Registration Act, 1908. A buyer under GPA/SA only acquires possessory or contractual protection "
            "under Section 53A of TPA if conditions are strictly satisfied."
        ),
        "source_url": "https://main.sci.gov.in/judgment/judis/38580.pdf",
        "disposition": "Clarified",
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    {
        "doc_id": "sc_prop_poona_ram_2019",
        "chunk_id": "sc_prop_poona_ram_2019_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Poona Ram v. Moti Ram (D) Th. Lrs. & Ors.",
        "citation": "(2019) 11 SCC 309",
        "year": 2019,
        "text": (
            "The Supreme Court ruled that in a suit for recovery of possession under Section 5 and 6 of the Specific Relief Act, 1963, "
            "a person who claims possessory title over immovable property must establish settled and continuous legal possession. "
            "Casual, sporadic acts of possession or trespass do not amount to settled possession. A true owner can only be dispossessed "
            "by due process of law, but a trespasser cannot seek an injunction against the rightful title holder."
        ),
        "source_url": "https://main.sci.gov.in/supremecourt/2008/17260/17260_2008_Judgement_29-Jan-2019.pdf",
        "disposition": "Allowed",
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    # Consumer Rights Violations
    {
        "doc_id": "sc_cpa_lucknow_dev_auth_1994",
        "chunk_id": "sc_cpa_lucknow_dev_auth_1994_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Lucknow Development Authority v. M.K. Gupta",
        "citation": "(1994) 1 SCC 243",
        "year": 1994,
        "text": (
            "Landmark decision expanding the ambit of 'service' and 'deficiency' under consumer law. The Supreme Court held that "
            "statutory development authorities, housing boards, and builders providing housing construction and allotment services "
            "fall strictly under consumer jurisdiction. Unreasonable delay in possession, poor quality of construction, or failure to deliver "
            "amenities constitutes deficiency in service under consumer protection statutes."
        ),
        "source_url": "https://main.sci.gov.in/judgment/judis/13303.pdf",
        "disposition": "Dismissed against Authority",
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    {
        "doc_id": "sc_cpa_experion_developers_2022",
        "chunk_id": "sc_cpa_experion_developers_2022_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Experion Developers Pvt. Ltd. v. Sushma Ashok Shiroor",
        "citation": "(2022) SCC OnLine SC 478",
        "year": 2022,
        "text": (
            "The Supreme Court held that under the Consumer Protection Act, flat purchasers are entitled to full refund with interest "
            "if the developer fails to deliver possession within the committed contract period. The consumer is not bound to accept "
            "an offer of delayed possession after substantial default. Terms in builder-buyer agreements that penalize consumers "
            "at 18% for payment delay while compensating consumers at only Rs. 5/sq.ft for construction delays constitute an "
            "unfair trade practice under Section 2(47) of the Consumer Protection Act, 2019."
        ),
        "source_url": "https://main.sci.gov.in/supremecourt/2020/22904/22904_2020_4_1501_34947_Judgement_07-Apr-2022.pdf",
        "disposition": "Affirmed with interest",
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
]

# Baseline fallback provisions for offline/test environments
FALLBACK_PROVISIONS: List[Dict[str, any]] = [
    # Motor Vehicles Act, 1988
    {
        "doc_id": "central_mva_1988_sec_134",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "134",
        "title": "Duty of driver in case of accident and injury to a person",
        "text": (
            "When any person is injured or any property of a third party is damaged, as a result of an accident in which a motor vehicle is involved, "
            "the driver of the vehicle or other person in charge of the vehicle shall: (a) take all reasonable steps to secure medical attention for the injured person, "
            "by conveying him to the nearest medical practitioner or hospital; (b) give on demand by a police officer any information required; "
            "(c) report the circumstances of occurrence, including the date, time and place of the accident, to the nearest police station within twenty-four hours."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1798?view_type=browse&sam_handle=123456789/1362",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    {
        "doc_id": "central_mva_1988_sec_166",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "166",
        "title": "Application for compensation",
        "text": (
            "An application for compensation arising out of an accident of the nature specified in sub-section (1) of section 165 may be made: "
            "(a) by the person who has sustained the injury; or (b) by the owner of the property; or (c) where death has resulted from the accident, "
            "by all or any of the legal representatives of the deceased; or (d) by any agent duly authorised by the person injured or all or any of the "
            "legal representatives of the deceased. Must be filed before the Claims Tribunal having jurisdiction over the place of accident or where the claimant resides."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1798?view_type=browse&sam_handle=123456789/1362",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    {
        "doc_id": "central_mva_1988_sec_161",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "161",
        "title": "Special provisions as to compensation in cases of hit and run motor accident",
        "text": (
            "Provides for payment of compensation in respect of the death of, or grievous hurt to, persons resulting from hit and run motor accidents. "
            "A sum of two lakh rupees or such higher amount as may be prescribed by the Central Government in respect of the death of any person, and "
            "fifty thousand rupees in respect of grievous hurt to any person."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1798?view_type=browse&sam_handle=123456789/1362",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    # Bharatiya Nyaya Sanhita, 2023
    {
        "doc_id": "central_bns_2023_sec_106",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "106",
        "title": "Causing death by negligence",
        "text": (
            "(1) Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide, shall be punished with "
            "imprisonment of either description for a term which may extend to five years, and shall also be liable to fine. "
            "(2) Whoever causes the death of any person by rash and negligent driving of vehicle not amounting to culpable homicide, and escapes without "
            "reporting it to a police officer or a Magistrate soon after the incident, shall be punished with imprisonment of either description of a term which may extend to ten years, and shall also be liable to fine."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/20062",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    {
        "doc_id": "central_bns_2023_sec_281",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "281",
        "title": "Rash driving or riding on a public way",
        "text": (
            "Whoever drives any vehicle, or rides, on any public way in a manner so rash or negligent as to endanger human life, or to be likely to cause hurt or injury to any other person, "
            "shall be punished with imprisonment of either description for a term which may extend to six months, or with fine which may extend to one thousand rupees, or with both."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/20062",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    # Bharatiya Nagarik Suraksha Sanhita, 2023
    {
        "doc_id": "central_bnss_2023_sec_173",
        "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "section_number": "173",
        "title": "Information in cognizable cases (First Information Report)",
        "text": (
            "Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, "
            "and be read over to the informant; and every such information shall be signed by the person giving it. Information may be given electronically (e-FIR) subject to physical signature within three days."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/20063",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    },
    # Transfer of Property Act, 1882
    {
        "doc_id": "central_tpa_1882_sec_54",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "54",
        "title": "Sale defined and sale how made",
        "text": (
            "Sale is a transfer of ownership in exchange for a price paid or promised or part-paid and part-promised. "
            "Such transfer, in the case of tangible immovable property of the value of one hundred rupees and upwards, or in the case of a reversion or other intangible thing, "
            "can be made only by a registered instrument. A contract for sale does not, of itself, create any interest in or charge on such property."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2338",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    {
        "doc_id": "central_tpa_1882_sec_106",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "106",
        "title": "Duration of certain leases in absence of written contract or local usage",
        "text": (
            "In the absence of a contract or local law or usage to the contrary, a lease of immovable property for agricultural or manufacturing purposes shall be deemed to be a lease from year to year, "
            "terminable, on the part of either lessor or lessee, by six months' notice; and a lease of immovable property for any other purpose shall be deemed to be a lease from month to month, "
            "terminable, on the part of either lessor or lessee, by fifteen days' notice. Notice must be in writing and duly served."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2338",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    # Specific Relief Act, 1963
    {
        "doc_id": "central_sra_1963_sec_6",
        "act_name": "Specific Relief Act, 1963",
        "section_number": "6",
        "title": "Suit by person dispossessed of immovable property",
        "text": (
            "If any person is dispossessed without his consent of immovable property otherwise than in due course of law, he or any person through whom he has been in possession or any person claiming "
            "through him may, by suit, recover possession thereof, notwithstanding any other title that may be set up in such suit. No suit under this section shall be brought after the expiry of six months from the date of dispossession."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1583",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    {
        "doc_id": "central_sra_1963_sec_38",
        "act_name": "Specific Relief Act, 1963",
        "section_number": "38",
        "title": "Perpetual injunction when granted",
        "text": (
            "Subject to the other provisions contained in or referred to by this Chapter, a perpetual injunction may be granted to the plaintiff to prevent the breach of an obligation existing in his favour, "
            "whether expressly or by implication. When the defendant invades or threatens to invade the plaintiff's right to, or enjoyment of, property, the court may grant a perpetual injunction."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/1583",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    # Consumer Protection Act, 2019
    {
        "doc_id": "central_cpa_2019_sec_2_11",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "2(11)",
        "title": "Definition of deficiency in service",
        "text": (
            "'Deficiency' means any fault, imperfection, shortcoming or inadequacy in the quality, nature and manner of performance which is required to be maintained by or under any law for the time being in force "
            "or has been undertaken to be performed by a person in pursuance of a contract or otherwise in relation to any service and includes: (i) any act of negligence or omission or commission by such person; "
            "(ii) deliberate withholding of relevant information by such person to the consumer."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/15256",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    {
        "doc_id": "central_cpa_2019_sec_35",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "35",
        "title": "Manner in which complaint shall be made before District Commission",
        "text": (
            "A complaint, in relation to any goods sold or delivered or agreed to be sold or delivered or any service provided or agreed to be provided, may be filed with a District Commission by: "
            "(a) the consumer; (b) any recognised consumer association; (c) one or more consumers having the same interest. "
            "District Commission has jurisdiction for value of goods/services up to fifty lakh rupees (revised limits). Limitation period is two years from cause of action under Section 69."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/15256",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    {
        "doc_id": "central_cpa_2019_sec_84",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "84",
        "title": "Liability of product manufacturer",
        "text": (
            "A product manufacturer shall be liable in a product liability action, if: (a) the product contains a manufacturing defect; "
            "(b) the product is defective in design; (c) there is a deviation from manufacturing specifications; (d) the product does not conform to the express warranty; "
            "(e) the product fails to contain adequate instructions for correct usage to prevent any harm or any warning regarding improper or incorrect usage."
        ),
        "source_url": "https://www.indiacode.nic.in/handle/123456789/15256",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
]


def load_legal_provisions_from_parquet(parquet_file_path: Path) -> List[LegalProvision]:
    """Parse legislation provisions from a local open-india-law parquet file.

    Filters sections matching target domain Acts. If parsing fails or yields 0 rows,
    falls back cleanly to the curated foundational provisions.

    Args:
        parquet_file_path: Path to in_central_legislation.parquet.

    Returns:
        List of validated LegalProvision objects.
    """
    if not parquet_file_path.exists():
        print(f"[Normalizer] Parquet file not found at {parquet_file_path}. Loading curated provisions.")
        return [LegalProvision(**p) for p in FALLBACK_PROVISIONS]

    try:
        conn = duckdb.connect()
        query = f"""
            SELECT 
                chunk_id as doc_id,
                title as act_name,
                section_number,
                section_title as title,
                text,
                COALESCE(source_url, mirror_url, 'https://www.indiacode.nic.in') as source_url,
                year
            FROM read_parquet('{parquet_file_path}')
            WHERE text IS NOT NULL AND length(trim(text)) > 15
        """
        rows = conn.execute(query).fetchall()

        provisions: List[LegalProvision] = []
        for row in rows:
            act_lower = str(row[1]).lower()
            # Ignore repealed acts
            if "(rep." in act_lower or "repealed" in act_lower:
                continue

            matched_domain = None
            for key_act, domain in TARGET_ACTS.items():
                if key_act in act_lower:
                    matched_domain = domain
                    break

            if matched_domain:
                year_val = None
                if row[6] is not None:
                    try:
                        year_val = int(row[6])
                    except (ValueError, TypeError):
                        pass

                provisions.append(
                    LegalProvision(
                        doc_id=str(row[0]),
                        act_name=str(row[1]),
                        section_number=str(row[2]) if row[2] else "",
                        title=str(row[3]) if row[3] else None,
                        text=str(row[4]),
                        source_url=str(row[5]),
                        jurisdiction="Central",
                        year=year_val,
                        domain=matched_domain,
                    )
                )

        if provisions:
            # Also append BNS and BNSS if not present in central parquet
            act_names_present = {p.act_name.lower() for p in provisions}
            for fallback in FALLBACK_PROVISIONS:
                if fallback["act_name"].lower() not in act_names_present:
                    provisions.append(LegalProvision(**fallback))

            print(f"[Normalizer] Extracted {len(provisions)} domain-specific provisions from Parquet & new criminal codes.")
            return provisions

    except Exception as err:
        print(f"[Normalizer] Warning during Parquet parsing: {err}. Using fallback provisions.")

    return [LegalProvision(**p) for p in FALLBACK_PROVISIONS]


def load_legal_precedents() -> List[LegalPrecedent]:
    """Retrieve curated landmark Supreme Court and High Court precedents.

    Returns:
        List of validated LegalPrecedent objects.
    """
    return [LegalPrecedent(**p) for p in CURATED_LANDMARK_PRECEDENTS]


def load_unified_corpus(parquet_path: Optional[Path] = None) -> Dict[str, List]:
    """Load both statutory provisions and judicial precedents into a unified container.

    Args:
        parquet_path: Optional path to in_central_legislation.parquet.

    Returns:
        Dictionary with keys 'provisions' and 'precedents'.
    """
    settings = get_settings()
    if parquet_path is None:
        parquet_path = settings.resolve_path(settings.data_cache_dir / "in_central_legislation.parquet")

    provisions = load_legal_provisions_from_parquet(parquet_path)
    precedents = load_legal_precedents()

    return {
        "provisions": provisions,
        "precedents": precedents,
    }
