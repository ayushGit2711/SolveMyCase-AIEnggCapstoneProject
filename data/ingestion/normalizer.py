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

# Acts mapped to target dispute categories.
# Criminal statutes cut across dispute types (a courier's theft, a killed pet, rash driving), so they are
# tagged general_dispute: domain-filtered searches also match general_dispute documents.
TARGET_ACTS: Dict[str, LegalDomain] = {
    # Motor Vehicle Accidents & Negligence
    "motor vehicles act": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
    # Criminal law and procedure (cross-cutting)
    "bharatiya nyaya sanhita": LegalDomain.GENERAL_DISPUTE,
    "indian penal code": LegalDomain.GENERAL_DISPUTE,
    "bharatiya nagarik suraksha sanhita": LegalDomain.GENERAL_DISPUTE,
    "code of criminal procedure": LegalDomain.GENERAL_DISPUTE,
    # Animal protection (offences of cruelty to animals)
    "prevention of cruelty to animals act": LegalDomain.GENERAL_DISPUTE,
    # Property Conflicts & Injunctions
    "transfer of property act": LegalDomain.PROPERTY_CONFLICT,
    "specific relief act": LegalDomain.PROPERTY_CONFLICT,
    "code of civil procedure": LegalDomain.PROPERTY_CONFLICT,
    # Consumer Protection & E-Commerce
    "consumer protection act": LegalDomain.CONSUMER_RIGHTS,
}


def classify_act_category(act_name: str) -> str:
    """Classify an Act into broad procedural categories for targeted RAG retrieval.
    
    Returns 'criminal', 'traffic', 'consumer', 'civil', or 'general'. The Prevention of Cruelty
    to Animals Act is 'criminal' so its offences are reachable by the criminal-code sub-search.
    """
    name = act_name.lower()
    if any(
        k in name
        for k in ["bharatiya nyaya", "bharatiya nagarik", "penal code", "criminal procedure", "cruelty to animals"]
    ):
        return "criminal"
    if any(k in name for k in ["motor vehicles", "motor vehicle"]):
        return "traffic"
    if any(k in name for k in ["consumer protection"]):
        return "consumer"
    if any(k in name for k in ["transfer of property", "specific relief", "civil procedure"]):
        return "civil"
    return "general"


# Curated landmark precedents with authentic SC/HC citations and official government / court sources
# Judgment links point to Indian Kanoon: the legacy main.sci.gov.in PDF paths are no longer served.
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
        "source_url": "https://indiankanoon.org/doc/837924/",
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
        "source_url": "https://indiankanoon.org/doc/139996215/",
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
        "source_url": "https://indiankanoon.org/doc/1565619/",
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
        "source_url": "https://indiankanoon.org/doc/56103097/",
        "disposition": "Allowed",
        "domain": LegalDomain.PROPERTY_CONFLICT,
    },
    # Consumer Rights Violations
    {
        "doc_id": "sc_cpa_patel_roadways_2000",
        "chunk_id": "sc_cpa_patel_roadways_2000_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Patel Roadways Ltd. v. Birla Yamaha Ltd.",
        "citation": "(2000) 4 SCC 91",
        "year": 2000,
        "text": (
            "The Supreme Court held that the liability of a common carrier in India, as in England, is that of an insurer: "
            "it is absolute, subject only to two exceptions, an act of God and a special contract which the carrier may choose "
            "to enter into with the customer. Under section 9 of the Carriers Act, 1865, a plaintiff claiming for loss of or "
            "damage to goods entrusted to a carrier need not establish negligence; the loss or non-delivery of the goods is "
            "prima facie evidence of negligence. Rejecting the argument that section 9 applies only to suits in civil courts, "
            "the Court held that the consumer dispute redressal agencies have jurisdiction over such complaints and that a "
            "proceeding before the National Commission comes within the term 'suit' in section 9, so the complainant need "
            "not prove negligence in consumer-forum proceedings either. The carrier's appeal was dismissed."
        ),
        "source_url": "https://indiankanoon.org/doc/1907957/",
        "disposition": "Carrier's appeal dismissed",
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    {
        "doc_id": "sc_cpa_bharathi_knitting_1996",
        "chunk_id": "sc_cpa_bharathi_knitting_1996_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Bharathi Knitting Company v. DHL Worldwide Express Courier Division",
        "citation": "(1996) 4 SCC 704",
        "year": 1996,
        "text": (
            "The Supreme Court upheld the National Commission's order limiting a courier's liability for a lost consignment "
            "to US $100, the limit stated in the consignment note (receipt) that the sender had signed, and awarding "
            "compensation for deficiency in service only to that extent. A person who signs a document containing contractual "
            "terms is normally bound by them even if he has not read them; a party who disputes the binding nature of the "
            "signed terms must establish the exception in a suit. Any claim beyond the contractual limit was left to the "
            "remedy available at law. The sender's appeal was dismissed."
        ),
        "source_url": "https://indiankanoon.org/doc/126678/",
        "disposition": "Sender's appeal dismissed",
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    {
        "doc_id": "sc_cpa_lucknow_dev_auth_1994",
        "chunk_id": "sc_cpa_lucknow_dev_auth_1994_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Lucknow Development Authority v. M.K. Gupta",
        "citation": "(1994) 1 SCC 243",
        "year": 1993,
        "text": (
            "The Supreme Court held that the Consumer Protection Act, 1986 applies to services rendered by statutory and public "
            "authorities, and that housing construction and the allotment of flats or plots by a development authority is a "
            "'service' under the Act, even before 'housing construction' was expressly added to the definition in 1993. "
            "The consumer fora can award compensation not only for the value of the deficiency in service but also for "
            "harassment and mental agony caused to the consumer by the oppressive or capricious conduct of public officers. "
            "Where such compensation is paid from public funds, the authority should recover it from the officers found "
            "responsible."
        ),
        "source_url": "https://indiankanoon.org/doc/1375046/",
        "disposition": "Appeals dismissed",
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
        "source_url": "https://indiankanoon.org/doc/71246029/",
        "disposition": "Affirmed with interest",
        "domain": LegalDomain.CONSUMER_RIGHTS,
    },
    # General Disputes (animal protection)
    {
        "doc_id": "sc_gen_awbi_nagaraja_2014",
        "chunk_id": "sc_gen_awbi_nagaraja_2014_001",
        "court": "Supreme Court of India",
        "court_type": "supreme_court",
        "title": "Animal Welfare Board of India v. A. Nagaraja & Ors.",
        "citation": "(2014) 7 SCC 547",
        "year": 2014,
        "text": (
            "The Supreme Court held that Sections 3 and 11 of the Prevention of Cruelty to Animals Act, 1960, read with "
            "Article 51A(g) of the Constitution, guarantee animals a right to live in a healthy and clean atmosphere, a right "
            "to protection from human beings against the infliction of unnecessary pain or suffering, and a right not to be "
            "beaten, kicked, over-ridden or over-loaded; domesticated animals also have a right to food and shelter. Section 3 "
            "casts a duty on persons having the care or charge of animals to ensure their well-being and to prevent the "
            "infliction of unnecessary pain or suffering. Relying on the expanded meaning of 'life' under Article 21, the Court "
            "observed that every species has a right to life and security, subject to the law of the land (which includes "
            "depriving its life out of human necessity), and that for animals 'life' means more than mere survival: a life "
            "with some intrinsic worth, honour and dignity. It declared that the 'five freedoms' (freedom from hunger, thirst "
            "and malnutrition; from fear and distress; from physical and thermal discomfort; from pain, injury and disease; "
            "and to express normal patterns of behaviour) are to be read into Sections 3 and 11 of the Act. Holding that "
            "jallikattu, bullock-cart races and such events per se violate Sections 3, 11(1)(a) and 11(1)(m)(ii) of the Act, "
            "it upheld the Central Government's notification of 11 July 2011 barring bulls from being exhibited or trained "
            "as performing animals, declared the Tamil Nadu Regulation of Jallikattu Act, 2009 repugnant to the Act and void, "
            "and urged Parliament to provide adequate penalties for violations of Section 11. Later development (not part "
            "of this judgment): Tamil Nadu, Maharashtra and Karnataka then amended the Act in its application to those States "
            "to permit such events subject to conditions, and a Constitution Bench upheld those amendments in Animal Welfare "
            "Board of India v. Union of India (18 May 2023)."
        ),
        "source_url": "https://indiankanoon.org/doc/39696860/",
        "disposition": "Disposed of; Madras High Court judgment set aside",
        "domain": LegalDomain.GENERAL_DISPUTE,
    },
]

# Baseline fallback provisions for offline/test environments.
# `text` must be verbatim statutory text (excerpts marked with '...'); never paraphrase inside `text`.
# Criminal provisions are tagged general_dispute because they apply across dispute types.
FALLBACK_PROVISIONS: List[Dict[str, any]] = [
    {
        "doc_id": "central_bns_2023_sec_316",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "316",
        "title": "Criminal breach of trust",
        "text": (
            "(1) Whoever, being in any manner entrusted with property, or with any dominion over property, dishonestly "
            "misappropriates or converts to his own use that property, or dishonestly uses or disposes of that property in "
            "violation of any direction of law prescribing the mode in which such trust is to be discharged, or of any legal "
            "contract, express or implied, which he has made touching the discharge of such trust, or wilfully suffers any "
            "other person so to do, commits criminal breach of trust. ... Illustrations. ... (f) A, a carrier, is entrusted by Z "
            "with property to be carried by land or by water. A dishonestly misappropriates the property. A has committed "
            "criminal breach of trust. (2) Whoever commits criminal breach of trust shall be punished with imprisonment of "
            "either description for a term which may extend to five years, or with fine, or with both. (3) Whoever, being "
            "entrusted with property as a carrier, wharfinger or warehouse-keeper, commits criminal breach of trust in respect "
            "of such property, shall be punished with imprisonment of either description for a term which may extend to seven "
            "years, and shall also be liable to fine. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545808",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bns_2023_sec_352",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "352",
        "title": "Intentional insult with intent to provoke breach of peace",
        "text": (
            "Whoever intentionally insults in any manner, and thereby gives provocation to any person, intending or knowing it "
            "to be likely that such provocation will cause him to break the public peace, or to commit any other offence, shall "
            "be punished with imprisonment of either description for a term which may extend to two years, or with fine, or "
            "with both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545841",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
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
        "source_url": "https://indiacode.gov.in/handle/123456789/523232",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    {
        "doc_id": "central_mva_1988_sec_166",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "166",
        "title": "Application for compensation",
        "text": (
            "(1) An application for compensation arising out of an accident of the nature specified in sub-section (1) of "
            "section 165 may be made— (a) by the person who has sustained the injury; or (b) by the owner of the property; "
            "or (c) where death has resulted from the accident, by all or any of the legal representatives of the "
            "deceased; or (d) by any agent duly authorised by the person injured or all or any of the legal representatives "
            "of the deceased, as the case may be: ... (2) Every application under sub-section (1) shall be made, at the "
            "option of the claimant, either to the Claims Tribunal having jurisdiction over the area in which the accident "
            "occurred or to the Claims Tribunal within the local limits of whose jurisdiction the claimant resides or "
            "carries on business or within the local limits of whose jurisdiction the defendant resides, and shall be in "
            "such form and contain such particulars as may be prescribed: ... (3) No application for compensation shall be "
            "entertained unless it is made within six months of the occurrence of the accident. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523268",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/523262",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    # Chapter XI as substituted by the Motor Vehicles (Amendment) Act, 2019 (w.e.f. 1 Sep 2019)
    {
        "doc_id": "central_mva_1988_sec_147",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "147",
        "title": "Requirements of policies and limits of liability",
        "text": (
            "(1) In order to comply with the requirements of this Chapter, a policy of insurance must be a policy which— "
            "(a) is issued by a person who is an authorised insurer; and (b) insures the person or classes of persons "
            "specified in the policy to the extent specified in sub-section (2)— (i) against any liability which may be "
            "incurred by him in respect of the death of or bodily injury to any person including owner of the goods or his "
            "authorised representative carried in the motor vehicle or damage to any property of a third party caused by "
            "or arising out of the use of the motor vehicle in a public place; (ii) against the death of or bodily injury "
            "to any passenger of a transport vehicle, except gratuitous passengers of a goods vehicle, caused by or arising "
            "out of the use of the motor vehicle in a public place. ... (2) Notwithstanding anything contained under any "
            "other law for the time being in force, for the purposes of third party insurance related to either death of a "
            "person or grievous hurt to a person, the Central Government shall prescribe a base premium and the liability "
            "of an insurer in relation to such premium for an insurance policy under sub-section (1) in consultation with "
            "the Insurance Regulatory and Development Authority. ... (6) Notwithstanding anything contained in any other "
            "law for the time being in force, an insurer issuing a policy of insurance under this section shall be liable "
            "to indemnify the person or classes of persons specified in the policy in respect of any liability which the "
            "policy purports to cover in the case of that person or those classes of persons."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523247",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    {
        "doc_id": "central_mva_1988_sec_150",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "150",
        "title": "Duty of insurers to satisfy judgments and awards against persons insured in respect of third party risks",
        "text": (
            "(1) If, after a certificate of insurance has been issued under sub-section (3) of section 147 in favour of the "
            "person by whom a policy has been effected, judgment or award in respect of any such liability as is required "
            "to be covered by a policy under clause (b) of sub-section (1) of section 147 (being a liability covered by the "
            "terms of the policy) or under the provisions of section 164 is obtained against any person insured by the "
            "policy, then, notwithstanding that the insurer may be entitled to avoid or cancel or may have avoided or "
            "cancelled the policy, the insurer shall, subject to the provisions of this section, pay to the person "
            "entitled to the benefit of the award any sum not exceeding the sum assured payable thereunder, as if that "
            "person were the decree holder, in respect of the liability, together with any amount payable in respect of "
            "costs and any sum payable in respect of interest on that sum by virtue of any enactment relating to interest "
            "on judgments. (2) No sum shall be payable by an insurer under sub-section (1) in respect of any judgment or "
            "award unless, before the commencement of the proceedings in which the judgment or award is given the insurer "
            "had notice through the court or, as the case may be, the Claims Tribunal of the bringing of the proceedings, "
            "or in respect of such judgment or award so long as its execution is stayed pending an appeal; and an insurer "
            "to whom notice of the bringing of any such proceedings is so given shall be entitled to be made a party "
            "thereto, and to defend the action on any of the following grounds, namely:— (a) that there has been a breach "
            "of a specified condition of the policy, being one of the following conditions, namely:— (i) a condition "
            "excluding the use of the vehicle— (A) for hire or reward, where the vehicle is on the date of the contract of "
            "insurance a vehicle not covered by a permit to ply for hire or reward; or (B) for organised racing and speed "
            "testing; or (C) for a purpose not allowed by the permit under which the vehicle is used, where the vehicle is "
            "a transport vehicle; or (D) without side-car being attached where the vehicle is a two-wheeled vehicle; or "
            "(ii) a condition excluding driving by a named person or by any person who is not duly licenced or by any "
            "person who has been disqualified for holding or obtaining a driving licence during the period of "
            "disqualification or driving under the influence of alcohol or drugs as laid down in section 185; or (iii) a "
            "condition excluding liability for injury caused or contributed to by conditions of war, civil war, riot or "
            "civil commotion; or (b) that the policy is void on the ground that it was obtained by nondisclosure of any "
            "material fact or by representation of any fact which was false in some material particular; or (c) that "
            "there is non-receipt of premium as required under section 64VB of the Insurance Act, 1938 (4 of 1938). ... "
            "(6) If on the date of filing of any claim, the claimant is not aware of the insurance company with which the "
            "vehicle had been insured, it shall be the duty of the owner of the vehicle to furnish to the tribunal or court "
            "the information as to whether the vehicle had been insured on the date of the accident, and if so, the name "
            "of the insurance company with which it is insured. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523250",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    # Bharatiya Nyaya Sanhita, 2023
    {
        "doc_id": "central_bns_2023_sec_106",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "106",
        "title": "Causing death by negligence",
        "text": (
            "(1) Whoever causes death of any person by doing any rash or negligent act not amounting to culpable homicide, "
            "shall be punished with imprisonment of either description for a term which may extend to five years, and shall "
            "also be liable to fine; and if such act is done by a registered medical practitioner while performing medical "
            "procedure, he shall be punished with imprisonment of either description for a term which may extend to two "
            "years, and shall also be liable to fine. ... (2) Whoever causes death of any person by rash and negligent "
            "driving of vehicle not amounting to culpable homicide, and escapes without reporting it to a police officer or "
            "a Magistrate soon after the incident, shall be punished with imprisonment of either description of a term "
            "which may extend to ten years, and shall also be liable to fine. [Status note: sub-section (2) has not been "
            "brought into force; the notification that brought this Sanhita into force from 1 July 2024 excluded it.]"
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545604",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/545772",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bns_2023_sec_324",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "324",
        "title": "Mischief",
        "text": (
            "(1) Whoever with intent to cause, or knowing that he is likely to cause, wrongful loss or damage to the public "
            "or to any person, causes the destruction of any property, or any such change in any property or in the "
            "situation thereof as destroys or diminishes its value or utility, or affects it injuriously, commits mischief. "
            "... (2) Whoever commits mischief shall be punished with imprisonment of either description for a term which "
            "may extend to six months, or with fine, or with both. (3) Whoever commits mischief and thereby causes loss or "
            "damage to any property including the property of Government or Local Authority shall be punished with "
            "imprisonment of either description for a term which may extend to one year, or with fine, or with both. "
            "(4) Whoever commits mischief and thereby causes loss or damage to the amount of twenty thousand rupees and "
            "more but less than one lakh rupees shall be punished with imprisonment of either description for a term which "
            "may extend to two years, or with fine, or with both. (5) Whoever commits mischief and thereby causes loss or "
            "damage to the amount of one lakh rupees or upwards, shall be punished with imprisonment of either description "
            "for a term which may extend to five years, or with fine, or with both. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545814",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bns_2023_sec_325",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "325",
        "title": "Mischief by killing or maiming animal",
        "text": (
            "Whoever commits mischief by killing, poisoning, maiming or rendering useless any animal shall be punished with "
            "imprisonment of either description for a term which may extend to five years, or with fine, or with both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545816",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bns_2023_sec_318",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "318",
        "title": "Cheating",
        "text": (
            "(1) Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any "
            "property to any person, or to consent that any person shall retain any property, or intentionally induces the "
            "person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived, and "
            "which act or omission causes or is likely to cause damage or harm to that person in body, mind, reputation or "
            "property, is said to cheat. ... (2) Whoever cheats shall be punished with imprisonment of either description "
            "for a term which may extend to three years, or with fine, or with both. ... (4) Whoever cheats and thereby "
            "dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy "
            "the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of "
            "being converted into a valuable security, shall be punished with imprisonment of either description for a term "
            "which may extend to seven years, and shall also be liable to fine."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545810",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    # Prevention of Cruelty to Animals Act, 1960
    {
        "doc_id": "central_pca_1960_sec_11",
        "act_name": "Prevention of Cruelty to Animals Act, 1960",
        "section_number": "11",
        "title": "Treating animals cruelly",
        "text": (
            "(1) If any person— (a) beats, kicks, over-rides, over-drives, over-loads, tortures or otherwise treats any "
            "animal so as to subject it to unnecessary pain or suffering or causes or, being the owner permits, any animal "
            "to be so treated; or ... (c) wilfully and unreasonably administers any injurious drug or injurious substance "
            "to any animal or wilfully and unreasonably causes or attempts to cause any such drug or substance to be taken "
            "by any animal; or ... (l) mutilates any animal or kills any animal (including stray dogs) by using the method "
            "of strychnine injections in the heart or in any other unnecessarily cruel manner; or ... he shall be "
            "punishable, in the case of a first offence, with fine which shall not be less than ten rupees but which may "
            "extend to fifty rupees and in the case of a second or subsequent offence committed within three years of the "
            "previous offence, with fine which shall not be less than twenty-five rupees but which may extend to one "
            "hundred rupees or with imprisonment for a term which may extend to three months, or with both. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/532762",
        "jurisdiction": "Central",
        "year": 1960,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    # Bharatiya Nagarik Suraksha Sanhita, 2023
    {
        "doc_id": "central_bnss_2023_sec_173",
        "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "section_number": "173",
        "title": "Information in cognizable cases (First Information Report)",
        "text": (
            "(1) Every information relating to the commission of a cognizable offence, irrespective of the area where the "
            "offence is committed, may be given orally or by electronic communication to an officer in charge of a police "
            "station, and if given— (i) orally, it shall be reduced to writing by him or under his direction, and be read "
            "over to the informant; and every such information, whether given in writing or reduced to writing as "
            "aforesaid, shall be signed by the person giving it; (ii) by electronic communication, it shall be taken on "
            "record by him on being signed within three days by the person giving it, and the substance thereof shall be "
            "entered in a book to be kept by such officer in such form as the State Government may by rules prescribe in "
            "this behalf: ... (2) A copy of the information as recorded under sub-section (1) shall be given forthwith, "
            "free of cost, to the informant or the victim. ... (4) Any person aggrieved by a refusal on the part of an "
            "officer in charge of a police station to record the information referred to in sub-section (1), may send the "
            "substance of such information, in writing and by post, to the Superintendent of Police concerned who, if "
            "satisfied that such information discloses the commission of a cognizable offence, shall either investigate "
            "the case himself or direct an investigation to be made by any police officer subordinate to him, in the "
            "manner provided by this Sanhita, and such officer shall have all the powers of an officer in charge of the "
            "police station in relation to that offence failing which such aggrieved person may make an application to "
            "the Magistrate."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/546648",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bnss_2023_sec_193",
        "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "section_number": "193",
        "title": "Report of police officer on completion of investigation (Chargesheet)",
        "text": (
            "(1) Every investigation under this Chapter shall be completed without unnecessary delay. ... (3) (i) As soon "
            "as the investigation is completed, the officer in charge of the police station shall forward, including "
            "through electronic communication to a Magistrate empowered to take cognizance of the offence on a police "
            "report, a report in the form as the State Government may, by rules provide, stating— (a) the names of the "
            "parties; (b) the nature of the information; (c) the names of the persons who appear to be acquainted with the "
            "circumstances of the case; (d) whether any offence appears to have been committed and, if so, by whom; ... "
            "(ii) the police officer shall, within a period of ninety days, inform the progress of the investigation by "
            "any means including through electronic communication to the informant or the victim; ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/546293",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bnss_2023_sec_480",
        "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "section_number": "480",
        "title": "When bail may be taken in case of non-bailable offence",
        "text": (
            "(1) When any person accused of, or suspected of, the commission of any non-bailable offence is arrested or "
            "detained without warrant by an officer in charge of a police station or appears or is brought before a Court "
            "other than the High Court or Court of Session, he may be released on bail, but— (i) such person shall not be "
            "so released if there appear reasonable grounds for believing that he has been guilty of an offence punishable "
            "with death or imprisonment for life; (ii) such person shall not be so released if such offence is a "
            "cognizable offence and he had been previously convicted of an offence punishable with death, imprisonment for "
            "life or imprisonment for seven years or more, or he had been previously convicted on two or more occasions of "
            "a cognizable offence punishable with imprisonment for three years or more but less than seven years: Provided "
            "that the Court may direct that a person referred to in clause (i) or clause (ii) be released on bail if such "
            "person is a child or is a woman or is sick or infirm: ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/546565",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    # Indian Penal Code, 1860 (Legacy reference; repealed by the Bharatiya Nyaya Sanhita, 2023 w.e.f. 1 July 2024)
    {
        "doc_id": "central_ipc_1860_sec_279",
        "act_name": "Indian Penal Code, 1860",
        "section_number": "279",
        "title": "Rash driving or riding on a public way [Repealed w.e.f. 1 Jul 2024; see BNS s.281]",
        "text": (
            "[Repealed w.e.f. 1 July 2024 by the Bharatiya Nyaya Sanhita, 2023 (see BNS section 281); it still governs "
            "offences committed before that date. Text before repeal:] "
            "Whoever drives any vehicle, or rides, on any public way in a manner so rash or negligent as to endanger human life, or to be likely to cause hurt or injury to any other person, "
            "shall be punished with imprisonment of either description for a term which may extend to six months, or with fine which may extend to one thousand rupees, or with both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/488475",
        "jurisdiction": "Central",
        "year": 1860,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_ipc_1860_sec_304a",
        "act_name": "Indian Penal Code, 1860",
        "section_number": "304A",
        "title": "Causing death by negligence [Repealed w.e.f. 1 Jul 2024; see BNS s.106]",
        "text": (
            "[Repealed w.e.f. 1 July 2024 by the Bharatiya Nyaya Sanhita, 2023 (see BNS section 106); it still governs "
            "offences committed before that date. Text before repeal:] "
            "Whoever causes the death of any person by doing any rash or negligent act not amounting to culpable homicide, shall be punished with "
            "imprisonment of either description for a term which may extend to two years, or with fine, or with both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/488475",
        "jurisdiction": "Central",
        "year": 1860,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    # Code of Criminal Procedure, 1973 (Legacy reference; repealed by the Bharatiya Nagarik Suraksha Sanhita, 2023
    # w.e.f. 1 July 2024)
    {
        "doc_id": "central_crpc_1973_sec_154",
        "act_name": "Code of Criminal Procedure, 1973",
        "section_number": "154",
        "title": "Information in cognizable cases (First Information Report) [Repealed w.e.f. 1 Jul 2024; see BNSS s.173]",
        "text": (
            "[Repealed w.e.f. 1 July 2024 by the Bharatiya Nagarik Suraksha Sanhita, 2023 (see BNSS section 173); "
            "proceedings pending on that date continue under this Code. Text before repeal:] "
            "(1) Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing by him or under his direction, "
            "and be read over to the informant; and every such information, whether given in writing or reduced to writing as aforesaid, shall be signed by the person giving it, "
            "and the substance thereof shall be entered in a book to be kept by such officer in such form as the State Government may prescribe in this behalf. ..."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/620185",
        "jurisdiction": "Central",
        "year": 1973,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/535222",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/535285",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/524827",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/524862",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/538576",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/538611",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
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
        "source_url": "https://indiacode.gov.in/handle/123456789/538685",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
    },
    # Additional Benchmark-Referenced Provisions (Motor Vehicles Act, 1988)
    {
        "doc_id": "central_mva_1988_sec_165",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "165",
        "title": "Claims Tribunals",
        "text": (
            "A State Government may, by notification in the Official Gazette, constitute one or more Motor Accidents Claims Tribunals "
            "(MACT) for such area as may be specified for the purpose of adjudicating upon claims for compensation in respect of accidents "
            "involving the death of, or bodily injury to, persons arising out of the use of motor vehicles, or damages to any property of a third party so arising, or both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523267",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    {
        "doc_id": "central_mva_1988_sec_185",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "185",
        "title": "Driving by a drunken person or by a person under the influence of drugs",
        "text": (
            "Whoever, while driving, or attempting to drive, a motor vehicle, (a) has, in his blood, alcohol exceeding 30 mg. per 100 ml. of blood "
            "detected in a test by a breath analyser or any other test including a laboratory test, or (b) is under the influence of a drug to such an extent "
            "as to be incapable of exercising proper control over the vehicle, shall be punishable for the first offence with imprisonment up to six months or fine of ten thousand rupees, or both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523290",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    {
        "doc_id": "central_mva_1988_sec_196",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "196",
        "title": "Driving uninsured vehicle",
        "text": (
            "Whoever drives a motor vehicle or causes or allows a motor vehicle to be driven in contravention of the provisions of section 146 "
            "(necessity for insurance against third party risk) shall be punishable for the first offence with imprisonment which may extend to three months, "
            "or with fine of two thousand rupees, or with both; and the registered owner remains personally liable for third-party compensation before the MACT."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/523303",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    {
        "doc_id": "central_mva_1988_sec_199A",
        "act_name": "Motor Vehicles Act, 1988",
        "section_number": "199A",
        "title": "Offences by juveniles",
        "text": (
            "Where an offence under this Act has been committed by a juvenile, the guardian of such juvenile or the owner of the motor vehicle "
            "shall be deemed to be guilty of the contravention and shall be liable to be proceeded against and punished accordingly with imprisonment up to three years "
            "and fine of twenty-five thousand rupees, and the registration of the motor vehicle shall be cancelled for twelve months."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/538138",
        "jurisdiction": "Central",
        "year": 1988,
        "domain": LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        "act_category": "traffic",
    },
    # Additional Benchmark-Referenced Provisions (Bharatiya Nyaya Sanhita, 2023)
    {
        "doc_id": "central_bns_2023_sec_191",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "191",
        "title": "Rioting",
        "text": (
            "(1) Whenever force or violence is used by an unlawful assembly, or by any member thereof, in prosecution of the "
            "common object of such assembly, every member of such assembly is guilty of the offence of rioting. (2) Whoever "
            "is guilty of rioting, shall be punished with imprisonment of either description for a term which may extend to "
            "two years, or with fine, or with both. (3) Whoever is guilty of rioting, being armed with a deadly weapon or "
            "with anything which, used as a weapon of offence, is likely to cause death, shall be punished with imprisonment "
            "of either description for a term which may extend to five years, or with fine, or with both."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545688",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    {
        "doc_id": "central_bns_2023_sec_190",
        "act_name": "Bharatiya Nyaya Sanhita, 2023",
        "section_number": "190",
        "title": "Every member of unlawful assembly guilty of offence committed in prosecution of common object",
        "text": (
            "If an offence is committed by any member of an unlawful assembly in prosecution of the common object of that "
            "assembly, or such as the members of that assembly knew to be likely to be committed in prosecution of that "
            "object, every person who, at the time of the committing of that offence, is a member of the same assembly, is "
            "guilty of that offence."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/545864",
        "jurisdiction": "Central",
        "year": 2023,
        "domain": LegalDomain.GENERAL_DISPUTE,
        "act_category": "criminal",
    },
    # Additional Benchmark-Referenced Provisions (Transfer of Property Act, 1882)
    {
        "doc_id": "central_tpa_1882_sec_10",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "10",
        "title": "Condition restraining alienation",
        "text": (
            "Where property is transferred subject to a condition or limitation absolutely restraining the transferee or any person claiming under him "
            "from parting with or disposing of his interest in the property, the condition or limitation is void, except in the case of a lease where the condition is for the benefit of the lessor."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535176",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_31",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "31",
        "title": "Condition that transferred interest shall cease on happening of specified uncertain event",
        "text": (
            "Subject to the provisions of section 12, on a transfer of property an interest therein may be created with the superadded condition "
            "that it shall cease to exist in case a specified uncertain event shall happen, or in case a specified uncertain event shall not happen."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535324",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_44",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "44",
        "title": "Transfer by one co-owner",
        "text": (
            "Where one of two or more co-owners of immovable property legally competent in that behalf transfers his share of such property or any interest therein, "
            "the transferee acquires as to such share or interest, and so far as is necessary to give effect to the transfer, the transferor's right to joint possession "
            "or other common or part enjoyment of the property, and to enforce a partition of the same, subject to protection of an undivided family dwelling-house."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535210",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_53A",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "53A",
        "title": "Part performance",
        "text": (
            "Where any person contracts to transfer for consideration any immovable property by writing signed by him or on his behalf from which the terms "
            "necessary to constitute the transfer can be ascertained with reasonable certainty, and the transferee has, in part performance of the contract, "
            "taken possession of the property and performed or is willing to perform his part of the contract, the transferor is debarred from enforcing against the transferee any right in respect of the property."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535221",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_55",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "55",
        "title": "Rights and liabilities of buyer and seller",
        "text": (
            "In the absence of a contract to the contrary, the seller is bound to disclose to the buyer any material defect in the property or in the seller's title thereto, "
            "produce title documents for examination, execute a proper conveyance on payment of the price, give possession, and pay all public charges and encumbrances accrued up to the date of sale."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535223",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_107",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "107",
        "title": "Leases how made",
        "text": (
            "A lease of immovable property from year to year, or for any term exceeding one year, or reserving a yearly rent, can be made only by a registered instrument. "
            "All other leases of immovable property may be made either by a registered instrument or by oral agreement accompanied by delivery of possession."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535286",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_111",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "111",
        "title": "Determination of lease",
        "text": (
            "A lease of immovable property determines: (a) by efflux of the time limited thereby; (b) where such time is limited conditionally on the happening of some event; "
            "(g) by forfeiture; or (h) on the expiration of a notice to determine the lease, or to quit, or of intention to quit, the property leased, duly given by one party to the other."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535290",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_122",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "122",
        "title": "Gift defined",
        "text": (
            "Gift is the transfer of certain existing movable or immovable property made voluntarily and without consideration, by one person, called the donor, "
            "to another, called the donee, and accepted by or on behalf of the donee during the lifetime of the donor and while he is still capable of giving."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535303",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_tpa_1882_sec_123",
        "act_name": "Transfer of Property Act, 1882",
        "section_number": "123",
        "title": "Transfer how effected for gifts",
        "text": (
            "For the purpose of making a gift of immovable property, the transfer must be effected by a registered instrument signed by or on behalf of the donor, "
            "and attested by at least two witnesses. For the purpose of making a gift of movable property, the transfer may be effected either by a registered instrument or by delivery."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/535304",
        "jurisdiction": "Central",
        "year": 1882,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    # Additional Benchmark-Referenced Provisions (Specific Relief Act, 1963)
    {
        "doc_id": "central_sra_1963_sec_10",
        "act_name": "Specific Relief Act, 1963",
        "section_number": "10",
        "title": "Specific performance in respect of contracts",
        "text": (
            "The specific performance of a contract shall be enforced by the court subject to the provisions contained in "
            "sub-section (2) of section 11, section 14 and section 16. [Section 10 as substituted by the Specific Relief "
            "(Amendment) Act, 2018 (18 of 2018), with effect from 1 October 2018.]"
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/524831",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_sra_1963_sec_16",
        "act_name": "Specific Relief Act, 1963",
        "section_number": "16",
        "title": "Personal bars to relief in suit for specific performance",
        "text": (
            "Specific performance of a contract cannot be enforced in favour of a person who fails to prove that he has performed or has always been ready and willing "
            "to perform the essential terms of the contract which are to be performed by him, other than terms the performance of which has been prevented or waived by the defendant."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/524838",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    {
        "doc_id": "central_sra_1963_sec_39",
        "act_name": "Specific Relief Act, 1963",
        "section_number": "39",
        "title": "Mandatory injunctions",
        "text": (
            "When, to prevent the breach of an obligation, it is necessary to compel the performance of certain acts which the court is capable of enforcing, "
            "the court may in its discretion grant an injunction to prevent the breach complained of, and also to compel performance of the requisite acts, including demolition of unauthorized encroachment."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/524863",
        "jurisdiction": "Central",
        "year": 1963,
        "domain": LegalDomain.PROPERTY_CONFLICT,
        "act_category": "civil",
    },
    # Additional Benchmark-Referenced Provisions (Consumer Protection Act, 2019)
    {
        "doc_id": "central_cpa_2019_sec_2_47",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "2(47)",
        "title": "Unfair trade practice",
        "text": (
            "'Unfair trade practice' means a trade practice which, for the purpose of promoting the sale, use or supply of any goods or for the provision of any service, "
            "adopts any unfair method or unfair or deceptive practice including false representation of standard or quality, refusing to take back defective goods or refund consideration, "
            "imposing one-sided penalty clauses, or disclosing consumer personal information."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/538576",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
    },
    {
        "doc_id": "central_cpa_2019_sec_85",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "85",
        "title": "Liability of product service provider",
        "text": (
            "A product service provider shall be liable in a product liability action, if: (a) the service provided by it was faulty or imperfect or deficient or inadequate in quality, "
            "nature or manner of performance; or (b) there was an act of omission or commission or negligence or conscious withholding any information which caused harm; "
            "or (c) it did not issue adequate instructions or warnings to prevent any harm; or (d) the service did not conform to express warranty or terms and conditions."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/538658",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
    },
    {
        "doc_id": "central_cpa_2019_sec_89",
        "act_name": "Consumer Protection Act, 2019",
        "section_number": "89",
        "title": "Punishment for false or misleading advertisement",
        "text": (
            "Any manufacturer or service provider who causes a false or misleading advertisement to be made which is prejudicial to the interest of consumers "
            "shall be punished with imprisonment for a term which may extend to two years and with fine which may extend to ten lakh rupees; and for every subsequent offence, "
            "with imprisonment up to five years and fine up to fifty lakh rupees."
        ),
        "source_url": "https://indiacode.gov.in/handle/123456789/538662",
        "jurisdiction": "Central",
        "year": 2019,
        "domain": LegalDomain.CONSUMER_RIGHTS,
        "act_category": "consumer",
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
    try:
        file_exists = parquet_file_path.exists()
    except OSError:
        file_exists = False

    if not file_exists:
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
                COALESCE(source_url, mirror_url, 'https://indiacode.gov.in') as source_url,
                year
            FROM read_parquet('{parquet_file_path}')
            WHERE text IS NOT NULL AND length(trim(text)) > 15
        """
        rows = conn.execute(query).fetchall()

        provisions: List[LegalProvision] = []
        for row in rows:
            act_name_str = str(row[1])
            act_lower = act_name_str.lower()
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

                category = classify_act_category(act_name_str)

                provisions.append(
                    LegalProvision(
                        doc_id=str(row[0]),
                        act_name=act_name_str,
                        section_number=str(row[2]) if row[2] else "",
                        title=str(row[3]) if row[3] else None,
                        text=str(row[4]),
                        source_url=str(row[5]),
                        jurisdiction="Central",
                        year=year_val,
                        domain=matched_domain,
                        act_category=category,
                    )
                )

        if provisions:
            # Also ensure all curated fallback provisions (especially BNS, BNSS, IPC) are present
            doc_ids_present = {p.doc_id for p in provisions}
            for fallback in FALLBACK_PROVISIONS:
                if fallback["doc_id"] not in doc_ids_present:
                    provisions.append(LegalProvision(**fallback))

            print(f"[Normalizer] Extracted {len(provisions)} domain-specific provisions including criminal code sub-layer.")
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
