"""Lawyer directory and consultation request components (dummy data for prototype)."""

import hashlib
from typing import Any, Dict, List, Optional

import streamlit as st

from solvemycase.data.ingestion.schema import DualOutputResponse, LegalDomain
from solvemycase.ui.components.formatting import escape_markdown

COMPLAINT_ALL = "All complaint types"
COMPLAINT_MVA = "🚗 Road & Motor Vehicle Accidents"
COMPLAINT_PROPERTY = "🏠 Property, Tenancy & Injunctions"
COMPLAINT_CONSUMER = "🛒 Consumer Rights & Builder Delays"
COMPLAINT_ANIMAL_CRIMINAL = "🐾 Animal Cruelty & Criminal Offences"
COMPLAINT_OTHER = "⚖️ Employment, Family, Cheque Bounce & Cyber Disputes"

COMPLAINT_CATEGORIES: List[str] = [
    COMPLAINT_ALL,
    COMPLAINT_MVA,
    COMPLAINT_PROPERTY,
    COMPLAINT_CONSUMER,
    COMPLAINT_ANIMAL_CRIMINAL,
    COMPLAINT_OTHER,
]

CITY_OPTIONS: List[str] = [
    "All cities / Online",
    "Bengaluru",
    "Delhi NCR",
    "Mumbai",
    "Pune",
    "Chennai",
    "Hyderabad",
    "Kolkata",
]

# Dummy advocate profiles for prototype demonstration.
LAWYER_DIRECTORY: List[Dict[str, Any]] = [
    {
        "id": "adv_mva_01",
        "name": "Adv. Rajeshwari Nair",
        "bar_id": "KAR/1482/2014 (Demo)",
        "categories": [COMPLAINT_MVA],
        "domains": [LegalDomain.MOTOR_VEHICLE_ACCIDENT.value],
        "city": "Bengaluru",
        "modes": ["Video call", "Phone", "In-person (Bengaluru)"],
        "experience_years": 11,
        "courts": "MACT Bengaluru, Karnataka High Court",
        "languages": "English, Kannada, Hindi, Malayalam",
        "fee_inr": "₹800 / 30 min",
        "rating": "4.9 ★ (120+ MACT claims)",
        "response_time": "Usually responds within 2 hours",
        "specialties": [
            "MACT compensation petitions (MV Act s.166)",
            "Third-party & own-damage insurer claim refusals (MV Act s.147/150)",
            "Permanent disability & fatal accident multiplier claims",
        ],
    },
    {
        "id": "adv_mva_02",
        "name": "Adv. Vikramaditya Deshmukh",
        "bar_id": "MAH/3910/2012 (Demo)",
        "categories": [COMPLAINT_MVA, COMPLAINT_ANIMAL_CRIMINAL],
        "domains": [LegalDomain.MOTOR_VEHICLE_ACCIDENT.value],
        "city": "Pune",
        "modes": ["Video call", "Phone", "In-person (Pune & Mumbai)"],
        "experience_years": 13,
        "courts": "MACT Pune, Sessions Court Pune, Bombay High Court",
        "languages": "English, Marathi, Hindi",
        "fee_inr": "₹900 / 30 min",
        "rating": "4.8 ★ (95+ cases)",
        "response_time": "Usually responds within 3 hours",
        "specialties": [
            "Hit-and-run Solatium Scheme claims (MV Act s.161) & BNS s.106/281 FIRs",
            "Commercial vehicle & bus accident claims",
            "Police investigation monitoring under BNSS s.173",
        ],
    },
    {
        "id": "adv_prop_01",
        "name": "Adv. Meenakshi Iyer",
        "bar_id": "KAR/2209/2011 (Demo)",
        "categories": [COMPLAINT_PROPERTY],
        "domains": [LegalDomain.PROPERTY_CONFLICT.value],
        "city": "Bengaluru",
        "modes": ["Video call", "Phone", "In-person (Bengaluru & Chennai)"],
        "experience_years": 14,
        "courts": "City Civil Court Bengaluru, Madras High Court",
        "languages": "English, Tamil, Kannada, Hindi",
        "fee_inr": "₹1,200 / 30 min",
        "rating": "4.9 ★ (150+ property suits)",
        "response_time": "Usually responds within 2 hours",
        "specialties": [
            "Unlawful eviction & summary recovery of possession (SRA s.6)",
            "Lease termination & tenant eviction suits (TPA s.106 & s.111)",
            "Perpetual, mandatory & temporary injunctions (SRA s.38/39, CPC Order 39)",
        ],
    },
    {
        "id": "adv_prop_02",
        "name": "Adv. Siddharth Malhotra",
        "bar_id": "D/1845/2015 (Demo)",
        "categories": [COMPLAINT_PROPERTY, COMPLAINT_CONSUMER],
        "domains": [LegalDomain.PROPERTY_CONFLICT.value, LegalDomain.CONSUMER_RIGHTS.value],
        "city": "Delhi NCR",
        "modes": ["Video call", "Phone", "In-person (Delhi NCR)"],
        "experience_years": 10,
        "courts": "Tis Hazari & Saket District Courts, Delhi High Court, UP/HR RERA",
        "languages": "English, Hindi, Punjabi",
        "fee_inr": "₹1,000 / 30 min",
        "rating": "4.8 ★ (110+ cases)",
        "response_time": "Usually responds within 4 hours",
        "specialties": [
            "Neighbour encroachment, driveway/easement obstruction & boundary suits",
            "Specific performance of agreement to sell (SRA s.10 & s.16, TPA s.53A)",
            "RERA builder possession delay & refund with interest (RERA s.18)",
        ],
    },
    {
        "id": "adv_con_01",
        "name": "Adv. Ananya Chatterjee",
        "bar_id": "WB/0942/2016 (Demo)",
        "categories": [COMPLAINT_CONSUMER],
        "domains": [LegalDomain.CONSUMER_RIGHTS.value],
        "city": "Kolkata",
        "modes": ["Video call", "Phone", "In-person (Kolkata)"],
        "experience_years": 9,
        "courts": "District & State Consumer Commissions (DCDRCs / SCDRCs), NCDRC",
        "languages": "English, Bengali, Hindi",
        "fee_inr": "₹600 / 30 min",
        "rating": "4.9 ★ (180+ consumer complaints)",
        "response_time": "Usually responds within 1 hour",
        "specialties": [
            "Defective electronics/vehicles & e-commerce refund refusals (CPA 2019 s.2(10), s.35)",
            "Courier loss, airline/telecom/banking deficiency in service (CPA 2019 s.2(11))",
            "E-Daakhil filing, legal notices & unfair trade practice claims (CPA 2019 s.2(47))",
        ],
    },
    {
        "id": "adv_con_02",
        "name": "Adv. Rohan Kulkarni",
        "bar_id": "MAH/5120/2017 (Demo)",
        "categories": [COMPLAINT_CONSUMER, COMPLAINT_PROPERTY],
        "domains": [LegalDomain.CONSUMER_RIGHTS.value],
        "city": "Mumbai",
        "modes": ["Video call", "Phone", "In-person (Mumbai & Hyderabad)"],
        "experience_years": 8,
        "courts": "Maharashtra State Consumer Commission, MahaRERA, NCDRC",
        "languages": "English, Marathi, Hindi, Telugu",
        "fee_inr": "₹750 / 30 min",
        "rating": "4.7 ★ (85+ cases)",
        "response_time": "Usually responds within 3 hours",
        "specialties": [
            "Product liability & manufacturer defect compensation (CPA 2019 s.84)",
            "Housing builder delay & MahaRERA / Consumer Commission complaints",
            "Health & motor insurance claim rejection complaints before Consumer Forum",
        ],
    },
    {
        "id": "adv_crim_01",
        "name": "Adv. Kavita Menon",
        "bar_id": "D/3310/2015 (Demo)",
        "categories": [COMPLAINT_ANIMAL_CRIMINAL],
        "domains": [LegalDomain.GENERAL_DISPUTE.value],
        "city": "Delhi NCR",
        "modes": ["Video call", "Phone", "In-person (Delhi NCR)"],
        "experience_years": 10,
        "courts": "Magistrate & Sessions Courts Delhi, Delhi High Court",
        "languages": "English, Hindi, Malayalam",
        "fee_inr": "₹700 / 30 min (Pro-bono initial triage for animal cruelty)",
        "rating": "4.9 ★ (90+ criminal & animal law matters)",
        "response_time": "Usually responds within 1 hour",
        "specialties": [
            "Animal cruelty, killing/poisoning of pets & street animals (BNS s.325, PCA Act 1960 s.11)",
            "Zero-FIR & Magistrate directions when police refuse FIR (BNSS s.173(4) / s.175(3))",
            "Cheating (BNS s.318), criminal breach of trust (BNS s.316) & unlawful assembly (BNS s.190/191)",
        ],
    },
    {
        "id": "adv_gen_01",
        "name": "Adv. Arvind Subramanian",
        "bar_id": "TN/2780/2010 (Demo)",
        "categories": [COMPLAINT_OTHER, COMPLAINT_ANIMAL_CRIMINAL],
        "domains": [LegalDomain.GENERAL_DISPUTE.value],
        "city": "Chennai",
        "modes": ["Video call", "Phone", "In-person (Chennai & Hyderabad)"],
        "experience_years": 15,
        "courts": "Madras High Court, Family Court, Labour Court & Metropolitan Magistrate Courts",
        "languages": "English, Tamil, Telugu, Hindi",
        "fee_inr": "₹1,100 / 30 min",
        "rating": "4.8 ★ (200+ civil & criminal matters)",
        "response_time": "Usually responds within 2 hours",
        "specialties": [
            "Unpaid salary, employment recovery & Labour Commissioner / NCLT claims",
            "Cheque dishonour under Negotiable Instruments Act s.138 & civil money recovery",
            "Matrimonial/Family Court petitions, Cyber Crime (IT Act) & Defamation (BNS s.356)",
        ],
    },
]

_ANIMAL_OR_COVERED_CRIM_HINTS = (
    "pet",
    "cat",
    "dog",
    "animal",
    "stray",
    "cruelty",
    "poison",
    "maim",
    "watchman",
    "watchmen",
    "fir",
    "cheating",
    "breach of trust",
    "unlawful assembly",
    "mob",
)


def infer_complaint_category(
    scenario: str = "",
    domain: Optional[LegalDomain] = None,
    coverage_gap: bool = False,
) -> str:
    """Map a user's scenario, classified domain, and coverage status to a lawyer complaint category."""
    lower = (scenario or "").lower()
    if domain == LegalDomain.MOTOR_VEHICLE_ACCIDENT:
        return COMPLAINT_MVA
    if domain == LegalDomain.PROPERTY_CONFLICT:
        return COMPLAINT_PROPERTY
    if domain == LegalDomain.CONSUMER_RIGHTS:
        return COMPLAINT_CONSUMER
    if coverage_gap:
        return COMPLAINT_OTHER
    if any(tok in lower for tok in _ANIMAL_OR_COVERED_CRIM_HINTS):
        return COMPLAINT_ANIMAL_CRIMINAL
    if domain == LegalDomain.GENERAL_DISPUTE and scenario.strip():
        return COMPLAINT_OTHER
    return COMPLAINT_ALL


def filter_lawyers(
    category: str = COMPLAINT_ALL,
    city: str = "All cities / Online",
) -> List[Dict[str, Any]]:
    """Filter the dummy lawyer directory by complaint category and city."""
    matched: List[Dict[str, Any]] = []
    for lawyer in LAWYER_DIRECTORY:
        cat_ok = category == COMPLAINT_ALL or category in lawyer["categories"]
        city_ok = (
            city == "All cities / Online"
            or lawyer["city"].lower() == city.lower()
            or any(city.lower() in m.lower() for m in lawyer["modes"])
        )
        if cat_ok and city_ok:
            matched.append(lawyer)
    return matched


def _booking_ref(lawyer_id: str, client_name: str, summary: str) -> str:
    digest = hashlib.sha1(f"{lawyer_id}:{client_name}:{summary}".encode("utf-8")).hexdigest()[:6].upper()
    return f"SMC-{digest}"


def render_lawyer_card(
    lawyer: Dict[str, Any],
    key_prefix: str,
    default_scenario: str = "",
    default_category: str = "",
) -> None:
    """Render a single advocate profile card with an expandable consultation booking form."""
    lid = lawyer["id"]
    card_key = f"{key_prefix}_{lid}"
    booking_state_key = f"lawyer_booking_{card_key}"

    with st.container(border=True):
        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.markdown(f"**👩‍⚖️ {escape_markdown(lawyer['name'])}** · `{escape_markdown(lawyer['bar_id'])}`")
            st.caption(
                f"📍 {escape_markdown(lawyer['city'])} (Online pan-India) · "
                f"**{lawyer['experience_years']} yrs experience** · {escape_markdown(lawyer['rating'])}"
            )
        with top_right:
            st.markdown(f"**{escape_markdown(lawyer['fee_inr'])}**")
            st.caption(escape_markdown(lawyer["response_time"]))

        st.markdown(
            "**Practice areas:** " + " · ".join(escape_markdown(cat) for cat in lawyer["categories"])
        )
        for spec in lawyer["specialties"]:
            st.markdown(f"- {escape_markdown(spec)}")
        st.caption(
            f"**Courts:** {escape_markdown(lawyer['courts'])}  |  "
            f"**Languages:** {escape_markdown(lawyer['languages'])}"
        )

        with st.expander(f"📨 Connect with {lawyer['name']}", expanded=False):
            with st.form(key=f"{card_key}_form", clear_on_submit=False):
                c1, c2 = st.columns(2)
                client_name = c1.text_input("Your name", key=f"{card_key}_name", placeholder="e.g. Aarav Sharma")
                client_contact = c2.text_input(
                    "Phone or email", key=f"{card_key}_contact", placeholder="e.g. +91 98XXXXXX10 or email"
                )
                mode = st.selectbox(
                    "Preferred consultation mode",
                    options=lawyer["modes"],
                    key=f"{card_key}_mode",
                )
                complaint_type = st.text_input(
                    "Complaint category",
                    value=default_category or lawyer["categories"][0],
                    key=f"{card_key}_category",
                )
                brief = st.text_area(
                    "Brief summary of your complaint (shared with the advocate)",
                    value=default_scenario,
                    height=90,
                    key=f"{card_key}_summary",
                    placeholder="Summarize what happened, city/date, and what relief you need...",
                )
                submitted = st.form_submit_button("Request consultation (Demo)", type="primary")
                if submitted:
                    if not client_name.strip() or not client_contact.strip():
                        st.warning("Please enter your name and a phone number or email so the advocate can reach you.")
                    else:
                        ref = _booking_ref(lid, client_name.strip(), brief.strip())
                        st.session_state[booking_state_key] = {
                            "ref": ref,
                            "lawyer_name": lawyer["name"],
                            "client_name": client_name.strip(),
                            "contact": client_contact.strip(),
                            "mode": mode,
                            "complaint_type": complaint_type.strip(),
                        }

            saved_booking = st.session_state.get(booking_state_key)
            if saved_booking:
                st.success(
                    f"✅ **Consultation request recorded (Demo Ref `{escape_markdown(saved_booking['ref'])}`)** — "
                    f"{escape_markdown(saved_booking['lawyer_name'])} has been matched for your "
                    f"**{escape_markdown(saved_booking['complaint_type'])}** matter via "
                    f"**{escape_markdown(saved_booking['mode'])}**. "
                    f"(Demo prototype: no external message is sent.)"
                )


def render_inline_lawyer_connect(
    scenario: str,
    response: DualOutputResponse,
    key_prefix: str,
) -> None:
    """Render a compact advocate-matching section at the bottom of a case result."""
    matched_category = infer_complaint_category(
        scenario=scenario,
        domain=response.domain,
        coverage_gap=bool(response.coverage_note),
    )
    lawyers = filter_lawyers(category=matched_category)[:2]
    if not lawyers:
        lawyers = LAWYER_DIRECTORY[:2]

    safe_scenario = escape_markdown(scenario)
    with st.container(border=True):
        st.markdown(f"#### 👩‍⚖️ Connect with a Lawyer for this Case ({escape_markdown(matched_category)})")
        st.caption(
            "Need an advocate to draft a legal notice, register an FIR, or file before the court/tribunal? "
            "Below are specialist advocates matched to your complaint type (demo profiles for prototype)."
        )
        cols = st.columns(len(lawyers))
        for col, lawyer in zip(cols, lawyers):
            with col:
                render_lawyer_card(
                    lawyer,
                    key_prefix=f"{key_prefix}_inline_lw",
                    default_scenario=safe_scenario,
                    default_category=matched_category,
                )
