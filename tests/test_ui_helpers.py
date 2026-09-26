"""Unit tests for pure UI helpers (formatting, deadlines, rejection parsing, Markdown export)."""

from datetime import date

from solvemycase.data.ingestion.schema import (
    DualOutputResponse,
    LegalDomain,
    PrecedentCitation,
    ProceduralActionStep,
    ProceduralPhase,
    StatutoryCitation,
)
from solvemycase.ui.components.export import response_to_markdown
from solvemycase.ui.components.formatting import (
    DISCLAIMER,
    domain_label,
    domain_label_from_value,
    escape_markdown,
    extract_deadlines,
    group_steps_by_phase,
    has_real_deadline,
    is_guardrail_rejection,
    markdown_table_cell,
    precedent_read_url,
    precedent_search_url,
    rejection_details,
    removed_references,
    safe_http_url,
    statute_search_url,
    verified_citation_count,
)


def _step(number, phase, priority="High", limitation=None, basis=None):
    return ProceduralActionStep(
        step_number=number,
        phase=phase,
        title=f"Step title {number}",
        description=f"Do thing {number}",
        forum_or_authority="Police Station",
        statutory_basis=basis,
        limitation_period=limitation,
        priority=priority,
    )


def _response(**overrides):
    base = dict(
        scenario_summary="Truck hit scooter",
        domain=LegalDomain.MOTOR_VEHICLE_ACCIDENT,
        action_plan=[
            _step(3, ProceduralPhase.FORUM_FILING, "High", "Within 6 months", "Section 166 MV Act"),
            _step(1, ProceduralPhase.IMMEDIATE_ACTION, "Critical", "Immediately"),
            _step(2, ProceduralPhase.POLICE_ADMINISTRATIVE, "Critical", "N/A"),
        ],
        statutory_citations=[
            StatutoryCitation(
                act_name="Motor Vehicles Act, 1988",
                section_number="166",
                summary_of_provision="Claim application",
                applicability_to_scenario="Compensation",
                source_url="https://indiacode.gov.in/x",
            )
        ],
        precedent_citations=[
            PrecedentCitation(
                case_title="Sarla Verma v. DTC",
                court="Supreme Court of India",
                year=2009,
                legal_principle="Multiplier method",
                source_url="https://main.sci.gov.in/y",
            )
        ],
        unverified_citations_stripped=["Statute: Fake Act Section 9999"],
    )
    base.update(overrides)
    return DualOutputResponse(**base)


def test_group_steps_by_phase_orders_chronologically():
    grouped = group_steps_by_phase(_response().action_plan)
    assert list(grouped) == [
        ProceduralPhase.IMMEDIATE_ACTION,
        ProceduralPhase.POLICE_ADMINISTRATIVE,
        ProceduralPhase.FORUM_FILING,
    ]


def test_extract_deadlines_skips_na_and_sorts_by_priority():
    deadlines = extract_deadlines(_response().action_plan)
    assert [d["Deadline"] for d in deadlines] == ["Immediately", "Within 6 months"]


def test_rejection_helpers():
    rejected = _response(
        scenario_summary="Query rejected: The query appears to be non-legal.",
        action_plan=[],
        statutory_citations=[],
        precedent_citations=[],
        unverified_citations_stripped=["Clarification: Please describe a legal dispute."],
    )
    assert is_guardrail_rejection(rejected)
    assert rejection_details(rejected) == {
        "reason": "The query appears to be non-legal.",
        "clarification": "Please describe a legal dispute.",
    }
    assert removed_references(rejected) == []
    assert not is_guardrail_rejection(_response())


def test_safe_http_url_blocks_non_http_schemes():
    assert safe_http_url("https://indiacode.gov.in/a") == "https://indiacode.gov.in/a"
    assert safe_http_url("javascript:alert(1)") is None
    assert safe_http_url("/relative/path") is None
    assert safe_http_url(None) is None


def test_escape_markdown_neutralizes_control_characters():
    assert escape_markdown("Pay $500 *now* [link](x)") == r"Pay \$500 \*now\* \[link\](x)"
    assert escape_markdown("<script>") == r"\<script\>"
    assert escape_markdown(None) == ""


def test_response_to_markdown_contains_all_sections():
    md = response_to_markdown("My father was hit by a truck.", _response(), generated_on=date(2026, 1, 2))

    assert "Generated on 2026-01-02" in md
    assert "My father was hit by a truck." in md
    assert md.index("1 · Immediate action") < md.index("5 · File your case")
    assert "| Immediately | Step title 1 | Police Station |" in md
    assert "Motor Vehicles Act, 1988, Section 166" in md
    assert "Sarla Verma v. DTC" in md
    assert "Fake Act" not in md  # Removed references must never be exported as advice.
    assert DISCLAIMER in md


def test_response_to_markdown_skips_placeholder_deadlines():
    md = response_to_markdown("x" * 20, _response(), generated_on=date(2026, 1, 2))
    assert "- Deadline: N/A" not in md
    assert "- Deadline: Immediately" in md


def test_response_to_markdown_escapes_table_cells():
    steps = [_step(1, ProceduralPhase.IMMEDIATE_ACTION, "Critical", "Within 30 days | or less\nurgent")]
    md = response_to_markdown("x" * 20, _response(action_plan=steps), generated_on=date(2026, 1, 2))
    assert "| Within 30 days \\| or less urgent | Step title 1 | Police Station |" in md


def test_markdown_table_cell_flattens_and_escapes_pipes():
    assert markdown_table_cell("a | b\n c") == "a \\| b c"
    assert markdown_table_cell(None) == ""


def test_has_real_deadline_rejects_placeholders():
    assert has_real_deadline(_step(1, ProceduralPhase.IMMEDIATE_ACTION, limitation="Within 6 months"))
    for placeholder in (None, "", " n/a ", "NA", "None", "-"):
        assert not has_real_deadline(_step(1, ProceduralPhase.IMMEDIATE_ACTION, limitation=placeholder))


def test_domain_label_from_value_handles_unknown_and_missing():
    assert domain_label_from_value("consumer_rights") == domain_label(LegalDomain.CONSUMER_RIGHTS)
    assert domain_label_from_value("family_law") == "Family Law"
    assert domain_label_from_value(None) == "Unknown"


def test_verified_citation_count_sums_statutes_and_precedents():
    assert verified_citation_count(_response()) == 2
    assert verified_citation_count(_response(statutory_citations=[], precedent_citations=[])) == 0


def test_statute_search_url_targets_exact_section():
    assert statute_search_url("Motor Vehicles Act, 1988", "166") == "https://indiankanoon.org/doc/136948773/"
    assert statute_search_url("Motor Vehicles Act, 1988", "Section 166") == "https://indiankanoon.org/doc/136948773/"
    assert statute_search_url("Arbitration Act, 1996", "11") == (
        "https://indiankanoon.org/search/?formInput=Section+11+in+Arbitration+Act%2C+1996+doctypes%3Alaws"
    )
    assert statute_search_url("", "166") is None
    assert statute_search_url("Motor Vehicles Act, 1988", None) is None


def test_precedent_links_prefer_stored_source_then_title_search():
    assert precedent_read_url("https://indiankanoon.org/doc/837924/", "Sarla Verma") == "https://indiankanoon.org/doc/837924/"
    fallback = "https://indiankanoon.org/search/?formInput=title%3A+Unknown+Case+v.+DTC"
    assert precedent_search_url("  Unknown Case   v. DTC ") == fallback
    assert precedent_read_url("javascript:alert(1)", "Unknown Case v. DTC") == fallback
    assert precedent_read_url(None, "") is None


def test_response_to_markdown_includes_working_links():
    md = response_to_markdown("x" * 20, _response(), generated_on=date(2026, 1, 2))
    assert "[Indian Kanoon](https://indiankanoon.org/doc/136948773/)" in md
    assert "[India Code](https://indiacode.gov.in/handle/123456789/523268)" in md
    assert "[Read judgment](https://indiankanoon.org/doc/837924/)" in md


def test_phase_b_statute_and_precedent_links():
    from solvemycase.ui.components.formatting import official_statute_url

    assert statute_search_url("Bharatiya Nyaya Sanhita, 2023", "325") == "https://indiankanoon.org/doc/186696080/"
    assert official_statute_url("Bharatiya Nyaya Sanhita, 2023", "325") == "https://indiacode.gov.in/handle/123456789/545816"
    assert statute_search_url("Prevention of Cruelty to Animals Act, 1960", "11") == "https://indiankanoon.org/doc/1763700/"
    assert official_statute_url("Prevention of Cruelty to Animals Act, 1960", "11") == "https://indiacode.gov.in/handle/123456789/532762"
    assert statute_search_url("Bharatiya Nyaya Sanhita, 2023", "190") == "https://indiankanoon.org/doc/53218156/"
    assert statute_search_url("Bharatiya Nyaya Sanhita, 2023", "191") == "https://indiankanoon.org/doc/175201984/"
    assert statute_search_url("Motor Vehicles Act, 1988", "147") == "https://indiankanoon.org/doc/87183818/"
    assert statute_search_url("Motor Vehicles Act, 1988", "150") == "https://indiankanoon.org/doc/185690380/"
    assert precedent_read_url(None, "Animal Welfare Board of India v. A. Nagaraja & Ors.") == "https://indiankanoon.org/doc/39696860/"
    assert precedent_read_url(None, "M. Nagarajan v. State") != "https://indiankanoon.org/doc/39696860/"
