"""Offline tests for the link checker's pure comparison functions and orchestration (no network)."""

from solvemycase.data.ingestion.link_checker import (
    FAIL,
    OK,
    WARN,
    UiLinks,
    acts_match,
    backoff_seconds,
    base_section,
    check_curated_precedent_title,
    check_india_code_record,
    check_indian_kanoon_statute_title,
    check_precedent_page,
    check_ui_map_consistency,
    india_code_handle,
    indian_kanoon_doc_id,
    is_marked_repealed,
    lead_party_name,
    load_ui_links,
    parse_indian_kanoon_section_title,
    run_checks,
    section_numbers_match,
    summarize,
)
from solvemycase.data.ingestion.normalizer import CURATED_LANDMARK_PRECEDENTS, FALLBACK_PROVISIONS


def _md(**fields):
    """Build DSpace-style metadata: {'dc.identifier.act_name': [{'value': ...}], ...}."""
    keys = {
        "act_name": "dc.identifier.act_name",
        "section_number": "dc.identifier.section_number",
        "state_name": "dc.identifier.state_name",
        "title": "dc.title",
    }
    return {keys[k]: [{"value": v}] for k, v in fields.items()}


def test_acts_match_ignores_articles_case_and_repeal_suffixes():
    assert acts_match("Bharatiya Nyaya Sanhita, 2023", "The Bharatiya Nyaya Sanhita, 2023")
    assert acts_match("Prevention of Cruelty to Animals Act, 1960", "The Prevention Of Cruelty To Animals Act, 1960")
    assert acts_match("Indian Penal Code, 1860", "The Indian Penal Code, 45 of 1860 (Rep., Act 45 of 2023)")
    assert acts_match("Code of Criminal Procedure, 1973", "The Code of Criminal Procedure 1973, 02 of 1974 (Rep., Act 46 of 2023)")


def test_acts_match_distinguishes_different_acts_and_versions():
    assert not acts_match("Bharatiya Nyaya Sanhita, 2023", "The Bharatiya Nagarik Suraksha Sanhita, 2023")
    assert not acts_match("Code of Criminal Procedure, 1973", "The Code of Civil Procedure, 1908")
    assert not acts_match("Consumer Protection Act, 2019", "The Consumer Protection Act, 1986")
    assert not acts_match("Motor Vehicles Act, 1988", "")


def test_section_matching_uses_the_base_section():
    assert base_section("2(11)") == "2"
    assert base_section("Section 304a") == "304A"
    assert section_numbers_match("2(47)", "2")
    assert section_numbers_match("199A", "199a")
    assert not section_numbers_match("147", "149")
    assert not section_numbers_match("304A", "304")
    assert not section_numbers_match("", "")


def test_url_parsers():
    assert india_code_handle("https://indiacode.gov.in/handle/123456789/545816") == "123456789/545816"
    assert india_code_handle("https://www.indiacode.nic.in/handle/123456789/2263?sam_handle=123456789/1362") == "123456789/2263"
    assert india_code_handle("https://indiankanoon.org/doc/1763700/") is None
    assert indian_kanoon_doc_id("https://indiankanoon.org/doc/1763700/") == "1763700"
    assert indian_kanoon_doc_id("https://indiankanoon.org/search/?formInput=x") is None


def test_is_marked_repealed():
    assert is_marked_repealed({"title": "Causing death by negligence [Repealed w.e.f. 1 Jul 2024; see BNS s.106]"})
    assert is_marked_repealed({"title": "x", "text": "[Repealed w.e.f. 1 July 2024 by ...] Whoever ..."})
    assert not is_marked_repealed({"title": "Causing death by negligence", "text": "Whoever causes death ..."})


def test_india_code_section_record_must_match_act_and_section():
    md = _md(act_name="The Bharatiya Nyaya Sanhita, 2023", section_number="325", state_name="CENTRAL",
             title="Mischief by killing or maiming animal.")
    assert check_india_code_record("Bharatiya Nyaya Sanhita, 2023", "325", md).status == OK

    wrong_section = check_india_code_record("Bharatiya Nyaya Sanhita, 2023", "147", md)
    assert wrong_section.status == FAIL and "section mismatch" in wrong_section.message

    wrong_act = check_india_code_record("Bharatiya Nagarik Suraksha Sanhita, 2023", "325", md)
    assert wrong_act.status == FAIL and "Act mismatch" in wrong_act.message


def test_india_code_state_copy_and_missing_metadata_fail():
    state_copy = _md(title="The Indian Penal Code, 1860", state_name="Chhattisgarh")
    result = check_india_code_record("Indian Penal Code, 1860", "279", state_copy, repealed=True)
    assert result.status == FAIL and "state copy" in result.message
    assert check_india_code_record("Indian Penal Code, 1860", "279", {}).status == FAIL


def test_whole_act_link_warns_only_for_entries_marked_repealed():
    repealed_ipc = _md(title="The Indian Penal Code, 45 of 1860 (Rep., Act 45 of 2023)")
    assert check_india_code_record("Indian Penal Code, 1860", "279", repealed_ipc, repealed=True).status == WARN
    assert check_india_code_record("Indian Penal Code, 1860", "279", repealed_ipc, repealed=False).status == FAIL

    tpa_act = _md(title="The Transfer of Property Act, 1882", state_name="CENTRAL")
    assert check_india_code_record("Transfer of Property Act, 1882", "54", tpa_act).status == FAIL
    wrong_whole_act = check_india_code_record("Code of Criminal Procedure, 1973", "154", tpa_act, repealed=True)
    assert wrong_whole_act.status == FAIL and "Act mismatch" in wrong_whole_act.message


def test_indian_kanoon_statute_titles():
    assert parse_indian_kanoon_section_title("Section 191 in Bharatiya Nyaya Sanhita, 2023") == (
        "191",
        "Bharatiya Nyaya Sanhita, 2023",
    )
    assert parse_indian_kanoon_section_title("Bharatiya Nyaya Sanhita, 2023") is None

    ok = check_indian_kanoon_statute_title(
        "Prevention of Cruelty to Animals Act, 1960", "11", "Section 11 in The Prevention Of Cruelty To Animals Act, 1960"
    )
    assert ok.status == OK
    whole_act_page = check_indian_kanoon_statute_title("Bharatiya Nyaya Sanhita, 2023", "318", "Bharatiya Nyaya Sanhita, 2023")
    assert whole_act_page.status == FAIL and "not a section page" in whole_act_page.message
    wrong_section = check_indian_kanoon_statute_title(
        "Transfer of Property Act, 1882", "123", "Section 107 in The Transfer Of Property Act, 1882"
    )
    assert wrong_section.status == FAIL and "section mismatch" in wrong_section.message
    wrong_act = check_indian_kanoon_statute_title(
        "Bharatiya Nyaya Sanhita, 2023", "173", "Section 173 in Bharatiya Nagarik Suraksha Sanhita, 2023"
    )
    assert wrong_act.status == FAIL


def test_precedent_page_must_contain_lead_party():
    assert lead_party_name("Animal Welfare Board of India v. A. Nagaraja & Ors.") == "Animal Welfare Board of India"
    assert lead_party_name("Patel Roadways Limited vs Birla Yamaha Limited") == "Patel Roadways Limited"

    assert check_precedent_page(
        "Animal Welfare Board of India v. A. Nagaraja & Ors.",
        "Animal Welfare Board Of India vs A. Nagaraja &amp; Ors on 7 May, 2014",
    ).status == OK
    # Corporate suffixes differ between the corpus ("Ltd.") and Indian Kanoon ("Limited").
    assert check_precedent_page(
        "Patel Roadways Ltd. v. Birla Yamaha Ltd.", "Patel Roadways Limited vs Birla Yamaha Limited on 28 March, 2000"
    ).status == OK
    # The party may appear only in the body text.
    assert check_precedent_page("Poona Ram v. Moti Ram", "Judgment", "... POONA RAM ... Appellant ...").status == OK
    assert check_precedent_page(
        "Sarla Verma & Ors. v. Delhi Transport Corporation", "Some Other Party vs State on 1 January, 2000", ""
    ).status == FAIL


def test_curated_precedent_title_check():
    title = "Animal Welfare Board Of India vs A. Nagaraja & Ors on 7 May, 2014"
    assert check_curated_precedent_title("nagaraja", title).status == OK
    assert check_curated_precedent_title("m.k. gupta", "Lucknow Development Authority vs M.K. Gupta on 5 November, 1993").status == OK
    assert check_curated_precedent_title("nagaraja", "M. Nagarajan vs State Of Tamil Nadu on 1 May, 2010").status == FAIL


def test_ui_map_consistency_flags_diverging_links():
    token_acts = {"nyaya": "Bharatiya Nyaya Sanhita, 2023"}
    provisions = [
        {"doc_id": "bns_325", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section_number": "325",
         "source_url": "https://indiacode.gov.in/handle/123456789/545816"},
        {"doc_id": "bns_190", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section_number": "190",
         "source_url": "https://indiacode.gov.in/handle/123456789/545864"},
    ]
    precedents = [
        {"doc_id": "awbi", "title": "Animal Welfare Board of India v. A. Nagaraja & Ors.",
         "source_url": "https://indiankanoon.org/doc/39696860/"},
    ]
    ic_map = {
        ("nyaya", "325"): "https://indiacode.gov.in/handle/123456789/545816",
        ("nyaya", "190"): "https://indiacode.gov.in/handle/123456789/545649",  # Diverges from the corpus.
        ("bogus", "1"): "https://indiacode.gov.in/handle/123456789/2",
    }
    ik_map = {("nyaya", "325"): "https://indiankanoon.org/doc/186696080/"}

    def resolve(act_name, section_number):
        act = (act_name or "").lower()
        token = next((t for t in token_acts if t in act), None)
        return token, base_section(section_number)

    ui = UiLinks(
        india_code_map=ic_map,
        indian_kanoon_map=ik_map,
        curated_precedents={},
        token_acts=token_acts,
        resolve_key=resolve,
        statute_url=lambda a, s: ik_map.get(resolve(a, s)),
        official_url=lambda a, s, u: ic_map.get(resolve(a, s)) or u,
        precedent_url=lambda _src, _title: "https://indiankanoon.org/doc/1/",
    )
    results = check_ui_map_consistency(provisions, precedents, ui)
    fails = {(r.subject, r.message) for r in results if r.status == FAIL}
    assert (
        "UI map vs corpus bns_190",
        "UI India Code link differs from corpus source_url https://indiacode.gov.in/handle/123456789/545864",
    ) in fails
    assert any(r.subject == "UI map key ('bogus', '1')" and r.status == FAIL for r in results)
    assert any(r.subject == "UI judgment link awbi" and r.status == FAIL for r in results)
    assert not any(r.subject == "UI map vs corpus bns_325" for r in results)


def test_real_ui_maps_and_corpus_are_consistent_offline():
    assert check_ui_map_consistency(FALLBACK_PROVISIONS, CURATED_LANDMARK_PRECEDENTS, load_ui_links()) == []


def test_run_checks_with_fake_fetcher_has_no_network_io():
    class FakeFetcher:
        def india_code_metadata(self, handle):
            assert handle == "123456789/545816"
            return _md(
                act_name="The Bharatiya Nyaya Sanhita, 2023",
                section_number="325",
                state_name="CENTRAL",
                title="Mischief by killing or maiming animal.",
            )

        def indian_kanoon_page(self, url):
            if "186696080" in url:
                return "Section 325 in Bharatiya Nyaya Sanhita, 2023 - Indian Kanoon", ""
            return "Animal Welfare Board Of India vs A. Nagaraja & Ors on 7 May, 2014", ""

    provisions = [
        {"doc_id": "bns_325", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section_number": "325",
         "title": "Mischief by killing or maiming animal",
         "source_url": "https://indiacode.gov.in/handle/123456789/545816"}
    ]
    precedents = [
        {"doc_id": "awbi", "title": "Animal Welfare Board of India v. A. Nagaraja & Ors.",
         "source_url": "https://indiankanoon.org/doc/39696860/"}
    ]
    results = run_checks(FakeFetcher(), provisions=provisions, precedents=precedents, include_ui=False)
    assert summarize(results) == {OK: 2, WARN: 0, FAIL: 0}


def test_backoff_seconds_respects_retry_after_and_rate_limits():
    assert backoff_seconds(0, rate_limited=False) == 1.5
    assert backoff_seconds(1, rate_limited=True) == 20.0
    assert backoff_seconds(0, rate_limited=True, retry_after="45") == 45.0
    assert backoff_seconds(20, rate_limited=True) == 120.0
