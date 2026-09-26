"""Manual link checker for the curated corpus and the links the UI renders (needs network access).

Usage:
    python -m solvemycase.data.ingestion.link_checker [--verbose] [--skip-ui] [--timeout SECONDS]

Checks:
    * Statutes: the India Code record behind each handle
      (``https://indiacode.gov.in/server/api/pid/find?id=123456789/<handle>``) must be a central record naming the
      same Act (``dc.identifier.act_name``) and section (``dc.identifier.section_number``). A link to a whole Act
      fails, except for entries explicitly marked repealed: repealed Codes have no section-level records, so that
      is only a warning.
    * Indian Kanoon statute pages must be titled "Section N in <Act>".
    * Precedents: the Indian Kanoon judgment page must contain the lead party's name.
    * UI (``ui/components/formatting.py``): the links the UI actually renders for every corpus document
      (``statute_search_url``, ``official_statute_url`` and ``precedent_read_url``, i.e. after its link maps and
      overrides) are checked as above, as is every other entry of its direct link maps and curated judgment URLs.
      Offline, the maps must agree with the corpus (same India Code handle for the same Act and section, same
      judgment URL). A section with no direct Indian Kanoon page, for which the UI shows a search link, is a warning.

Exits with status 1 if any check fails. The comparison logic is pure (no I/O); it and the check orchestration (with
a fake fetcher) are unit-tested offline in ``tests/test_link_checker.py``.
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

OK, WARN, FAIL = "ok", "warn", "fail"

INDIA_CODE_PID_API = "https://indiacode.gov.in/server/api/pid/find?id="
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

_ACT_NAME_STOPWORDS = {"the", "of", "to", "and", "act", "rep", "repealed"}
# Words dropped from party names before matching ("Ltd." vs "Limited", "Corp." vs "Corporation", "& Ors").
_PARTY_NOISE_WORDS = {
    "the", "and", "ltd", "limited", "pvt", "private", "co", "company", "corp", "corporation", "inc",
    "ors", "anr", "others", "another",
}
_YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_SECTION_RE = re.compile(r"^\s*(?:sections?|secs?\.?|s\.)?\s*(\d+[A-Za-z]*)", re.IGNORECASE)
_IK_SECTION_TITLE_RE = re.compile(r"^\s*Section\s+(\d+[A-Za-z]*)\s+in\s+(.+?)\s*$", re.IGNORECASE)
_IK_TITLE_SUFFIX_RE = re.compile(r"\s*[-|]\s*Indian Kanoon\s*$", re.IGNORECASE)
_INDIA_CODE_HANDLE_RE = re.compile(r"indiacode\.(?:gov|nic)\.in/handle/(\d+/\d+)", re.IGNORECASE)
_INDIAN_KANOON_DOC_RE = re.compile(r"indiankanoon\.org/doc/(\d+)", re.IGNORECASE)
_INDIAN_KANOON_SEARCH_RE = re.compile(r"indiankanoon\.org/search/", re.IGNORECASE)
_VERSUS_RE = re.compile(r"\s+(?:v\.?|vs\.?|versus)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one link check."""

    status: str  # OK, WARN or FAIL
    subject: str  # What was checked, e.g. "corpus central_bns_2023_sec_325"
    url: str
    message: str


@dataclass(frozen=True)
class UiLinks:
    """The UI's link tables and resolvers (see ``load_ui_links``; tests pass fakes)."""

    india_code_map: Mapping[Tuple[str, str], str]  # (act token, section) -> India Code handle URL
    indian_kanoon_map: Mapping[Tuple[str, str], str]  # (act token, section) -> Indian Kanoon section page
    curated_precedents: Mapping[str, str]  # title key phrase -> Indian Kanoon judgment URL
    token_acts: Mapping[str, str]  # act token -> the Act it denotes
    resolve_key: Callable[[Optional[str], Optional[str]], Tuple[Optional[str], str]]  # (act, section) -> map key
    statute_url: Callable[[Optional[str], Optional[str]], Optional[str]]  # (act, section) -> Indian Kanoon link
    official_url: Callable[[Optional[str], Optional[str], Optional[str]], Optional[str]]  # + source_url -> India Code
    precedent_url: Callable[[Optional[str], Optional[str]], Optional[str]]  # (source_url, title) -> judgment link


# ---------------------------------------------------------------------------
# Pure comparison helpers (offline, unit-tested)
# ---------------------------------------------------------------------------


def _act_words_and_years(name: str) -> Tuple[frozenset, frozenset]:
    lowered = (name or "").lower()
    words = frozenset(w for w in re.findall(r"[a-z]+", lowered) if w not in _ACT_NAME_STOPWORDS)
    return words, frozenset(_YEAR_RE.findall(lowered))


def acts_match(expected_act: str, actual_act: str) -> bool:
    """True if two Act names denote the same Act.

    The significant words must be identical (so BNS and BNSS differ) and the expected year, if any, must appear in
    the actual name. Case, a leading "The" and suffixes such as ", 45 of 1860 (Rep., Act 45 of 2023)" are ignored.
    """
    exp_words, exp_years = _act_words_and_years(expected_act)
    act_words, act_years = _act_words_and_years(actual_act)
    if not exp_words or exp_words != act_words:
        return False
    return not exp_years or bool(exp_years & act_years)


def base_section(section: Optional[str]) -> str:
    """Section number without sub-section or clause: '2(11)' -> '2', 'Section 304a' -> '304A'."""
    m = _SECTION_RE.match(section or "")
    return m.group(1).upper() if m else (section or "").strip().upper()


def section_numbers_match(expected: Optional[str], actual: Optional[str]) -> bool:
    """True if both refer to the same section (sub-sections and clauses of the expected section are ignored)."""
    exp, act = base_section(expected), base_section(actual)
    return bool(exp) and exp == act


def india_code_handle(url: Optional[str]) -> Optional[str]:
    """'https://indiacode.gov.in/handle/123456789/545816' -> '123456789/545816' (None if not an India Code handle)."""
    m = _INDIA_CODE_HANDLE_RE.search(url or "")
    return m.group(1) if m else None


def indian_kanoon_doc_id(url: Optional[str]) -> Optional[str]:
    """'https://indiankanoon.org/doc/1763700/' -> '1763700' (None if not an Indian Kanoon document URL)."""
    m = _INDIAN_KANOON_DOC_RE.search(url or "")
    return m.group(1) if m else None


def is_indian_kanoon_search(url: Optional[str]) -> bool:
    """True for an Indian Kanoon search-results link (the UI's fallback when no direct page is known)."""
    return bool(_INDIAN_KANOON_SEARCH_RE.search(url or ""))


def is_marked_repealed(entry: Mapping) -> bool:
    """True if a corpus provision is explicitly marked repealed in its title or text."""
    title = str(entry.get("title") or "").lower()
    text = str(entry.get("text") or "").lstrip().lower()
    return "[repealed" in title or text.startswith("[repealed")


def metadata_value(metadata: Optional[Mapping], key: str) -> str:
    """First value of a DSpace metadata field ('' if missing)."""
    values = (metadata or {}).get(key) or []
    if values and isinstance(values[0], Mapping):
        return str(values[0].get("value") or "").strip()
    return ""


def check_india_code_record(
    expected_act: str,
    expected_section: str,
    metadata: Optional[Mapping],
    repealed: bool = False,
    subject: str = "",
    url: str = "",
) -> CheckResult:
    """Compare an India Code record (DSpace metadata) with the Act and section it is supposed to show."""
    if not metadata:
        return CheckResult(FAIL, subject, url, "India Code returned no metadata for this handle")
    state = metadata_value(metadata, "dc.identifier.state_name")
    act_name = metadata_value(metadata, "dc.identifier.act_name")
    section = metadata_value(metadata, "dc.identifier.section_number")
    title = metadata_value(metadata, "dc.title")
    if state and state.upper() != "CENTRAL":
        return CheckResult(FAIL, subject, url, f"record is a state copy ({state}): {act_name or title!r}")
    if section:
        if not acts_match(expected_act, act_name):
            return CheckResult(FAIL, subject, url, f"Act mismatch: expected {expected_act!r}, record has {act_name!r}")
        if not section_numbers_match(expected_section, section):
            return CheckResult(
                FAIL, subject, url, f"section mismatch: expected s.{expected_section}, record is s.{section} ({title!r})"
            )
        return CheckResult(OK, subject, url, f"{act_name} s.{section}: {title}")
    whole_act = act_name or title
    if not acts_match(expected_act, whole_act):
        return CheckResult(FAIL, subject, url, f"Act mismatch: expected {expected_act!r}, record is {whole_act!r}")
    if repealed:
        return CheckResult(
            WARN, subject, url, f"links to the whole repealed Act {whole_act!r} (no section-level record exists)"
        )
    return CheckResult(FAIL, subject, url, f"links to the whole Act {whole_act!r}, not to s.{expected_section}")


def clean_indian_kanoon_title(title: Optional[str]) -> str:
    """Unescape and trim an Indian Kanoon <title> (drops a trailing '- Indian Kanoon' if present)."""
    cleaned = " ".join(html.unescape(title or "").split())
    return _IK_TITLE_SUFFIX_RE.sub("", cleaned).strip()


def parse_indian_kanoon_section_title(title: Optional[str]) -> Optional[Tuple[str, str]]:
    """'Section 11 in The Prevention Of Cruelty To Animals Act, 1960' -> ('11', 'The Prevention ... 1960')."""
    m = _IK_SECTION_TITLE_RE.match(clean_indian_kanoon_title(title))
    return (m.group(1), m.group(2)) if m else None


def check_indian_kanoon_statute_title(
    expected_act: str,
    expected_section: str,
    page_title: Optional[str],
    subject: str = "",
    url: str = "",
) -> CheckResult:
    """An Indian Kanoon statute page must be titled 'Section N in <Act>' for the expected Act and section."""
    parsed = parse_indian_kanoon_section_title(page_title)
    shown = clean_indian_kanoon_title(page_title)
    if not parsed:
        return CheckResult(FAIL, subject, url, f"not a section page: title is {shown!r}")
    section, act = parsed
    if not acts_match(expected_act, act):
        return CheckResult(FAIL, subject, url, f"Act mismatch: expected {expected_act!r}, page is {shown!r}")
    if not section_numbers_match(expected_section, section):
        return CheckResult(FAIL, subject, url, f"section mismatch: expected s.{expected_section}, page is {shown!r}")
    return CheckResult(OK, subject, url, shown)


def _normalize_party_text(text: Optional[str]) -> str:
    words = re.findall(r"[a-z0-9]+", html.unescape(text or "").lower())
    return " ".join(w for w in words if w not in _PARTY_NOISE_WORDS)


def lead_party_name(case_title: Optional[str]) -> str:
    """First party of a case title: 'Patel Roadways Ltd. v. Birla Yamaha Ltd.' -> 'Patel Roadways Ltd.'."""
    return _VERSUS_RE.split((case_title or "").strip(), maxsplit=1)[0].strip()


def check_precedent_page(
    case_title: str,
    page_title: Optional[str],
    page_text: Optional[str] = "",
    subject: str = "",
    url: str = "",
) -> CheckResult:
    """An Indian Kanoon judgment page must contain the lead party's name (corporate suffixes are ignored)."""
    lead = _normalize_party_text(lead_party_name(case_title))
    shown = clean_indian_kanoon_title(page_title)
    if not lead:
        return CheckResult(FAIL, subject, url, f"cannot derive a lead party from {case_title!r}")
    for haystack in (page_title, page_text):
        if f" {lead} " in f" {_normalize_party_text(haystack)} ":
            return CheckResult(OK, subject, url, shown)
    return CheckResult(FAIL, subject, url, f"lead party {lead_party_name(case_title)!r} not found; page is {shown!r}")


def check_curated_precedent_title(
    key_phrase: str, page_title: Optional[str], subject: str = "", url: str = ""
) -> CheckResult:
    """A curated UI judgment link must lead to a judgment whose title contains the key phrase."""
    shown = clean_indian_kanoon_title(page_title)
    if f" {_normalize_party_text(key_phrase)} " in f" {_normalize_party_text(page_title)} ":
        return CheckResult(OK, subject, url, shown)
    return CheckResult(FAIL, subject, url, f"key phrase {key_phrase!r} not in judgment title {shown!r}")


def check_ui_map_consistency(
    provisions: Sequence[Mapping], precedents: Sequence[Mapping], ui: UiLinks
) -> List[CheckResult]:
    """Offline checks that the UI's link tables agree with the corpus (returns failures only).

    Fails when the UI would link a corpus provision to a different India Code handle than its ``source_url``, when a
    map key uses an act token with no known Act, or when the judgment link the UI renders for a corpus precedent
    differs from its ``source_url``.
    """
    results: List[CheckResult] = []
    for entry in provisions:
        token, section = ui.resolve_key(entry.get("act_name"), entry.get("section_number"))
        ic_url = ui.india_code_map.get((token, section)) if token else None
        corpus_url = entry.get("source_url") or ""
        if ic_url is not None and india_code_handle(ic_url) != india_code_handle(corpus_url):
            results.append(
                CheckResult(
                    FAIL, f"UI map vs corpus {entry.get('doc_id')}", ic_url,
                    f"UI India Code link differs from corpus source_url {corpus_url}",
                )
            )
    for key, map_url in list(ui.india_code_map.items()) + list(ui.indian_kanoon_map.items()):
        if key[0] not in ui.token_acts:
            results.append(CheckResult(FAIL, f"UI map key {key}", map_url, "unknown act token (no expected Act)"))
    for entry in precedents:
        corpus_url = entry.get("source_url") or ""
        ui_url = ui.precedent_url(corpus_url, entry.get("title"))
        if indian_kanoon_doc_id(ui_url) != indian_kanoon_doc_id(corpus_url):
            results.append(
                CheckResult(
                    FAIL, f"UI judgment link {entry.get('doc_id')}", ui_url or "",
                    f"UI link differs from corpus source_url {corpus_url}",
                )
            )
    return results


def summarize(results: Iterable[CheckResult]) -> Dict[str, int]:
    """Count results by status."""
    counts = {OK: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


_MAX_BACKOFF_SECONDS = 120.0


def backoff_seconds(attempt: int, rate_limited: bool = False, retry_after: Optional[str] = None) -> float:
    """Seconds to wait before retrying after failed attempt number ``attempt`` (0-based).

    A numeric ``Retry-After`` header wins; otherwise the wait doubles per attempt, starting at 10 s when the server
    is rate-limiting (HTTP 429) and 1.5 s for other transient errors. Capped at two minutes.
    """
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after.strip()), _MAX_BACKOFF_SECONDS)
    base = 10.0 if rate_limited else 1.5
    return min(base * (2 ** attempt), _MAX_BACKOFF_SECONDS)


# ---------------------------------------------------------------------------
# I/O layer
# ---------------------------------------------------------------------------


class HttpFetcher:
    """Cached HTTP GETs with a browser User-Agent, retries (honouring HTTP 429) and a polite delay.

    ``urlopen`` and ``sleep`` are injectable so the retry logic can be tested offline.
    """

    def __init__(
        self,
        timeout: float = 40.0,
        delay: float = 1.0,
        retries: int = 4,
        urlopen: Callable = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.timeout, self.delay, self.retries = timeout, delay, retries
        self._urlopen, self._sleep = urlopen, sleep
        self._cache: Dict[str, str] = {}

    def get(self, url: str) -> str:
        if url in self._cache:
            return self._cache[url]
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            rate_limited, retry_after = False, None
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with self._urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8", "replace")
                self._cache[url] = body
                self._sleep(self.delay)
                return body
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    raise
                last_error = err
                rate_limited = err.code == 429
                retry_after = err.headers.get("Retry-After") if err.headers else None
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last_error = err
            if attempt < self.retries:
                self._sleep(backoff_seconds(attempt, rate_limited, retry_after))
        raise last_error  # type: ignore[misc]

    def india_code_metadata(self, handle: str) -> Dict:
        return json.loads(self.get(INDIA_CODE_PID_API + handle)).get("metadata") or {}

    def indian_kanoon_page(self, url: str) -> Tuple[str, str]:
        page = self.get(url)
        title = re.search(r"(?is)<title>(.*?)</title>", page)
        body = re.sub(r"(?is)<(script|style).*?</\1>", " ", page)
        body = " ".join(html.unescape(re.sub(r"(?s)<[^>]+>", " ", body)).split())
        return (title.group(1) if title else ""), body


def load_ui_links() -> UiLinks:
    """The link tables and resolvers of ``ui/components/formatting.py`` (imports the UI package)."""
    from solvemycase.ui.components import formatting as f

    return UiLinks(
        india_code_map=f._DIRECT_INDIACODE_SECTION_URLS,
        indian_kanoon_map=f._DIRECT_INDIAN_KANOON_SECTION_URLS,
        curated_precedents=f._CURATED_PRECEDENT_URLS,
        token_acts={token: act for token, (_phrase, act) in f._ACT_LINK_TOKENS.items()},
        resolve_key=f._resolve_act_section_key,
        statute_url=f.statute_search_url,
        official_url=f.official_statute_url,
        precedent_url=f.precedent_read_url,
    )


# ---------------------------------------------------------------------------
# Orchestration (the fetcher is injected: anything with india_code_metadata(handle) and indian_kanoon_page(url))
# ---------------------------------------------------------------------------


def _guard(subject: str, url: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except urllib.error.HTTPError as err:
        hint = " (rate-limited; re-run later)" if err.code == 429 else ""
        return CheckResult(FAIL, subject, url, f"HTTP {err.code}{hint}")
    except Exception as err:  # Network errors must not abort the whole run.
        return CheckResult(FAIL, subject, url, f"fetch failed: {type(err).__name__}: {err}")


def _check_statute_url(
    fetcher, subject: str, url: Optional[str], act: str, section: str, repealed: bool, allow_search: bool = False
) -> CheckResult:
    url = url or ""
    handle = india_code_handle(url)
    if handle:
        return _guard(
            subject, url,
            lambda: check_india_code_record(act, section, fetcher.india_code_metadata(handle), repealed, subject, url),
        )
    if indian_kanoon_doc_id(url):
        return _guard(
            subject, url,
            lambda: check_indian_kanoon_statute_title(act, section, fetcher.indian_kanoon_page(url)[0], subject, url),
        )
    if allow_search and is_indian_kanoon_search(url):
        return CheckResult(WARN, subject, url, "no direct Indian Kanoon section page; the UI shows a search link")
    return CheckResult(FAIL, subject, url, "not an India Code handle or Indian Kanoon document URL")


def _check_precedent_url(fetcher, subject: str, url: Optional[str], case_title: str) -> CheckResult:
    url = url or ""
    if not indian_kanoon_doc_id(url):
        return CheckResult(FAIL, subject, url, "not an Indian Kanoon judgment URL")
    return _guard(subject, url, lambda: check_precedent_page(case_title, *fetcher.indian_kanoon_page(url), subject, url))


def run_checks(
    fetcher,
    provisions: Optional[Sequence[Mapping]] = None,
    precedents: Optional[Sequence[Mapping]] = None,
    ui: Optional[UiLinks] = None,
    include_ui: bool = True,
) -> List[CheckResult]:
    """Check every corpus link and (unless ``include_ui`` is False) every link the UI renders or maps.

    ``provisions`` / ``precedents`` default to the curated corpus in ``normalizer``; ``ui`` defaults to
    ``load_ui_links()``. Returns one result per check.
    """
    if provisions is None or precedents is None:
        from solvemycase.data.ingestion.normalizer import CURATED_LANDMARK_PRECEDENTS, FALLBACK_PROVISIONS

        provisions = FALLBACK_PROVISIONS if provisions is None else provisions
        precedents = CURATED_LANDMARK_PRECEDENTS if precedents is None else precedents

    results: List[CheckResult] = []
    for entry in provisions:
        results.append(
            _check_statute_url(
                fetcher, f"corpus {entry['doc_id']}", entry["source_url"], entry["act_name"],
                entry["section_number"], is_marked_repealed(entry),
            )
        )
    for entry in precedents:
        results.append(_check_precedent_url(fetcher, f"corpus {entry['doc_id']}", entry["source_url"], entry["title"]))

    if not include_ui:
        return results
    if ui is None:
        try:
            ui = load_ui_links()
        except Exception as err:  # The UI package imports the LangGraph pipeline, which can fail in a broken checkout.
            results.append(
                CheckResult(
                    FAIL, "UI links", "", f"cannot import ui.components.formatting ({type(err).__name__}: {err}); "
                    "use --skip-ui",
                )
            )
            return results

    results.extend(check_ui_map_consistency(provisions, precedents, ui))

    # The links the UI renders for each corpus document.
    rendered: Set[Tuple[str, Tuple[Optional[str], str]]] = set()
    for entry in provisions:
        act, section, doc_id = entry["act_name"], entry["section_number"], entry["doc_id"]
        repealed, key = is_marked_repealed(entry), ui.resolve_key(act, section)
        for label, url in (
            ("Indian Kanoon", ui.statute_url(act, section)),
            ("India Code", ui.official_url(act, section, entry.get("source_url"))),
        ):
            results.append(
                _check_statute_url(fetcher, f"UI link {doc_id} ({label})", url, act, section, repealed, True)
            )
            rendered.add((url or "", key))
    for entry in precedents:
        url = ui.precedent_url(entry.get("source_url"), entry.get("title"))
        results.append(_check_precedent_url(fetcher, f"UI judgment link {entry['doc_id']}", url, entry["title"]))

    # Remaining map entries (sections the UI can link that are not in the curated corpus).
    repealed_keys = {ui.resolve_key(e["act_name"], e["section_number"]) for e in provisions if is_marked_repealed(e)}
    for label, link_map in (("India Code", ui.india_code_map), ("Indian Kanoon", ui.indian_kanoon_map)):
        for key, url in sorted(link_map.items()):
            act = ui.token_acts.get(key[0])
            if act is None or (url, key) in rendered:
                continue  # Unknown tokens are reported by check_ui_map_consistency.
            results.append(_check_statute_url(fetcher, f"UI {label} map {key}", url, act, key[1], key in repealed_keys))
    for key_phrase, url in sorted(ui.curated_precedents.items()):
        subject = f"UI curated judgment {key_phrase!r}"
        results.append(
            _guard(
                subject, url,
                lambda k=key_phrase, u=url, s=subject: check_curated_precedent_title(
                    k, fetcher.indian_kanoon_page(u)[0], s, u
                ),
            )
        )
    return results


def main(argv: Optional[Sequence[str]] = None, fetcher=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--verbose", action="store_true", help="also print passing checks")
    parser.add_argument("--skip-ui", action="store_true", help="skip the UI links (corpus only)")
    parser.add_argument("--timeout", type=float, default=40.0, help="per-request timeout in seconds")
    args = parser.parse_args(argv)

    results = run_checks(fetcher or HttpFetcher(timeout=args.timeout), include_ui=not args.skip_ui)
    for r in results:
        if r.status != OK or args.verbose:
            print(f"[{r.status.upper():4s}] {r.subject}: {r.message}" + (f" <{r.url}>" if r.url else ""))
    counts = summarize(results)
    print(f"\nLink check: {counts[OK]} ok, {counts[WARN]} warning(s), {counts[FAIL]} failure(s) in {len(results)} checks.")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
