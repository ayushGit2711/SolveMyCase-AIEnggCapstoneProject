"""Export a DualOutputResponse as a shareable Markdown document (pure function)."""

from datetime import date
from typing import List, Optional

from solvemycase.data.ingestion.schema import DualOutputResponse
from solvemycase.ui.components.formatting import (
    DISCLAIMER,
    PHASE_LABELS,
    domain_label,
    extract_deadlines,
    group_steps_by_phase,
    has_real_deadline,
    markdown_table_cell,
    precedent_read_url,
    safe_http_url,
    statute_search_url,
)


def response_to_markdown(scenario: str, response: DualOutputResponse, generated_on: Optional[date] = None) -> str:
    """Render the scenario, action plan, deadlines, and verified citations as Markdown.

    Args:
        scenario: The user's original scenario text.
        response: Pipeline output to export.
        generated_on: Date stamp (defaults to today; injectable for tests).

    Returns:
        Markdown string suitable for st.download_button.
    """
    generated_on = generated_on or date.today()
    lines: List[str] = [
        "# solvemycase — Legal Action Plan",
        "",
        f"*Generated on {generated_on.isoformat()} · Category: {domain_label(response.domain)}*",
        "",
        "## Your situation",
        "",
        scenario.strip(),
        "",
        "## Action plan",
        "",
    ]

    for phase, steps in group_steps_by_phase(response.action_plan).items():
        lines += [f"### {PHASE_LABELS[phase]}", ""]
        for step in steps:
            lines.append(f"**Step {step.step_number}: {step.title}** ({step.priority})")
            lines.append("")
            lines.append(step.description)
            lines.append("")
            lines.append(f"- Where: {step.forum_or_authority}")
            if has_real_deadline(step):
                lines.append(f"- Deadline: {step.limitation_period.strip()}")
            if step.statutory_basis:
                lines.append(f"- Legal basis: {step.statutory_basis}")
            lines.append("")

    deadlines = extract_deadlines(response.action_plan)
    if deadlines:
        lines += ["## Deadlines", "", "| Deadline | Action | Where |", "|---|---|---|"]
        lines += [
            f"| {markdown_table_cell(d['Deadline'])} | {markdown_table_cell(d['Action'])} | {markdown_table_cell(d['Where'])} |"
            for d in deadlines
        ]
        lines.append("")

    if response.statutory_citations:
        lines += ["## Verified laws", ""]
        for cit in response.statutory_citations:
            links = []
            kanoon = statute_search_url(cit.act_name, cit.section_number)
            if kanoon:
                links.append(f"[Indian Kanoon]({kanoon})")
            official = safe_http_url(cit.source_url)
            if official:
                links.append(f"[India Code]({official})")
            suffix = f" ({' · '.join(links)})" if links else ""
            lines.append(f"- **{cit.act_name}, Section {cit.section_number}**: {cit.summary_of_provision}{suffix}")
        lines.append("")

    if response.precedent_citations:
        lines += ["## Verified court judgments", ""]
        for prec in response.precedent_citations:
            ref = prec.citation or (str(prec.year) if prec.year else "")
            read_url = precedent_read_url(prec.source_url, prec.case_title)
            suffix = f" ([Read judgment]({read_url}))" if read_url else ""
            lines.append(f"- **{prec.case_title}** {ref} ({prec.court}): {prec.legal_principle}{suffix}")
        lines.append("")

    lines += ["---", "", f"> {DISCLAIMER}", ""]
    return "\n".join(lines)
