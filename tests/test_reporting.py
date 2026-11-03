import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from secom.reporting import inject_sections, markdown_table, parse_sections, write_results


def test_markdown_table_formats_floats_and_missing_values() -> None:
    frame = pd.DataFrame({"model": ["a"], "score": [0.12345], "other": [np.nan]})

    assert markdown_table(frame).splitlines() == [
        "| model | score | other |",
        "|---|---|---|",
        "| a | 0.123 | n/a |",
    ]


def test_write_results_creates_both_files(tmp_path: Path) -> None:
    write_results({"Title": "content"}, {"value": 1}, directory=tmp_path)

    assert "## Title" in (tmp_path / "results.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8")) == {"value": 1}


def test_sections_are_injected_between_markers() -> None:
    sections = parse_sections("# Title\n\n## First\n\nnew table\n\n## Second\n\nother\n")
    document = "intro\n<!-- BEGIN: First -->\nold\n<!-- END: First -->\noutro"

    assert inject_sections(document, sections) == (
        "intro\n<!-- BEGIN: First -->\nnew table\n<!-- END: First -->\noutro"
    )


def test_marker_without_a_matching_section_is_an_error() -> None:
    with pytest.raises(KeyError, match="Missing"):
        inject_sections("<!-- BEGIN: Missing -->x<!-- END: Missing -->", {})
