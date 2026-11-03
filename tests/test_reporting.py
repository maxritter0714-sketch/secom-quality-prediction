import json
from pathlib import Path

import numpy as np
import pandas as pd

from secom.reporting import markdown_table, write_results


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
