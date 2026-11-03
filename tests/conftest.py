"""Shared synthetic fixtures; CI never downloads the real dataset."""

from pathlib import Path

import pytest


@pytest.fixture
def raw_secom_dir(tmp_path: Path) -> Path:
    """Write a tiny dataset in the raw SECOM format, deliberately out of time order."""
    (tmp_path / "secom.data").write_text(
        "1.0 NaN 3.0\n4.0 5.0 NaN\n7.0 8.0 9.0\n10.0 11.0 12.0\n", encoding="utf-8"
    )
    (tmp_path / "secom_labels.data").write_text(
        '-1 "19/07/2008 12:00:00"\n'
        '1 "19/07/2008 11:00:00"\n'
        '-1 "01/08/2008 09:30:00"\n'
        '1 "18/07/2008 23:59:59"\n',
        encoding="utf-8",
    )
    return tmp_path
