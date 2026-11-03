from pathlib import Path

import numpy as np
import pytest

from secom.data import load_secom


def test_shape_and_column_names(raw_secom_dir: Path) -> None:
    features, labels, timestamps = load_secom(raw_secom_dir)

    assert features.shape == (4, 3)
    assert list(features.columns) == ["f000", "f001", "f002"]
    assert len(labels) == len(timestamps) == 4


def test_labels_are_binary_with_fail_as_positive(raw_secom_dir: Path) -> None:
    _, labels, _ = load_secom(raw_secom_dir)

    assert set(labels.unique()) == {0, 1}
    assert labels.sum() == 2


def test_rows_are_sorted_by_time_and_stay_aligned(raw_secom_dir: Path) -> None:
    features, labels, timestamps = load_secom(raw_secom_dir)

    assert timestamps.is_monotonic_increasing
    assert features["f000"].tolist() == [10.0, 4.0, 1.0, 7.0]
    assert labels.tolist() == [1, 1, 0, 0]


def test_timestamps_are_parsed_day_first(raw_secom_dir: Path) -> None:
    _, _, timestamps = load_secom(raw_secom_dir)

    assert timestamps.iloc[-1].month == 8
    assert timestamps.iloc[-1].day == 1


def test_missing_values_are_kept_as_nan(raw_secom_dir: Path) -> None:
    features, _, _ = load_secom(raw_secom_dir)

    assert np.isnan(features.to_numpy()).sum() == 2


def test_missing_files_raise_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_secom(tmp_path)


def test_unknown_label_is_rejected(raw_secom_dir: Path) -> None:
    labels_path = raw_secom_dir / "secom_labels.data"
    labels_path.write_text(
        labels_path.read_text(encoding="utf-8").replace('-1 "01', '2 "01'), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unexpected label"):
        load_secom(raw_secom_dir)


def test_row_mismatch_is_rejected(raw_secom_dir: Path) -> None:
    (raw_secom_dir / "secom.data").write_text("1.0 2.0 3.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Row mismatch"):
        load_secom(raw_secom_dir)
