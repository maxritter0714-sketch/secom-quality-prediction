import numpy as np
import pandas as pd
import pytest

from secom.validation import temporal_split


def _hourly(n_rows: int) -> pd.Series:
    return pd.Series(pd.date_range("2008-07-19", periods=n_rows, freq="h"))


def test_holdout_is_the_latest_period() -> None:
    split = temporal_split(_hourly(100), test_fraction=0.2)

    assert split.train.tolist() == list(range(80))
    assert split.test.tolist() == list(range(80, 100))


def test_shared_timestamp_is_not_split_across_periods() -> None:
    timestamps = _hourly(10)
    timestamps.iloc[8] = timestamps.iloc[7]

    split = temporal_split(timestamps, test_fraction=0.2)

    assert np.array_equal(split.test, [9])
    assert timestamps.iloc[split.train].max() < timestamps.iloc[split.test].min()


def test_unsorted_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        temporal_split(_hourly(10).iloc[::-1])


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_invalid_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        temporal_split(_hourly(10), test_fraction=fraction)
