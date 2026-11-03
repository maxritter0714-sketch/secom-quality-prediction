import numpy as np
import pandas as pd

from secom.eda import (
    constant_feature_counts,
    correlated_pair_counts,
    missing_fraction,
    top_separating_features,
    weekly_fail_rate,
)


def test_missing_fraction_per_feature() -> None:
    features = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0]})

    assert missing_fraction(features).to_dict() == {"a": 0.5, "b": 0.0}


def test_constant_counts_partition_all_features() -> None:
    features = pd.DataFrame(
        {
            "constant": [1.0] * 100,
            "near_constant": [1.0] * 96 + [2.0, 3.0, 4.0, 5.0],
            "varied": np.arange(100.0),
        }
    )

    assert constant_feature_counts(features) == {"constant": 1, "near_constant": 1, "other": 1}


def test_correlated_pairs_counts_each_pair_once() -> None:
    base = np.arange(20.0)
    noise = np.tile([1.0, -1.0], 10)
    features = pd.DataFrame({"a": base, "b": -2 * base, "c": noise})

    assert correlated_pair_counts(features, thresholds=(0.99,)) == {0.99: 1}


def test_weekly_fail_rate_aggregates_counts() -> None:
    timestamps = pd.Series(pd.to_datetime(["2008-07-21", "2008-07-22", "2008-07-29"]))
    weekly = weekly_fail_rate(pd.Series([1, 0, 0]), timestamps)

    assert weekly["units"].tolist() == [2, 1]
    assert weekly["fail_rate"].tolist() == [0.5, 0.0]


def test_top_separating_feature_is_ranked_first() -> None:
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    features = pd.DataFrame(
        {"noise": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], "signal": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2]}
    )

    assert top_separating_features(features, labels, n_features=1, min_class_count=2) == ["signal"]


def test_sparsely_observed_feature_is_not_ranked() -> None:
    labels = pd.Series([0, 0, 0, 1, 1, 1])
    features = pd.DataFrame(
        {"sparse": [0.0, 0.1, 0.2, 50.0, np.nan, np.nan], "dense": [0.0, 1.0, 2.0, 1.0, 2.0, 3.0]}
    )

    assert top_separating_features(features, labels, min_class_count=2) == ["dense"]
