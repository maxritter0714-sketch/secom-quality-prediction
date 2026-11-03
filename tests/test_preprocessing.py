import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from secom.preprocessing import (
    CorrelationFilter,
    MissingnessFilter,
    PreprocessingConfig,
    TopKSelector,
    build_pipeline,
)


@pytest.fixture
def train_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n_rows = 60
    labels = pd.Series(np.tile([0, 0, 1], n_rows // 3))
    frame = pd.DataFrame(
        {
            "signal": labels + rng.normal(0, 0.1, n_rows),
            "noise": rng.normal(0, 1, n_rows),
            "constant": np.full(n_rows, 7.0),
            "mostly_missing": np.where(np.arange(n_rows) < 50, np.nan, 1.0),
            "some_missing": np.where(np.arange(n_rows) % 10 == 0, np.nan, rng.normal(5, 1, n_rows)),
        }
    )
    return frame, labels


def _preprocessor(frame: pd.DataFrame, labels: pd.Series, **config: object) -> object:
    classifier = LogisticRegression()
    pipeline = build_pipeline(classifier, PreprocessingConfig(**config))
    return pipeline.fit(frame, labels)[:-1]


def test_imputer_statistics_come_from_training_rows_only(train_frame) -> None:
    frame, labels = train_frame
    fitted = build_pipeline(LogisticRegression(), PreprocessingConfig(k_features=None))
    fitted.fit(frame, labels)

    imputer = fitted.named_steps["impute"]
    expected = frame["some_missing"].median()
    kept = list(fitted.named_steps["drop_missing"].get_feature_names_out())

    assert imputer.statistics_[kept.index("some_missing")] == pytest.approx(expected)


def test_extreme_unseen_rows_do_not_change_the_fitted_transform(train_frame) -> None:
    """A transform learned on training rows must not depend on later rows at all."""
    frame, labels = train_frame
    query = frame.iloc[:5]
    outlier_rows = frame.iloc[:20] * 1e6

    before = _preprocessor(frame, labels, k_features=2, scaler="standard")
    transformed_before = before.transform(query)
    before.transform(outlier_rows)
    transformed_after = before.transform(query)
    refitted = _preprocessor(frame, labels, k_features=2, scaler="standard")

    np.testing.assert_array_equal(transformed_before, transformed_after)
    np.testing.assert_array_equal(transformed_before, refitted.transform(query))


def test_selected_features_ignore_a_leak_available_only_outside_training(train_frame) -> None:
    frame, labels = train_frame
    train, holdout = frame.iloc[:45], frame.iloc[45:].copy()
    holdout["noise"] = labels.iloc[45:] * 1000.0

    pipeline = build_pipeline(LogisticRegression(), PreprocessingConfig(k_features=1))
    pipeline.fit(train, labels.iloc[:45])
    selected = pipeline[:-1].get_feature_names_out()

    assert list(selected) == ["signal"]
    assert pipeline.predict_proba(holdout).shape == (15, 2)


def test_constant_columns_are_removed(train_frame) -> None:
    frame, labels = train_frame
    names = _preprocessor(frame, labels, k_features=None).get_feature_names_out()

    assert "constant" not in names


def test_output_contains_no_missing_values(train_frame) -> None:
    frame, labels = train_frame
    output = _preprocessor(frame, labels, k_features=None).transform(frame)

    assert not np.isnan(output).any()


def test_missingness_filter_uses_threshold_from_fit(train_frame) -> None:
    frame, _ = train_frame
    selector = MissingnessFilter(max_missing=0.5).fit(frame)

    assert "mostly_missing" not in selector.get_feature_names_out()
    assert "some_missing" in selector.get_feature_names_out()


def test_correlation_filter_keeps_first_of_a_correlated_pair() -> None:
    values = np.arange(20.0)
    frame = pd.DataFrame(
        {"first": values, "copy": 2 * values + 1, "other": np.tile([1.0, -1.0], 10)}
    )

    selector = CorrelationFilter(threshold=0.95).fit(frame)

    assert list(selector.get_feature_names_out()) == ["first", "other"]


def test_top_k_tolerates_k_above_remaining_features(train_frame) -> None:
    frame, labels = train_frame
    usable = frame[["signal", "noise"]]

    selector = TopKSelector(k=50).fit(usable, labels)

    assert list(selector.get_feature_names_out()) == ["signal", "noise"]


def test_smote_changes_only_the_fit_and_keeps_prediction_size(train_frame) -> None:
    frame, labels = train_frame
    config = PreprocessingConfig(k_features=2, scaler="standard", use_smote=True)
    pipeline = build_pipeline(LogisticRegression(), config).fit(frame, labels)

    assert pipeline.predict_proba(frame).shape == (len(frame), 2)
