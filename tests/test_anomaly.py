import numpy as np
import pandas as pd
import pytest

from secom.anomaly import (
    DETECTOR_NAMES,
    GoodOnlyDetector,
    cross_validate_detector,
    make_detector,
)
from secom.validation import time_series_folds


@pytest.fixture
def outlier_data() -> tuple[pd.DataFrame, pd.Series]:
    """Fails lie far outside the distribution of good rows."""
    rng = np.random.default_rng(7)
    labels = pd.Series(np.tile([0, 0, 0, 0, 1], 40))
    frame = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"f{i}" for i in range(10)])
    frame.loc[labels == 1, :] += 8
    frame.loc[::13, "f2"] = np.nan
    return frame, labels


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_higher_scores_mean_more_anomalous(name, outlier_data) -> None:
    frame, labels = outlier_data

    scores = GoodOnlyDetector(make_detector(name)).fit(frame, labels).score(frame)

    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_fit_uses_only_good_rows(outlier_data) -> None:
    frame, labels = outlier_data
    good_rows = int((labels == 0).sum())

    detector = GoodOnlyDetector(make_detector("isolation_forest")).fit(frame, labels)

    assert detector.fitted_.named_steps["impute"].n_features_in_ == frame.shape[1]
    assert detector.fitted_.named_steps["detector"].max_samples_ == min(256, good_rows)


def test_fail_rows_have_no_influence_on_the_fit(outlier_data) -> None:
    frame, labels = outlier_data
    altered = frame.copy()
    altered.loc[labels == 1, :] = 1e6

    original = GoodOnlyDetector(make_detector("one_class_svm")).fit(frame, labels).score(frame)
    refit = GoodOnlyDetector(make_detector("one_class_svm")).fit(altered, labels).score(frame)

    np.testing.assert_allclose(original, refit)


def test_cross_validation_scores_only_validated_rows(outlier_data) -> None:
    frame, labels = outlier_data
    folds = time_series_folds(labels, 3)

    fold_scores, out_of_fold = cross_validate_detector("isolation_forest", frame, labels, folds)

    assert fold_scores.shape == (3,)
    assert np.isnan(out_of_fold[: folds[0][1].min()]).all()
    assert fold_scores.mean() > 0.9


def test_unknown_detector_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        make_detector("lof")
