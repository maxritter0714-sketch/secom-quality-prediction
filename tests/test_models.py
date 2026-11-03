import numpy as np
import pandas as pd
import pytest

from secom.models import MODEL_NAMES, default_preprocessing, make_classifier
from secom.preprocessing import build_pipeline
from secom.validation import cross_validate, repeated_stratified_folds, time_series_folds


@pytest.fixture
def noise_data() -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels are independent, so any real skill would indicate leakage."""
    rng = np.random.default_rng(1)
    n_rows, n_features = 120, 400
    frame = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    labels = pd.Series(rng.permutation(np.r_[np.ones(30, dtype=int), np.zeros(90, dtype=int)]))
    return frame, labels


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_runs_through_the_pipeline(name, noise_data) -> None:
    frame, labels = noise_data
    pipeline = build_pipeline(make_classifier(name), default_preprocessing(name))

    scores = pipeline.fit(frame, labels).predict_proba(frame)[:, 1]

    assert scores.shape == (len(frame),)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        make_classifier("svm")


def test_smote_strategy_enables_oversampling_and_disables_weights() -> None:
    assert default_preprocessing("lightgbm", "smote").use_smote
    assert make_classifier("logreg", "smote").class_weight is None


def test_feature_selection_inside_cv_shows_no_skill_on_pure_noise(noise_data) -> None:
    """With k-best selection on 400 noise features, selecting before splitting inflates scores."""
    frame, labels = noise_data
    pipeline = build_pipeline(make_classifier("logreg"), default_preprocessing("logreg"))

    result = cross_validate(pipeline, frame, labels, repeated_stratified_folds(labels))

    prevalence = labels.mean()
    assert abs(result.mean_pr_auc - prevalence) < 0.07


def test_time_series_folds_only_validate_on_later_rows() -> None:
    labels = pd.Series(np.tile([0, 0, 1], 40))

    folds = time_series_folds(labels, n_splits=4)

    assert all(train.max() < validation.min() for train, validation in folds)


def test_fold_without_positives_is_rejected() -> None:
    labels = pd.Series(np.r_[np.tile([0, 1], 30), np.zeros(60, dtype=int)])

    with pytest.raises(ValueError, match="only one class"):
        time_series_folds(labels, n_splits=3)


def test_out_of_fold_scores_cover_only_validated_rows(noise_data) -> None:
    frame, labels = noise_data
    pipeline = build_pipeline(make_classifier("dummy"), default_preprocessing("dummy"))
    folds = time_series_folds(labels, n_splits=3)

    result = cross_validate(pipeline, frame, labels, folds)

    assert np.isnan(result.out_of_fold_scores[: folds[0][1].min()]).all()
    assert not np.isnan(result.out_of_fold_scores[folds[0][1].min() :]).any()
    assert len(result.fold_pr_aucs) == 3
