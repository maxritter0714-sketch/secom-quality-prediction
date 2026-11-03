import numpy as np
import pandas as pd
import pytest

from secom.tuning import TUNED_MODELS, build_tuned_pipeline, tune_model
from secom.validation import time_series_folds


@pytest.fixture
def small_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(3)
    labels = pd.Series(np.tile([0, 0, 0, 1], 40))
    frame = pd.DataFrame(rng.normal(size=(160, 15)), columns=[f"f{i}" for i in range(15)])
    frame["f0"] += labels * 2
    frame.loc[::9, "f1"] = np.nan
    return frame, labels


@pytest.mark.parametrize("model", TUNED_MODELS)
def test_short_study_returns_a_design_that_can_be_refitted(model, small_data) -> None:
    frame, labels = small_data

    result = tune_model(model, frame, labels, time_series_folds(labels, 3), n_trials=3)

    assert 0 <= result.best_pr_auc <= 1
    assert len(result.study.trials) == 3
    fitted = build_tuned_pipeline(model, result.design).fit(frame, labels)
    assert fitted.predict_proba(frame).shape == (len(frame), 2)


def test_studies_are_reproducible_with_the_fixed_seed(small_data) -> None:
    frame, labels = small_data
    folds = time_series_folds(labels, 3)

    first = tune_model("logreg", frame, labels, folds, n_trials=4)
    second = tune_model("logreg", frame, labels, folds, n_trials=4)

    assert first.study.best_params == second.study.best_params
    assert first.best_pr_auc == pytest.approx(second.best_pr_auc)


def test_unknown_model_cannot_be_tuned(small_data) -> None:
    frame, labels = small_data

    with pytest.raises(ValueError, match="cannot be tuned"):
        tune_model("dummy", frame, labels, time_series_folds(labels, 3), n_trials=1)
