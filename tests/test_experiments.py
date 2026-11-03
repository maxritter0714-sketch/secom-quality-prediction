import numpy as np
import pandas as pd
import pytest

from secom.experiments import (
    build_variants,
    evaluate_holdout,
    reference_costs,
    run_detectors,
    select_best_tree_variant,
    select_best_variant,
    tune_all,
    validation_table,
)
from secom.validation import time_series_folds


@pytest.fixture
def small_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(5)
    labels = pd.Series(np.tile([0, 0, 0, 1], 40))
    frame = pd.DataFrame(rng.normal(size=(160, 12)), columns=[f"f{i}" for i in range(12)])
    frame["f0"] += labels * 3
    return frame, labels


def test_best_variant_excludes_dummy_and_threshold_uses_validated_rows(small_data) -> None:
    frame, labels = small_data
    folds = time_series_folds(labels, 3)

    variants = build_variants(frame, labels, folds, tune_all(frame, labels, folds, n_trials=1))
    table = validation_table(variants, labels)

    assert select_best_variant(variants).label != "dummy"
    assert {"dummy", "logreg (default)", "lightgbm (tuned)"} <= set(table["model"])
    validated_rows = int((~np.isnan(variants[0].cv.out_of_fold_scores)).sum())
    assert (table[["tn", "fp", "fn", "tp"]].sum(axis=1) == validated_rows).all()


def test_holdout_evaluation_covers_models_and_detectors_once(small_data) -> None:
    frame, labels = small_data
    train, test = (frame.iloc[:120], labels.iloc[:120]), (frame.iloc[120:], labels.iloc[120:])
    folds = time_series_folds(train[1], 3)
    variants = build_variants(*train, folds, tune_all(*train, folds, n_trials=1))
    detectors = run_detectors(*train, folds)

    result = evaluate_holdout(variants, detectors, train, test)

    expected = {variant.label for variant in variants} | {detector.name for detector in detectors}
    assert set(result.table["model"]) == expected
    assert (result.table[["tn", "fp", "fn", "tp"]].sum(axis=1) == len(test[1])).all()
    low, high = result.table["test_pr_auc_ci_low"], result.table["test_pr_auc_ci_high"]
    assert (low <= high).all()


def test_tree_selection_only_considers_tree_models(small_data) -> None:
    frame, labels = small_data
    folds = time_series_folds(labels, 3)
    variants = build_variants(frame, labels, folds, tune_all(frame, labels, folds, n_trials=1))

    assert select_best_tree_variant(variants).label.startswith(("random_forest", "lightgbm"))


def test_reference_costs_by_hand() -> None:
    table = reference_costs(pd.Series([0, 0, 0, 1]))

    assert table["cost_per_unit"].tolist() == pytest.approx([10 / 4, 3 / 4])
