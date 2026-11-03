import numpy as np
import pandas as pd
import pytest

from secom.experiments import build_variants, select_best_variant, tune_all, validation_table
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
