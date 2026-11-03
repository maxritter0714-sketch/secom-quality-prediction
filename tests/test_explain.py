import numpy as np
import pandas as pd
import pytest

from secom.explain import (
    explain_tree_model,
    permutation_table,
    single_case_table,
    tree_importance_table,
)
from secom.models import default_preprocessing, make_classifier
from secom.preprocessing import build_pipeline


@pytest.fixture
def signal_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(11)
    labels = pd.Series(np.tile([0, 0, 0, 1], 50))
    frame = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    frame["f3"] = labels * 4 + rng.normal(0, 0.1, 200)
    return frame, labels


@pytest.mark.parametrize("model", ["random_forest", "lightgbm"])
def test_informative_feature_ranks_first_in_all_tree_explanations(model, signal_data) -> None:
    frame, labels = signal_data
    fitted = build_pipeline(make_classifier(model), default_preprocessing(model)).fit(frame, labels)

    explanation = explain_tree_model(fitted, frame)
    names = explanation.transformed_features.columns
    assert explanation.values.shape == (len(frame), len(names))
    assert names[np.abs(explanation.values).mean(axis=0).argmax()] == "f3"
    assert tree_importance_table(fitted)["feature"].iloc[0] == "f3"
    assert permutation_table(fitted, frame, labels)["feature"].iloc[0] == "f3"


def test_single_case_table_sorts_by_absolute_contribution(signal_data) -> None:
    frame, labels = signal_data
    model = "lightgbm"
    fitted = build_pipeline(make_classifier(model), default_preprocessing(model)).fit(frame, labels)

    table = single_case_table(explain_tree_model(fitted, frame), row=3, n_top=4)

    assert len(table) == 4
    assert table["shap"].abs().is_monotonic_decreasing
    assert table["feature"].iloc[0] == "f3"


def test_explanations_work_when_the_pipeline_contains_smote(signal_data) -> None:
    frame, labels = signal_data
    config = default_preprocessing("lightgbm", "smote")
    fitted = build_pipeline(make_classifier("lightgbm", "smote"), config).fit(frame, labels)

    assert permutation_table(fitted, frame, labels)["feature"].iloc[0] == "f3"
    assert explain_tree_model(fitted, frame).values.shape[0] == len(frame)
