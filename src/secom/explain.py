"""Model explanations. Features are anonymous, so these rank signals without physical meaning."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from secom.config import SEED

PERMUTATION_REPEATS = 10
SAMPLER_STEP = "oversample"


@dataclass(frozen=True)
class ShapExplanation:
    """SHAP values for the fail class on the features the tree model actually received."""

    values: np.ndarray
    base_value: float
    transformed_features: pd.DataFrame


def split_fitted_pipeline(fitted: Pipeline) -> tuple[Pipeline, object]:
    """Separate the fitted transforming steps from the model; samplers only act during fit."""
    transforming_steps = [step for step in fitted.steps[:-1] if step[0] != SAMPLER_STEP]
    return Pipeline(transforming_steps), fitted[-1]


def transformed_frame(preprocessor: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted preprocessing and keep the original feature names."""
    return pd.DataFrame(
        preprocessor.transform(features),
        columns=preprocessor.get_feature_names_out(),
        index=features.index,
    )


def permutation_table(
    fitted: Pipeline, features: pd.DataFrame, labels: pd.Series, n_top: int = 15
) -> pd.DataFrame:
    """Rank the model's input features by the PR-AUC lost when each one is shuffled.

    Features removed by preprocessing have an importance of exactly zero, so only
    the features that reach the model are permuted.
    """
    preprocessor, model = split_fitted_pipeline(fitted)
    frame = transformed_frame(preprocessor, features)
    result = permutation_importance(
        model,
        frame.to_numpy(),
        labels,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
    )
    table = pd.DataFrame(
        {
            "feature": frame.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).head(n_top)


def tree_importance_table(fitted: Pipeline, n_top: int = 15) -> pd.DataFrame:
    """Rank features by the tree model's own impurity or gain importance."""
    preprocessor, model = split_fitted_pipeline(fitted)
    booster = getattr(model, "booster_", None)
    importance = (
        booster.feature_importance(importance_type="gain")
        if booster is not None
        else model.feature_importances_
    )
    table = pd.DataFrame(
        {"feature": preprocessor.get_feature_names_out(), "importance": importance}
    )
    return table.sort_values("importance", ascending=False).head(n_top)


def _fail_class_values(raw: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Normalize the shapes SHAP returns across model types to (rows, features) for class 1."""
    if isinstance(raw, list):
        return raw[1]
    return raw[:, :, 1] if raw.ndim == 3 else raw


def _fail_class_base(expected: float | np.ndarray) -> float:
    values = np.atleast_1d(expected)
    return float(values[-1])


def explain_tree_model(fitted: Pipeline, features: pd.DataFrame) -> ShapExplanation:
    """Compute TreeExplainer SHAP values for the fail class."""
    preprocessor, model = split_fitted_pipeline(fitted)
    frame = transformed_frame(preprocessor, features)
    explainer = shap.TreeExplainer(model)
    return ShapExplanation(
        values=_fail_class_values(explainer.shap_values(frame)),
        base_value=_fail_class_base(explainer.expected_value),
        transformed_features=frame,
    )


def single_case_table(explanation: ShapExplanation, row: int, n_top: int = 8) -> pd.DataFrame:
    """List a unit's strongest SHAP contributions, sorted by absolute size."""
    contribution = pd.Series(
        explanation.values[row], index=explanation.transformed_features.columns
    )
    top = contribution.reindex(contribution.abs().sort_values(ascending=False).index).head(n_top)
    return pd.DataFrame(
        {
            "feature": top.index,
            "value": explanation.transformed_features.iloc[row][top.index].to_numpy(),
            "shap": top.to_numpy(),
        }
    )
