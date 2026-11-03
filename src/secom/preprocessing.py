"""Preprocessing steps that learn only from the data passed to fit.

The pipeline is always fitted inside a training fold, so statistics such as medians,
kept features or selected features never see validation or hold-out rows.
"""

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectorMixin, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from secom.config import SEED

Scaler = Literal["standard", "robust"]


class _NanTolerantSelector(SelectorMixin, BaseEstimator):
    """Base for selectors that run before imputation and must accept missing values."""

    def __sklearn_tags__(self):  # noqa: ANN204 - the Tags type is an sklearn internal
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_


class MissingnessFilter(_NanTolerantSelector):
    """Drop features whose fraction of missing values exceeds max_missing."""

    def __init__(self, max_missing: float = 0.5) -> None:
        self.max_missing = max_missing

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:  # noqa: N803
        values = validate_data(self, X, ensure_all_finite="allow-nan", dtype="float64")
        self.support_ = np.isnan(values).mean(axis=0) <= self.max_missing
        return self


class CorrelationFilter(_NanTolerantSelector):
    """Keep the first feature of every group whose absolute correlation reaches threshold."""

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:  # noqa: N803
        values = validate_data(self, X, ensure_all_finite="allow-nan", dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = np.abs(np.corrcoef(values, rowvar=False))
        correlation = np.atleast_2d(np.nan_to_num(correlation))
        is_redundant = np.triu(correlation >= self.threshold, k=1).any(axis=0)
        self.support_ = ~is_redundant
        return self


class TopKSelector(SelectorMixin, BaseEstimator):
    """Keep the k features with the highest ANOVA F-score.

    Unlike SelectKBest, k may exceed the number of remaining features, which can vary
    between folds and trials because earlier filter steps depend on the data.
    """

    def __init__(self, k: int = 100) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:  # noqa: N803
        values = validate_data(self, X, dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.nan_to_num(f_classif(values, y)[0])
        top_indices = np.argsort(scores, kind="stable")[::-1][: self.k]
        self.support_ = np.zeros(values.shape[1], dtype=bool)
        self.support_[top_indices] = True
        return self

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_


@dataclass(frozen=True)
class PreprocessingConfig:
    """All tunable preprocessing choices; None disables an optional step."""

    max_missing: float = 0.5
    add_missing_indicator: bool = False
    correlation_threshold: float | None = None
    scaler: Scaler | None = None
    k_features: int | None = 100
    use_smote: bool = False


def _make_scaler(kind: Scaler) -> StandardScaler | RobustScaler:
    return {"standard": StandardScaler, "robust": RobustScaler}[kind]()


def build_pipeline(classifier: ClassifierMixin, config: PreprocessingConfig) -> Pipeline:
    """Assemble preprocessing and classifier so everything is fitted as one unit.

    The imblearn Pipeline applies SMOTE only while fitting, never while predicting.
    """
    steps: list[tuple[str, object]] = [
        ("drop_missing", MissingnessFilter(config.max_missing)),
        (
            "impute",
            SimpleImputer(strategy="median", add_indicator=config.add_missing_indicator),
        ),
        ("drop_constant", VarianceThreshold()),
    ]
    if config.correlation_threshold is not None:
        steps.append(("drop_correlated", CorrelationFilter(config.correlation_threshold)))
    if config.scaler is not None:
        steps.append(("scale", _make_scaler(config.scaler)))
    if config.k_features is not None:
        steps.append(("select", TopKSelector(config.k_features)))
    if config.use_smote:
        steps.append(("oversample", SMOTE(random_state=SEED)))
    steps.append(("model", classifier))
    return Pipeline(steps)
