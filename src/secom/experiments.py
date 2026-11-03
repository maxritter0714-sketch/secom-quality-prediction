"""Experiment stages. Stages before the final evaluation receive training-period data only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from secom.anomaly import DETECTOR_NAMES, GoodOnlyDetector, cross_validate_detector, make_detector
from secom.evaluation import (
    bootstrap_interval,
    confusion_counts,
    cost_per_unit,
    select_cost_threshold,
    threshold_free_metrics,
    threshold_metrics,
)
from secom.models import ImbalanceStrategy, ModelName, default_preprocessing, make_classifier
from secom.preprocessing import build_pipeline
from secom.tuning import TUNED_MODELS, TuningResult, build_tuned_pipeline, tune_model
from secom.validation import (
    CrossValidationResult,
    Folds,
    cross_validate,
    repeated_stratified_folds,
    time_series_folds,
)

Candidate = tuple[ModelName, ImbalanceStrategy]

IMBALANCE_STRATEGIES: tuple[ImbalanceStrategy, ...] = ("none", "weighted", "smote")
CANDIDATES: tuple[Candidate, ...] = (
    ("dummy", "none"),
    *(
        (model, strategy)
        for model in ("logreg", "random_forest", "lightgbm")
        for strategy in IMBALANCE_STRATEGIES
    ),
)


@dataclass(frozen=True)
class Variant:
    """A pipeline with its cross-validation result and the threshold chosen on it."""

    label: str
    pipeline: Pipeline
    cv: CrossValidationResult
    threshold: float


@dataclass(frozen=True)
class DetectorVariant:
    """An anomaly detector with its cross-validation result and threshold."""

    name: str
    mean_pr_auc: float
    std_pr_auc: float
    threshold: float
    out_of_fold_scores: np.ndarray


@dataclass(frozen=True)
class HoldoutResult:
    """Metrics, scores and fitted pipelines from the single hold-out evaluation."""

    table: pd.DataFrame
    scores: dict[str, np.ndarray]
    fitted: dict[str, Pipeline]


def _default_pipeline(model: ModelName, imbalance: ImbalanceStrategy) -> Pipeline:
    return build_pipeline(
        make_classifier(model, imbalance), default_preprocessing(model, imbalance)
    )


def compare_candidates(
    features: pd.DataFrame, labels: pd.Series, candidates: tuple[Candidate, ...]
) -> pd.DataFrame:
    """Score each candidate with temporal and with shuffled cross-validation."""
    temporal = time_series_folds(labels)
    shuffled = repeated_stratified_folds(labels)
    rows = []
    for model, imbalance in candidates:
        pipeline = _default_pipeline(model, imbalance)
        temporal_result = cross_validate(pipeline, features, labels, temporal)
        shuffled_result = cross_validate(pipeline, features, labels, shuffled)
        rows.append(
            {
                "model": model,
                "imbalance": imbalance,
                "pr_auc_temporal_mean": temporal_result.mean_pr_auc,
                "pr_auc_temporal_std": temporal_result.std_pr_auc,
                "pr_auc_shuffled_mean": shuffled_result.mean_pr_auc,
                "pr_auc_shuffled_std": shuffled_result.std_pr_auc,
            }
        )
    return pd.DataFrame(rows)


def tune_all(
    features: pd.DataFrame, labels: pd.Series, folds: Folds, n_trials: int
) -> list[TuningResult]:
    """Run one Optuna study per tunable model."""
    return [tune_model(model, features, labels, folds, n_trials) for model in TUNED_MODELS]


def _make_variant(
    label: str, pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series, folds: Folds
) -> Variant:
    """Cross-validate a pipeline and pick its threshold from out-of-fold scores only."""
    cv = cross_validate(pipeline, features, labels, folds)
    validated = ~np.isnan(cv.out_of_fold_scores)
    threshold = select_cost_threshold(
        labels.to_numpy()[validated], cv.out_of_fold_scores[validated]
    )
    return Variant(label, pipeline, cv, threshold)


def build_variants(
    features: pd.DataFrame, labels: pd.Series, folds: Folds, tuned: list[TuningResult]
) -> list[Variant]:
    """Create the dummy baseline plus default and tuned versions of every model."""
    variants = [_make_variant("dummy", _default_pipeline("dummy", "none"), features, labels, folds)]
    for result in tuned:
        pipelines = {
            f"{result.model} (default)": _default_pipeline(result.model, "weighted"),
            f"{result.model} (tuned)": build_tuned_pipeline(result.model, result.design),
        }
        variants += [
            _make_variant(label, pipeline, features, labels, folds)
            for label, pipeline in pipelines.items()
        ]
    return variants


def select_best_variant(variants: list[Variant]) -> Variant:
    """Choose the model for detailed reporting by its cross-validation score alone."""
    real_models = [variant for variant in variants if variant.label != "dummy"]
    return max(real_models, key=lambda variant: variant.cv.mean_pr_auc)


def validation_table(variants: list[Variant], labels: pd.Series) -> pd.DataFrame:
    """Summarize each variant on its out-of-fold scores from the training period."""
    rows = []
    for variant in variants:
        scores = variant.cv.out_of_fold_scores
        validated = ~np.isnan(scores)
        fold_labels, fold_scores = labels.to_numpy()[validated], scores[validated]
        rows.append(
            {
                "model": variant.label,
                "cv_pr_auc_mean": variant.cv.mean_pr_auc,
                "cv_pr_auc_std": variant.cv.std_pr_auc,
                "threshold": variant.threshold,
                **threshold_metrics(fold_labels, fold_scores, variant.threshold),
                **confusion_counts(fold_labels, fold_scores, variant.threshold),
            }
        )
    return pd.DataFrame(rows)


def run_detectors(features: pd.DataFrame, labels: pd.Series, folds: Folds) -> list[DetectorVariant]:
    """Cross-validate detectors and pick their thresholds from out-of-fold scores only."""
    variants = []
    for name in DETECTOR_NAMES:
        fold_pr_aucs, scores = cross_validate_detector(name, features, labels, folds)
        validated = ~np.isnan(scores)
        threshold = select_cost_threshold(labels.to_numpy()[validated], scores[validated])
        variants.append(
            DetectorVariant(name, fold_pr_aucs.mean(), fold_pr_aucs.std(), threshold, scores)
        )
    return variants


def _holdout_row(
    label: str, cv_mean: float, threshold: float, labels: np.ndarray, scores: np.ndarray
) -> dict[str, object]:
    pr_auc = bootstrap_interval(average_precision_score, labels, scores)
    cost = bootstrap_interval(lambda y, s: cost_per_unit(y, s, threshold), labels, scores)
    metrics = threshold_free_metrics(labels, scores)
    return {
        "model": label,
        "cv_pr_auc": cv_mean,
        "test_pr_auc": pr_auc.estimate,
        "test_pr_auc_ci_low": pr_auc.low,
        "test_pr_auc_ci_high": pr_auc.high,
        "roc_auc": metrics["roc_auc"],
        "recall_at_precision_0.5": metrics["recall_at_precision"],
        "precision_at_recall_0.5": metrics["precision_at_recall"],
        **threshold_metrics(labels, scores, threshold),
        "cost_ci_low": cost.low,
        "cost_ci_high": cost.high,
        **confusion_counts(labels, scores, threshold),
    }


def evaluate_holdout(
    variants: list[Variant],
    detectors: list[DetectorVariant],
    train: tuple[pd.DataFrame, pd.Series],
    test: tuple[pd.DataFrame, pd.Series],
) -> HoldoutResult:
    """Refit on the full training period and score the hold-out once, thresholds unchanged."""
    train_features, train_labels = train
    test_features, test_labels = test
    labels = test_labels.to_numpy()
    rows, scores, fitted = [], {}, {}
    for variant in variants:
        model = clone(variant.pipeline).fit(train_features, train_labels)
        fitted[variant.label] = model
        scores[variant.label] = model.predict_proba(test_features)[:, 1]
        rows.append(
            _holdout_row(
                variant.label,
                variant.cv.mean_pr_auc,
                variant.threshold,
                labels,
                scores[variant.label],
            )
        )
    for detector in detectors:
        fitted_detector = GoodOnlyDetector(make_detector(detector.name))
        fitted_detector.fit(train_features, train_labels)
        scores[detector.name] = fitted_detector.score(test_features)
        rows.append(
            _holdout_row(
                detector.name,
                detector.mean_pr_auc,
                detector.threshold,
                labels,
                scores[detector.name],
            )
        )
    return HoldoutResult(pd.DataFrame(rows), scores, fitted)


def reference_costs(labels: pd.Series) -> pd.DataFrame:
    """Return the cost of the two trivial policies, as a yardstick for the cost column."""
    values = labels.to_numpy()
    placeholder_scores = np.zeros(len(values))
    return pd.DataFrame(
        {
            "policy": ["flag no unit", "flag every unit"],
            "cost_per_unit": [
                cost_per_unit(values, placeholder_scores, threshold=np.inf),
                cost_per_unit(values, placeholder_scores, threshold=0.0),
            ],
        }
    )


def select_best_tree_variant(variants: list[Variant]) -> Variant:
    """Choose the tree model to explain by cross-validation score, not by hold-out score."""
    trees = [
        variant for variant in variants if variant.label.startswith(("random_forest", "lightgbm"))
    ]
    return max(trees, key=lambda variant: variant.cv.mean_pr_auc)
