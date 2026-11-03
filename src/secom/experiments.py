"""Experiment stages. Stages before the final evaluation receive training-period data only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline

from secom.evaluation import confusion_counts, select_cost_threshold, threshold_metrics
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
