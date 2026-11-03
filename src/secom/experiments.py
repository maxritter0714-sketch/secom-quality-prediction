"""Experiment stages. Stages before the final evaluation receive training-period data only."""

import pandas as pd

from secom.models import ImbalanceStrategy, ModelName, default_preprocessing, make_classifier
from secom.preprocessing import build_pipeline
from secom.validation import cross_validate, repeated_stratified_folds, time_series_folds

Candidate = tuple[ModelName, ImbalanceStrategy]

BASELINE_CANDIDATES: tuple[Candidate, ...] = (
    ("dummy", "none"),
    ("logreg", "none"),
    ("logreg", "weighted"),
)


def compare_candidates(
    features: pd.DataFrame, labels: pd.Series, candidates: tuple[Candidate, ...]
) -> pd.DataFrame:
    """Score each candidate with temporal and with shuffled cross-validation."""
    temporal = time_series_folds(labels)
    shuffled = repeated_stratified_folds(labels)
    rows = []
    for model, imbalance in candidates:
        pipeline = build_pipeline(
            make_classifier(model, imbalance), default_preprocessing(model, imbalance)
        )
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
