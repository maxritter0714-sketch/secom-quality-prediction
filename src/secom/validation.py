"""Time-aware splitting of the dataset and cross-validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, TimeSeriesSplit

from secom.config import SEED, TEST_FRACTION

Folds = list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class TemporalSplit:
    """Row positions of the earlier training period and the later hold-out period."""

    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class CrossValidationResult:
    """Scores per fold and out-of-fold scores, which are NaN for rows never validated."""

    fold_pr_aucs: np.ndarray
    out_of_fold_scores: np.ndarray

    @property
    def mean_pr_auc(self) -> float:
        return float(self.fold_pr_aucs.mean())

    @property
    def std_pr_auc(self) -> float:
        return float(self.fold_pr_aucs.std())


def temporal_split(timestamps: pd.Series, test_fraction: float = TEST_FRACTION) -> TemporalSplit:
    """Reserve the most recent rows as hold-out without splitting a shared timestamp.

    Units with an identical timestamp all stay in the training period, so nothing
    from the hold-out period can be concurrent with training data.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in increasing order")

    n_rows = len(timestamps)
    cut = int(round(n_rows * (1 - test_fraction)))
    values = timestamps.to_numpy()
    while 0 < cut < n_rows and values[cut] == values[cut - 1]:
        cut += 1
    if cut <= 0 or cut >= n_rows:
        raise ValueError(f"Split leaves an empty period ({n_rows=}, {test_fraction=})")
    return TemporalSplit(train=np.arange(cut), test=np.arange(cut, n_rows))


def _require_both_classes(labels: np.ndarray, folds: Folds) -> None:
    """Fail loudly instead of silently averaging over folds where a metric is undefined."""
    for index, (train, validation) in enumerate(folds):
        for part, rows in (("training", train), ("validation", validation)):
            if len(np.unique(labels[rows])) < 2:
                raise ValueError(f"Fold {index}: {part} rows contain only one class")


def time_series_folds(labels: pd.Series, n_splits: int = 5) -> Folds:
    """Create expanding-window folds; rows must already be sorted by time."""
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(labels))
    _require_both_classes(labels.to_numpy(), folds)
    return folds


def repeated_stratified_folds(labels: pd.Series, n_splits: int = 5, n_repeats: int = 5) -> Folds:
    """Create shuffled folds that ignore time; used only to compare against temporal folds."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    _require_both_classes(labels.to_numpy(), folds)
    return folds


def cross_validate(
    pipeline: Pipeline, features: pd.DataFrame, labels: pd.Series, folds: Folds
) -> CrossValidationResult:
    """Fit a fresh pipeline clone per fold; a row validated repeatedly gets its mean score."""
    score_sum = np.zeros(len(labels))
    score_count = np.zeros(len(labels))
    fold_pr_aucs = []
    for train, validation in folds:
        fitted = clone(pipeline).fit(features.iloc[train], labels.iloc[train])
        scores = fitted.predict_proba(features.iloc[validation])[:, 1]
        fold_pr_aucs.append(average_precision_score(labels.iloc[validation], scores))
        score_sum[validation] += scores
        score_count[validation] += 1
    out_of_fold = np.divide(
        score_sum,
        score_count,
        out=np.full(len(labels), np.nan),
        where=score_count > 0,
    )
    return CrossValidationResult(np.array(fold_pr_aucs), out_of_fold)
