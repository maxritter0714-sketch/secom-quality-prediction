"""Metrics, cost-based threshold selection and bootstrap intervals.

Convention: a unit is flagged as a fail when its score is at or above the threshold.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from secom.config import COSTS, SEED, CostMatrix

MIN_PRECISION = 0.5
MIN_RECALL = 0.5
BOOTSTRAP_SAMPLES = 2000
CONFIDENCE = 0.95

Metric = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    """Point estimate on the original sample with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float


def recall_at_precision(
    labels: np.ndarray, scores: np.ndarray, min_precision: float = MIN_PRECISION
) -> float:
    """Return the best recall reachable while precision stays at least min_precision."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    reachable = recall[precision >= min_precision]
    return float(reachable.max()) if reachable.size else 0.0


def precision_at_recall(
    labels: np.ndarray, scores: np.ndarray, min_recall: float = MIN_RECALL
) -> float:
    """Return the best precision reachable while recall stays at least min_recall."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(precision[recall >= min_recall].max())


def confusion_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """Return confusion counts with fail as the positive class."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def cost_per_unit(
    labels: np.ndarray, scores: np.ndarray, threshold: float, costs: CostMatrix = COSTS
) -> float:
    """Return the mean misclassification cost per unit at a threshold."""
    counts = confusion_counts(labels, scores, threshold)
    total = counts["fn"] * costs.false_negative + counts["fp"] * costs.false_positive
    return total / len(labels)


def cost_curve(
    labels: np.ndarray, scores: np.ndarray, costs: CostMatrix = COSTS
) -> tuple[np.ndarray, np.ndarray]:
    """Return every distinct threshold and its cost per unit.

    The extra threshold above the maximum score represents flagging no unit at all.
    """
    thresholds = np.append(np.unique(scores), np.inf)
    per_unit = np.array([cost_per_unit(labels, scores, t, costs) for t in thresholds])
    return thresholds, per_unit


def select_cost_threshold(
    labels: np.ndarray, scores: np.ndarray, costs: CostMatrix = COSTS
) -> float:
    """Return the threshold with minimal cost; ties go to the one raising the fewest alarms."""
    thresholds, per_unit = cost_curve(labels, scores, costs)
    best_cost = per_unit.min()
    return float(thresholds[np.flatnonzero(per_unit == best_cost)[-1]])


def threshold_free_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return the ranking metrics, which do not depend on any threshold."""
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "recall_at_precision": recall_at_precision(labels, scores),
        "precision_at_recall": precision_at_recall(labels, scores),
    }


def threshold_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float, costs: CostMatrix = COSTS
) -> dict[str, float]:
    """Return the metrics at a threshold. BER as used by UCI equals 1 - balanced accuracy."""
    predictions = (scores >= threshold).astype(int)
    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    return {
        "balanced_accuracy": balanced_accuracy,
        "balanced_error_rate": 1 - balanced_accuracy,
        "cost_per_unit": cost_per_unit(labels, scores, threshold, costs),
    }


def bootstrap_interval(
    metric: Metric,
    labels: np.ndarray,
    scores: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> Interval:
    """Estimate a percentile interval, skipping resamples that lack one of the classes."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        raise ValueError("Bootstrap requires both classes in the labels")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        rows = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[rows])) == 2:
            values.append(metric(labels[rows], scores[rows]))
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return Interval(metric(labels, scores), float(low), float(high))
