import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from secom.config import CostMatrix
from secom.evaluation import (
    bootstrap_interval,
    confusion_counts,
    cost_curve,
    cost_per_unit,
    precision_at_recall,
    recall_at_precision,
    select_cost_threshold,
    threshold_metrics,
)

COSTS = CostMatrix(false_negative=10.0, false_positive=1.0)
LABELS = np.array([0, 0, 0, 0, 1, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


def test_confusion_counts_treat_threshold_as_inclusive() -> None:
    assert confusion_counts(LABELS, SCORES, 0.6) == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}


def test_cost_per_unit_by_hand() -> None:
    # At 0.6: one missed fail costs 10, one false alarm costs 1, over 8 units.
    assert cost_per_unit(LABELS, SCORES, 0.6, COSTS) == pytest.approx(11 / 8)


def test_threshold_selection_minimizes_cost() -> None:
    # Catching the fail at 0.5 costs one extra alarm (1) and saves one miss (10).
    assert select_cost_threshold(LABELS, SCORES, COSTS) == pytest.approx(0.5)


def test_threshold_selection_flags_nothing_when_alarms_are_too_expensive() -> None:
    costs = CostMatrix(false_negative=1.0, false_positive=100.0)
    labels = np.array([1, 0, 0, 0])
    scores = np.array([0.1, 0.4, 0.5, 0.6])

    assert select_cost_threshold(labels, scores, costs) == np.inf


def test_threshold_selection_prefers_fewer_alarms_on_ties() -> None:
    costs = CostMatrix(false_negative=1.0, false_positive=1.0)
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    # Total cost is 2 at 0.1 (two alarms), at 0.3 (one alarm) and when flagging nothing.
    assert select_cost_threshold(labels, scores, costs) == np.inf


def test_cost_curve_includes_flag_nothing_option() -> None:
    thresholds, costs = cost_curve(LABELS, SCORES, COSTS)

    assert thresholds[-1] == np.inf
    assert costs[-1] == pytest.approx(3 * 10 / 8)
    assert len(thresholds) == len(SCORES) + 1


def test_recall_at_fixed_precision_by_hand() -> None:
    # Flagging 0.5 and above gives precision 3/4 with recall 1; precision never falls below 0.5.
    assert recall_at_precision(LABELS, SCORES, 0.75) == pytest.approx(1.0)
    # Precision 1.0 is reached only by flagging the top two scores.
    assert recall_at_precision(LABELS, SCORES, 1.0) == pytest.approx(2 / 3)


def test_recall_at_unreachable_precision_is_zero() -> None:
    labels = np.array([1, 0, 0])
    scores = np.array([0.1, 0.8, 0.9])

    # Only the trivial point on the curve reaches this; there the recall is zero.
    assert recall_at_precision(labels, scores, 0.9) == 0.0


def test_precision_at_fixed_recall_by_hand() -> None:
    assert precision_at_recall(LABELS, SCORES, 2 / 3) == pytest.approx(1.0)
    assert precision_at_recall(LABELS, SCORES, 1.0) == pytest.approx(3 / 4)


def test_balanced_error_rate_complements_balanced_accuracy() -> None:
    metrics = threshold_metrics(LABELS, SCORES, 0.6, COSTS)

    assert metrics["balanced_error_rate"] == pytest.approx(1 - metrics["balanced_accuracy"])
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_bootstrap_interval_brackets_estimate_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    labels = rng.binomial(1, 0.2, 300)
    scores = labels * 0.5 + rng.random(300)

    first = bootstrap_interval(average_precision_score, labels, scores, n_samples=200)
    second = bootstrap_interval(average_precision_score, labels, scores, n_samples=200)

    assert first == second
    assert first.low <= first.estimate <= first.high


def test_bootstrap_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        bootstrap_interval(average_precision_score, np.zeros(5), np.arange(5.0))
