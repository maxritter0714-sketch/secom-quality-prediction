"""Single entry point that regenerates every table and figure in reports/.

Only run_holdout_stage receives hold-out rows, and it runs exactly once, after every
model, parameter and threshold has been fixed on the training period.
"""

import argparse
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from secom.config import ARTIFACTS_DIR, COSTS, REPORTS_DIR
from secom.data import load_secom
from secom.evaluation import cost_curve
from secom.experiments import (
    CANDIDATES,
    DetectorVariant,
    Variant,
    build_variants,
    compare_candidates,
    evaluate_holdout,
    reference_costs,
    run_detectors,
    select_best_tree_variant,
    select_best_variant,
    tune_all,
    validation_table,
)
from secom.explain import (
    explain_tree_model,
    permutation_table,
    single_case_table,
    tree_importance_table,
)
from secom.plots import (
    apply_style,
    plot_candidate_comparison,
    plot_cost_curve,
    plot_optuna_history,
    plot_optuna_importance,
    plot_pr_curves,
    plot_shap_cases,
    plot_shap_summary,
    save_figure,
)
from secom.reporting import markdown_table, write_results
from secom.tuning import N_TRIALS
from secom.validation import temporal_split, time_series_folds

FAST_TRIALS = 3


@dataclass
class Report:
    """Collects Markdown sections, machine-readable data and the figure location."""

    figures_dir: Path
    sections: dict[str, str]
    data: dict[str, Any]

    def add_table(self, title: str, key: str, table: pd.DataFrame) -> None:
        self.sections[title] = markdown_table(table)
        self.data[key] = table.to_dict(orient="records")


@dataclass(frozen=True)
class TrainingOutcome:
    """Everything fixed on the training period before the hold-out is touched."""

    variants: list[Variant]
    detectors: list[DetectorVariant]
    best: Variant
    best_tree: Variant


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=f"smoke test with {FAST_TRIALS} trials, writing to artifacts/fast instead of reports/",
    )
    return parser.parse_args()


def _validated(labels: pd.Series, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~np.isnan(scores)
    return labels.to_numpy()[mask], scores[mask]


def run_training_stage(
    features: pd.DataFrame, labels: pd.Series, n_trials: int, report: Report
) -> TrainingOutcome:
    """Compare, tune and choose thresholds using the training period only."""
    folds = time_series_folds(labels)

    comparison = compare_candidates(features, labels, CANDIDATES)
    report.add_table("Candidate comparison", "candidate_comparison", comparison)
    save_figure(
        plot_candidate_comparison(comparison), "cv_temporal_vs_shuffled", report.figures_dir
    )

    tuned = tune_all(features, labels, folds, n_trials)
    studies = {result.model: result.study for result in tuned}
    save_figure(plot_optuna_history(studies), "optuna_history", report.figures_dir)
    save_figure(plot_optuna_importance(studies), "optuna_importance", report.figures_dir)
    report.data["best_params"] = {result.model: result.study.best_params for result in tuned}

    variants = build_variants(features, labels, folds, tuned)
    report.add_table("Default vs tuned", "validation", validation_table(variants, labels))
    best = select_best_variant(variants)
    report.sections["Selected model"] = f"`{best.label}`"
    report.data["selected_model"] = best.label

    best_labels, best_scores = _validated(labels, best.cv.out_of_fold_scores)
    thresholds, costs = cost_curve(best_labels, best_scores)
    save_figure(
        plot_cost_curve(thresholds, costs, best.threshold, best.label),
        "cost_vs_threshold",
        report.figures_dir,
    )
    return TrainingOutcome(
        variants, run_detectors(features, labels, folds), best, select_best_tree_variant(variants)
    )


def _add_explanations(
    fitted, label: str, features: pd.DataFrame, labels: pd.Series, report: Report
) -> None:
    report.add_table(
        "Permutation importance", "permutation", permutation_table(fitted, features, labels)
    )
    report.add_table("Tree importance", "tree_importance", tree_importance_table(fitted))

    explanation = explain_tree_model(fitted, features)
    save_figure(
        plot_shap_summary(explanation.values, explanation.transformed_features, label),
        "shap_summary",
        report.figures_dir,
    )
    scores = fitted.predict_proba(features)[:, 1]
    fail_rows = np.flatnonzero(labels.to_numpy() == 1)
    case_rows = {
        "Fail with the highest score": fail_rows[scores[fail_rows].argmax()],
        "Fail with the lowest score": fail_rows[scores[fail_rows].argmin()],
    }
    cases = {title: single_case_table(explanation, row) for title, row in case_rows.items()}
    save_figure(plot_shap_cases(cases), "shap_cases", report.figures_dir)


def run_holdout_stage(
    outcome: TrainingOutcome,
    train: tuple[pd.DataFrame, pd.Series],
    test: tuple[pd.DataFrame, pd.Series],
    report: Report,
) -> None:
    """Evaluate everything fixed so far on the hold-out, once."""
    test_features, test_labels = test
    result = evaluate_holdout(outcome.variants, outcome.detectors, train, test)
    report.add_table("Hold-out results", "holdout", result.table)
    report.add_table("Reference costs", "reference_costs", reference_costs(test_labels))

    labels = test_labels.to_numpy()
    curve_names = [outcome.best.label, *[detector.name for detector in outcome.detectors]]
    save_figure(
        plot_pr_curves(
            {name: (labels, result.scores[name]) for name in curve_names},
            "PR curves on the hold-out: supervised vs anomaly detection",
        ),
        "pr_curves_holdout",
        report.figures_dir,
    )
    label = outcome.best_tree.label
    _add_explanations(result.fitted[label], label, test_features, test_labels, report)


def _run_information(
    features: pd.DataFrame, train_rows: int, test_labels: pd.Series, n_trials: int, seconds: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item": [
                "Python",
                "Units / features",
                "Training units",
                "Hold-out units / fails",
                "Optuna trials per model",
                "Assumed cost (missed fail / false alarm)",
                "Runtime (minutes)",
            ],
            "value": [
                platform.python_version(),
                f"{features.shape[0]} / {features.shape[1]}",
                train_rows,
                f"{len(test_labels)} / {int(test_labels.sum())}",
                n_trials,
                f"{COSTS.false_negative:g} / {COSTS.false_positive:g}",
                f"{seconds / 60:.1f}",
            ],
        }
    )


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    apply_style()
    n_trials = FAST_TRIALS if args.fast else N_TRIALS
    output_dir = ARTIFACTS_DIR / "fast" if args.fast else REPORTS_DIR
    report = Report(output_dir / "figures", {}, {})

    features, labels, timestamps = load_secom()
    split = temporal_split(timestamps)
    train = features.iloc[split.train], labels.iloc[split.train]
    test = features.iloc[split.test], labels.iloc[split.test]

    outcome = run_training_stage(*train, n_trials, report)
    run_holdout_stage(outcome, train, test, report)

    seconds = time.perf_counter() - started
    information = _run_information(features, len(split.train), test[1], n_trials, seconds)
    report.add_table("Run information", "run_information", information)
    write_results(report.sections, report.data, directory=output_dir)


if __name__ == "__main__":
    main()
