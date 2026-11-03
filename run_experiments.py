"""Single entry point that regenerates every table and figure in reports/."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from secom.config import ARTIFACTS_DIR, REPORTS_DIR
from secom.data import load_secom
from secom.evaluation import cost_curve
from secom.experiments import (
    CANDIDATES,
    build_variants,
    compare_candidates,
    select_best_variant,
    tune_all,
    validation_table,
)
from secom.plots import (
    apply_style,
    plot_candidate_comparison,
    plot_cost_curve,
    plot_optuna_history,
    plot_optuna_importance,
    plot_pr_curves,
    save_figure,
)
from secom.reporting import markdown_table, write_results
from secom.tuning import N_TRIALS
from secom.validation import temporal_split, time_series_folds

FAST_TRIALS = 3


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


def main() -> None:
    args = _parse_args()
    apply_style()
    n_trials = FAST_TRIALS if args.fast else N_TRIALS
    output_dir: Path = ARTIFACTS_DIR / "fast" if args.fast else REPORTS_DIR
    figures_dir = output_dir / "figures"

    features, labels, timestamps = load_secom()
    split = temporal_split(timestamps)
    train_features = features.iloc[split.train]
    train_labels = labels.iloc[split.train]
    folds = time_series_folds(train_labels)

    comparison = compare_candidates(train_features, train_labels, CANDIDATES)
    save_figure(plot_candidate_comparison(comparison), "cv_temporal_vs_shuffled", figures_dir)

    tuned = tune_all(train_features, train_labels, folds, n_trials)
    studies = {result.model: result.study for result in tuned}
    save_figure(plot_optuna_history(studies), "optuna_history", figures_dir)
    save_figure(plot_optuna_importance(studies), "optuna_importance", figures_dir)

    variants = build_variants(train_features, train_labels, folds, tuned)
    best = select_best_variant(variants)
    validation = validation_table(variants, train_labels)

    oof_scores = {
        variant.label: _validated(train_labels, variant.cv.out_of_fold_scores)
        for variant in variants
        if "(tuned)" in variant.label
    }
    save_figure(
        plot_pr_curves(oof_scores, "Out-of-fold PR curves (training period)"),
        "pr_curves_cv",
        figures_dir,
    )
    best_labels, best_scores = _validated(train_labels, best.cv.out_of_fold_scores)
    thresholds, costs = cost_curve(best_labels, best_scores)
    save_figure(
        plot_cost_curve(thresholds, costs, best.threshold, best.label),
        "cost_vs_threshold",
        figures_dir,
    )

    write_results(
        sections={
            "Candidate comparison (training period)": markdown_table(comparison),
            "Default vs tuned (out-of-fold, training period)": markdown_table(validation),
            "Selected by cross-validation": f"`{best.label}`",
        },
        data={
            "candidate_comparison": comparison.to_dict(orient="records"),
            "validation": validation.to_dict(orient="records"),
            "best_params": {result.model: result.study.best_params for result in tuned},
            "selected_variant": best.label,
            "n_trials": n_trials,
        },
        directory=output_dir,
    )


if __name__ == "__main__":
    main()
