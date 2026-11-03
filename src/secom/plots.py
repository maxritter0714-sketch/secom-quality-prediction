"""Shared chart style and figure helpers. Every chart is saved under reports/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from secom.config import FIGURES_DIR, SEED  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GOOD_COLOR = SERIES[0]
FAIL_COLOR = SERIES[1]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e4e3df"
FIGURE_DPI = 130


def apply_style() -> None:
    """Set a recessive grid, thin lines and neutral ink for all figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED_INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "xtick.color": MUTED_INK,
            "ytick.color": MUTED_INK,
            "font.size": 10,
        }
    )


def save_figure(figure: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save a figure as PNG, close it and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def _mark_holdout_start(axis: Axes, holdout_start: pd.Timestamp) -> None:
    axis.axvline(holdout_start, color=MUTED_INK, linewidth=1, linestyle="--")


def plot_weekly_fail_rate(weekly: pd.DataFrame, holdout_start: pd.Timestamp) -> Figure:
    """Show unit volume and fail rate per week as two aligned panels."""
    figure, (rate_axis, units_axis) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, height_ratios=[2, 1], constrained_layout=True
    )
    rate_axis.plot(weekly.index, weekly["fail_rate"] * 100, color=FAIL_COLOR, marker="o", ms=4)
    rate_axis.set(title="Fail rate per week", ylabel="Fail rate (%)")
    _mark_holdout_start(rate_axis, holdout_start)
    rate_axis.annotate(
        "hold-out starts",
        xy=(holdout_start, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        color=MUTED_INK,
    )
    units_axis.bar(weekly.index, weekly["units"], width=5, color=GOOD_COLOR)
    units_axis.set(title="Units per week", ylabel="Units")
    _mark_holdout_start(units_axis, holdout_start)
    units_axis.tick_params(axis="x", rotation=30)
    return figure


def plot_missing_histogram(missing: pd.Series, threshold: float) -> Figure:
    """Show how missingness is distributed across features."""
    figure, axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    axis.hist(missing * 100, bins=40, color=GOOD_COLOR, edgecolor=SURFACE, linewidth=1)
    axis.set_yscale("symlog", linthresh=10)
    axis.axvline(threshold * 100, color=MUTED_INK, linewidth=1, linestyle="--")
    axis.annotate(
        f"default drop threshold {threshold:.0%}",
        xy=(threshold * 100, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        color=MUTED_INK,
    )
    axis.set(
        title="Missing values per feature (training period)",
        xlabel="Missing values (%)",
        ylabel="Features (symlog scale)",
    )
    return figure


def plot_counts(counts: dict[str, int], title: str, xlabel: str) -> Figure:
    """Draw one horizontal bar per category with its count as a direct label."""
    figure, axis = plt.subplots(figsize=(7, 0.6 * len(counts) + 1.2), constrained_layout=True)
    names = list(counts)[::-1]
    values = [counts[name] for name in names]
    bars = axis.barh(names, values, color=GOOD_COLOR, height=0.6)
    axis.bar_label(bars, padding=4, color=INK)
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.12)
    axis.set(title=title, xlabel=xlabel)
    return figure


def plot_feature_distributions(features: pd.DataFrame, labels: pd.Series) -> Figure:
    """Overlay class-conditional histograms as small multiples, one panel per feature."""
    columns = list(features.columns)
    n_rows = (len(columns) + 1) // 2
    figure, axes = plt.subplots(
        n_rows, 2, figsize=(8, 2.6 * n_rows), constrained_layout=True, squeeze=False
    )
    good = labels.to_numpy() == 0
    for axis, column in zip(axes.flat, columns, strict=False):
        values = features[column].dropna()
        bins = pd.cut(values, bins=25, retbins=True)[1]
        for mask, color, label in ((good, GOOD_COLOR, "good"), (~good, FAIL_COLOR, "fail")):
            axis.hist(
                features.loc[mask, column].dropna(),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=label,
            )
        axis.set(title=column, ylabel="Density")
    axes.flat[0].legend()
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    return figure


def plot_candidate_comparison(comparison: pd.DataFrame) -> Figure:
    """Compare temporal and shuffled CV scores per candidate on one shared axis."""
    labels = (comparison["model"] + " / " + comparison["imbalance"]).tolist()[::-1]
    positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.4), constrained_layout=True)
    series = (
        ("pr_auc_temporal_mean", GOOD_COLOR, "o", "Temporal folds"),
        ("pr_auc_shuffled_mean", FAIL_COLOR, "s", "Shuffled folds"),
    )
    for column, color, marker, label in series:
        axis.scatter(
            comparison[column].to_numpy()[::-1],
            positions,
            color=color,
            marker=marker,
            s=60,
            edgecolor=SURFACE,
            linewidth=1,
            label=label,
            zorder=3,
        )
    axis.set_yticks(list(positions), labels)
    axis.grid(axis="y", visible=False)
    axis.set(title="Mean PR-AUC in cross-validation (training period)", xlabel="PR-AUC")
    axis.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)
    return figure


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str) -> Figure:
    """Draw one precision-recall curve per model against the no-skill prevalence line."""
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for (label, (labels, scores)), color in zip(curves.items(), SERIES, strict=False):
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        axis.step(
            recall,
            precision,
            where="post",
            color=color,
            label=f"{label}, AP = {average_precision:.3f}",
        )
    prevalence = next(iter(curves.values()))[0].mean()
    axis.axhline(prevalence, color=MUTED_INK, linewidth=1, linestyle="--")
    axis.annotate(
        f"no skill = {prevalence:.3f}",
        xy=(1, prevalence),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=MUTED_INK,
    )
    axis.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="upper right")
    return figure


def plot_cost_curve(
    thresholds: np.ndarray, costs: np.ndarray, chosen: float, model_label: str
) -> Figure:
    """Show the assumed cost per unit over the threshold and mark the chosen threshold."""
    finite = np.isfinite(thresholds)
    figure, axis = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    axis.plot(thresholds[finite], costs[finite], color=GOOD_COLOR)
    axis.axhline(costs[-1], color=MUTED_INK, linewidth=1, linestyle=":")
    axis.annotate(
        "flag nothing",
        xy=(thresholds[finite].min(), costs[-1]),
        xytext=(0, 4),
        textcoords="offset points",
        color=MUTED_INK,
    )
    if np.isfinite(chosen):
        axis.axvline(chosen, color=FAIL_COLOR, linewidth=1, linestyle="--")
        axis.annotate(
            f"chosen threshold {chosen:.3f}",
            xy=(chosen, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            color=MUTED_INK,
        )
    axis.set(
        title=f"Assumed cost over threshold, out-of-fold: {model_label}",
        xlabel="Score threshold",
        ylabel="Cost per unit",
    )
    return figure


def plot_optuna_history(studies: dict[str, optuna.Study]) -> Figure:
    """Show every trial and the best value so far, one panel per study on a shared axis."""
    figure, axes = plt.subplots(
        1, len(studies), figsize=(4 * len(studies), 3.4), sharey=True, constrained_layout=True
    )
    for axis, (name, study) in zip(np.atleast_1d(axes), studies.items(), strict=True):
        values = np.array([trial.value for trial in study.trials])
        numbers = np.arange(len(values))
        axis.scatter(numbers, values, color=GRID, edgecolor=MUTED_INK, linewidth=0.5, s=20)
        axis.plot(numbers, np.maximum.accumulate(values), color=GOOD_COLOR, label="Best so far")
        axis.set(title=name, xlabel="Trial")
    np.atleast_1d(axes)[0].set_ylabel("Mean PR-AUC, temporal folds")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    return figure


def plot_optuna_importance(studies: dict[str, optuna.Study], n_params: int = 6) -> Figure:
    """Show the most important hyperparameters per study by fANOVA importance."""
    figure, axes = plt.subplots(
        1, len(studies), figsize=(4.4 * len(studies), 3.4), constrained_layout=True
    )
    for axis, (name, study) in zip(np.atleast_1d(axes), studies.items(), strict=True):
        importance = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED)
        )
        top = list(importance.items())[:n_params][::-1]
        axis.barh([key for key, _ in top], [value for _, value in top], color=GOOD_COLOR)
        axis.grid(axis="y", visible=False)
        axis.set(title=name, xlabel="Importance")
    return figure


def plot_shap_summary(values: np.ndarray, features: pd.DataFrame, model_label: str) -> Figure:
    """Draw the SHAP summary plot for the hold-out units."""
    shap.summary_plot(values, features, max_display=15, show=False, plot_size=(8, 5.5))
    figure = plt.gcf()
    figure.axes[0].set_title(f"SHAP summary on the hold-out: {model_label}")
    return figure


def plot_shap_cases(cases: dict[str, pd.DataFrame]) -> Figure:
    """Show signed SHAP contributions for individual units, one panel each."""
    figure, axes = plt.subplots(
        1, len(cases), figsize=(5 * len(cases), 3.6), constrained_layout=True
    )
    for axis, (title, table) in zip(np.atleast_1d(axes), cases.items(), strict=True):
        ordered = table.iloc[::-1]
        colors = [FAIL_COLOR if value > 0 else GOOD_COLOR for value in ordered["shap"]]
        axis.barh(ordered["feature"], ordered["shap"], color=colors, height=0.6)
        axis.axvline(0, color=MUTED_INK, linewidth=1)
        axis.grid(axis="y", visible=False)
        axis.set(title=title, xlabel="SHAP value (positive pushes toward fail)")
    return figure
