"""Single entry point that regenerates every table and figure in reports/."""

from secom.data import load_secom
from secom.experiments import CANDIDATES, compare_candidates
from secom.plots import apply_style, plot_candidate_comparison, save_figure
from secom.reporting import markdown_table, write_results
from secom.validation import temporal_split


def main() -> None:
    apply_style()
    features, labels, timestamps = load_secom()
    split = temporal_split(timestamps)
    train_features = features.iloc[split.train]
    train_labels = labels.iloc[split.train]

    comparison = compare_candidates(train_features, train_labels, CANDIDATES)
    save_figure(plot_candidate_comparison(comparison), "cv_temporal_vs_shuffled")

    write_results(
        sections={"Candidate comparison (training period)": markdown_table(comparison)},
        data={"candidate_comparison": comparison.to_dict(orient="records")},
    )


if __name__ == "__main__":
    main()
