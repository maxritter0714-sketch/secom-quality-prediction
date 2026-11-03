"""Single entry point that regenerates every table and figure in reports/."""

from secom.data import load_secom
from secom.experiments import BASELINE_CANDIDATES, compare_candidates
from secom.reporting import markdown_table, write_results
from secom.validation import temporal_split


def main() -> None:
    features, labels, timestamps = load_secom()
    split = temporal_split(timestamps)
    train_features = features.iloc[split.train]
    train_labels = labels.iloc[split.train]

    comparison = compare_candidates(train_features, train_labels, BASELINE_CANDIDATES)

    write_results(
        sections={"Candidate comparison (training period)": markdown_table(comparison)},
        data={"candidate_comparison": comparison.to_dict(orient="records")},
    )


if __name__ == "__main__":
    main()
