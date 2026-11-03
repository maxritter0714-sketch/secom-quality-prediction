"""Load the SECOM dataset into time-ordered features, labels and timestamps."""

from pathlib import Path

import pandas as pd

from secom.config import RAW_DATA_DIR

FEATURES_FILE = "secom.data"
LABELS_FILE = "secom_labels.data"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
RAW_TO_BINARY_LABEL = {-1: 0, 1: 1}


def _read_labels(path: Path) -> pd.DataFrame:
    """Read the label file, whose lines look like: -1 "19/07/2008 11:55:00"."""
    return pd.read_csv(path, sep=" ", header=None, names=["raw_label", "timestamp"], quotechar='"')


def _encode_labels(raw_labels: pd.Series) -> pd.Series:
    """Map raw labels to 0 = good and 1 = fail, rejecting anything unexpected."""
    unknown = set(raw_labels.unique()) - set(RAW_TO_BINARY_LABEL)
    if unknown:
        raise ValueError(f"Unexpected label values: {sorted(unknown)}")
    return raw_labels.map(RAW_TO_BINARY_LABEL).astype("int64").rename("fail")


def load_secom(data_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, labels (1 = fail) and timestamps, sorted by time.

    The sort is stable, so units sharing a timestamp keep their original order.
    """
    features_path = data_dir / FEATURES_FILE
    labels_path = data_dir / LABELS_FILE
    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"SECOM files not found in {data_dir}. Run: python scripts/download_data.py"
        )

    features = pd.read_csv(features_path, sep=" ", header=None, na_values="NaN", dtype="float64")
    labels = _read_labels(labels_path)
    if len(features) != len(labels):
        raise ValueError(f"Row mismatch: {len(features)} features, {len(labels)} labels")

    features.columns = [f"f{index:03d}" for index in range(features.shape[1])]
    timestamps = pd.to_datetime(labels["timestamp"], format=TIMESTAMP_FORMAT).rename("timestamp")
    order = timestamps.argsort(kind="stable").to_numpy()

    return (
        features.iloc[order].reset_index(drop=True),
        _encode_labels(labels["raw_label"]).iloc[order].reset_index(drop=True),
        timestamps.iloc[order].reset_index(drop=True),
    )
