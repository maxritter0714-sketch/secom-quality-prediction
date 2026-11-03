"""Time-aware splitting of the dataset."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from secom.config import TEST_FRACTION


@dataclass(frozen=True)
class TemporalSplit:
    """Row positions of the earlier training period and the later hold-out period."""

    train: np.ndarray
    test: np.ndarray


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
