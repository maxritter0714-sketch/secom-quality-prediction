"""Descriptive statistics for exploration. Nothing here feeds into model fitting."""

import numpy as np
import pandas as pd

NEAR_CONSTANT_DOMINANCE = 0.95


def missing_fraction(features: pd.DataFrame) -> pd.Series:
    """Fraction of missing values per feature."""
    return features.isna().mean()


def constant_feature_counts(features: pd.DataFrame) -> dict[str, int]:
    """Count features with a single value, and those dominated by one value."""
    n_unique = features.nunique(dropna=True)
    dominance = features.apply(
        lambda column: column.value_counts(normalize=True).iloc[0] if column.notna().any() else 1.0
    )
    constant = n_unique <= 1
    return {
        "constant": int(constant.sum()),
        "near_constant": int(((dominance >= NEAR_CONSTANT_DOMINANCE) & ~constant).sum()),
        "other": int(((dominance < NEAR_CONSTANT_DOMINANCE) & ~constant).sum()),
    }


def correlated_pair_counts(
    features: pd.DataFrame, thresholds: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99)
) -> dict[float, int]:
    """Count distinct feature pairs whose absolute Pearson correlation reaches each threshold."""
    correlation = features.corr().abs().to_numpy()
    upper_triangle = correlation[np.triu_indices_from(correlation, k=1)]
    return {threshold: int(np.nansum(upper_triangle >= threshold)) for threshold in thresholds}


def weekly_fail_rate(labels: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    """Aggregate unit counts and the fail rate per calendar week."""
    frame = pd.DataFrame({"fail": labels.to_numpy()}, index=pd.DatetimeIndex(timestamps))
    weekly = frame["fail"].resample("W").agg(["count", "sum"])
    weekly = weekly.rename(columns={"count": "units", "sum": "fails"})
    weekly["fail_rate"] = weekly["fails"] / weekly["units"].where(weekly["units"] > 0)
    return weekly


def top_separating_features(
    features: pd.DataFrame, labels: pd.Series, n_features: int = 4, min_class_count: int = 30
) -> list[str]:
    """Rank features by absolute standardized mean difference between the classes.

    Features observed fewer than min_class_count times in either class are skipped,
    because a mean over a handful of values can look separating purely by chance.
    """
    fail_mask = labels.to_numpy() == 1
    fails, goods = features[fail_mask], features[~fail_mask]
    enough_data = (fails.notna().sum() >= min_class_count) & (
        goods.notna().sum() >= min_class_count
    )
    pooled_std = features.std().where(lambda std: std > 0)
    effect = ((fails.mean() - goods.mean()) / pooled_std).abs()
    return effect[enough_data].dropna().sort_values(ascending=False).head(n_features).index.tolist()
