"""Unsupervised anomaly detection, scored with the same metrics as the classifiers.

Variant used: detectors are fitted on training rows labelled good ("novelty detection").
This is the semi-supervised setting of a line with known-good reference production.
Fitting on all rows would be fully label-free but treats known fails as normal; labels
are used here only to filter the fit data and to evaluate, never as a learning target.
"""

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from secom.config import SEED
from secom.preprocessing import MissingnessFilter
from secom.validation import Folds

DetectorName = Literal["isolation_forest", "one_class_svm"]
DETECTOR_NAMES: tuple[DetectorName, ...] = ("isolation_forest", "one_class_svm")
EXPECTED_ANOMALY_FRACTION = 0.1


def make_detector(name: DetectorName) -> Pipeline:
    """Build a detector pipeline; all steps are fitted on the rows passed to fit."""
    detector: BaseEstimator
    if name == "isolation_forest":
        detector = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1)
    elif name == "one_class_svm":
        detector = OneClassSVM(kernel="rbf", gamma="scale", nu=EXPECTED_ANOMALY_FRACTION)
    else:
        raise ValueError(f"Unknown detector: {name}")
    return Pipeline(
        [
            ("drop_missing", MissingnessFilter()),
            ("impute", SimpleImputer(strategy="median")),
            ("drop_constant", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("detector", detector),
        ]
    )


@dataclass
class GoodOnlyDetector:
    """Fit a detector on good rows and return scores where higher means more anomalous."""

    pipeline: Pipeline

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> Self:
        self.fitted_ = clone(self.pipeline).fit(features[labels.to_numpy() == 0])
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return -self.fitted_.decision_function(features)


def cross_validate_detector(
    name: DetectorName, features: pd.DataFrame, labels: pd.Series, folds: Folds
) -> tuple[np.ndarray, np.ndarray]:
    """Return fold PR-AUCs and out-of-fold anomaly scores, NaN for rows never validated."""
    fold_pr_aucs = []
    out_of_fold = np.full(len(labels), np.nan)
    for train, validation in folds:
        detector = GoodOnlyDetector(make_detector(name)).fit(
            features.iloc[train], labels.iloc[train]
        )
        scores = detector.score(features.iloc[validation])
        fold_pr_aucs.append(average_precision_score(labels.iloc[validation], scores))
        out_of_fold[validation] = scores
    return np.array(fold_pr_aucs), out_of_fold
