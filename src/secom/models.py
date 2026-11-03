"""Factory for the supervised classifiers compared in this project."""

from typing import Any, Literal

from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from secom.config import SEED
from secom.preprocessing import PreprocessingConfig

ModelName = Literal["dummy", "logreg", "random_forest", "lightgbm"]
ImbalanceStrategy = Literal["none", "weighted", "smote"]

MODEL_NAMES: tuple[ModelName, ...] = ("dummy", "logreg", "random_forest", "lightgbm")
TREE_MODELS: frozenset[ModelName] = frozenset({"random_forest", "lightgbm"})
LIGHTGBM_THREADS = 4


def make_classifier(
    name: ModelName, imbalance: ImbalanceStrategy = "weighted", **params: Any
) -> ClassifierMixin:
    """Create a classifier with a fixed seed; weights are computed per fit from the fold's labels.

    LightGBM uses class_weight="balanced", which equals scale_pos_weight but is derived
    from each training fold at fit time and therefore cannot leak.
    """
    class_weight = "balanced" if imbalance == "weighted" else None
    if name == "dummy":
        return DummyClassifier(strategy="prior", **params)
    if name == "logreg":
        settings = {"C": 1.0, "max_iter": 5000, **params}
        return LogisticRegression(class_weight=class_weight, random_state=SEED, **settings)
    if name == "random_forest":
        settings = {"n_estimators": 300, "min_samples_leaf": 2, **params}
        return RandomForestClassifier(
            class_weight=class_weight, random_state=SEED, n_jobs=-1, **settings
        )
    if name == "lightgbm":
        settings = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, **params}
        return LGBMClassifier(
            class_weight=class_weight,
            random_state=SEED,
            n_jobs=LIGHTGBM_THREADS,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            **settings,
        )
    raise ValueError(f"Unknown model: {name}")


def default_preprocessing(
    name: ModelName, imbalance: ImbalanceStrategy = "weighted"
) -> PreprocessingConfig:
    """Return default preprocessing: only linear models and SMOTE need scaled features."""
    needs_scaling = name not in TREE_MODELS or imbalance == "smote"
    return PreprocessingConfig(
        scaler="standard" if needs_scaling else None,
        use_smote=imbalance == "smote",
    )
