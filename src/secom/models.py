"""Factory for the supervised classifiers compared in this project."""

from typing import Any, Literal

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from secom.config import SEED
from secom.preprocessing import PreprocessingConfig

ModelName = Literal["dummy", "logreg"]
ImbalanceStrategy = Literal["none", "weighted", "smote"]

MODEL_NAMES: tuple[ModelName, ...] = ("dummy", "logreg")


def make_classifier(
    name: ModelName, imbalance: ImbalanceStrategy = "weighted", **params: Any
) -> ClassifierMixin:
    """Create a seeded classifier; class weights are derived per fit from the fold's labels."""
    class_weight = "balanced" if imbalance == "weighted" else None
    if name == "dummy":
        return DummyClassifier(strategy="prior", **params)
    if name == "logreg":
        settings = {"C": 1.0, "max_iter": 5000, **params}
        return LogisticRegression(class_weight=class_weight, random_state=SEED, **settings)
    raise ValueError(f"Unknown model: {name}")


def default_preprocessing(
    name: ModelName, imbalance: ImbalanceStrategy = "weighted"
) -> PreprocessingConfig:
    """Return default preprocessing for a model and imbalance strategy."""
    return PreprocessingConfig(scaler="standard", use_smote=imbalance == "smote")
