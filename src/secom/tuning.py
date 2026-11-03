"""Hyperparameter optimization with Optuna on temporal folds of the training period."""

import warnings
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning

from secom.config import SEED
from secom.models import ImbalanceStrategy, ModelName, make_classifier
from secom.preprocessing import PreprocessingConfig, build_pipeline
from secom.validation import Folds, cross_validate

N_TRIALS = 100
TUNED_MODELS: tuple[ModelName, ...] = ("logreg", "random_forest", "lightgbm")


@dataclass(frozen=True)
class TrialDesign:
    """Everything one trial decides: model parameters, preprocessing and imbalance strategy."""

    model_params: dict[str, Any]
    preprocessing: PreprocessingConfig
    imbalance: ImbalanceStrategy


@dataclass(frozen=True)
class TuningResult:
    """Best design of a model's study, together with the study for plots."""

    model: ModelName
    design: TrialDesign
    best_pr_auc: float
    study: optuna.Study


def _suggest_model_params(trial: optuna.Trial, model: ModelName) -> dict[str, Any]:
    if model == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 10.0, log=True)}
    if model == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.05, 1.0),
        }
    if model == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 3, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(f"Model cannot be tuned: {model}")


def _suggest_design(trial: optuna.Trial, model: ModelName) -> TrialDesign:
    imbalance = trial.suggest_categorical("imbalance", ["none", "weighted", "smote"])
    use_correlation_filter = trial.suggest_categorical("use_correlation_filter", [False, True])
    # Scaling is needed by the linear model and gives SMOTE's distances a common unit.
    needs_scaling = model == "logreg" or imbalance == "smote"
    scaler = trial.suggest_categorical("scaler", ["standard", "robust"]) if needs_scaling else None
    preprocessing = PreprocessingConfig(
        max_missing=trial.suggest_float("max_missing", 0.1, 0.9),
        add_missing_indicator=trial.suggest_categorical("add_missing_indicator", [False, True]),
        correlation_threshold=0.95 if use_correlation_filter else None,
        scaler=scaler,
        k_features=trial.suggest_int("k_features", 10, 300, log=True),
        use_smote=imbalance == "smote",
    )
    return TrialDesign(_suggest_model_params(trial, model), preprocessing, imbalance)


def build_tuned_pipeline(model: ModelName, design: TrialDesign) -> Pipeline:
    """Create an unfitted pipeline from a trial design."""
    classifier = make_classifier(model, design.imbalance, **design.model_params)
    return build_pipeline(classifier, design.preprocessing)


def tune_model(
    model: ModelName,
    features: pd.DataFrame,
    labels: pd.Series,
    folds: Folds,
    n_trials: int = N_TRIALS,
) -> TuningResult:
    """Maximize mean PR-AUC over the folds with a seeded TPE sampler.

    No pruner is used: folds differ strongly in their number of fails, so a poor
    early fold says little about a trial's final score.
    """

    def objective(trial: optuna.Trial) -> float:
        design = _suggest_design(trial, model)
        pipeline = build_tuned_pipeline(model, design)
        # A trial that does not fully converge is still judged by its real validation score.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return cross_validate(pipeline, features, labels, folds).mean_pr_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=model,
    )
    study.optimize(objective, n_trials=n_trials)
    best_design = _suggest_design(study.best_trial, model)
    return TuningResult(model, best_design, study.best_value, study)
