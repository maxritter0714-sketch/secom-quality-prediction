# Design decisions

Short records of choices that are not obvious from the code.

## Chronological hold-out, split at a timestamp boundary

The last 20 % of units form the hold-out. The cut moves forward until no timestamp is shared
between both periods, because several units carry an identical timestamp.

## Temporal folds for tuning and threshold selection

`TimeSeriesSplit` with an expanding window is the primary validation, since the fail rate drifts
strongly over the recording period (see `reports/figures/eda_fail_rate_over_time.png`). Shuffled
5x5 cross-validation is reported on the training period for comparison only.

A fold without any fail raises an error rather than being skipped, because silently averaging over
fewer folds would hide how little evidence there is.

## Exploration uses the training period for feature statistics

The feature analyses in the notebook only use the training period. The one full-period view is the
label timeline, which is descriptive and sets no parameter.

## Own selector classes instead of `SelectKBest`

The number of features that reach selection varies between folds and trials, because the earlier
filters depend on the data. `TopKSelector` accepts a `k` above the remaining feature count.

## LightGBM as the boosting library

`TreeExplainer` supports it directly, which keeps the SHAP analysis simple.
`class_weight="balanced"` replaces a hand-set `scale_pos_weight`: the weight is computed from each
training fold during fit, so it cannot leak.

## SMOTE runs inside an imblearn pipeline, after scaling and selection

The sampler runs during fit only. Features are always scaled first, because SMOTE interpolates
between neighbours and would otherwise be dominated by large-valued features. It runs after feature
selection, so that selection only sees real units.

## No Optuna pruner

Folds differ strongly in their number of fails, so a poor early fold says little about the final
mean. Trials therefore always run all folds. The optional `MedianPruner` from the brief is unused.

## Tuned CV scores are optimistically biased

The best trial is selected on the same folds that report its score, and with few fails per fold this
selection effect can be large. Tuned CV scores are therefore not an independent estimate. Only the
hold-out provides one, and its interval is wide.

## Thresholds come from out-of-fold scores

Expanding-window folds leave the earliest rows without out-of-fold scores, so the threshold is chosen
on the validated rows only. Scores from different folds come from differently trained models, and
the fail rate drifts between them; a threshold can therefore transfer poorly to the hold-out. The
dummy classifier shows the effect in the validation table: its only score is the fold's prior, so a
threshold on it effectively selects whole early, fail-rich folds.

## Anomaly detectors are fitted on good units only

This matches a line with known-good reference production. Fitting on all training units would be
fully label-free, but would treat known fails as normal. Labels filter the fit data and evaluate the
scores; they are never a learning target. This is therefore a semi-supervised setting, which is
stated wherever results are compared.

Detector hyperparameters stay at sensible fixed values, because tuning them on labels would further
weaken the claim of how far one gets with few labels.

## Permutation importance runs on the model's input features

Features removed by preprocessing have an importance of exactly zero, so permuting them would only
cost time. The importance is measured after the fitted preprocessing, on the hold-out.

## The hold-out is evaluated in one stage

`run_holdout_stage` is the only code that receives hold-out rows. Everything it uses is fixed before
it starts. `--fast` also passes through this stage as a smoke test, but writes to the ignored
`artifacts/` directory, and its numbers are not used for any decision.
