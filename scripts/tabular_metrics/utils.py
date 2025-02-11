"""
===============================
Metrics calculation
===============================
Includes a few metric as well as functions composing metrics on results files.

"""
from __future__ import annotations

import numpy as np
from typing import Any, Callable, Literal, TypedDict
import pandas as pd
import warnings

from .classification import *
from.fairness import *


class MetricDefinition(TypedDict):
    name: str
    func: Callable
    aggregator: Literal["mean", "sum"]


def get_scoring_direction(metric_used):
    if type(metric_used) == str:
        metric_used = metric_used
    else:
        metric_used = get_scoring_string(metric_used, usage="sklearn_cv")
    if metric_used in {
        "r2",
        "neg_log_loss",
        "neg_root_mean_squared_error",
        "neg_mean_absolute_error",
        "neg_mean_absolute_error",
        "pinball_loss",
        "expected_calibration_error",
        "cross_entropy",
        "ce",
        "ece",
        "time",
        "count",
        "normalized_mae",
        "normalized_mse",
        "normalized_rmse",
        "rmse",
        "mse",
        "mae",
        "uncensored_mse",
        "fairness_ddsp",
        "causal_fairness_total_effect",
        "causal_fairness_direct_effect",
        "causal_fairness_indirect_effect",
    }:
        return -1
    elif metric_used in {
        "roc_auc_ovo",
        "roc_auc_ovr",
        "roc_auc",
        "spearman",
        "c_index",
        "roc",
        "acc",
        "f1",
        "balanced_acc",
        "automl",
        "cindex",
        "censoring_acc",
    }:
        return 1
    else:
        raise Exception("No scoring direction found for metric {}".format(metric_used))


def get_task_type(metric_used):
    if metric_used.__name__ in [
        auc_metric.__name__,
        cross_entropy.__name__,
        accuracy_metric.__name__,
        f1_metric.__name__,
        balanced_accuracy_metric.__name__,
        auc_metric_ovr.__name__,
        auc_metric_ovo.__name__,
        automl_benchmark_metric.__name__,
    ]:
        return "multiclass"
    elif metric_used.__name__ in [
        auc_metric.__name__,
        fairness_ddsp.__name__,
        causal_fairness_direct_effect.__name__,
        causal_fairness_indirect_effect.__name__,
        causal_fairness_total_effect.__name__
    ]:
        return "fairness_multiclass"
    else:
        raise NotImplementedError(f"Unknown metric {metric_used.__name__}")


def _get_autosklearn_scoring_str(
    metric_used: Callable,
    multiclass: bool = True,
) -> Any:
    import autosklearn.classification

    _mapping = {
        cross_entropy: autosklearn.metrics.log_loss,
        r2_metric: autosklearn.metrics.r2,
        root_mean_squared_error_metric: autosklearn.metrics.root_mean_squared_error,
        mean_absolute_error_metric: autosklearn.metrics.mean_absolute_error,
    }
    # roc_auc only works for binary, use logloss instead
    if multiclass:
        _mapping.update(
            {
                auc_metric_ovr: autosklearn.metrics.log_loss,
                auc_metric_ovo: autosklearn.metrics.log_loss,
                automl_benchmark_metric: autosklearn.metrics.log_loss,
            }
        )
    else:
        _mapping.update(
            {
                auc_metric_ovr: autosklearn.metrics.roc_auc,
                auc_metric_ovo: autosklearn.metrics.roc_auc,
                automl_benchmark_metric: autosklearn.metrics.roc_auc,
            }
        )

    return _mapping[metric_used]


# Loss
def get_scoring_string(
    metric_used: Callable,
    multiclass: bool = True,
    usage: (
        Literal[
            "xgb",
            "sklearn_cv",
            "autogluon",
            "tabnet",
            "autosklearn",
            "catboost",
            "lightgbm",
        ]
        | None
    ) = "sklearn_cv",
) -> str:
    # TODO: Not sure we should do this! Might lead to unexpected metrics being used
    metric_remapping = {
        auc_metric: auc_metric_ovr,
    }
    if metric_used in metric_remapping:
        metric_used = metric_remapping[metric_used]

    # Exit out early if it's autosklearn since it requires a different handling
    if usage == "autosklearn":
        return _get_autosklearn_scoring_str(metric_used, multiclass=multiclass)

    # All of these are identical for binary classification
    ag_auc_metric_ovr = ag_auc_metric_ovo = ag_automl_benchmark_metric = "roc_auc"

    _mapping = {
        auc_metric_ovr.__name__: {
            None: "roc_auc_ovr",
            "sklearn_cv": "roc_auc_ovr",
            "catboost": "MultiClass",
            "xgb": "binary:logistic",  # Changed for multiclass below
            "autogluon": ag_auc_metric_ovr,
            "tabnet": "auc",
            "lightgbm": "binary",
        },
        auc_metric_ovo.__name__: {
            None: "roc_auc",  # TODO: Maybe this should be `roc_auc_ovo`?
            "sklearn_cv": "roc_auc_ovo",
            "catboost": "MultiClass",
            "xgb": "binary:logistic",  # Changed for multiclass below
            "autogluon": ag_auc_metric_ovo,
            "tabnet": "auc",
            "lightgbm": "binary",
        },
        automl_benchmark_metric.__name__: {
            None: "roc_auc_ovr",
            "sklearn_cv": "roc_auc_ovr",
            "catboost": "MultiClass",
            "xgb": "binary:logistic",
            "autogluon": ag_automl_benchmark_metric,
            "tabnet": "auc",
            "lightgbm": "binary",
        },
        cross_entropy.__name__: {
            None: "logloss",
            "sklearn_cv": "neg_log_loss",
            "autogluon": "log_loss",
            "tabnet": "logloss",
            "catboost": "MultiClass",
        },
        r2_metric.__name__: {
            None: "r2",
            "sklearn_cv": "r2",  # neg_r2
            "autogluon": "r2",
            "xgb": "reg:squarederror",  # XGB cannot directly optimize r2
            "catboost": "RMSE",  # Catboost cannot directly optimize r2 ("Can't be used for optimization." - docu)
            "lightgbm": "regression",
        },
        root_mean_squared_error_metric.__name__: {
            None: "neg_root_mean_squared_error",
            "sklearn_cv": "neg_root_mean_squared_error",
            "autogluon": "rmse",
            "xgb": "reg:squarederror",
            "catboost": "RMSE",
            "lightgbm": "regression",
        },
        mean_absolute_error_metric.__name__: {
            None: "neg_mean_absolute_error",
            "sklearn_cv": "neg_mean_absolute_error",
            "autogluon": "mae",
            "xgb": "mae",
            "lightgbm": "regression_l1",
            "catboost": "MAE",
        },
        PinballLossMetric.__name__: {
            None: "pinball_loss",
            "autogluon": None,
        },
        NormalizedPinballLossMetric.__name__: {
            None: "normalized_pinball_loss",
            "autogluon": None,
        },
    }

    if multiclass:
        _mapping[automl_benchmark_metric.__name__].update(
            {
                None: "logloss",
                "sklearn_cv": "neg_log_loss",
                "autogluon": "log_loss",
                "tabnet": "logloss",
                "catboost": "MultiClass",
            }
        )
        _mapping[auc_metric_ovr.__name__].update(
            {
                "autogluon": "roc_auc_ovr_macro",
                "tabnet": "logloss",
                "lightgbm": "multiclass",  # ovr objective gives unnormalized prediction probabilities (e.g. sum over classes = 0.3 or 1.4)
                "xgb": "multi:softprob",
            },
        )
        _mapping[auc_metric_ovo.__name__].update(
            {
                "autogluon": "roc_auc_ovo_macro",
                "tabnet": "logloss",
                "lightgbm": "multiclass",  # Doesn't seem to have an OVO option
                "xgb": "multi:softprob",
            },
        )

    metric_group = _mapping.get(metric_used.__name__, None)
    if metric_group is None:
        raise NotImplementedError(f"Unknown metric {metric_used.__name__}")

    metric_str = metric_group.get(usage, None)
    if metric_str is None:
        metric_str = metric_group[None]

    return metric_str


def time_metric():
    """
    Dummy function, will just be used as a handler.
    """
    pass


def count_metric(x, y):
    """
    Dummy function, returns one count per dataset.
    """
    return 1


def get_main_eval_metric(task_type):
    if task_type == "multiclass":
        metric_used = auc_metric_ovr
    elif task_type == "fairness_multiclass":
        metric_used = auc_metric
    else:
        raise NotImplementedError("Unknown task type")
    return metric_used


def get_metric_name(metric):
    if metric.__name__ == time_metric.__name__:
        return "time"
    elif metric.__name__ == count_metric.__name__:
        return "count"
    elif metric.__name__ == accuracy_metric.__name__:
        return "acc"
    elif metric.__name__ == cross_entropy.__name__:
        return "ce"
    elif metric.__name__ == automl_benchmark_metric.__name__:
        return "automl"
    elif (
        metric.__name__ == auc_metric.__name__
        or metric.__name__ == auc_metric_ovr.__name__
    ):
        return "roc"
    elif metric.__name__ == auc_metric_ovo.__name__:
        return "roc_ovo"
    elif metric.__name__ == f1_metric.__name__:
        return "f1"
    elif metric.__name__ == balanced_accuracy_metric.__name__:
        return "balanced_acc"
    elif metric.__name__ == r2_metric.__name__:
        return "r2"
    elif metric.__name__ == normalized_root_mean_squared_error_metric.__name__:
        return "normalized_rmse"
    elif metric.__name__ == root_mean_squared_error_metric.__name__:
        return "rmse"
    elif metric.__name__ == mean_absolute_error_metric.__name__:
        return "mae"
    elif metric.__name__ == mean_squared_error_metric.__name__:
        return "mse"
    elif metric.__name__ == normalized_mean_squared_error_metric.__name__:
        return "normalized_mse"
    elif metric.__name__ == normalized_mean_absolute_error_metric.__name__:
        return "normalized_mae"
    elif metric.__name__ == spearman_metric.__name__:
        return "spearman"
    elif metric.__name__ == expected_calibration_error.__name__:
        return "ece"
    elif metric.__name__ == PinballLossMetric.__name__:
        return f"pinball_loss_q={str(metric.quantiles):.10}"
    elif metric.__name__ == NormalizedPinballLossMetric.__name__:
        return f"normalized_pinball_loss_q={str(metric.quantiles):.10}"
    elif metric.__name__ == QuantileCalibrationMetric.__name__:
        return f"quantile_calibration_q={str(metric.quantiles):.10}"
    elif metric.__name__ == SharpnessMetric.__name__:
        return f"sharpness_q={str(metric.quantiles):.10}"
    elif metric.__name__ == QuantileMAEMetric.__name__:
        return f"quantile_mae"
    elif metric.__name__ == NormalizedQuantileMAEMetric.__name__:
        return f"normalized_quantile_mae"
    elif metric.__name__ == MeanIntervalScoreMetric.__name__:
        return f"mean_interval_score_q={str(metric.quantiles):.10}"
    elif metric.__name__ == MeanNormalizedIntervalScoreMetric.__name__:
        return f"mean_normalized_interval_score_q={str(metric.quantiles):.10}"
    elif metric.__name__ == fairness_ddsp.__name__:
        return f"fairness_ddsp"
    elif metric.__name__ == causal_fairness_direct_effect.__name__:
        return f"causal_fairness_direct_effect"
    elif metric.__name__ == causal_fairness_indirect_effect.__name__:
        return f"causal_fairness_indirect_effect"
    elif metric.__name__ == causal_fairness_total_effect.__name__:
        return f"causal_fairness_total_effect"
    raise NotImplementedError(f"Unknown metric {metric.__name__}")


def check_metric_fits_task_type(metric_used, task_type):
    metrics_for_task_type = [
        metric["name"] for metric in get_standard_eval_metrics(task_type)
    ]
    if get_metric_name(metric_used) not in metrics_for_task_type:
        return False
    else:
        return True


def get_standard_eval_baselines(
    task_type: Literal["multiclass"]
) -> list[str]:
    if task_type == "multiclass":
        return [
            # "knn", - currently files missing
            "logistic",
            "xgb_default",
            "xgb",
            "catboost_default",
            "catboost",
            "lgb_default",
            "lightgbm",
            "autogluon",
        ]
    elif task_type == "fairness_multiclass":
        return [
            "tabpfn_1",
            "unaware",
            "egr",
            "level_one",
            "level_two"
        ]
    else:
        raise NotImplementedError(f"Unknown task type {task_type}")


def get_standard_eval_metrics(
    task_type: Literal["multiclass"]
) -> typinglist[MetricDefinition]:
    generic_metrics = [
        {"func": time_metric, "aggregator": "sum"},
        {"func": time_metric, "aggregator": "mean"},
        {"func": count_metric, "aggregator": "nansum"},
    ]
    task_metrics = {
        "multiclass": [
            {"func": auc_metric_ovr, "aggregator": "mean"},
            {"func": accuracy_metric, "aggregator": "mean"},
            {"func": cross_entropy, "aggregator": "mean"},
            {"func": f1_metric, "aggregator": "mean"},
            {"func": balanced_accuracy_metric, "aggregator": "mean"},
            {"func": expected_calibration_error, "aggregator": "mean"},
            {"func": automl_benchmark_metric, "aggregator": "mean"},
        ],
        "fairness_multiclass": [
            {"func": fairness_ddsp, "aggregator": "mean"},
            {"func": auc_metric, "aggregator": "mean"},
            {"func": causal_fairness_direct_effect, "aggregator": "mean"},
            {"func": causal_fairness_indirect_effect, "aggregator": "mean"},
            {"func": causal_fairness_total_effect, "aggregator": "mean"},
        ],
    }

    all_metrics = generic_metrics + task_metrics[task_type]
    return [
        MetricDefinition(
            name=get_metric_name(metric["func"]),
            func=metric["func"],
            aggregator=metric["aggregator"],
        )
        for metric in all_metrics
    ]


"""
===============================
Metrics composition
===============================
"""


def calculate_score_per_method(
    metric: Callable,
    name: str,
    global_results: dict,
    ds: list,
    aggregator: Literal["mean", "sum"] = "mean",
    subgroups: dict[str, list] = {},
):
    """Calculates the metric given by 'metric' and saves it under 'name' in
    the 'global_results'

    :param metric: Metric function
    :param name: Name of metric in 'global_results'
    :param global_results: Dicrtonary containing the results for current method for
        a collection of datasets
    :param ds: Dataset to calculate metrics on, a list of dataset properties
    :param eval_positions: List of positions to calculate metrics on
    :param aggregator: Specifies way to aggregate results across evaluation positions
    :param subgroups: Specifies groups of datasets along with a key that is used to aggregate the results for each group separately
    :return:
    """

    aggregator_f = get_aggregator_f(aggregator)
    for d in ds:
        # In case of a DatasetEvaluationCollection, we need to calculate the metric for each dataset in the collection
        # and then aggregate the results using the aggregator given by the user.
        global_results[d.get_dataset_identifier()].calculate_metric(
            metric, name, aggregator
        )

    global_results[f"{aggregator}_{name}"] = aggregator_f(
        [
            global_results[d.get_dataset_identifier()].metrics[f"{aggregator}_{name}"]
            for d in ds
        ]
    )

    # Filter warnings for empty slices e.g. np.nanmean([])
    # We only catch that for subgroup calculations, because we want to know,
    # whether someone forgot to pass any datasets to the method at all
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        for subgroup, subgroup_ds in subgroups.items():
            global_results[f"{subgroup}_{aggregator}_{name}"] = aggregator_f(
                [
                    global_results[d.get_dataset_identifier()].metrics[
                        f"{aggregator}_{name}"
                    ]
                    for d in subgroup_ds
                ]
            )


def get_aggregator_f(aggregator):
    if aggregator == "mean":
        return np.mean
    elif aggregator == "sum":
        return np.sum
    elif aggregator == "nansum":
        return np.nansum
    raise NotImplementedError(f"Unknown aggregator {aggregator}")


def calculate_score(metric, name, global_results, ds, aggregator="mean", limit_to=""):
    """
    Calls calculate_metrics_by_method with a range of methods. See arguments of that method.
    :param limit_to: This method will not get metric calculations.
    """
    for scoring_str in global_results:
        for m in global_results[scoring_str]:
            for time in global_results[scoring_str][m]:
                for split in global_results[scoring_str][m][time]:
                    if limit_to not in m:
                        continue
                    calculate_score_per_method(
                        metric,
                        name,
                        global_results[scoring_str][m][time][split],
                        ds,
                        aggregator=aggregator,
                    )


def make_metric_matrix(global_results, methods, name, ds):
    result = []
    for m in global_results:
        try:
            result += [
                [global_results[m][d.get_dataset_identifier() + "_" + name] for d in ds]
            ]
        except Exception as e:
            result += [[np.nan]]
    result = np.array(result)
    result = pd.DataFrame(
        result.T,
        index=[d.get_dataset_identifier() for d in ds],
        columns=[k for k in list(global_results.keys())],
    )

    matrix_means, matrix_stds = [], []

    for method in methods:
        matrix_means += [
            result.iloc[
                :, [c.startswith(method + "_time") for c in result.columns]
            ].mean(axis=1)
        ]
        matrix_stds += [
            result.iloc[
                :, [c.startswith(method + "_time") for c in result.columns]
            ].std(axis=1)
        ]

    matrix_means = pd.DataFrame(matrix_means, index=methods).T
    matrix_stds = pd.DataFrame(matrix_stds, index=methods).T

    return matrix_means, matrix_stds


def make_ranks_and_wins_table(matrix):
    from scipy.stats import rankdata

    for dss in matrix.T:
        matrix.loc[dss] = rankdata(-matrix.round(3).loc[dss], method="average")
    ranks_acc = matrix.mean()

    wins_acc = (matrix == 1).sum()

    return ranks_acc, wins_acc


metric_renamer = {
    "ce": "Cross entropy",
    "roc": "ROC AUC",
    "cross_entropy": "Cross entropy",
    "rank_roc": "Mean ROC AUC Rank",
    "rank_cross_entropy": "Mean Cross entropy Rank",
    "wins_roc": "Mean ROC AUC Wins",
    "wins_cross_entropy": "Mean Cross entropy Wins",
    "time": "actual time taken",
    "rmse": "RMSE",
    "r2": "R2",
    "rank_rmse": "RMSE Rank",
    "rank_r2": "R2 Rank",
    "rank_spearman": "Spearman Rank",
    "spearman": "Spearman",
    "wins_rmse": "RMSE Wins",
    "wins_r2": "R2 Wins",
    "acc": "Accuracy",
    "ece": "Exp. Calibration Err.",
    "rank_acc": "Accuracy Rank",
    "wins_acc": "Accuracy Wins",
    "rank_ece": "ECE Rank",
    "wins_ece": "ECE Wins",
    "f1": "F1",
    "rank_f1": "F1 Rank",
    "wins_f1": "F1 Wins",
    "balanced_acc": "Balanced Accuracy",
    "rank_balanced_acc": "Balanced Accuracy Rank",
    "wins_balanced_acc": "Balanced Accuracy Wins",
    "automl": "AutoML Benchmark Metric",
    "rank_automl": "AutoML Benchmark Metric Rank",
    "wins_automl": "AutoML Benchmark Metric Wins",
    "cindex": "C-index",
    "rank_cindex": "C-index Rank",
    "wins_cindex": "C-index Wins",
    "censoring_acc": "Censoring Accuracy",
    "rank_censoring_acc": "Censoring Accuracy Rank",
    "wins_censoring_acc": "Censoring Accuracy Wins",
    "uncensored_mse": "Uncensored MSE",
    "rank_uncensored_mse": "Uncensored MSE Rank",
    "wins_uncensored_mse": "Uncensored MSE Wins",
    "normalized_rmse": "Normalized RMSE",
    "rank_normalized_rmse": "Normalized RMSE Rank",
    "wins_normalized_rmse": "Normalized RMSE Wins",
    "normalized_mse": "Normalized MSE",
    "rank_normalized_mse": "Normalized MSE Rank",
    "wins_normalized_mse": "Normalized MSE Wins",
    "demographic_statistical Parity": "Demographic Statistical Parity"
}
