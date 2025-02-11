from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    f1_score,
)

"""
===============================
Classification calculation
===============================
"""


def automl_benchmark_metric(target, pred, numpy=False, should_raise=False):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    if len(lib.unique(target)) > 2:
        return -cross_entropy(target, pred)
    else:
        return auc_metric_ovr(target, pred, numpy=numpy, should_raise=should_raise)


def auc_metric_ovr(target, pred, numpy=False, should_raise=False):
    return auc_metric(
        target, pred, multi_class="ovr", numpy=numpy, should_raise=should_raise
    )


def auc_metric_ovo(target, pred, numpy=False, should_raise=False):
    return auc_metric(
        target, pred, multi_class="ovo", numpy=numpy, should_raise=should_raise
    )


def auc_metric(target, pred, multi_class="ovo", numpy=False, should_raise=False):
    lib = np if numpy else torch

    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    if len(lib.unique(target)) > 2:
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
        return roc_auc_score(target, pred, multi_class=multi_class)
    else:
        if len(pred.shape) == 2:
            pred = pred[:, 1] # FIXME-Jake
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred))
        return roc_auc_score(target, pred)


def accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(accuracy_score(target, pred[:, 1] > 0.5))


def f1_metric(target, pred, multi_class="micro"):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(
            f1_score(target, torch.argmax(pred, -1), average=multi_class)
        )
    else:
        return torch.tensor(f1_score(target, pred[:, 1] > 0.5))


def average_precision_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(average_precision_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(average_precision_score(target, pred[:, 1] > 0.5))


def balanced_accuracy_metric(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        return torch.tensor(balanced_accuracy_score(target, torch.argmax(pred, -1)))
    else:
        return torch.tensor(balanced_accuracy_score(target, pred[:, 1] > 0.5))


def cross_entropy(target, pred):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(torch.unique(target)) > 2:
        ce = torch.nn.CrossEntropyLoss()
        return ce(pred.float().log(), target.long())
    else:
        bce = torch.nn.BCELoss()
        return bce(pred[:, 1].float(), target.float())


def is_classification(metric_used):
    if metric_used == auc_metric or metric_used == cross_entropy:
        return True
    return False


def nll_bar_dist(target, pred, bar_dist):
    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    target, pred = target.unsqueeze(0).to(bar_dist.borders.device), pred.unsqueeze(
        1
    ).to(bar_dist.borders.device)

    l = bar_dist(pred.log(), target).mean().cpu()
    return l


def expected_calibration_error(target, pred, norm="l1", n_bins=10):
    import torchmetrics

    target = torch.tensor(target) if not torch.is_tensor(target) else target
    pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred

    ece = torchmetrics.classification.MulticlassCalibrationError(
        n_bins=n_bins,
        norm=norm,
        num_classes=len(torch.unique(target)),
    )
    return ece(
        target=target,
        preds=pred,
    )


def is_imbalanced(y, threshold=0.8):
    """
    Determine if a numpy array of class labels is imbalanced based on Gini impurity.

    Parameters:
    - y (numpy.ndarray): A 1D numpy array containing class labels.
    - threshold (float): Proportion of the maximum Gini impurity to consider as the boundary
                         between balanced and imbalanced. Defaults to 0.8.

    Returns:
    - bool: True if the dataset is imbalanced, False otherwise.

    Example:
    >>> y = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    >>> is_imbalanced(y)
    True
    """

    # Calculate class proportions
    _, class_counts = np.unique(y, return_counts=True)
    class_probs = class_counts / len(y)

    # Calculate Gini impurity
    gini = 1 - np.sum(class_probs**2)

    # Determine max possible Gini for the number of classes
    C = len(class_probs)
    max_gini = 1 - 1 / C

    # Check if the Gini impurity is less than the threshold of the maximum possible Gini
    return gini < threshold * max_gini
