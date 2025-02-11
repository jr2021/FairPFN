from __future__ import annotations

from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
)
import numpy as np


class SafePowerTransformer(PowerTransformer):
    """
    Power Transformer which reverts features back to their original values if they are
    transformed to very large values or the output column does not have unit variance.
    This happens e.g. when the input data has a large number of outliers.
    """

    def __init__(self, variance_threshold=1e-3, large_value_threshold=100, **kwargs):
        super().__init__(**kwargs)
        self.variance_threshold = variance_threshold
        self.large_value_threshold = large_value_threshold

    def _check_and_revert_features(self, transformed_X, original_X):
        # Calculate the variance for each feature in the transformed data
        variances = np.nanvar(transformed_X, axis=0)

        # Identify features where the variance is not close to 1
        non_unit_variance_indices = np.where(
            np.abs(variances - 1) > self.variance_threshold
        )[0]

        # Identify features with values greater than the large_value_threshold
        large_value_indices = np.any(transformed_X > self.large_value_threshold, axis=0)
        large_value_indices = np.nonzero(large_value_indices)[0]

        # Identify features to revert based on either condition
        revert_indices = np.unique(
            np.concatenate([non_unit_variance_indices, large_value_indices])
        )

        # Replace these features with the original features
        if len(revert_indices) > 0:
            transformed_X[:, revert_indices] = original_X[:, revert_indices]

        return transformed_X

    def fit_transform(self, X, y=None):
        # Fit the model and transform the input data
        transformed_X = super().fit_transform(X, y)

        # Check and revert features as necessary
        return self._check_and_revert_features(transformed_X, X)

    def transform(self, X):
        # Transform the input data
        transformed_X = super().transform(X)

        # Check and revert features as necessary
        return self._check_and_revert_features(transformed_X, X)


class NanHandlingPolynomialFeatures(PolynomialFeatures):
    """
    PolynomialFeatures class which handles NaN values in the input data.
    Nans are propagated to the polynomial features if either of the original features is NaN.
    """

    def fit_transform(self, X, y=None, **fit_params):
        X_ = X.copy()
        X[np.isnan(X)] = 0

        # Call the parent class's fit_transform method
        poly_features = super().fit_transform(X, **fit_params)

        # Identify the indices of NaN values in the original array
        nan_indices = np.isnan(X_)

        # Iterate over polynomial feature columns to identify those that should be NaN
        n_base_features = X.shape[-1]
        for i in range(n_base_features):
            for j in range(i, n_base_features):
                col_idx = self._get_col_idx(i, j, n_base_features)
                nan_idx_i = nan_indices[:, i]
                nan_idx_j = nan_indices[:, j]

                # If either feature i or j has a NaN, set the corresponding poly feature to NaN
                poly_features[nan_idx_i | nan_idx_j, col_idx] = np.nan

        return poly_features

    def _get_col_idx(self, i, j, n_base_features):
        # Method to calculate the column index in the polynomial features array
        return n_base_features + i * (2 * n_base_features - i + 1) // 2 + j - i
