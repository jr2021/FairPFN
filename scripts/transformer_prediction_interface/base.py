from __future__ import annotations

import traceback
import warnings
import torch
import random
import tqdm
from copy import copy
import math
import os

import utils
from utils import normalize_data, print_once, NOP, to_tensor
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Union, Tuple, Any, Dict, List, Literal

from .configs import TabPFNModelPathsConfig

from .model_builder import load_model

from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    PolynomialFeatures,
    OrdinalEncoder,
)
from .preprocessing import SafePowerTransformer, NanHandlingPolynomialFeatures
import pickle as pkl


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
import itertools
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import (
    check_array,
    check_consistent_length,
)
from .configs import PreprocessorConfig, get_params_from_config
from model.bar_distribution import FullSupportBarDistribution

from model.encoders import ProtAttrEncoderStep

LOG_MEMORY_USAGE_PATH = None  # Disables logging, could be "/work/dlclarge1/hollmann-PFN_Tabular/memory_usage.csv"

# a flag needed to allow displaying the user a warning if we need to change the model to work on CPU
# to be removed when pytorch fixes the issue
did_change_model = False


class TabPFNBaseModel(BaseEstimator):
    def _more_tags(self):
        return {"allow_nan": True}

    models_in_memory = {}
    semisupervised_indicator = np.nan

    def __init__(
        self,
        model: Optional[Any] = None,
        device: str = "cpu",
        model_string: str = "",
        batch_size_inference: int = None,
        fp16_inference: bool = False,
        inference_mode: bool = True,
        c: Optional[Dict] = None,
        N_ensemble_configurations: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: Literal[
            "shuffle", "none", "local_shuffle", "rotate", "auto_rotate"
        ] = "none",  # local_shuffle breaks, because no local configs are generated with high feature number # FairPFN
        normalize_with_test: bool = False,
        average_logits: bool = False,
        categorical_features: Tuple[str, ...] = tuple(),
        optimize_metric: Optional[str] = None,
        seed: Optional[int] = 0,
        transformer_predict_kwargs: Optional[Dict] = None,
        show_progress: bool = True,
        sklearn_compatible_precision: bool = False,
        model_name: str = "tabpfn",  # This name will be tracked on wandb
        softmax_temperature: Optional[float] = math.log(0.8),
        save_peak_memory: Literal["True", "False", "auto"] = "True",
        maximum_free_memory_in_gb: Optional[float] = None,
        apply_pca=False,
        use_poly_features=False,
        max_poly_features=None,
        sample_random_poly_features=True,
        computational_budget=1e14,
        transductive=False,
        random_feature_scaling_strength=0.0,
        auxiliary_clf: tuple(str, str) | None = None,
        regression_y_preprocess_transforms: Optional[Tuple[Optional[str], ...]] = (
            None,
            "power",
        ),
        num_prot_attrs: int = 1
    ) -> None:
        """
        You need to specify a model either by setting the `model_string` or by setting `model` and `c`,
        where the latter is the config.
        :param model_string: The model string is the path to the model
        :param no_preprocess_mode: If True, the input will not be preprocessed
        :param preprocess_transforms: A tuple of strings, specifying the preprocessing steps to use.
        You can use the following strings as elements '(none|power|quantile|robust)[_all][_and_none]', where the first
        part specifies the preprocessing step and the second part specifies the features to apply it to and
        finally '_and_fnone' specifies that the original features should be added back to the features in plain.
        Finally, you can combine all strings without `_all` with `_onehot` to apply one-hot encoding to the categorical
        features specified with `self.fit(..., categorical_features=...)`.
        :param feature_shift_decoder: ["False", "True", "auto"] Whether to shift features for each ensemble configuration
        :param model: The model, if you want to specify it directly, this is used in combination with c
        :param c: The config, if you want to specify it directly, this is used in combination with model
        :param seed: The default seed to use for the order of the ensemble configurations, a seed of None will not
        :param device: The device to use for inference
        :param fp16_inference: Whether to use fp16 for inference on GPU, does not affect CPU inference.
        :param device: The device to use for inference
        :param inference_mode: Whether to use inference mode, which does not allow to backpropagate through the model.
        :param N_ensemble_configurations: The number of ensemble configurations to use, the most important setting
        :param batch_size_inference: The batch size to use for inference, this does not affect the results, just the
            memory usage and speed. A higher batch size is faster but uses more memory. Setting the batch size to None
            means that the batch size is automatically determined based on the memory usage and the maximum free memory
            specified with `maximum_free_memory_in_gb`.
        :param normalize_with_test: If True, the test set is used to normalize the data, otherwise the training set is used only.
        :param average_logits: Whether to average logits or probabilities for ensemble members
        :param sklearn_compatible_precision: This rounds predictions to 8 decimals, so that they are deterministic (numeric instability in torch)
            check_methods_sample_order_invariance does not pass, because numerically the order of samples  has an effect of the order of 1e-8
        :param save_peak_memory: Whether to save the peak memory usage of the model, can enable up to 8 times larger datasets to fit into memory
        :param maximum_free_memory_in_gb: How much memory to use at most in GB, if None, the memory usage will be calculated based on
           an estimation of the systems free memory. For CUDA will use the free memory of the GPU. For CPU will default to 32GB.
        :param apply_pca: Whether to apply PCA to the features
        :param use_bagging: If True, does bagging with the samples, with n_estimators number of bags and max_samples*samples samples per bag
        :param use_poly_features: Whether to use polynomial features as the last preprocessing step
        :param max_poly_features: Maximum number of polynomial features to use, None means unlimited
        :param sample_random_poly_features: Whether to randomly sample the polynomial features if the number of of polynomial features exceed max_poly_features. If False, the first max_poly_features are used whenever limit is exceeded.

        set the seed, this yields non-deterministic results but improves yields clearer results
        (smaller conf intervals across seeds)
        """
        self.device = device
        self.model = model
        self.c = c
        self.N_ensemble_configurations = N_ensemble_configurations
        self.model_string = model_string
        self.batch_size_inference = batch_size_inference
        self.fp16_inference = fp16_inference

        self.feature_shift_decoder = feature_shift_decoder
        self.seed = seed
        self.inference_mode = inference_mode
        self.softmax_temperature = softmax_temperature
        self.normalize_with_test = normalize_with_test
        self.average_logits = average_logits
        self.categorical_features = categorical_features
        self.optimize_metric = optimize_metric
        self.transformer_predict_kwargs = transformer_predict_kwargs
        self.preprocess_transforms = preprocess_transforms
        self.show_progress = show_progress
        self.sklearn_compatible_precision = sklearn_compatible_precision
        self.model_name = model_name

        self.save_peak_memory = save_peak_memory
        self.maximum_free_memory_in_gb = maximum_free_memory_in_gb

        self.apply_pca = apply_pca
        self.use_poly_features = use_poly_features
        self.max_poly_features = max_poly_features
        self.sample_random_poly_features = sample_random_poly_features
        self.computational_budget = computational_budget
        self.transductive = transductive
        self.random_feature_scaling_strength = random_feature_scaling_strength

        self.auxiliary_clf = auxiliary_clf
        self.regression_y_preprocess_transforms = regression_y_preprocess_transforms

        self.num_prot_attrs = num_prot_attrs

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features

    def set_protected_attributes(self, num_prot_attrs):
        self.num_prot_attrs = num_prot_attrs
        self.init_prot_attr_encoder_step()

    def _init_rnd(self):
        seed = self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
        self.rnd = np.random.default_rng(seed)

    def init_prot_attr_encoder_step(self) -> None:
        if self.model_processed_ is None:
            raise ValueError("Model needs to be initialized first")
        else:
            for encoder in self.model_processed_.encoder: # TODO check if name is correct
                
                if isinstance(encoder, ProtAttrEncoderStep):
                    encoder.num_prot_attrs = self.num_prot_attrs

    def init_model_and_get_model_config(self) -> None:
        """
        Initialize the model and its associated configuration.

        It loads the model and its configuration into memory. If the device is CPU, it also resets the attention bias
        in PyTorch's MultiheadAttention module if needed.
        """
        self.normalize_std_only_ = False
        self._init_rnd()
        # Model file specification (Model name, Epoch)
        if self.model is None:
            model_key = (self.model_string, self.device)
            if model_key in TabPFNBaseModel.models_in_memory:
                (
                    self.model_processed_,
                    self.c_processed_,
                ) = TabPFNBaseModel.models_in_memory[model_key]
            else:
                self.model_processed_, self.c_processed_ = load_model(
                    self.model_string, self.device, verbose=False
                )
                TabPFNBaseModel.models_in_memory[model_key] = (
                    self.model_processed_,
                    self.c_processed_,
                )
        else:
            assert (
                self.c is not None and len(self.c.keys()) > 0
            ), "If you specify the model you need to set the config, c"
            self.model_processed_ = self.model
            self.c_processed_ = self.c
        # style, temperature = self.load_result_minimal(style_file, i, e)

        # Set variables
        self.is_classification_ = True

        if not self.is_classification_:
            self.num_classes_ = None  # this defines the end of the indexes used in softmax, we just want all and [:None] is the whole range.

        if self.device.startswith("cpu"):
            global did_change_model
            did_change_model = False

            def reset_attention_bias(module):
                global did_change_model
                if isinstance(module, torch.nn.MultiheadAttention):
                    if module.in_proj_bias is None:
                        module.in_proj_bias = torch.nn.Parameter(
                            torch.zeros(3 * module.kdim)
                        )
                        did_change_model = True
                    if module.out_proj.bias is None:
                        module.out_proj.bias = torch.nn.Parameter(
                            torch.zeros(1 * module.kdim)
                        )
                        did_change_model = True

            self.model_processed_.apply(reset_attention_bias)
            if did_change_model:
                print_once(
                    "Changed model to be compatible with CPU, this is needed for the current version of "
                    "PyTorch, see issue: https://github.com/pytorch/pytorch/issues/97128. "
                    "The model will be slower if reused on GPU."
                )

        if hasattr(self.model_processed_, "to_device_for_forward"):
            self.model_processed_.to_device_for_forward(self.device)
        else:
            self.model_processed_.to(self.device)
        self.model_processed_.eval()

        self.max_num_features_ = self.c_processed_.get("num_features")
        if self.max_num_features_ is None:
            self.max_num_features_ = self.c_processed_.get(
                "max_num_features_in_training"
            )
            if self.max_num_features_ is None:
                print("falling back to unlimited features")
                self.max_num_features_ = -1
        if self.c_processed_.get("use_per_feature_transformer", False):
            self.max_num_features_ = -1  # Unlimited features

        self.differentiable_hps_as_style_ = self.c_processed_[
            "differentiable_hps_as_style"
        ]

        if self.feature_shift_decoder in ("local_shuffle", "auto_rotate"):
            self.feature_shift_decoder_max_perm_ = (
                0
                if self.c_processed_.get("features_per_group", self.max_num_features_)
                == 1
                else self.c_processed_.get("features_per_group", self.max_num_features_)
            )
        elif self.feature_shift_decoder in ("none", "shuffle", "rotate"):
            self.feature_shift_decoder_max_perm_ = 100000
        else:
            raise ValueError(
                f"Unknown feature_shift_decoder {self.feature_shift_decoder}"
            )

        style = None  # Currently we do not support style, code left for later usage
        self.num_styles_, self.style_ = self.init_style(
            style, differentiable_hps_as_style=self.differentiable_hps_as_style_
        )

        self._poly_ignore_nan_features = False
        self._poly_degree = 2
        self.semisupervised_enabled_ = self.c_processed_.get(
            "semisupervised_enabled", False
        )
        self.adapt_encoder_with_feature_scaling_strength()

        self.preprocess_transforms = [
            PreprocessorConfig(**p) if type(p) is dict else p
            for p in self.preprocess_transforms
        ]

        self.init_prot_attr_encoder_step()

    def adapt_encoder_with_feature_scaling_strength(self):
        if self.random_feature_scaling_strength == 0.0:
            return

        norm_layer = [
            e
            for e in self.model_processed_.encoder
            if "InputNormalizationEncoderStep" in str(e.__class__)
        ][0]
        norm_layer.random_feature_scaling_strength = (
            self.random_feature_scaling_strength
        )
        norm_layer.seed = self.seed
        norm_layer.reset_seed()

    def get_save_peak_memory(self, X, **overwrite_params) -> bool:
        if self.save_peak_memory == "True":
            return True
        elif self.save_peak_memory == "False":
            return False
        elif self.save_peak_memory == "auto":
            return (
                self.estimate_memory_usage(
                    X, "gb", save_peak_mem_factor=False, **overwrite_params
                )
                > self.get_max_free_memory_in_gb()
            )
        else:
            raise ValueError(
                f"Unknown value for save_peak_memory {self.save_peak_memory}"
            )

    def get_batch_size_inference(self, X, **overwrite_params) -> int:
        safety_factor = 2.0

        if self.batch_size_inference is None:
            if self.device.startswith("cpu"):
                return 1  # No batching on CPU
            capacity = self.get_max_free_memory_in_gb()
            usage = self.estimate_memory_usage(
                X, "gb", batch_size_inference=1, **overwrite_params
            )
            estimated_max_size = math.floor(capacity / usage / safety_factor)
            estimated_max_size = max(1, estimated_max_size)

            return estimated_max_size
        else:
            return self.batch_size_inference

    def is_initialized(self):
        return hasattr(self, "model_processed_")

    @staticmethod
    def check_training_data(
        clf: TabPFNBaseModel, X: np.ndarray, y: np.ndarray
    ) -> Tuple:
        """
        Validates the training data X and y.

        Raises:
            ValueError: If the number of features in X exceeds the maximum allowed features.

        Args:
            X (ndarray): The feature matrix.
            y (ndarray): The target vector.
        """
        if clf.max_num_features_ > -1 and X.shape[1] > clf.max_num_features_:
            raise ValueError(
                "The number of features for this classifier is restricted to ",
                clf.max_num_features_,
            )

        X = check_array(
            X, accept_sparse="csr", dtype=np.float32, force_all_finite=False
        )
        y = check_array(y, ensure_2d=False, dtype=np.float32, force_all_finite=False)

        check_consistent_length(X, y)

        return X, y

    def fit(self, X, y, additional_y=None) -> TabPFNBaseModel:
        """
        This function only checks the inputs and stores them in the object.
        :param X: A numpy array of shape (n_samples, n_features)
        :param y: A numpy array of shape (n_samples,)
        :param categorical_features: A tuple of the indexes of categorical features in [0, n_features - 1]
        :param categorical_features: A tuple of the indexes of categorical features in [0, n_features - 1]
        :return: self
        """
        # Must not modify any input parameters
        self.init_model_and_get_model_config()

        X, y = self.check_training_data(self, X, y)
        self.n_features_in_ = X.shape[1]

        X = X.astype(np.float32)
        self.X_ = X if torch.is_tensor(X) else torch.tensor(X)
        self.y_ = y if torch.is_tensor(y) else torch.tensor(y)

        if additional_y is not None:
            assert type(additional_y) == dict
            for k, v in additional_y.items():
                additional_y[k] = v if torch.is_tensor(v) else torch.tensor(v)
        self.additional_y_ = additional_y

        if self.auxiliary_clf:
            import sklearn

            self.auxiliary_clf_ = self.get_aux_clf()
            self.auxiliary_clf_feature_train = (
                sklearn.model_selection.cross_val_predict(
                    self.auxiliary_clf_,
                    X,
                    y,
                    method=self.auxiliary_clf[1],
                    cv=10,
                    verbose=1,
                )
            )
            self.auxiliary_clf_.fit(X, y)

        # Return the classifier
        return self

    def get_aux_clf(self):
        raise NotImplementedError

    def _transform_with_PCA(self, X):
        try:
            U, S, _ = torch.pca_lowrank(torch.squeeze(X))
            return torch.unsqueeze(U * S, 1)
        except:
            return X

    def _get_columns_with_nan(self, eval_xs: torch.Tensor) -> np.ndarray:
        nan_columns = np.isnan(eval_xs).any(axis=0)
        return np.where(nan_columns)[0]

    def _compute_poly_features(self, eval_xs: torch.Tensor):
        if eval_xs.shape[0] == 0 or eval_xs.shape[-1] == 0:
            return eval_xs

        try:
            # Custom Poly Implementation handles nan values
            poly_features = NanHandlingPolynomialFeatures(
                degree=self._poly_degree, include_bias=False
            )
        # PolynomialFeatures crashes for float32 overflows, which we want to catch
        except FloatingPointError:
            return eval_xs
        n_base_features = eval_xs.shape[-1]

        eval_xs_no_nan = eval_xs
        nan_columns = self._get_columns_with_nan(eval_xs)

        if self._poly_ignore_nan_features:
            if len(nan_columns) == eval_xs.shape[-1]:
                warnings.warn("All columns have NaNs. Skipping Polynomial Features.")
                return eval_xs
            elif len(nan_columns) > 0:
                warnings.warn(
                    f"There are {len(nan_columns)} columns with NaNs in the data ({nan_columns}). Ignoring these columns when computing Polynomial Features"
                )
                eval_xs_no_nan = eval_xs[
                    :, [i for i in range(eval_xs.shape[-1]) if i not in nan_columns]
                ]

        poly_features_xs = poly_features.fit_transform(eval_xs_no_nan)[
            :, n_base_features:
        ]

        if self.max_poly_features is None:
            return np.hstack((eval_xs, poly_features_xs))

        n_poly_features = poly_features_xs.shape[-1]

        if n_poly_features > self.max_poly_features:
            if self.sample_random_poly_features:
                random_poly_features_idx = self.rnd.choice(
                    np.arange(n_base_features, n_base_features + n_poly_features),
                    size=self.max_poly_features,
                    replace=False,
                ).tolist()
                keep_features_idx = (
                    list(range(n_base_features)) + random_poly_features_idx
                )
            else:
                keep_features_idx = list(
                    range(n_base_features + self.max_poly_features)
                )

            eval_xs = np.hstack((eval_xs, poly_features_xs))[:, keep_features_idx]
            return eval_xs

        eval_xs = np.hstack((eval_xs, poly_features_xs))

        return eval_xs

    def preprocess_input(
        self,
        eval_xs: torch.Tensor,
        preprocess_transform: PreprocessorConfig,
        eval_position: int,
    ) -> torch.Tensor:
        """
        Preprocesses the input feature tensor based on the specified transformation.

        Args:
            eval_xs (Tensor): Input feature tensor for evaluation.
            preprocess_transform (str): The type of preprocessing transform to apply.
            eval_position (int): Index position where the training set ends and the test set begins.

        Returns:
            Tensor: The preprocessed input tensor.

        Raises:
            Exception: For unsupported preprocess_transform or dimension mismatch.

        Should be invariant to the order of the samples, features and be deterministic:
        ```
        order = np.arange(X.shape[0])
        np.random.shuffle(order)

        t = estimator.preprocess_input(estimator.X_.unsqueeze(1), preprocess_transform='none', eval_position=30)
        t1 = estimator.preprocess_input(estimator.X_[order].unsqueeze(1), preprocess_transform='none', eval_position=30)
        t[order] == t1
        ```
        """
        assert self.normalize_with_test is False, "Not implemented yet"
        import warnings

        if self.feature_shift_decoder != "none":
            assert [
                "onehot" not in preprocess_transform.categorical_name
            ], "Feature shift decoder is not compatible with one hot encoding"

        if len(eval_xs.shape) != 3:
            raise Exception("Input must be 3D tensor (seq_len, batch_size, num_feats)")
        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")

        all_preprocessors = {
            "power": PowerTransformer(standardize=True),
            "safepower": SafePowerTransformer(standardize=True),
            "power_box": PowerTransformer(standardize=True, method="box-cox"),
            "safepower_box": SafePowerTransformer(standardize=True, method="box-cox"),
            "quantile_uni_coarse": QuantileTransformer(
                output_distribution="uniform", n_quantiles=eval_xs.shape[0] // 10
            ),
            "quantile_norm_coarse": QuantileTransformer(
                output_distribution="normal", n_quantiles=eval_xs.shape[0] // 10
            ),
            "quantile_uni": QuantileTransformer(
                output_distribution="uniform", n_quantiles=eval_xs.shape[0] // 5
            ),
            "quantile_norm": QuantileTransformer(
                output_distribution="normal", n_quantiles=eval_xs.shape[0] // 5
            ),
            "quantile_uni_fine": QuantileTransformer(
                output_distribution="uniform", n_quantiles=eval_xs.shape[0]
            ),
            "quantile_norm_fine": QuantileTransformer(
                output_distribution="normal", n_quantiles=eval_xs.shape[0]
            ),
            "robust": RobustScaler(unit_variance=True),
            "none": NOP(),
            # "robust": RobustScaler(unit_variance=True),
        }

        # if not per feature select provided preprocess transform
        if "per_feature" not in preprocess_transform.name:
            pt = all_preprocessors[preprocess_transform.name]

        eval_xs = eval_xs[:, 0, :]
        eval_xs_original = eval_xs.clone()

        # print('preprocess_input_transforms before', eval_xs[:5])
        # print('preprocess_input_transforms before SHAPE', eval_xs.shape)

        # removes empty features with all the same value
        sel = [
            len(torch.unique(eval_xs[0 : eval_xs.shape[0], col])) > 1
            for col in range(eval_xs.shape[1])
        ]

        # records indices of categorical
        categorical_features = (
            [] if self.categorical_features is None else self.categorical_features
        )

        # records indices of non-empty categorical features
        categorical_feats = [
            i for i, idx in enumerate(np.where(sel)[0]) if idx in categorical_features
        ]

        # selects only non-empty features
        eval_xs = eval_xs[:, sel].cpu().numpy().astype(np.float64)

        warnings.simplefilter("error")

        # print('preprocess_transform', preprocess_transform)

        # performs preporcessing
        if preprocess_transform.name != "none":

            # preprocess all features
            if preprocess_transform.categorical_name == "numeric":
                feats = set(range(eval_xs.shape[1]))
                if not preprocess_transform.append_original:
                    categorical_feats = []
            # preprocess only numerical features
            else:
                feats = set(range(eval_xs.shape[1])) - set(categorical_feats)
            extra_cols = []

            # iterate over selected features
            for col in feats:

                # if per feature select a random preprocessing transform
                if "per_feature" == preprocess_transform.name:
                    #
                    random_preprocessor = self.rnd.choice(
                        list(all_preprocessors.keys())
                    )
                    pt = all_preprocessors[random_preprocessor]


                if pt.__class__.__name__ == "NOP":
                    continue
                
                # apply 
                try:
                    pt.fit(eval_xs[:eval_position, col : col + 1])
                    trans = pt.transform(eval_xs[:, col : col + 1])

                    # save original columns
                    if preprocess_transform.append_original:
                        extra_cols += [trans]

                    # replace original columns
                    else:
                        eval_xs[:, col : col + 1] = trans
                except Exception as e:
                    pass
                    # print(
                    #    f"Failed to transform column {preprocess_transform} {np.where(sel)[0][col]}. Ignoring. {e}"
                    # )
            eval_xs = np.concatenate([eval_xs] + extra_cols, axis=1)

        if preprocess_transform.categorical_name == "ordinal":
            # Create a column transformer
            ct = ColumnTransformer(
                [
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),  # 'sparse' has been deprecated
                        categorical_feats,
                    )
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            ct.fit(eval_xs[:eval_position, :])
            eval_xs = ct.transform(eval_xs)
        
        elif preprocess_transform.categorical_name == "onehot":
            # Create a column transformer
            ct = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            sparse_output=False, handle_unknown="ignore"
                        ),  # 'sparse' has been deprecated
                        categorical_feats,
                    )
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            ct.fit(eval_xs[:eval_position, :])
            eval_xs_ = ct.transform(eval_xs)
            if eval_xs_.size < 1_000_000:
                print(
                    f"applied onehot encoding {eval_xs.shape} -> {eval_xs_.shape} (Categorical features: {categorical_feats})"
                )
                eval_xs = eval_xs_
                categorical_feats = list(range(eval_xs.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
            else:
                print("Because of memory constraints, onehot encoding is not applied.")

        if self.use_poly_features:
            eval_xs = self._compute_poly_features(eval_xs)

        try:
            eval_xs = torch.tensor(eval_xs.astype(np.float32)).float()
        except Exception:
            # This can fail due to overflow errors which would end the entire evaluation
            eval_xs = eval_xs_original

        warnings.simplefilter("default")

        eval_xs = eval_xs.unsqueeze(1)

        if self.apply_pca is True:
            eval_xs = self._transform_with_PCA(eval_xs)

        # print('preprocess_input_transforms after', eval_xs[:5])
        # print('preprocess_input_transforms after SHAPE', eval_xs.shape)

        return eval_xs.detach(), categorical_feats

    def init_style(self, style, differentiable_hps_as_style=True):
        if not differentiable_hps_as_style:
            style = None

        if style is not None:
            style = style
            style = style.unsqueeze(0) if len(style.shape) == 1 else style
            num_styles = style.shape[0]
        else:
            num_styles = 1
            style = None

        return num_styles, style

    @staticmethod
    def generate_shufflings(
        feature_n, n_configurations, rnd, shuffle_method="local_shuffle", max_step=10000
    ):
        if (
            max_step == 0
        ):  # this means that we use a per feature arch, which does not care about shuffling
            shuffle_method = "none"

        shuffle_method = "none" # FIXME 

        # print('shuffle method', shuffle_method)

        if shuffle_method == "auto_rotate" or shuffle_method == "rotate":

            def generate_shifting_permutations(n_features):
                initial = list(range(n_features))
                rotations = [initial]

                for i in range(1, min(min(n_features, max_step), n_configurations)):
                    rotated = initial[i:] + initial[:i]
                    rotations.append(rotated)

                return torch.tensor(rotations)

            feature_shift_configurations = generate_shifting_permutations(feature_n)
        elif shuffle_method == "shuffle":
            features_indices = list(range(feature_n))

            # Generate all permutations
            # all_permutations = itertools.permutations(features_indices)

            unique_shufflings = set()
            iterations = 0
            while (
                len(unique_shufflings) < n_configurations
                and iterations < n_configurations * 3
            ):
                shuffled = rnd.choice(
                    features_indices, size=len(features_indices), replace=False
                )
                unique_shufflings.add(tuple(shuffled))
                iterations += 1
            unique_shufflings = list(unique_shufflings)
            # Convert to PyTorch tensor
            feature_shift_configurations = torch.tensor(unique_shufflings)
            
        elif shuffle_method == "local_shuffle":
            unique_shufflings = set()
            features_indices = list(range(feature_n))
            iterations = 0

            while (
                len(unique_shufflings) < n_configurations
                and iterations < n_configurations * 3
            ):
                local_shuffle = [features_indices[0]]
                remaining_indices = features_indices[1:]

                for idx in local_shuffle:
                    possible_next_indices = [
                        x for x in remaining_indices if abs(idx - x) <= max_step
                    ]

                    if (
                        not possible_next_indices
                    ):  # No options within max_step, break the constraint
                        next_idx = rnd.choice(remaining_indices)
                    else:
                        next_idx = rnd.choice(possible_next_indices)

                    local_shuffle.append(next_idx)
                    remaining_indices.remove(next_idx)

                    if len(local_shuffle) == feature_n:
                        break

                unique_shufflings.add(tuple(local_shuffle))
                iterations += 1

            unique_shufflings = list(unique_shufflings)

            # Convert to PyTorch tensor
            feature_shift_configurations = torch.tensor(unique_shufflings)
        elif shuffle_method == "none":
            feature_shift_configurations = torch.tensor(
                list(range(feature_n))
            ).unsqueeze(0)
        else:
            raise ValueError(f"Unknown feature_shift_decoder {shuffle_method}")

        return feature_shift_configurations

    def get_ensemble_configurations(
        self,
        eval_xs: torch.Tensor,
        eval_ys: torch.Tensor,
    ) -> List[Tuple]:
        """
        Generate configurations for preprocessing that can be used for ensembling.

        Parameters:
            eval_xs (torch.Tensor): Input feature tensor.
            eval_ys (torch.Tensor): Input label tensor.
        Returns:
            List[Tuple]: List of ensemble configurations.
        """
        # Generate random configurations
        styles_configurations = range(0, self.num_styles_)

        max_extra_feats = self.max_poly_features if self.use_poly_features else 0
        feature_shift_configurations = TabPFNBaseModel.generate_shufflings(
            eval_xs.shape[-1] + max_extra_feats,
            rnd=self.rnd,
            n_configurations=self.get_max_N_ensemble_configurations(eval_xs),
            shuffle_method=self.feature_shift_decoder,
            max_step=self.feature_shift_decoder_max_perm_,
        )

        if self.is_classification_:
            class_shift_configurations = TabPFNBaseModel.generate_shufflings(
                self.num_classes_,
                rnd=self.rnd,
                n_configurations=self.get_max_N_ensemble_configurations(eval_xs),
                shuffle_method=self.multiclass_decoder,
                max_step=2,
            )
        else:
            class_shift_configurations = self.regression_y_preprocess_transforms

        shift_configurations = list(
            itertools.product(class_shift_configurations, feature_shift_configurations)
        )
        preprocess_transforms = self.preprocess_transforms
        if "per_feature" in [p.name for p in preprocess_transforms]:
            assert (
                len(preprocess_transforms) == 1
            ), "per_feature must be the only preprocess transform"
            preprocess_transforms = (
                "per_feature",
            ) * 9  # as each per_feature transform internally selects the feature at random, we want to allow to use bigger ensembles where multiple ensemble members have a preprocess_transform with the same name.

        self.rnd.shuffle(shift_configurations)
        ensemble_configurations = list(
            itertools.product(
                shift_configurations,
                preprocess_transforms,
                styles_configurations,
            )
        )

        return ensemble_configurations

    def preprocess_y(self, eval_ys, eval_position, configuration, bar_dist):
        assert (len(eval_ys.shape) == 2) and (
            eval_ys.shape[1] == 1
        ), f"only support (N, 1) shape, but got {eval_ys.shape}"

        eval_ys_orig = eval_ys

        if "power" in configuration:
            pt = PowerTransformer(standardize=True)
        elif "safepower" in configuration:
            pt = SafePowerTransformer(standardize=True)
        elif "quantile" in configuration:
            pt = QuantileTransformer(
                output_distribution="normal", n_quantiles=len(eval_ys) // 10
            )
        elif "robust" in configuration:
            pt = RobustScaler(unit_variance=True)
        else:
            raise ValueError(f"Unknown preprocessing {configuration}")
        try:
            print(f"before {eval_ys.squeeze(1)=}")
            pt.fit(eval_ys[:eval_position])
            eval_ys = pt.transform(eval_ys)
            new_borders = pt.inverse_transform(bar_dist.borders[:, None].cpu().numpy())[
                :, 0
            ]
            print(f"after {eval_ys.squeeze(1)=}, {eval_position=}, {eval_ys.shape=}")
            # print("transformed:", eval_ys.min(), eval_ys.max(), new_borders)
        except Exception as e:
            # print whole traceback
            traceback.print_exc()
            print("Failed to transform with", configuration)

        try:
            assert not np.isnan(eval_ys).any(), f"NaNs in transformed ys: {eval_ys}"
            assert not np.isnan(
                new_borders
            ).any(), f"NaNs in transformed borders: {new_borders}"
            assert not np.isinf(eval_ys).any(), f"Infs in transformed ys: {eval_ys}"
            assert not np.isinf(
                new_borders
            ).any(), f"Infs in transformed borders: {new_borders}"
            return (
                torch.tensor(eval_ys.astype(np.float32)),
                FullSupportBarDistribution(
                    torch.tensor(new_borders.astype(np.float32))
                ),
                pt,
            )
        except Exception:
            traceback.print_exc()
            print("Failed to go back with", configuration)
            # This can fail due to overflow errors which would end the entire evaluation
        return eval_ys_orig, bar_dist, None

    def build_transformer_input_for_each_configuration(
        self,
        ensemble_configurations: List[Tuple],
        eval_xs: torch.Tensor,
        eval_ys: torch.Tensor,
        eval_position: int,
        additional_ys: dict = None,
        bar_dist: Optional[FullSupportBarDistribution] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[int]],
        List,
        List,
        Dict[List[torch.Tensor]],
    ]:
        """
        Builds transformer inputs and labels based on given ensemble configurations.

        Parameters:
            ensemble_configurations (List[Tuple]): List of ensemble configurations.
            eval_xs (torch.Tensor): Input feature tensor.
            eval_ys (torch.Tensor): Input label tensor.
            eval_position (int): Position where the training set ends and the test set begins.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Transformed inputs and labels.
        """
        eval_xs_and_categorical_inds_transformed = {}
        inputs, labels, categorical_inds_list, adapted_bar_dists, additional_ys_out = (
            [],
            [],
            [],
            [],
            {k: [] for k in additional_ys} if additional_ys is not None else {},
        )
        y_transformers = []
        for (
            (class_shift_configuration, feature_shift_configuration),
            preprocess_transform_configuration,
            styles_configuration,
        ) in ensemble_configurations:
            # The feature shift configuration is a permutation of the features, but
            # during preprocessing additional features could be added
            assert feature_shift_configuration.shape[0] >= eval_xs.shape[-1]

            if additional_ys is not None:
                additional_ys_out = {
                    k: additional_ys_out[k] + [additional_ys[k]] for k in additional_ys
                }
            else:
                additional_ys_out = {}

            if self.is_classification_:
                # The class shift configuration is a permutation of the classes
                assert class_shift_configuration is None or (
                    class_shift_configuration.shape[0] == self.num_classes_
                )

            eval_ys_ = eval_ys.clone()
            if (
                preprocess_transform_configuration
                in eval_xs_and_categorical_inds_transformed
            ):
                eval_xs_, categorical_feats = eval_xs_and_categorical_inds_transformed[
                    preprocess_transform_configuration
                ]
                eval_xs_ = eval_xs_.clone()
                categorical_feats = copy(categorical_feats)
            else:
                eval_xs_ = eval_xs.clone()

                # print('calling preprocess_input')

                eval_xs_, categorical_feats = self.preprocess_input(
                    eval_xs_,
                    preprocess_transform=preprocess_transform_configuration,
                    eval_position=eval_position,
                )
                # eval_xs_, categorical_feats = eval_xs_.detach(), []
                eval_xs_and_categorical_inds_transformed[
                    preprocess_transform_configuration
                ] = (eval_xs_.clone(), copy(categorical_feats))

            # TODO: Now categorical feats are messed up, need to be reordered
            # Important: We are shuffling after preprocessing, so the shuffling is done on the transformed features
            # Potentially those are more than the input features
            feature_shift_configuration = feature_shift_configuration[
                np.where(feature_shift_configuration < eval_xs_.shape[-1])[0]
            ]
            eval_xs_ = eval_xs_[..., feature_shift_configuration]

            new_bar_dist = bar_dist
            y_transformer = None
            if class_shift_configuration is not None:
                if self.is_classification_:

                    def map_classes(tensor, permutation):
                        permutation_tensor = permutation.long()
                        return permutation_tensor[tensor.long()]

                    eval_ys_ = eval_ys_.long()
                    eval_ys_[eval_ys_ != self.semisupervised_indicator] = map_classes(
                        eval_ys_[eval_ys_ != self.semisupervised_indicator],
                        class_shift_configuration,
                    )
                else:
                    if isinstance(class_shift_configuration, float):
                        eval_ys_ = eval_ys_ * class_shift_configuration
                        new_bar_dist = FullSupportBarDistribution(
                            bar_dist.borders / class_shift_configuration
                        )
                    else:
                        assert bar_dist is not None
                        eval_ys_, new_bar_dist, y_transformer = self.preprocess_y(
                            eval_ys_, eval_position, class_shift_configuration, bar_dist
                        )

            # these are performed on the transformed inputs, which is not perfect, but should be fine
            # one hots might be split up for example

            inputs += [eval_xs_]
            labels += [eval_ys_]
            categorical_inds_list += [categorical_feats]
            adapted_bar_dists += [new_bar_dist]
            y_transformers += [y_transformer]

        return (
            inputs,
            labels,
            categorical_inds_list,
            adapted_bar_dists,
            y_transformers,
            additional_ys_out,
        )

    def build_transformer_input_in_batches(
        self,
        inputs: List[torch.Tensor],
        labels: List[torch.Tensor],
        categorical_inds_list: List[List[int]],
        batch_size_inference: int,
        additional_ys: dict = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, tuple, List[dict]]:
        """
        Partition inputs and labels into batches for transformer.

        Parameters:
            inputs (List[torch.Tensor]): List of input tensors.
            labels (List[torch.Tensor]): List of label tensors.
            batch_size_inference (int): Size of each inference batch.

        Returns:
            Tuple[inputs: List[torch.Tensor], labels: List[torch.Tensor], implied_permutation: torch.Tensor]: Partitioned inputs, labels, and permutations.
            Each items contains a list of tensors, where each tensor is a batch, with maximally batch_size_inference elements and the same number of features.
        """
        from collections import defaultdict, OrderedDict

        inds_for_each_feat_dim = defaultdict(list)
        for i, inp in enumerate(inputs):
            inds_for_each_feat_dim[inp.shape[-1]].append(i)
        inds_for_each_feat_dim = OrderedDict(inds_for_each_feat_dim)
        implied_permutation = torch.tensor(
            sum((inds for inds in inds_for_each_feat_dim.values()), [])
        )
        inputs = [
            torch.cat([inputs[i] for i in inds], dim=1)
            for inds in inds_for_each_feat_dim.values()
        ]

        inputs = sum(
            (torch.split(inp, batch_size_inference, dim=1) for inp in inputs), tuple()
        )

        labels = [
            torch.cat([labels[i] for i in inds], dim=1)
            for inds in inds_for_each_feat_dim.values()
        ]
        labels = sum(
            (torch.split(lab, batch_size_inference, dim=1) for lab in labels), tuple()
        )

        additional_ys_chunked = []
        if additional_ys is not None:
            additional_ys_chunked = {}
            for k in additional_ys.keys():
                additional_ys_key = additional_ys[k]
                additional_ys_key = [
                    torch.cat([additional_ys_key[i] for i in inds], dim=1)
                    for inds in inds_for_each_feat_dim.values()
                ]
                additional_ys_key = sum(
                    (
                        torch.split(additional_y, batch_size_inference, dim=1)
                        for additional_y in additional_ys_key
                    ),
                    tuple(),
                )
                additional_ys_chunked[k] = additional_ys_key
            # Assume all lists in the dictionary are of the same length
            list_length = len(labels)

            # Convert to list of dictionaries
            additional_ys_chunked = [
                {key: additional_ys_chunked[key][i] for key in additional_ys_chunked}
                for i in range(list_length)
            ]
        categorical_inds_list = [
            [categorical_inds_list[i] for i in inds]
            for inds in inds_for_each_feat_dim.values()
        ]

        categorical_inds_list = sum(
            (
                tuple(utils.chunks(cis, batch_size_inference))
                for cis in categorical_inds_list
            ),
            tuple(),
        )

        return (
            inputs,
            labels,
            implied_permutation,
            categorical_inds_list,
            additional_ys_chunked,
        )

    def reweight_probs_based_on_train_(
        self,
        eval_ys: torch.Tensor,
        output: torch.Tensor,
        num_classes: int,
        device: str,
    ) -> torch.Tensor:
        """
        Reweight class probabilities based on training set.

        Parameters:
            eval_ys (torch.Tensor): Label tensor for evaluation.
            output (torch.Tensor): Output probability tensor.
            num_classes (int): Number of classes.
            device (str): Computing device.

        Returns:
            torch.Tensor: Reweighted output tensor.
        """
        # make histogram of counts of each class in train set
        train_class_probs = torch.zeros(num_classes, device=device)
        train_class_probs.scatter_add_(
            0,
            eval_ys.flatten().long(),
            torch.ones_like(eval_ys.flatten(), device=device),
        )
        train_class_probs = (
            train_class_probs / train_class_probs.sum()
        )  # shape: num_classes
        output /= train_class_probs
        # make sure outputs last dim sums to 1
        output /= output.sum(dim=-1, keepdim=True)

        return output

    @staticmethod
    def reverse_permutation(output, permutation):
        """
        At the beginning of a prediction we create a permutation of the classes.
        We do this by simply indexing into the permutation tensor with the classes, i.e. `permutation[classes]`.

        Now output contains logits for the classes, but we want to get the logits for the original classes.
        That is why we reverse this permutation by indexing the other way around, `logits[permutation]`.

        The property we want is
        ```
        classes = torch.randint(100, (200,)) # 200 examples with random classes out of 100
        permutation = torch.randperm(100) # a random permutation of the classes

        # forward permutation
        permuted_classes = permutation[classes]
        perfect_predictions_on_permuted_classes = torch.nn.functional.one_hot(permuted_classes, 100)

        # backward permutation (this function)
        perfect_predictions_on_original_classes = perfect_predictions_on_permuted_classes[:, permutation]

        # now we want
        assert (perfect_predictions_on_original_classes == torch.nn.functional.one_hot(classes, 100)).all()
        ```

        We use the second dimension of the output tensor to index into the permutation tensor, as we have both batch and seq dimensions.

        :param output: tensor with shape (seq_len, batch_size, num_classes)
        :param permutation: tensor with shape (num_classes)
        :return:
        """
        return output[:, :, permutation]

    def predict_common_setup(self, X_eval, additional_y_eval=None):
        if additional_y_eval is None:
            additional_y_eval = {}

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X_eval = check_array(
            X_eval.cpu().detach().numpy(), accept_sparse="csr", dtype=np.float32, force_all_finite=False
        )

        if X_eval.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in the input data ({X_eval.shape[1]}) must match the number of features during training ({self.n_features_in_})."
            )

        X_eval = torch.tensor(X_eval)

        X_train, y_train, additional_y_train = (
            self.X_,
            self.y_,
            deepcopy(self.additional_y_),
        )

        ### Extend X, y and additional_y with the test set for transductive learning
        if self.auxiliary_clf is not None:
            auxiliary_clf_features_test = getattr(
                self.auxiliary_clf_, self.auxiliary_clf[1]
            )(X_eval.numpy())
            X_train = torch.cat(
                [X_train, torch.tensor(self.auxiliary_clf_feature_train)], dim=-1
            )
            X_eval = torch.cat(
                [X_eval, torch.tensor(auxiliary_clf_features_test)], dim=-1
            )

        ### Extend X, y and additional_y with the test set for transductive learning
        if self.transductive:
            X_train = torch.cat([X_train, X_eval], dim=0)
            y_train = torch.cat(
                [
                    y_train,
                    torch.ones_like(X_eval[:, 0]) * self.semisupervised_indicator,
                ],
                dim=0,
            )
            if additional_y_train is not None and len(additional_y_train.keys()) > 0:
                for k, v in additional_y_train.items():
                    additional_y_train[k] = torch.cat([v, additional_y_eval[k]], dim=0)

        ### Join train and test sets
        X_full = torch.cat([X_train, X_eval], dim=0).float().unsqueeze(1)
        y_full = (
            torch.cat([y_train, torch.zeros_like(X_eval[:, 0])], dim=0)
            .float()
            .unsqueeze(1)
        )
        if additional_y_eval:
            assert (
                additional_y_train
            ), "additional_y_train is None in fit but not predict"
            for k, v in additional_y_eval.items():
                additional_y_eval[k] = torch.cat(
                    [additional_y_train[k], v], dim=0
                ).float()
        else:
            assert (
                self.additional_y_ is None
            ), "additional_y is None in predict but not fit"

        ### Memory optimization
        save_peak_memory = self.get_save_peak_memory(X_full)
        if save_peak_memory:
            self.model_processed_.reset_save_peak_mem_factor(8)

        if self.estimate_memory_usage(X_full, "gb") > self.get_max_free_memory_in_gb():
            raise ValueError(
                f"Memory usage of the model is too high (Approximated Memory Usage (GB): {self.estimate_memory_usage(X_full, 'gb')}, Capacity {self.get_max_free_memory_in_gb()})."
            )

        eval_pos = X_train.shape[0]

        return X_full, y_full, additional_y_eval, eval_pos

    def get_max_N_ensemble_configurations(self, eval_xs):
        """
        Computes the maximum number of ensemble configurations that can be evaluated given the computational budget.
        If the number of ensemble configurations is specified, this is returned.

        Heuristics such as max_iterations and min_iterations are not properly tested yet.

        :param eval_xs:
        :return:
        """
        # TODO: The time depends a lot on the number of batches needed to evaluate, i.e.
        #    N_ensemble_configurations / self.get_batch_size_inference(eval_xs), should depend more on that
        if self.N_ensemble_configurations is not None:
            return self.N_ensemble_configurations

        max_iterations, min_iterations = 4096, 32
        batch_size_inference = self.get_batch_size_inference(eval_xs)

        per_sample_cost = self.estimate_computation_usage(eval_xs)
        per_sample_cost = per_sample_cost / batch_size_inference

        N_ensemble_configurations = int(self.computational_budget / per_sample_cost)
        N_ensemble_configurations = min(
            max_iterations, max(min_iterations, N_ensemble_configurations)
        )

        # Fill up the batch
        N_ensemble_configurations = max(batch_size_inference, N_ensemble_configurations)

        return N_ensemble_configurations

    def transformer_predict(
        self,
        eval_xs,  # shape (num_examples, [1], num_features)
        eval_ys,  # shape (num_examples, [1], [1])
        eval_position,
        bar_distribution: FullSupportBarDistribution | None = None,
        reweight_probs_based_on_train=False,
        additional_ys=None,
    ):
        """
        This function is used to make predictions inside `predict_proba`.
        :param eval_xs: A torch tensor of shape (num_examples, [1], num_features), where the first dim is optional
        :param eval_ys: A torch tensor of shape (num_examples, [1], [1]), where the first and second dim are optional. In the classification setting these are the labels in [0,num_classes).
        :param eval_position: The cut between training and test set, i.e. the number of training examples, train_xs = eval_xs[:eval_position]
        :param return_logits: Whether to return logits or probabilities
        :param reweight_probs_based_on_train: Whether to reweight the probabilities based on the training set, this is useful for imbalanced datasets and metrics that overemphasize the minority classes.
        :param kwargs:
        :return:
        """

        # Input validation
        assert (
            self.normalize_with_test is False
        ), "not supported right now, look into preprocess_input"

        assert (
            bar_distribution is None
        ) == self.is_classification_, (
            "bar_distribution needs to be set if and only if the model is a regressor"
        )

        # Store the classes seen during fit

        # Handle inputs with optional batch dim
        if len(eval_ys.shape) == 1:
            eval_ys = eval_ys.unsqueeze(1)
        if len(eval_xs.shape) == 2:
            eval_xs = eval_xs.unsqueeze(1)

        # Initialize model and device
        eval_ys = eval_ys[:eval_position]

        # Initialize list of preprocessings to check
        ensemble_configurations = self.get_ensemble_configurations(
            eval_xs,
            eval_ys,
        )
        N_ensemble_configurations = self.get_max_N_ensemble_configurations(eval_xs)

        ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]

        # print('transformer_predict', ensemble_configurations)

        ## Compute and save transformed inputs for each configuration
        (
            inputs,
            labels,
            categorical_inds,
            adapted_bar_dists,
            y_transformers,
            additional_ys,
        ) = self.build_transformer_input_for_each_configuration(
            ensemble_configurations=ensemble_configurations,
            eval_xs=eval_xs,
            eval_ys=eval_ys,
            eval_position=eval_position,
            bar_dist=bar_distribution,
            additional_ys=additional_ys,
        )

        ## Split inputs to model into chunks that can be calculated batchwise for faster inference
        ## We split based on the dimension of the features, so that we can feed in a batch
        (
            inputs,
            labels,
            implied_permutation,
            categorical_inds,
            additional_ys_inputs,
        ) = self.build_transformer_input_in_batches(
            inputs,
            labels,
            categorical_inds_list=categorical_inds,
            batch_size_inference=self.get_batch_size_inference(inputs[0]),
            additional_ys=additional_ys,
        )

        softmax_temperature = utils.to_tensor(
            self.softmax_temperature, device=self.device
        )

        style_ = self.style_.to(self.device) if self.style_ is not None else None

        ## Get predictions and save in intermediate tensor
        outputs = []
        for (
            batch_input,
            batch_label,
            batch_categorical_inds,
            batch_additional_ys,
        ) in tqdm.auto.tqdm(
            list(zip(inputs, labels, categorical_inds, additional_ys_inputs)),
            desc="Running inference",
            disable=not self.show_progress,
            unit="batch",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="None of the inputs have requires_grad=True. Gradients will be None",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
                )
                with torch.cuda.amp.autocast(enabled=self.fp16_inference):
                    inference_mode_call = (
                        torch.inference_mode() if self.inference_mode else NOP()
                    )
                    with inference_mode_call:
                        style_expanded = (
                            style_.repeat(batch_input.shape[1], 1)
                            if self.style_ is not None
                            else None
                        )
                        batch_additional_ys = (
                            {
                                k: v.to(self.device) if torch.is_tensor(v) else v
                                for k, v in batch_additional_ys.items()
                            }
                            if batch_additional_ys is not None
                            else {}
                        )
                        if self.is_classification_:
                            batch_label = batch_label.float()
                            batch_label[
                                batch_label == self.semisupervised_indicator
                            ] = np.nan
                        output = self.model_processed_(
                            (
                                style_expanded,
                                {"main": batch_input.to(self.device)},
                                {
                                    "main": batch_label.float().to(self.device),
                                    **batch_additional_ys,
                                },
                            ),
                            single_eval_pos=eval_position,
                            only_return_standard_out=False,
                            categorical_inds=batch_categorical_inds,
                        )
                        if isinstance(output, tuple):
                            output, output_once = output
                        standard_prediction_output = output["standard"]

                        # cut off additional logits for classes that do not exist in the dataset
                    standard_prediction_output = standard_prediction_output[
                        :, :, : self.num_classes_
                    ].float() / torch.exp(softmax_temperature)

            outputs += [standard_prediction_output]
        outputs = torch.cat(outputs, 1)
        # argsort of a permutation index yields the inverse
        outputs = outputs[:, torch.argsort(implied_permutation), :]

        ## Combine predictions
        ensemble_outputs = []
        for i, ensemble_configuration in enumerate(ensemble_configurations):
            (
                (class_shift_configuration, feature_shift_configuration),
                preprocess_transform_configuration,
                styles_configuration,
            ) = ensemble_configuration
            output_ = outputs[:, i : i + 1, :]

            if class_shift_configuration is not None:
                if self.is_classification_:
                    output_ = self.reverse_permutation(
                        output_, class_shift_configuration
                    )

            if not self.average_logits and self.is_classification_:
                output_ = torch.nn.functional.softmax(output_, dim=-1)
            ensemble_outputs += [output_]

        if self.is_classification_:
            outputs = torch.cat(ensemble_outputs, 1)
            output = torch.mean(outputs, 1, keepdim=True)

            if self.average_logits:
                output = torch.nn.functional.softmax(output, dim=-1)

            if reweight_probs_based_on_train:
                # This is used to reweight probabilities for when the optimized metric should give balanced predictions
                output = self.reweight_probs_based_on_train_(
                    eval_ys, output, self.num_classes_, self.device
                )

        else:
            # assert all(not torch.isnan(o).any() for o in ensemble_outputs)
            # assert all(not torch.isnan(d.borders).any() for d in adapted_bar_dists)
            output = bar_distribution.average_bar_distributions_into_this(
                [d.to(bar_distribution.borders.device) for d in adapted_bar_dists],
                ensemble_outputs,
            )
            # assert all(not torch.isnan(o).any() for o in output)
        output = torch.transpose(output, 0, 1)

        if not self.device.startswith("cpu") and LOG_MEMORY_USAGE_PATH:
            self._log_memory_usage(eval_xs)

        if not self.is_classification_:
            return output, y_transformers

        return output

    def get_max_free_memory_in_gb(self) -> float:
        """
        How much memory to use at most in GB, if None, the memory usage will be calculated based on
        an estimation of the systems free memory. For CUDA will use the free memory of the GPU.
        For CPU will default to 32 GB.

        Returns:
            float: The maximum memory usage in GB.
        """
        if self.maximum_free_memory_in_gb is None:
            # TODO: Get System Stats and adapt to free memory for default case

            if self.device.startswith("cpu"):
                try:
                    total_memory = (
                        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                    )
                except ValueError:
                    utils.print_once(
                        "Could not determine size of memory of this system, using default 8 GB"
                    )
                    total_memory = 8
                return total_memory
            elif self.device.startswith("cuda"):
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                f = t - a  # free inside reserved

                return f / 1e9
            else:
                raise ValueError(f"Unknown device {self.device}")
        else:
            return self.maximum_free_memory_in_gb

    def estimate_memory_usage(
        self,
        X: np.ndarray | torch.tensor,
        unit: Literal["b", "mb", "gb"] = "gb",
        **overwrite_params,
    ) -> float | None:
        """
        Estimates the memory usage of the model.

        Peak memory usage is accurate for ´save_peak_mem_factor´ in O(n_feats, n_samples) on average but with
        significant outliers (2x). Also this calculation does not include baseline usage and constant offsets.
        Baseline memory usage can be ignored if we set the maximum memory usage to the default None which uses
        the free memory of the system. The constant offsets are not significant for large datasets.

        Args:
            X (ndarray): The feature matrix.
            unit (str): The unit to return the memory usage in.

        Returns:
            int: The estimated memory usage in bytes.
        """
        byte_usage = self.estimate_model_usage(X, "memory", **overwrite_params)
        if byte_usage is None:
            return None

        if unit == "mb":
            return byte_usage / 1e6
        elif unit == "gb":
            return byte_usage / 1e9
        elif unit == "b":
            return byte_usage
        else:
            raise ValueError(f"Unknown unit {unit}")

    def estimate_computation_usage(self, X: np.ndarray) -> float | None:
        """
        Estimates the computation usage of the model per ensemble member.

        TODO: Computation usage is untested

        Args:
            X (ndarray): The feature matrix.

        Returns:
            int: The estimated computation usage in FLOPs.
        """
        return self.estimate_model_usage(X, "computation")

    def estimate_model_usage(
        self,
        X: np.ndarray | torch.tensor,
        estimation_task: Literal["memory", "computation"],
        **overwrite_params,
    ) -> int | None:
        if not self.is_initialized():
            self.init_model_and_get_model_config()
        fp_16 = overwrite_params.get(
            "fp_16", self.fp16_inference and not self.device.startswith("cpu")
        )
        num_heads = overwrite_params.get("num_heads", self.c_processed_.get("nhead"))
        num_layers = overwrite_params.get(
            "num_layers", self.c_processed_.get("nlayers")
        )
        embedding_size = overwrite_params.get(
            "embedding_size", self.c_processed_.get("emsize")
        )

        # These settings can be auto adapted
        if "save_peak_mem_factor" in overwrite_params:
            save_peak_mem_factor = overwrite_params["save_peak_mem_factor"]
        else:
            save_peak_mem_factor = self.get_save_peak_memory(X)
        per_feature_transformer = overwrite_params.get(
            "per_feature_transformer",
            self.c_processed_.get("use_per_feature_transformer"),
        )
        if "batch_size_inference" in overwrite_params:
            batch_size_inference = overwrite_params["batch_size_inference"]
        else:
            batch_size_inference = self.get_batch_size_inference(
                X, save_peak_mem_factor=save_peak_mem_factor
            )

        if len(X.shape) == 3:
            num_features = X.shape[2]
            num_samples = X.shape[0]
        elif len(X.shape) == 2:
            num_features = X.shape[1]
            num_samples = X.shape[0]
        else:
            raise ValueError(f"Unknown shape {X.shape}")

        if self.use_poly_features:
            num_features += self.max_poly_features

        num_cells = (num_features + 1) * num_samples
        bytes_per_float = 2 if fp_16 else 4

        if estimation_task == "memory":
            if per_feature_transformer:
                if save_peak_mem_factor:
                    overhead_factor = 2  # this is an approximative constant, which should give an upper bound
                    return (
                        num_cells
                        * embedding_size
                        * bytes_per_float
                        * overhead_factor
                        * batch_size_inference
                    )
                else:
                    # TODO: Check if this is correct
                    print("Warning: memory usage is untested")
                    return (
                        num_cells
                        * embedding_size
                        * bytes_per_float
                        * num_heads
                        * num_layers
                        * batch_size_inference
                    )
            else:
                # TODO: Check if this is correct
                print("Warning: memory usage is untested")
                return num_samples * embedding_size * bytes_per_float
        elif estimation_task == "computation":
            # TODO: Check if this is correct
            if per_feature_transformer:
                return (
                    ((num_samples**2) + (num_features**2))
                    * (embedding_size**2)
                    * num_heads
                    * num_layers
                )
            else:
                return (
                    num_samples
                    * num_samples
                    * embedding_size
                    * embedding_size
                    * num_heads
                    * num_layers
                )

    def _log_memory_usage(self, eval_xs: torch.Tensor) -> None:
        """
        Logs the memory usage of the model.
        The values are also saved on disk and can be accessed with:
        >>> mem_data = pd.read_csv(LOG_MEMORY_USAGE_PATH, delimiter="\t", header=None)
        >>> sns.scatterplot(mem_data, x=0, y=1)

        :param eval_xs: The input feature tensor.
        """
        if not self.device.startswith("cuda"):
            print("Memory logging only works for cuda.")
            return
        peak_mem = torch.cuda.max_memory_allocated()
        estimated_mem = self.estimate_memory_usage(eval_xs, "b")
        num_samples, batch_size, num_features = (
            eval_xs.shape[0],
            eval_xs.shape[1],
            eval_xs.shape[2],
        )

        print(
            f"Peak Memory usage (GB): {peak_mem / 1e9},"
            f" Estimated Memory usage (GB): {estimated_mem / 1e9},"
            f" Num Samples: {num_samples}, Batch Size: {batch_size},"
            f" Num Features: {num_features}"
        )
        with open(LOG_MEMORY_USAGE_PATH, "a") as myfile:
            myfile.write(
                f"{peak_mem}\t{estimated_mem}\t{num_samples}\t{num_features}\n"
            )
        torch.cuda.reset_peak_memory_stats()


class TabPFNClassifier(TabPFNBaseModel, ClassifierMixin):
    semisupervised_indicator = -100

    def __init__(
        self,
        model: Optional[Any] = None,
        device: str = "cuda:0",
        model_string: str = "",
        batch_size_inference: int = None,
        fp16_inference: bool = False,
        inference_mode: bool = True,
        c: Optional[Dict] = None,
        N_ensemble_configurations: int = 10,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig("none"),
            PreprocessorConfig("power", categorical_name="numeric"),
        ),
        feature_shift_decoder: str = "none", 
        normalize_with_test: bool = False,
        average_logits: bool = False,
        categorical_features: Tuple[str, ...] = tuple(),
        optimize_metric: str = "roc",
        seed: Optional[int] = 0,
        transformer_predict_kwargs: Optional[Dict] = None,
        show_progress: bool = True,
        multiclass_decoder="shuffle",
        sklearn_compatible_precision: bool = False,
        softmax_temperature: Optional[float] = math.log(0.8),
        save_peak_memory: Literal["True", "False", "auto"] = "True",
        maximum_free_memory_in_gb: Optional[float] = None,
        apply_pca=False,
        use_poly_features=False,
        max_poly_features=None,
        sample_random_poly_features=True,
        transductive=False,
        random_feature_scaling_strength=0.0,
        auxiliary_clf: tuple(str, str) | None = None,
        regression_y_preprocess_transforms: None = None,
    ):
        """
        According to Sklearn API we need to pass all parameters to the super class constructor without **kwargs or *args
        Also we must not do any input validation or programmatic changes to the parameters here.
        """
        self.multiclass_decoder = multiclass_decoder
        # Capture all local variables (which includes all arguments)
        args_dict = locals()

        # Remove 'self' from dictionary
        del args_dict["self"]
        del args_dict["__class__"]
        # Remove 'multiclass_decoder' from dictionary
        del args_dict["multiclass_decoder"]
        

        # Pass all parameters to super class constructor
        super().__init__(**args_dict)

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)

        cls, y_[y_ != self.semisupervised_indicator] = np.unique(
            y_[y_ != self.semisupervised_indicator], return_inverse=True
        )
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls

        return np.asarray(y_, dtype=np.float32, order="C")

    @staticmethod
    def check_training_data(clf: TabPFNClassifier, X, y) -> Tuple:
        unique_labels_ = unique_labels(y)

        has_nan_class_count = 0
        if TabPFNClassifier.semisupervised_indicator in unique_labels_:
            assert clf.c_processed_.get("semisupervised_enabled", False)
            has_nan_class_count = 1
            print_once(
                "Found nan class in training data, will be used as semisupervsied"
            )

        if len(unique_labels_) > clf.max_num_classes_ + has_nan_class_count:
            raise ValueError(
                "The number of classes for this classifier is restricted to ",
                clf.max_num_classes_,
            )

        clf.num_classes_ = len(unique_labels_) - has_nan_class_count

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, force_all_finite=False)

        # Store the classes seen during fit
        y = clf._validate_targets(y)
        clf.label_encoder_ = LabelEncoder()
        y[
            y != TabPFNClassifier.semisupervised_indicator
        ] = clf.label_encoder_.fit_transform(
            y[y != TabPFNClassifier.semisupervised_indicator]
        )

        X, y = TabPFNBaseModel.check_training_data(clf, X, y)

        return X, y

    def init_model_and_get_model_config(self):
        super().init_model_and_get_model_config()
        assert self.is_classification_, "This should be a classification model"
        self.max_num_classes_ = self.c_processed_["max_num_classes"]

    def predict_proba(self, X, additional_y=None):
        """
        Calls the transformer to predict the probabilities of the classes of the X test inputs given the previous set
        training dataset

        :param X: test datapoints
        :param normalize_with_test: if the test datapoints should be incorporated in calculating the normalization
                                    parameters mean and std together with the training datapoints.
                                    Is per default set to false.
        """

        # print('predict_proba X', X)

        X_full, y_full, additional_y, eval_pos = super().predict_common_setup(
            X, additional_y_eval=additional_y
        )

        if self.transformer_predict_kwargs is None:
            transformer_predict_kwargs = {}
        else:
            transformer_predict_kwargs = self.transformer_predict_kwargs

        prediction = self.transformer_predict(
            X_full,
            y_full,
            eval_pos,
            additional_ys=additional_y,
            **get_params_from_config(self.c_processed_),
            **transformer_predict_kwargs,
        )

        prediction = prediction.squeeze(0)
        prediction = prediction.detach().cpu().numpy()

        if self.sklearn_compatible_precision:
            print("SKLEARN COMPATIBLE PREDICTION FOR DEBUG PURPOSE")
            prediction = np.around(
                prediction, decimals=16
            )  # TODO: Do we want this, its just for sklearn
            prediction[prediction < 0.001] = 0
            prediction = prediction / prediction.sum(axis=1, keepdims=True)

        return prediction

    def predict(self, X, return_winning_probability=False):
        p = self.predict_proba(X)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=int))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss

        if self.optimize_metric == "roc":

            def safe_roc_auc_score(y_true, y_score, **kwargs):
                try:
                    return roc_auc_score(y_true, y_score, **kwargs)
                except ValueError:
                    try:
                        unique_classes = np.unique(y_true)
                        missing_classes = [
                            i
                            for i in range(y_score.shape[1])
                            if i not in unique_classes
                        ]

                        # Modify y_score to exclude columns corresponding to missing classes
                        y_score_adjusted = np.delete(y_score, missing_classes, axis=1)
                        y_score_adjusted = y_score_adjusted / y_score_adjusted.sum(
                            axis=1, keepdims=True
                        )

                        return roc_auc_score(y_true, y_score_adjusted, **kwargs)
                    except Exception:
                        return 0.5

            pred = self.predict_proba(X)
            if len(np.unique(y)) == 2:
                pred = pred[:, 1]

            return safe_roc_auc_score(
                y, pred, sample_weight=sample_weight, multi_class="ovr"
            )
        elif self.optimize_metric == "accuracy":
            return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        elif self.optimize_metric == "f1":
            return f1_score(y, self.predict(X), sample_weight=sample_weight)
        elif self.optimize_metric == "log_loss":
            return log_loss(y, self.predict_proba(X), sample_weight=sample_weight)
        else:
            raise ValueError(f"Unknown metric {self.optimize_metric}")

    def get_aux_clf(self):
        if self.auxiliary_clf[0] == "xgb":
            from xgboost import XGBClassifier

            auxiliary_clf_ = XGBClassifier()
        elif self.auxiliary_clf[0] == "rf":
            from sklearn.ensemble import RandomForestClassifier

            auxiliary_clf_ = RandomForestClassifier()
        else:
            raise ValueError(f"Unknown auxiliary classifier {self.auxiliary_clf[0]}")
        return auxiliary_clf_

    def get_shap_values(self, test_x):
        import shap

        class CustomIndependent(shap.maskers.Independent):
            def __call__(self, mask, x):
                return super().__call__(mask, x)

            def invariants(self, x):
                i = super().invariants(x)
                return i

        masker = CustomIndependent(
            np.ones_like(test_x.values) * float("inf"), max_samples=1
        )

        explainer = shap.Explainer(self.predict_proba, masker)
        shap_values = explainer(test_x)

        return shap_values

    def plot_shap(self, test_x=None, shap_values=None):
        import shap

        assert (
            test_x is not None or shap_values is not None
        ), "Either test_x or shap_values must be provided"

        if shap_values is None:
            shap_values = self.get_shap_values(test_x)

        shap_values_ = shap_values[:, :, 0]

        shap.plots.bar(shap_values=shap_values_)
        shap.summary_plot(shap_values=shap_values_)

        most_important = shap_values_.abs.mean(0).values.argsort()[-1]
        self.plot_shap_feature(shap_values_, most_important)

    def plot_shap_feature(self, shap_values_, feature_name):
        import shap

        # we can use shap.approximate_interactions to guess which features
        # may interact with age
        inds = shap.utils.potential_interactions(
            shap_values_[:, feature_name], shap_values_
        )

        # make plots colored by each of the top three possible interacting features
        for i in range(3):
            shap.plots.scatter(
                shap_values_[:, feature_name], color=shap_values_[:, inds[i]]
                
            )


class FairPFNClassifier(TabPFNClassifier):
    def __init__(self):

        with open('artifacts/fairpfn_config.pkl', 'rb') as f:
            config = pkl.load(f)

        super().__init__(**config.to_kwargs())

