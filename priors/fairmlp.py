import random
import math

import torch
from torch import nn
import numpy as np

from utils import default_device
from .prior import Batch

import copy

from scipy.stats import entropy

import logging

from utils import timing_start, timing_end, lambda_time_flush

from datasets import TabularDataset

SEED = 0

class GaussianNoise(nn.Module):
    def __init__(self, std, device):
        super().__init__()
        self.std = std
        self.device = device

    def forward(self, x):
        return x + torch.normal(torch.zeros_like(x), self.std)


def causes_sampler_f(num_causes):
    means = np.random.normal(0, 1, (num_causes))
    std = np.abs(np.random.normal(0, 1, (num_causes)) * means)
    return means, std

def get_batch(
    batch_size,
    seq_len,
    num_features,
    hyperparameters,
    device,
    num_outputs=1,
    num_prot_attrs=1,
    sampling="normal",
    epoch=None,
    **kwargs,
):

    if (
        "multiclass_type" in hyperparameters
        and hyperparameters["multiclass_type"] == "multi_node"
    ):
        num_outputs = num_outputs * hyperparameters["num_classes"]

    # if not (
    #     ("mix_activations" in hyperparameters) and hyperparameters["mix_activations"]
    # ):
    #     hyperparameters["prior_mlp_activations"]
    #     hyperparameters["prior_mlp_activations"]

    class FairMLP(torch.nn.Module):
        def __init__(self, hyperparameters):
            super(FairMLP, self).__init__()

            with torch.no_grad():
                for key in hyperparameters:
                    setattr(self, key, hyperparameters[key])
                
                self.is_causal = True

                assert self.num_layers >= 2

                if "verbose" in hyperparameters and self.verbose:
                    print(
                        {
                            k: hyperparameters[k]
                            for k in [
                                "is_causal",
                                "num_causes",
                                "prior_mlp_hidden_dim",
                                "num_layers",
                                "noise_std",
                                "y_is_effect",
                                "pre_sample_weights",
                                "prior_mlp_dropout_prob",
                                "pre_sample_causes",
                            ]
                        }
                    )

                if self.is_causal:
                    self.prior_mlp_hidden_dim = max(
                        self.prior_mlp_hidden_dim, num_outputs + 2 * num_features
                    )
                else:
                    self.num_causes = num_features

                # This means that the mean and standard deviation of each cause is determined in advance
                if self.pre_sample_causes:
                    self.causes_mean, self.causes_std = causes_sampler_f(
                        self.num_causes
                    )
                    self.causes_mean = (
                        torch.tensor(self.causes_mean, device=device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .tile((seq_len, 1, 1))
                    )
                    self.causes_std = (
                        torch.tensor(self.causes_std, device=device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .tile((seq_len, 1, 1))
                    )

                def generate_module(layer_idx, out_dim):
                    # Determine std of each noise term in initialization, so that is shared in runs
                    # torch.abs(torch.normal(torch.zeros((out_dim)), self.noise_std)) - Change std for each dimension?
                    noise = (
                        GaussianNoise(
                            torch.abs(
                                torch.normal(
                                    torch.zeros(size=(1, out_dim), device=device),
                                    float(self.noise_std),
                                )
                            ),
                            device=device,
                        )
                        if self.pre_sample_weights
                        else GaussianNoise(float(self.noise_std), device=device)
                    )

                    return [
                        nn.Sequential(
                            *[
                                self.prior_mlp_activations,
                                nn.Linear(self.prior_mlp_hidden_dim, out_dim),
                                noise,
                            ]
                        )
                    ]

                self.layers = [
                    nn.Linear(self.num_causes, self.prior_mlp_hidden_dim, device=device)
                ]
                self.layers += [
                    module
                    for layer_idx in range(self.num_layers - 1)
                    for module in generate_module(layer_idx, self.prior_mlp_hidden_dim)
                ]
                if not self.is_causal:
                    self.layers += generate_module(-1, num_outputs)
                self.layers = nn.Sequential(*self.layers)

                # Initialize Model parameters
                for i, (n, p) in enumerate(self.layers.named_parameters()):
                    if self.block_wise_dropout:
                        if (
                            len(p.shape) == 2
                        ):  # Only apply to weight matrices and not bias
                            nn.init.zeros_(p)
                            # TODO: N blocks should be a setting
                            n_blocks = random.randint(
                                1, math.ceil(math.sqrt(min(p.shape[0], p.shape[1])))
                            )
                            w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                            keep_prob = (n_blocks * w * h) / p.numel()
                            for block in range(0, n_blocks):
                                nn.init.normal_(
                                    p[
                                        w * block : w * (block + 1),
                                        h * block : h * (block + 1),
                                    ],
                                    std=self.init_std
                                    / keep_prob
                                    ** (
                                        1 / 2
                                        if self.prior_mlp_scale_weights_sqrt
                                        else 1
                                    ),
                                )
                    else:
                        if (
                            len(p.shape) == 2
                        ):  # Only apply to weight matrices and not bias
                            dropout_prob = (
                                self.prior_mlp_dropout_prob if i > 0 else 0.0
                            )  # Don't apply dropout in first layer
                            dropout_prob = min(dropout_prob, 0.99)
                            nn.init.normal_(
                                p,
                                std=self.init_std
                                / (
                                    1.0
                                    - dropout_prob
                                    ** (
                                        1 / 2
                                        if self.prior_mlp_scale_weights_sqrt
                                        else 1
                                    )
                                ),
                            )
                            p *= torch.bernoulli(
                                torch.zeros_like(p) + 1.0 - dropout_prob
                            )

        def catch_nan(self, X, y):

            if bool(torch.any(torch.isnan(X)).detach().cpu().numpy()) or bool(torch.any(torch.isnan(y)).detach().cpu().numpy()) or bool(torch.any(torch.isinf(X)).detach().cpu().numpy()) or bool(torch.any(torch.isinf(X)).detach().cpu().numpy()):
                print('Caught nan in prior')
                X[:] = 0.0
                y[:] = -100  # default ignore index for CE
            
            return X, y

        def forward(self, seed=0): 
            def sample_normal():
                if self.pre_sample_causes:
                    self.causes = torch.normal(
                        self.causes_mean, self.causes_std.abs()
                    ).float()
                else:
                    self.causes = torch.normal(
                        0.0, 1.0, (seq_len, 1, self.num_causes), device=device
                    ).float()
                return self.causes

            if self.sampling == "normal":
                self.causes = sample_normal()
            elif self.sampling == "mixed":
                zipf_p, multi_p, normal_p = (
                    random.random() * 0.66,
                    random.random() * 0.66,
                    random.random() * 0.66,
                )

                def sample_cause(n):
                    if random.random() > normal_p:
                        if self.pre_sample_causes:
                            return torch.normal(
                                self.causes_mean[:, :, n],
                                self.causes_std[:, :, n].abs(),
                            ).float()
                        else:
                            return torch.normal(
                                0.0, 1.0, (seq_len, 1), device=device
                            ).float()
                    elif random.random() > multi_p:
                        x = (
                            torch.multinomial(
                                torch.rand((random.randint(2, 10))),
                                seq_len,
                                replacement=True,
                            )
                            .to(device)
                            .unsqueeze(-1)
                            .float()
                        )
                        x = (x - torch.mean(x)) / torch.std(x)
                        return x
                    else:
                        x = torch.minimum(
                            torch.tensor(
                                np.random.zipf(
                                    2.0 + random.random() * 2, size=(seq_len)
                                ),
                                device=device,
                            )
                            .unsqueeze(-1)
                            .float(),
                            torch.tensor(10.0, device=device),
                        )
                        return x - torch.mean(x)

                self.causes = torch.cat(
                    [sample_cause(n).unsqueeze(-1) for n in range(self.num_causes)], -1
                )
            elif self.sampling == "uniform":
                self.causes = torch.rand((seq_len, 1, self.num_causes), device=device)
            else:
                raise ValueError(f"Sampling is set to invalid setting: {sampling}.")

            # sets seed for gaussian noise
            torch.manual_seed(seed)

            causes_perm = torch.randperm(
                self.causes.shape[-1] - 1, device=device
            )
            
            self.random_idx_A, self.random_idx_U = causes_perm[0:num_prot_attrs], causes_perm[num_prot_attrs:] 

            if hyperparameters["binary_prot_attr"]: 
                # print('sampling from binary real world')
                a_0 = np.random.choice(self.causes.flatten().numpy())
                a_1 = np.random.choice(self.causes.flatten().numpy())
                p_0 = np.random.uniform(low=0, high=1)
                A = torch.tensor(np.random.choice([float(a_0), float(a_1)], size=(self.causes.shape[0], 1, 1), p=[p_0, 1-p_0])).float()
                self.causes[:, :, self.random_idx_A] = A
            else:
                # print('sampling from continuous real world')
                A = self.causes[:, :, self.random_idx_A]
            
            U_fair = self.causes[:, :, self.random_idx_U]

            # generate biased output
            outputs_biased = [self.causes]

            # print(f"weighing prot_attr by factor of {hyperparameters['prot_attr_weight']}")

            for i, layer in enumerate(self.layers):
                if i == 0:   
                    for _, p in layer.named_parameters():
                        if len(p.shape) == 2:
                            with torch.no_grad():
                                p[:, self.random_idx_A] *= float(hyperparameters["prot_attr_weight"])

                output_biased = layer(outputs_biased[-1])
                outputs_biased.append(output_biased)

            outputs_biased = outputs_biased[2:]
            outputs_biased_flat = torch.cat(outputs_biased, -1)

            if self.in_clique:
                outputs_perm = random.randint(
                    0, outputs_biased_flat.shape[-1] - num_outputs - num_features
                ) + torch.randperm(num_outputs + num_features, device=device)
            else:
                outputs_perm = torch.randperm(
                    outputs_biased_flat.shape[-1] - 1, device=device
                )

            self.random_idx_y = (
                list(range(-num_outputs, -0))
                if self.y_is_effect
                else outputs_perm[0:num_outputs]
            )
            self.random_idx_X = outputs_perm[num_outputs:num_outputs+num_features]
            
            # optionally sort features
            if self.sort_features:
                self.random_idx_X, _ = torch.sort(self.random_idx_X)

            # sample biased dataset
            y_biased = outputs_biased_flat[:, :, self.random_idx_y]
            X_biased = outputs_biased_flat[:, :, self.random_idx_X]

            del outputs_biased_flat
            del output_biased

            # ensures same gaussian noise
            torch.manual_seed(seed)

            layers = copy.deepcopy(self.layers)

            # generate fair/counterfactual output
            X_cntf = None
            if hyperparameters["removal_type"] == "dropout":
                outputs_fair = [self.causes]
                for i, layer in enumerate(layers):
                    if i == 0:   
                        for _, p in layer.named_parameters():
                            if len(p.shape) == 2:
                                with torch.no_grad():
                                    p[:, self.random_idx_A] *= 0.0

                    output_fair = layer(outputs_fair[-1])
                    outputs_fair.append(output_fair)
                
                outputs_fair = outputs_fair[2:]
                outputs_fair_flat = torch.cat(outputs_fair, -1)

                y_fair = outputs_fair_flat[:, :, self.random_idx_y]
            elif hyperparameters["removal_type"] == "alignment":
                if hyperparameters["binary_prot_attr"]:
                    # print('sampling from binary counterfactual world')
                    p_0 = np.random.uniform(low=0, high=1)
                    A_cntf = torch.tensor(np.random.choice([float(a_0), float(a_1)], size=(self.causes.shape[0], 1, 1), p=[p_0, 1-p_0])).float()
                    self.causes[:, :, self.random_idx_A] = A_cntf
                else:
                    # print('sampling from continuous counterfactual world')
                    A_cntf = torch.tensor(np.random.choice(A.unique(), size=(self.causes.shape[0], 1, 1))).float()
                    self.causes[:, :, self.random_idx_A] = A_cntf

                outputs_cntf = [self.causes]
                for i, layer in enumerate(layers):
                    output_cntf = layer(outputs_cntf[-1])
                    outputs_cntf.append(output_cntf)
                
                outputs_cntf = outputs_cntf[2:]
                outputs_cntf_flat = torch.cat(outputs_cntf, -1)

                X_cntf = outputs_cntf_flat[:, :, self.random_idx_X]
                y_fair = copy.deepcopy(y_biased)

                del outputs_cntf_flat
                del output_cntf
                
            elif hyperparameters["removal_type"] == "level_two":
                y_fair = copy.deepcopy(y_biased)

            # random feature rotation
            if self.random_feature_rotation:
                arange = torch.arange(X_biased.shape[-1], device=device)
                rand_range = random.randrange(X_biased.shape[-1])

                X_biased = X_biased[
                    ...,
                    (
                        arange 
                        + rand_range
                    )
                    % X_biased.shape[-1],
                ]
                if X_cntf is not None:
                    X_cntf = X_cntf[
                    ...,
                    (
                        arange
                        + rand_range
                    )
                    % X_cntf.shape[-1],
                ]

            # concat protected attribute column
            X_biased = torch.cat((A, X_biased), axis=-1)

            if X_cntf is not None:
                X_cntf = torch.cat((A_cntf, X_cntf), axis=-1)

            # check for nans
            X_biased, y_biased = self.catch_nan(X_biased, y_biased)
            X_biased, y_fair = self.catch_nan(X_biased, y_fair)
            if X_cntf is not None:
                X_cntf, y_fair = self.catch_nan(X_cntf, y_fair)

            return X_biased, y_biased, y_fair, U_fair, X_cntf

    if hyperparameters.get("new_mlp_per_example", False):
        get_model = lambda: FairMLP(hyperparameters).to(device)
    else:
        model = FairMLP(hyperparameters).to(device)
        get_model = lambda: model

    sample = [get_model().forward(seed=batch) for batch in range(0, batch_size)]

    X_biased, y_biased, y_fair, U_fair, X_cntf = zip(*sample)

    X_biased = torch.cat(X_biased, 1).detach().float()
    y_biased = torch.cat(y_biased, 1).detach().float()
    y_fair = torch.cat(y_fair, 1).detach().float()
    U_fair = torch.cat(U_fair, 1).detach().float()

    if X_cntf[0] != None:
        X_cntf = torch.cat(X_cntf, 1).detach()

    return Batch(
        x=X_biased, 
        y=y_biased, 
        target_y=y_fair,
        U_fair=U_fair,
        X_cntf=X_cntf,
        num_prot_attrs=num_prot_attrs,
        additional_x=[],
    )