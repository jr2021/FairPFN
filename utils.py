from __future__ import annotations

import contextlib
import dataclasses
import os
import math
import argparse
import random
import datetime
import itertools
import typing as tp
import warnings

import torch
from torch import nn
import numpy as np
import scipy.stats as stats
import pandas as pd
import re
import scipy.stats
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
import logging

if tp.TYPE_CHECKING:
    import model.transformer

logger = logging.getLogger(__name__)


def mean_nested_structures(nested_structures):
    """
    Computes the mean of a list of nested structures. Supports lists, tuples, and dicts.
    E.g. mean_nested_structures([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]) == {'a': 2, 'b': 3}
    It also works for torch.tensor in the leaf nodes.
    :param nested_structures: List of nested structures
    :return: Mean of nested structures
    """
    if isinstance(nested_structures[0], dict):
        assert all(
            [
                set(arg.keys()) == set(nested_structures[0].keys())
                for arg in nested_structures
            ]
        )
        return {
            k: mean_nested_structures([arg[k] for arg in nested_structures])
            for k in nested_structures[0]
        }
    elif isinstance(nested_structures[0], list):
        assert all([len(arg) == len(nested_structures[0]) for arg in nested_structures])
        return [mean_nested_structures(elems) for elems in zip(*nested_structures)]
    elif isinstance(nested_structures[0], tuple):
        assert all([len(arg) == len(nested_structures[0]) for arg in nested_structures])
        return tuple(mean_nested_structures(elems) for elems in zip(*nested_structures))
    else:  # Assume leaf node is a tensor-like object that supports addition
        return sum(nested_structures) / len(nested_structures)


def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr



@dataclasses.dataclass
class BetaDistributionParameters:
    k: float = 0.95
    b: float = 8.0


@dataclasses.dataclass
class NumFeaturesSamplerConfig:
    type: str = "constant"
    min_num_features: int = None
    max_num_features: int = 100
    sampler_parameters: object = None

    def __post_init__(self):
        # this code can go whenever we switch to the new configs, as it is done automatically then.
        if self.type == "beta":
            self.sampler_parameters = BetaDistributionParameters(
                **self.sampler_parameters
            )
        elif self.type == "constant":
            assert (
                self.min_num_features is None
            ), "Please do not set `min_num_features`, only `max_num_features`, when using `constant`."
            self.min_num_features = self.max_num_features


class NumFeaturesSampler:
    def __init__(self, config):
        self.config: NumFeaturesSamplerConfig = (
            NumFeaturesSamplerConfig(**config) if isinstance(config, dict) else config
        )

    def __call__(self, step, max_num_features=None):
        """
        Sample the number of features for a given step.
        Do so by sampling but not higher than `max_num_features`.
        :param step: Current step, used for random number generation to make sure every worker gets the same num features in a step
        :param max_num_features: Maximum number of features to sample
        :return:
        """
        if max_num_features is None:
            max_num_features = self.config.max_num_features

        rng = np.random.RandomState(step)
        if self.config.type == "beta":
            assert isinstance(
                self.config.sampler_parameters, BetaDistributionParameters
            )
            sample = self.config.min_num_features + round(
                (max_num_features - self.config.min_num_features)
                * rng.beta(
                    self.config.sampler_parameters.k,
                    self.config.sampler_parameters.b,
                )
            )
        elif self.config.type == "constant":
            sample = max_num_features
        else:
            raise NotImplementedError(f"Unknown shape type {self.config.type}")

        assert (
            sample <= max_num_features
        ), f"Sampled {sample} but max is {max_num_features}, config: {self.config}"

        return sample


class SeqBN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)
        self.d_model = d_model

    def forward(self, x):
        assert self.d_model == x.shape[-1]
        flat_x = x.view(-1, self.d_model)
        flat_x = self.bn(flat_x)
        return flat_x.view(*x.shape)



def set_locals_in_self(locals):
    """
    Call this function like `set_locals_in_self(locals())` to set all local variables as object variables.
    Especially useful right at the beginning of `__init__`.
    :param locals: `locals()`
    """
    self = locals["self"]
    for var_name, val in locals.items():
        if var_name != "self":
            setattr(self, var_name, val)


def get_default_device():
    """
    Functional version of default_device, very helpful with submitit.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu:0"


default_device = get_default_device()


def is_cuda(device_or_device_str: tp.Union[torch.device, str]):
    return torch.device(device_or_device_str).type == "cuda"


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to_device(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to_device(v, device))
        return res
    else:
        return obj


# Copied from StackOverflow, but we do an eval on the values additionally
class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            try:
                my_dict[k] = eval(v)
            except NameError:
                my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        print("dict values: {}".format(my_dict))


def get_nan_value(v, set_value_to_nan=1.0):
    if random.random() < set_value_to_nan:
        return v
    else:
        return random.choice([-999, 0, 1, 999])


def to_ranking(data):
    x = data >= data.unsqueeze(-3)
    x = x.sum(0)
    return x


# TODO: Is there a better way to do this?
#   1. Comparing to unique elements: When all values are different we still get quadratic blowup
#   2. Argsort(Argsort()) returns ranking, but with duplicate values there is an ordering which is problematic
#   3. Argsort(Argsort(Unique))->Scatter seems a bit complicated, doesn't have quadratic blowup, but how fast?
def to_ranking_low_mem(data):
    x = torch.zeros_like(data)
    for col in range(data.shape[-1]):
        x_ = data[:, :, col] >= data[:, :, col].unsqueeze(-2)
        x_ = x_.sum(0)
        x[:, :, col] = x_
    return x


def nan_handling_missing_for_unknown_reason_value(nan_prob=1.0):
    return get_nan_value(float("nan"), nan_prob)


def nan_handling_missing_for_no_reason_value(nan_prob=1.0):
    return get_nan_value(float("-inf"), nan_prob)


def nan_handling_missing_for_a_reason_value(nan_prob=1.0):
    return get_nan_value(float("inf"), nan_prob)


def torch_nanmean(x, axis=0, return_nanshare=False, eps=0.0, include_inf=False):
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1.0 - num / x.shape[axis]
    return value / (num + eps)


def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(
        axis=axis
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis), x.shape[axis], dim=axis
    )
    return torch.sqrt(
        torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1)
    )


def normalize_data(
    data, normalize_positions=-1, return_scaling=False, clip=True, std_only=False
):
    """
    Normalize data to mean 0 and std 1

    :param data: T,B,H
    :param normalize_positions: If > 0, only use the first `normalize_positions` positions for normalization
    :param return_scaling: If True, return the scaling parameters as well (mean, std)
    :param clip: If True, clip values to [-100, 100]
    :param std_only: If True, only divide by std
    """
    if normalize_positions is not None and normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], axis=0)
        std = torch_nanstd(data[:normalize_positions], axis=0) + 0.000001
    else:
        mean = torch_nanmean(data, axis=0)
        std = torch_nanstd(data, axis=0) + 0.000001

    if len(data) == 1 or normalize_positions == 1:
        std = 1.0

    if std_only:
        mean = 0
    data = (data - mean) / std

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    data = X if normalize_positions == -1 else X[:normalize_positions]
    data_clean = data[:].clone()
    data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
    data_mean, data_std = torch_nanmean(data_clean, axis=0), torch_nanstd(
        data_clean, axis=0
    )
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    # print(ds[1][data < lower, col], ds[1][data > upper, col], ds[1][~np.isnan(data), col].shape, data_mean, data_std)
    return X


def bool_mask_to_att_mask(mask):
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )


def print_on_master_only(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if is_master or args[0].startswith("ALL"):
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_slurm_hostnames():
    """
    Expands a compressed hostlist string into a list of individual hosts.

    :return: A list of strings containing the expanded hostnames.

    Example for hostlist 'node[02-4]':
    >>> get_slurm_hostnames()
    ['node02', 'node03', 'node04']
    """
    import subprocess

    command = "scontrol show hostnames $SLURM_JOB_NODELIST"
    return subprocess.check_output(command, text=True, shell=True).splitlines()


def get_random_port(min_port=20000, max_port=60000, seed=None):
    import random

    assert (
        min_port >= 1023 and max_port <= 65535 and min_port <= max_port
    ), "Invalid port range."

    rng = random.Random(time.time_ns() if seed is None else seed)

    return rng.randint(min_port, max_port)


def init_dist(device):
    """
    Returns a tuple of (is_distributed, global_rank, local_rank, process_gpu (e.g. `cuda:3`))
    """
    # print("init dist")
    if "LOCAL_RANK" in os.environ:
        # launched with torch.distributed.launch
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        if "MASTER_PORT" not in os.environ:
            print("MASTER_PORT not in os.environ, better set it.")
        if not os.environ.get("NCCL_ASYNC_ERROR_HANDLING"):
            print("NCCL_ASYNC_ERROR_HANDLING=1 should be set s.t. runs do not hang.")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(
                hours=1
            ),  # a lot of timeout needed, as the evaluation runs for a long time on node 0 only
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {torch.cuda.device_count()} GPUs, this is rank {rank}, "
            "only I can print, but when using print(..., force=True) it will print on all ranks."
        )
        return True, rank, rank, f"cuda:{rank}"
    elif (
        "SLURM_PROCID" in os.environ
        and "SLURM_NTASKS" in os.environ
        and int(os.environ["SLURM_NTASKS"]) > 1
    ):
        # this is for multi gpu when starting with submitit
        assert device != "cpu:0"
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        if "SLURM_JOB_NODELIST" in os.environ:
            os.environ["MASTER_ADDR"] = get_slurm_hostnames()[0]
        else:
            os.environ["MASTER_ADDR"] = "localhost"

        os.environ["MASTER_PORT"] = str(
            get_random_port(seed=os.environ["SLURM_JOB_ID"])
        )

        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        print(
            "distributed submitit launch and my rank is",
            rank,
            "using gpu",
            local_rank,
            "and master port",
            os.environ["MASTER_PORT"],
        )
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(
                hours=1
            ),  # a lot of timeout needed, as the evaluation runs for a long time on node 0 only
            world_size=world_size,
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        print(
            f"Distributed training on {world_size} GPUs, this is rank {rank}, "
            "only I can print, but when starting the print statement with `ALL` like print('ALL...') it will print on all ranks."
        )

        return True, rank, local_rank, f"cuda:{local_rank}"
    else:
        # print("Not using distributed")
        # will not change any of the behavior of print, but allows putting the force=True in the print calls
        print_on_master_only(True)
        return False, 0, 0, device


# NOP decorator for python with statements (x = NOP(); with x:)
class NOP:
    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def check_compatibility(dl):
    if hasattr(dl, "num_outputs"):
        print(
            "`num_outputs` for the DataLoader is deprecated. It is assumed to be 1 from now on."
        )
        assert dl.num_outputs != 1, (
            "We assume num_outputs to be 1. Instead of the num_ouputs change your loss."
            "We specify the number of classes in the CE loss."
        )


def product_dict(dic):
    keys = dic.keys()
    vals = dic.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def to_tensor(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, device=device)


printed_already = set()


def print_once(*msgs: tp.Any):
    """
    Print a message or multiple messages, but only once.
    It has the same behavior on the first call as standard print.
    If you call it again with the exact same input, it won't print anymore, though.
    """
    msg = " ".join([(m if isinstance(m, str) else repr(m)) for m in msgs])
    if msg not in printed_already:
        print(msg)
        printed_already.add(msg)


def compare_nested_dicts(old_dict, new_dict):
    print("only in new\n")
    for k in new_dict.keys() - old_dict.keys():
        print(k, new_dict[k])

    print("only in old\n")
    for k in old_dict.keys() - new_dict.keys():
        print(k, old_dict[k])

    print("in both\n")
    for k in old_dict.keys() & new_dict.keys():
        if old_dict[k] != new_dict[k]:
            if isinstance(new_dict[k], dict):
                print("\ngoing into", k, "\n")
                compare_nested_dicts(old_dict[k], new_dict[k])
            else:
                print(k, "old:", old_dict[k], "new:", new_dict[k])


def get_all_times(j):
    if j.stdout() is not None:
        time_strs = re.findall(r"time:[ ]+([.0-9]+).*\|", j.stdout())
        if time_strs:
            return [float(t) for t in time_strs]
    print("ignore job", j)
    return None


def get_all_losses(j):
    if stdout := j.stdout():
        try:
            return [float(v) for v in re.findall("mean loss (.*) \|", stdout)]
        except Exception as e:
            print(stdout)
            raise e
    return None


def average_multiple_epochs(stdout, last_k=100):
    mean_losses = get_all_losses(stdout)
    print("avg over last", last_k, "of", len(mean_losses))
    return torch.tensor(mean_losses[-last_k:]).mean()


def window_average(lis, window_size=10):
    return [
        sum(lis[i - window_size : i]) / window_size
        for i in range(window_size, len(lis))
    ]


def np_load_if_exists(path):
    """Checks if a numpy file exists. Returns None if not, else returns the loaded file."""
    if os.path.isfile(path):
        # print(f'loading results from {path}')
        with open(path, "rb") as f:
            try:
                return np.load(f, allow_pickle=True).tolist()
            except Exception as e:
                logger.warning(f"Could not load {path} because {e}")
                return None
    return None


def rank_values(x):
    s = torch.argsort(x, descending=False, dim=0)
    x = torch.arange(x.numel(), device=x.device)[torch.argsort(s, dim=0)]
    return x


def map_unique_to_order(x):
    # Alternative implementation:
    # boolean = output[:, None] == torch.unique(output)
    # output = torch.nonzero(boolean)[:, -1]
    if len(x.shape) != 2:
        raise ValueError("map_unique_to_order only works with 2D tensors")
    if x.shape[1] > 1:
        return torch.cat(
            [map_unique_to_order(x[:, i : i + 1]) for i in range(x.shape[1])], dim=1
        )
    print(x.shape)
    return (x > torch.unique(x).unsqueeze(0)).sum(1).unsqueeze(-1)


"""timings with GPU involved are potentially wrong.
TODO: a bit of documentation on how to use these.
maybe include torch.cuda.synchronize!? but might make things slower..
maybe better write that timings with GPU involved are potentially wrong.
"""
import time

timing_dict_aggregation, timing_dict, timing_meta_dict = {}, {}, {}


def timing_start(name="", enabled=True, meta=None):
    if not enabled:
        return
    timing_dict[name] = time.time()
    timing_meta_dict[name] = meta


def timing_end(name="", enabled=True, collect=False):
    if not enabled:
        return
    if collect:
        timing_dict_aggregation[name] = (
            timing_dict_aggregation.get(name, 0) + time.time() - timing_dict[name]
        )
    else:
        print("Timing", name, time.time() - timing_dict[name], timing_meta_dict[name])
        timing_meta_dict[name] = None


def lambda_time(f, name="", enabled=True, collect=False, meta=None):
    timing_start(name, enabled, meta=meta)
    r = f()
    timing_end(name, enabled, collect)
    return r


def lambda_time_flush(name="", enabled=True):
    if not enabled or name not in timing_dict_aggregation:
        return
    print("Timing", name, timing_dict_aggregation[name])
    timing_dict_aggregation[name] = 0


def add_direct_connections(G):
    added_connection = False
    # Get the list of nodes
    nodes = list(G.nodes)

    # Iterate over each node
    for node in nodes:
        # Get the direct neighbors of the current node
        neighbors = list(G.neighbors(node))

        # Iterate over the neighbors of the current node
        for neighbor in neighbors:
            # Get the neighbors of the neighbor
            second_neighbors = list(G.neighbors(neighbor))

            # Iterate over the neighbors of the neighbor
            for second_neighbor in second_neighbors:
                # Add a direct edge from the current node to the second neighbor,
                # if it doesn't exist already
                if second_neighbor not in G.neighbors(node):
                    G.add_edge(node, second_neighbor)

                    added_connection = True
    return added_connection


import networkx as nx


def add_pos_emb(graph, is_undirected=False, k=20):
    from scipy.sparse.linalg import eigs, eigsh

    eig_fn = eigs if not is_undirected else eigsh

    # normalization='sym',
    L = nx.directed_laplacian_matrix(graph)

    if np.isnan(L).any():
        # print("nan in laplacian")
        # print(L)
        # print(graph.nodes)
        # print(graph.edges)
        L[np.isnan(L)] = 0.0

    # L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eig_vals, eig_vecs = eig_fn(
            L,
            k=k + 1,
            which="SR" if not is_undirected else "SA",
            return_eigenvectors=True,
        )

        # print(f"{np.isnan(eig_vals).any()=} {np.isnan(eig_vecs).any()=}")

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe_ = torch.from_numpy(eig_vecs[:, 1 : k + 1])
        # print(pe.std(0), pe.mean(0))
        pe = torch.zeros(len(eig_vecs), k)
        pe[:, : pe_.shape[1]] = pe_
        sign = -1 + 2 * torch.randint(0, 2, (k,))
        pe *= sign

        # print(f"{torch.isnan(pe).any()=} {torch.isinf(pe).any()=}")

        # TODO Double check the ordering is right
        for n, pe_ in zip(graph.nodes(), pe):
            graph.nodes[n]["positional_encoding"] = pe_


def add_grouped_confidence_interval(
    grouped: pd.core.groupby.generic.SeriesGroupBy, confidence=0.95
) -> pd.DataFrame:
    """
    Add a confidence interval to a SeriesGroupBy object,
    calculated separately for each group.

    Usage example:
    result_df # a DataFrame with columns for lr, batch_size, and accuracy
    grouped = result_df.groupby(['lr', 'batch_size']).accuracy
    ci_df = add_grouped_confidence_interval(grouped)

    Args:
    grouped (pd.core.groupby.generic.SeriesGroupBy): Input grouped Series.
    confidence (float): The confidence level to use in the confidence interval calculation.

    Returns:
    pd.DataFrame: The DataFrame with the added confidence interval columns.
    """

    # Function to calculate confidence intervals
    def ci(group, confidence):
        mean = group.mean()
        std_err = stats.sem(group)
        return pd.Series(
            {
                "mean": mean,
                "CI_lower": stats.t.interval(
                    confidence, len(group) - 1, loc=mean, scale=std_err
                )[0],
                "CI_upper": stats.t.interval(
                    confidence, len(group) - 1, loc=mean, scale=std_err
                )[1],
            }
        )

    # Calculate confidence intervals for each group
    ci_df = grouped.apply(ci, confidence).unstack()

    return ci_df


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for the mean of a given data set.

    Parameters:
    - data (array-like): The input data set, assumed to be a sample from a larger population.
    - confidence (float): The desired confidence level, default is 0.95.

    Assumptions:
    1. Data should be a random sample from the underlying population.
    2. If data has a normal (Gaussian) distribution or if sample size is large (>30), then the calculated confidence interval is valid.
    3. Missing values (NaNs) are excluded from calculations.

    Math Explained:
    - Mean (m) is calculated using NumPy's mean function.
    - Standard Error of the Mean (se) is calculated using scipy's sem function.
    - The t-distribution is used for calculating the confidence interval, making it more robust for small sample sizes compared to a z-distribution.
    - The margin of error (h) is calculated using the formula: h = se * t_alpha/2(n-1), where t_alpha/2(n-1) is the critical t-value for the given confidence level and degrees of freedom (n-1).

    Returns:
    - h (float): The calculated margin of error for the given data set at the desired confidence level.
    """
    data = data[~np.isnan(data)]
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def _save_stratified_splits(
    _splitter: StratifiedKFold | RepeatedStratifiedKFold,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    auto_fix_stratified_splits: bool = False,
) -> list[list[list[int], list[int]]]:
    """Fix from AutoGluon to avoid unsafe splits for classification if less than n_splits instances exist for all classes.

    https://github.com/autogluon/autogluon/blob/0ab001a1193869a88f7af846723d23245781a1ac/core/src/autogluon/core/utils/utils.py#L70.
    """
    try:
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(x, y)
        ]
    except ValueError as e:
        x = pd.DataFrame(x)
        y = pd.Series(y)
        y_dummy = pd.concat([y, pd.Series([-1] * n_splits)], ignore_index=True)
        X_dummy = pd.concat([x, x.head(n_splits)], ignore_index=True)
        invalid_index = set(y_dummy.tail(n_splits).index)
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(X_dummy, y_dummy)
        ]
        len_out = len(splits)
        for i in range(len_out):
            train_index, test_index = splits[i]
            splits[i][0] = [
                index for index in train_index if index not in invalid_index
            ]
            splits[i][1] = [index for index in test_index if index not in invalid_index]

        # only rais afterward because only now we know that we cannot fix it
        if not auto_fix_stratified_splits:
            raise AssertionError(
                "Cannot split data in a stratifed way with each class in each subset of the data."
            ) from e
    except UserWarning as e:
        # Assume UserWarning for not enough classes for correct stratified splitting.
        raise e

    return splits


def fix_split_by_dropping_classes(
    x: np.ndarray, y: np.ndarray, n_splits: int, spliter_kwargs: dict
) -> list[list[list[int], list[int]]]:
    """Fixes stratifed splits for edge case.

    For each class that has fewer instances than number of splits, we oversample before split to n_splits and then remove all oversamples and
    original samples from the splits; effectively removing the class from the data without touching the indices.
    """
    val, counts = np.unique(y, return_counts=True)
    too_low = val[counts < n_splits]
    too_low_counts = counts[counts < n_splits]

    y_dummy = pd.Series(y.copy())
    X_dummy = pd.DataFrame(x.copy())
    org_index_max = len(X_dummy)
    invalid_index = []

    for c_val, c_count in zip(too_low, too_low_counts):
        fill_missing = n_splits - c_count
        invalid_index.extend(np.where(y == c_val)[0])
        y_dummy = pd.concat(
            [y_dummy, pd.Series([c_val] * fill_missing)], ignore_index=True
        )
        X_dummy = pd.concat(
            [X_dummy, pd.DataFrame(x).head(fill_missing)], ignore_index=True
        )

    invalid_index.extend(list(range(org_index_max, len(y_dummy))))
    splits = _save_stratified_splits(
        _splitter=StratifiedKFold(**spliter_kwargs),
        x=X_dummy,
        y=y_dummy,
        n_splits=n_splits,
    )
    len_out = len(splits)
    for i in range(len_out):
        train_index, test_index = splits[i]
        splits[i][0] = [index for index in train_index if index not in invalid_index]
        splits[i][1] = [index for index in test_index if index not in invalid_index]

    return splits


def get_cv_split_for_data(
    x: np.ndarray,
    y: np.ndarray,
    splits_seed: int,
    n_splits: int,
    stratified_split: bool,
    safety_shuffle: bool = True,
    auto_fix_stratified_splits: bool = False,
) -> tuple[list[list[list[int], list[int]]] | str, np.ndarray, np.ndarray]:
    """Safety shuffle and generate (safe) splits.

    If it returns str at the first entry, no valid split could be generated and the str is the reason why.
    Due to the safety shuffle, the original x and y are also returned and must be used.

    Test with:

    ```python
        if __name__ == "__main__":
        print(
            get_cv_split_for_data(
                x=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T,
                y=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]),
                splits_seed=42,
                n_splits=3,
                stratified_split=True,
                auto_fix_stratified_splits=True,
            )
        )
    ```
    """
    kow_rng = np.random.RandomState(splits_seed)
    if safety_shuffle:
        p = kow_rng.permutation(len(x))
        x, y = x[p], y[p]
    spliter_kwargs = {"n_splits": n_splits, "shuffle": True, "random_state": kow_rng}

    if not stratified_split:
        return list(KFold(**spliter_kwargs).split(x, y)), x, y

    warnings.filterwarnings("error")
    try:
        splits = _save_stratified_splits(
            _splitter=StratifiedKFold(**spliter_kwargs),
            x=x,
            y=y,
            n_splits=n_splits,
            auto_fix_stratified_splits=auto_fix_stratified_splits,
        )
    except UserWarning as e:
        logger.debug(e)
        if auto_fix_stratified_splits:
            logger.debug("Trying to fix stratified splits automatically...")
            splits = fix_split_by_dropping_classes(
                x=x, y=y, n_splits=n_splits, spliter_kwargs=spliter_kwargs
            )
        else:
            splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"
    except AssertionError as e:
        logger.debug(e)
        splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"

    warnings.resetwarnings()

    return splits, x, y
