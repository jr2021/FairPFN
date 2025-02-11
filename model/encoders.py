import math
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import utils
from utils import normalize_data, to_ranking_low_mem, remove_outliers
from utils import torch_nanmean


class InputEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        single_eval_pos: int,
    ):
        raise NotImplementedError


## ENCODER COMPONENTS
class SequentialEncoder(torch.nn.Sequential, InputEncoder):
    """
    SequentialEncoder allows to build an encoder from a sequence of EncoderSteps.
    """

    def __init__(self, *args, output_key="output", **kwargs):
        """

        :param args: A list of EncoderSteps
        :param output_key: The key of the output of the encoder, defaults to "output", that means state["output"] will be returned
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input, **kwargs):
        if type(input) != dict:
            # If the input is not a dict and the first layer expects one input, mapping the
            #   input to the first input key must be correct
            if len(self[0].in_keys) == 1:
                input = {self[0].in_keys[0]: input}

        # print('SequentialEncoder.forward', input['main'])

        for module in self:
            
            # print('input', module, input['main']) # FIXME
            
            input = module(input, **kwargs)
            
        # print('output', module, input[self.output_key] if self.output_key is not None else input) # FIXME

        return input[self.output_key] if self.output_key is not None else input


class LinearInputEncoder(torch.nn.Module):
    """
    LinearInputEncoder is a simple linear layer that takes the input and applies a linear layer to it.
    """

    def __init__(self, num_features, emsize, replace_nan_by_zero=False, bias=True):
        super().__init__()
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, *x, **kwargs):
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return (self.layer(x),)


class DecomposedNumberInputEncoder(torch.nn.Module):
    # Following https://arxiv.org/pdf/2112.01898.pdf
    def __init__(self, **args):
        embed_size_per_number = 3
        raise NotImplementedError("TODO DecomposedNumberInputEncoder")

    def forward(self, x, single_eval_pos: int, **kwargs):
        sign, mantissa, exponent = (
            torch.sign(torch.frexp(x).mantissa),
            torch.frexp(x).mantissa,
            torch.frexp(x).exponent,
        )
        x = torch.cat([sign, mantissa, exponent], -1)
        return (x,)


class SeqEncStep(torch.nn.Module, metaclass=ABCMeta):
    """
    SequentialEncoderStep is a wrapper around a module, that defines which input_keys it expects and
    which output_keys it produces. Outputs of EncoderSteps are supposed to be a tuple which is assigned to the
    output_keys in the order of the out_keys list.
    """

    def __init__(self, in_keys=("main",), out_keys=("main",)):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    def forward(self, state, **kwargs):
        try:
            out = self._forward(*[state[in_key] for in_key in self.in_keys], **kwargs)
        except KeyError:
            raise KeyError(
                f"EncoderStep expected input keys {self.in_keys}, but got {list(state.keys())}"
            )

        assert (
            type(out) == tuple
        ), "EncoderStep must return a tuple of values (can be size 1)"
        assert len(out) == len(
            self.out_keys
        ), f"EncoderStep outputs don't match out_keys {len(out)} (out) != {len(self.out_keys)} (out_keys = {self.out_keys})"

        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass


class LinearInputEncoderStep(SeqEncStep):
    """
    LinearInputEncoder is a simple linear layer that takes the input and applies a linear layer to it.
    """

    def __init__(
        self,
        num_features,
        emsize,
        replace_nan_by_zero=False,
        bias=True,
        in_keys=("main",),
        out_keys=("output",),
    ):
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _forward(self, *x, **kwargs):
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)

        return (self.layer(x),)

############################## FairPFN ####################################

class ProtAttrEncoderStep(SeqEncStep):
    prot_attr_indicator = 1.0

    def __init__(
        self,
        num_prot_attrs=1,
        in_keys=("main",),
        out_keys=("main", "prot_attrs_indicator"),
    ):
        assert len(in_keys) == 1
        super().__init__(in_keys, out_keys)
        self.num_prot_attrs = num_prot_attrs

    def _forward(self, x, single_eval_pos: int, **kwargs):

        # print('ProtAttrEncoderStep', x[:, :, :self.num_prot_attrs])
        # print('ProtAttrEncoderStep SHAPE', x.shape)

        prot_attrs_indicator = torch.zeros(x.shape).to(x.dtype)
        prot_attrs_indicator[:, :, :self.num_prot_attrs] = self.prot_attr_indicator

        # print('ProtarrtEncoderStep', prot_attrs_indicator)

        return x, prot_attrs_indicator


class NanHandlingEncoderStep(SeqEncStep):
    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans=True,
        in_keys=("main",),
        out_keys=("main", "nan_indicators"),
    ):
        assert len(in_keys) == 1
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans

    def _forward(self, x, single_eval_pos: int, **kwargs):
        nans_indicator = None
        if self.keep_nans:
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        feature_means = torch_nanmean(
            x[:single_eval_pos], axis=0, eps=1e-10, include_inf=True
        )
        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations

        x[nan_mask] = feature_means.unsqueeze(0).expand_as(x)[nan_mask]

        return x, nans_indicator


class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, x, **kwargs):
        sel = (x[1:] == x[0]).sum(0) != (
            x.shape[0] - 1
        )  # Indicator of empty features, i.e. all values same (Shape S, B, F)

        print('RemoveEmptyFeaturesEncoderStep', x.shape)

        new_x = select_features(x, sel)

        return (new_x,)


def select_features(x, sel):
    new_x = x.clone()
    for B in range(x.shape[1]):
        if x.shape[1] > 1:
            new_x[:, B, :] = torch.cat(
                [
                    x[:, B, sel[B]],
                    torch.zeros(
                        x.shape[0],
                        x.shape[-1] - sel[B].sum(),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                ],
                -1,
            )
        else:
            # If B == 1, we don't need to append zeros, as the number of features can change
            new_x[:, B, :] = x[:, B, sel[B]]
    return new_x


class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
    def __init__(self, normalize_on_train_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _forward(self, x, single_eval_pos: int, **kwargs):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x_norm = normalize_data(x[:, :normalize_position])
        sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        for B in range(x_norm.shape[1]):
            cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.999).float()
            cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
            sel[B] = cov_mat_sum_below_trace == 1.0

        new_x = select_features(x, sel)

        return (new_x,)


class HandleDuplicatedSamplesEncoderStep(SeqEncStep):
    def __init__(self, normalize_on_train_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _forward(self, x, single_eval_pos: int, **kwargs):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x_norm = normalize_data(x[:, :normalize_position])
        sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        for B in range(x.shape[1]):
            cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.99).float()
            cov_mat, cov_mat.sum(dim=0)
            cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
            sel[B] = cov_mat_sum_below_trace == 1.0

        new_x = select_features(x, sel)

        return (new_x,)


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """
    Transforms the input to a fixed number of features by appending zeros. Also normalizes the input by the number of
    used features in order to keep the variance of the input to 1, even when zeros are appended.
    """

    def __init__(
        self,
        num_features,
        normalize_by_used_features=True,
        normalize_by_sqrt=True,
        **kwargs,
    ):
        """

        :param num_features: The number of features that the input should be transformed to.
        :param normalize_by_used_features: Whether to normalize by the number of used features.
        :param normalize_by_sqrt: Legacy option to not normalize by sqrt, which yields a non constant variance.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt

    def _forward(self, x, **kwargs):
        """
        :param x: Input of shape (seq_len, batch_size, num_features_old)
        :return: A tuple containing the transformed input of shape (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return torch.zeros(
                x.shape[0],
                x.shape[1],
                self.num_features,
                device=x.device,
                dtype=x.dtype,
            )
        if self.normalize_by_used_features:
            sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

            modified_sel = torch.clip(sel.sum(-1).unsqueeze(-1), min=1)
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(self.num_features / modified_sel)
            else:
                x = x * (self.num_features / modified_sel)

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)

        # print('VariableNumFeaturesEncoderStep,', x[:5, :, :6])

        return (x,)


class InputNormalizationEncoderStep(SeqEncStep):
    """
    Normalizes the input in different ways. Can be used to normalize the input to a ranking, remove outliers or
    normalize the input to have unit variance.
    """

    def __init__(
        self,
        normalize_on_train_only,
        normalize_to_ranking,
        normalize_x,
        remove_outliers,
        random_feature_scaling_strength=0.0,
        seed=0,
        **kwargs,
    ):
        """
        :param normalize_on_train_only: If True, the normalization is calculated only on the training set and applied
        to the test set as well. If False, the normalization is calculated on the entire dataset.
        :param normalize_to_ranking: If True, the input is normalized to a ranking.
        :param normalize_x: If True, the input is normalized to have unit variance.
        :param remove_outliers: If True, outliers are removed from the input.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_to_ranking = normalize_to_ranking
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.random_feature_scaling_strength = random_feature_scaling_strength
        self.seed = seed
        # self.gen = torch.Generator()
        self.reset_seed()

    def reset_seed(self):
        pass
        # self.gen.manual_seed(self.seed)

    def _forward(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs,
    ):
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        if self.normalize_to_ranking:
            x = to_ranking_low_mem(x)

        elif self.remove_outliers:
            x = remove_outliers(x, normalize_positions=normalize_position)

        if self.normalize_x:
            x = normalize_data(x, normalize_positions=normalize_position)

        if self.random_feature_scaling_strength > 0.0:
            # if self.gen.device != x.device:
            #    self.gen = torch.Generator(device=x.device)
            #    self.reset_seed()
            x = x * (
                1.0
                + torch.repeat_interleave(
                    torch.randn(
                        x.shape[0],
                        1,
                        x.shape[2],
                        device=x.device,  # , generator=self.gen
                    ),
                    x.shape[1],
                    dim=1,
                )
                * self.random_feature_scaling_strength
            )

        return (x,)


class FrequencyFeatureEncoderStep(SeqEncStep):
    def __init__(self, num_features, num_frequencies, freq_power_base=2.0, **kwargs):
        """
        This is an encoder step that will change the the input x, to also contain sin(x * freq) and cos(x * freq)
        for multiple frequencies.

        :param num_features: The number of features of the input. Only needed to determine `self.num_features_out`.
        :param num_frequencies: The number of frequencies to add, both sin and cos.
        :param freq_power_base: The base of the frequencies. The frequencies will be `freq_power_base`^i for i in range(num_frequencies).
        """
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.num_features = num_features
        self.num_features_out = num_features + 2 * num_frequencies * num_features
        self.freq_power_base = freq_power_base
        # 2.3 * 2 is .99 percentile, so we only add smaller frequencies
        # we add frequencies with a factor of 2
        frequencies = torch.tensor(
            [freq_power_base**i for i in range(num_frequencies)], dtype=torch.float
        )
        frequencies = frequencies / (frequencies[-1] / 2.3 / 2 / 2)
        self.register_buffer("frequencies", frequencies)
        print("using frequencies", frequencies)

    def _forward(
        self, x: torch.Tensor, single_eval_pos: int = None, categorical_inds=None
    ):
        """

        :param x: input of shape (seq_len, batch_size, num_features)
        :param single_eval_pos: - not needed
        :return: A tuple containing the transformed input of shape (seq_len, batch_size, num_features + 2 * num_frequencies * num_features).
        """
        # x.shape = (seq_len, batch_size, num_features)

        extended = x[..., None] / self.frequencies[None, None, None, :] * 2 * torch.pi
        new_features = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)), -1
        )
        new_features = new_features.reshape(*x.shape[:-1], -1)
        return (new_features,)


class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
    """
    Expects input of size 1.
    """

    def __init__(self, num_features, emsize, base_encoder, num_embs=1_000, **kwargs):
        super().__init__(**kwargs)
        assert num_features == 1
        self.num_features = num_features
        self.emsize = emsize
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embs, emsize)
        self.base_encoder = base_encoder

    def _forward(
        self,
        x,
        single_eval_pos: int,
        categorical_inds: list[int],
    ):
        
        print('CategoricalInputEncoderPerFeature', x.shape)

        if categorical_inds is None:
            is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        else:
            utils.print_once("using cateogircal inds")
            assert all(ci in ([0], []) for ci in categorical_inds), categorical_inds
            is_categorical = torch.tensor(
                [ci == [0] for ci in categorical_inds], device=x.device
            )
        if is_categorical.any():
            lx = x[:, is_categorical]
            nan_mask = torch.isnan(lx) | torch.isinf(lx)
            lx = lx.long().clamp(0, self.num_embs - 2)
            lx[nan_mask] = self.num_embs - 1
            categorical_embs = self.embedding(lx.squeeze(-1))
        else:
            categorical_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        if (~is_categorical).any():
            lx = x[:, ~is_categorical]
            continuous_embs = self.base_encoder(lx, single_eval_pos=single_eval_pos)[0]
        else:
            continuous_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        # return (torch.cat((continuous_embs, categorical_embs), dim=1),)
        # above is wrong as we need to preserve order in the batch dimension
        embs = torch.zeros(
            x.shape[0], x.shape[1], self.emsize, device=x.device, dtype=torch.float
        )
        embs[:, is_categorical] = categorical_embs.float()
        embs[:, ~is_categorical] = continuous_embs.float()
        return (embs,)


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


class StyleEmbEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size, num_embeddings=100):
        super().__init__()
        assert num_hyperparameters == 1
        self.em_size = em_size
        self.embedding = nn.Embedding(num_embeddings, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters.squeeze(1))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.device_test_tensor = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # T x B x num_features
        assert self.d_model % x.shape[-1] * 2 == 0
        d_per_feature = self.d_model // x.shape[-1]
        pe = torch.zeros(*x.shape, d_per_feature, device=self.device_test_tensor.device)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        interval_size = 10
        div_term = (
            (1.0 / interval_size)
            * 2
            * math.pi
            * torch.exp(
                torch.arange(
                    0, d_per_feature, 2, device=self.device_test_tensor.device
                ).float()
                * math.log(math.sqrt(2))
            )
        )
        # print(div_term/2/math.pi)
        pe[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        return self.dropout(pe).view(x.shape[0], x.shape[1], self.d_model)


Positional = lambda _, emsize: _PositionalEncoding(d_model=emsize)


class EmbeddingEncoder(nn.Module):
    def __init__(self, num_features, em_size, num_embs=100):
        super().__init__()
        self.num_embs = num_embs
        self.embeddings = nn.Embedding(num_embs * num_features, em_size, max_norm=True)
        self.init_weights(0.1)
        self.min_max = (-2, +2)

    @property
    def width(self):
        return self.min_max[1] - self.min_max[0]

    def init_weights(self, initrange):
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def discretize(self, x):
        split_size = self.width / self.num_embs
        return (x - self.min_max[0] // split_size).int().clamp(0, self.num_embs - 1)

    def forward(self, x):  # T x B x num_features
        x_idxs = self.discretize(x)
        x_idxs += (
            torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) * self.num_embs
        )
        # print(x_idxs,self.embeddings.weight.shape)
        return self.embeddings(x_idxs).mean(-2)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class SqueezeBetween0and1(nn.Module):  # take care of test set here
    def forward(self, x):
        width = x.max(0).values - x.min(0).values
        result = (x - x.min(0).values) / width
        result[(width == 0)[None].repeat(len(x), *[1] * (len(x.shape) - 1))] = 0.5
        return result


def get_normalized_uniform_encoder(encoder_creator):
    """
    This can be used to wrap an encoder that is fed uniform samples in [0,1] and normalizes these to 0 mean and 1 std.
    For example, it can be used as `encoder_creator = get_normalized_uniform_encoder(encoders.Linear)`, now this can
    be initialized with `encoder_creator(feature_dim, in_dim)`.
    :param encoder:
    :return:
    """
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.5, math.sqrt(1 / 12)), encoder_creator(in_dim, out_dim)
    )


def get_normalized_encoder(encoder_creator, data_std):
    return lambda in_dim, out_dim: nn.Sequential(
        Normalize(0.0, data_std), encoder_creator(in_dim, out_dim)
    )


def get_log_dims(x, eps=1e-10):
    logged_x = ((x + eps).log() - math.log(eps)) / (math.log(1.0 + eps) - math.log(eps))
    return logged_x


def add_log_neglog_dims(x, eps=1e-10):
    logged_x = get_log_dims(x, eps) / 2.0
    neglogged_x = 1 - get_log_dims(1 - x, eps) / 2.0
    logged_x[x > 0.5] = neglogged_x[x > 0.5]
    return torch.stack([x, logged_x], -1).view(*x.shape[:-1], -1)


class AddLogNegLogDims(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return add_log_neglog_dims(x, self.eps)


def get_logdim_encoder(encoder_creator, eps=1e-10):
    return lambda in_dim, out_dim: nn.Sequential(
        AddLogNegLogDims(eps), encoder_creator(in_dim * 2, out_dim)
    )


class ZNormalize(nn.Module):
    def forward(self, x):
        std = x.std(-1, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(-1, keepdim=True)) / std


class ZNormalizePerDataset(nn.Module):
    def forward(self, x):
        std = x.std(0, keepdim=True)
        std[std == 0.0] = 1.0
        return (x - x.mean(0, keepdim=True)) / std


class AppendEmbeddingEncoder(nn.Module):
    def __init__(self, base_encoder, num_features, emsize):
        super().__init__()
        self.num_features = num_features
        self.base_encoder = base_encoder
        self.emb = nn.Parameter(torch.zeros(emsize))

    def forward(self, x):
        if (x[-1] == 1.0).all():
            append_embedding = True
        else:
            assert (x[-1] == 0.0).all(), (
                "You need to specify as last position whether to append embedding. "
                "If you don't want this behavior, please use the wrapped encoder instead."
            )
            append_embedding = False
        x = x[:-1]
        encoded_x = self.base_encoder(x)
        if append_embedding:
            encoded_x = torch.cat(
                [encoded_x, self.emb[None, None, :].repeat(1, encoded_x.shape[1], 1)], 0
            )
        return encoded_x


def get_append_embedding_encoder(encoder_creator):
    return lambda num_features, emsize: AppendEmbeddingEncoder(
        encoder_creator(num_features, emsize), num_features, emsize
    )


class NoMeanEncoder(nn.Module):
    """
    This can be useful for any prior that is translation invariant in x or y.
    A standard GP for example is translation invariant in x.
    That is, GP(x_test+const,x_train+const,y_train) = GP(x_test,x_train,y_train).
    """

    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, x):
        return self.base_encoder(x - x.mean(0, keepdim=True))


def get_no_mean_encoder(encoder_creator):
    return lambda num_features, emsize: NoMeanEncoder(
        encoder_creator(num_features, emsize)
    )


def get_linear_encoder_generator(in_keys):
    def get_linear_encoder(num_features, emsize):
        encoder = SequentialEncoder(
            LinearInputEncoderStep(
                num_features, emsize, in_keys=in_keys, out_keys=["output"]
            ),
            output_key="output",
        )
        return encoder

    return get_linear_encoder


class Linear(nn.Linear):
    def __init__(self, num_features, emsize, replace_nan_by_zero=False):
        super().__init__(num_features, emsize)
        self.num_features = num_features
        self.emsize = emsize
        self.replace_nan_by_zero = replace_nan_by_zero

    def forward(self, x):
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)
        return super().forward(x)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("replace_nan_by_zero", True)


class Conv(nn.Module):
    def __init__(self, input_size, emsize):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [nn.Conv2d(64 if i else 1, 64, 3) for i in range(5)]
        )
        self.linear = nn.Linear(64, emsize)

    def forward(self, x):
        size = math.isqrt(x.shape[-1])
        assert size * size == x.shape[-1]
        x = x.reshape(*x.shape[:-1], 1, size, size)
        for conv in self.convs:
            if x.shape[-1] < 4:
                break
            x = conv(x)
            x.relu_()
        x = nn.AdaptiveAvgPool2d((1, 1))(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class CanEmb(nn.Embedding):
    def __init__(
        self, num_features, num_embeddings: int, embedding_dim: int, *args, **kwargs
    ):
        assert embedding_dim % num_features == 0
        embedding_dim = embedding_dim // num_features
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)

    def forward(self, x):
        lx = x.long()
        assert (lx == x).all(), "CanEmb only works with tensors of whole numbers"
        x = super().forward(lx)
        return x.view(*x.shape[:-2], -1)


def get_Canonical(num_classes):
    return lambda num_features, emsize: CanEmb(num_features, num_classes, emsize)


def get_Embedding(num_embs_per_feature=100):
    return lambda num_features, emsize: EmbeddingEncoder(
        num_features, emsize, num_embs=num_embs_per_feature
    )


##### TARGET ENCODERS #####
# TODO: Merge with InputEncoder
class TargetEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        single_eval_pos: int,
    ):
        raise NotImplementedError


class MulticlassClassificationTargetEncoder(SeqEncStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def flatten_targets(y: torch.Tensor):

        y = (y.unsqueeze(-1) > torch.unique(y)).sum(axis=-1)

        return y

    def _forward(self, y: torch.Tensor, single_eval_pos: int = None):
        
        assert not (
            y.isnan().any() and self.training
        ), "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
        y_new = y.clone()
        for B in range(y.shape[1]):
            y_new[:, B, :] = MulticlassClassificationTargetEncoder.flatten_targets(
                y[:, B, :]
            )
        return (y_new,)
