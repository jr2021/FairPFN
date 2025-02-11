import copy
import math

import einops
import torch
import torch.nn as nn
from torch.nn import Module

import utils
from . import encoders
from .layer import TransformerEncoderLayer, PerFeatureEncoderLayer
from utils import SeqBN, bool_mask_to_att_mask, mean_nested_structures, print_once


def make_decoder_dict(decoder_description_dict, ninp, nhid):
    if decoder_description_dict is None or len(decoder_description_dict) == 0:
        return None
    initialized_decoder_dict = {}
    for decoder_key in decoder_description_dict:
        decoder_model, decoder_n_out = decoder_description_dict[decoder_key]
        if decoder_model is None:
            initialized_decoder_dict[decoder_key] = nn.Sequential(
                nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, decoder_n_out)
            )
        else:
            initialized_decoder_dict[decoder_key] = decoder_model(
                ninp, nhid, decoder_n_out
            )
        # print(
        #     "Initialized decoder for",
        #     decoder_key,
        #     "with",
        #     decoder_description_dict[decoder_key],
        #     " and nout",
        #     decoder_n_out,
        # )
    return torch.nn.ModuleDict(initialized_decoder_dict)


class TransformerModel(nn.Module):
    """
    This will perform a forward-pass (possibly recording gradients) of the model.
    We have multiple interfaces we support with this model:

    model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
    model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
    model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
    """

    def __init__(
        self,
        encoder,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.0,
        style_encoder=None,
        y_encoder=None,
        pos_encoder=None,
        decoder_dict=None,
        input_normalization=False,
        init_method=None,
        pre_norm=False,
        activation="gelu",
        recompute_attn=False,
        num_global_att_tokens=0,
        full_attention=False,
        efficient_eval_masking=True,
        decoder_once_dict=None,
        return_all_outputs=False,
        save_trainingset_representations=False,
        use_flash_attention=False,
        use_separate_decoder=False,
        nlayers_decoder=None,
        custom_attention_style_and_activation_and_scale=None,
        use_zero_attention=False,
        use_encoder_compression_layer=False,
        share_key_and_value_attention_proj=False,
        scale_softmax_w_dataset_size=False,
        min_num_layers_layer_dropout=None,
        repeat_same_layer=False,  # this means the weights of all layers are tied
        recompute_layer=False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        layer_creator = lambda: TransformerEncoderLayer(
            ninp,
            nhead,
            nhid,
            dropout,
            activation=activation,
            pre_norm=pre_norm,
            recompute_attn=recompute_attn,
            save_trainingset_representations=save_trainingset_representations,
            use_flash_attention=use_flash_attention,
            custom_attention_style_and_activation_and_scale=custom_attention_style_and_activation_and_scale,
            use_zero_attention=use_zero_attention,
            share_kv_proj_weights=share_key_and_value_attention_proj,
            scale_softmax_w_dataset_size=scale_softmax_w_dataset_size,
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator,
            nlayers_encoder,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
            recompute_each_layer=recompute_layer,
        )
        self.transformer_decoder = None
        if use_separate_decoder:
            self.transformer_decoder = LayerStack(
                layer_creator,
                nlayers_decoder,
                min_num_layers_layer_dropout=min_num_layers_layer_dropout,
                recompute_each_layer=recompute_layer,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert use_separate_decoder

            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression, ninp
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator,
                2,
            )

        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder
        self.return_all_outputs = return_all_outputs

        self.decoder_dict = make_decoder_dict(decoder_dict, ninp, nhid)
        self.decoder_dict_once = make_decoder_dict(decoder_once_dict, ninp, nhid)

        # N(0,1) is the initialization as the default of nn.Embedding
        self.decoder_dict_once_embeddings = (
            torch.nn.Parameter(torch.randn((len(self.decoder_dict_once), 1, ninp)))
            if self.decoder_dict_once is not None
            else None
        )
        # nn.Embedding(len(self.decoder_dict.keys()), nhid)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        if num_global_att_tokens is not None:
            assert not full_attention
        self.global_att_embeddings = (
            nn.Embedding(num_global_att_tokens, ninp) if num_global_att_tokens else None
        )
        self.full_attention = full_attention
        self.efficient_eval_masking = efficient_eval_masking

        self.nhid = nhid

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("efficient_eval_masking", False)
        if not hasattr(self, "decoder_dict_once"):
            self.__dict__.setdefault("decoder_dict_once", None)
        if hasattr(self, "decoder") and not hasattr(self, "decoder_dict"):
            self.add_module("decoder_dict", nn.ModuleDict({"standard": self.decoder}))
        self.__dict__.setdefault("return_all_outputs", False)
        if not hasattr(self, "transformer_decoder"):
            self.__dict__.setdefault("transformer_decoder", None)

        def add_approximate_false(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault("approximate", "none")

        self.apply(add_approximate_false)

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return bool_mask_to_att_mask(mask)

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        return bool_mask_to_att_mask(mask)

    def init_weights(self):
        initrange = 1.0
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            attns = (
                layer.self_attn
                if isinstance(layer.self_attn, nn.ModuleList)
                else [layer.self_attn]
            )
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

    def forward(self, *args, **kwargs):
        """
        This will perform a forward-pass (possibly recording gradients) of the model.
        We have multiple interfaces we support with this model:

        model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
        model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        """

        if "train_x" in kwargs:
            assert len(args) == 0
            args = [kwargs["train_x"], kwargs["train_y"], kwargs["test_x"]]
            del kwargs["train_x"]
            del kwargs["train_y"]
            del kwargs["test_x"]

        if len(args) == 3:
            # case model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
            accepted_kwargs = {
                "src_mask",
                "style",
                "only_return_standard_out",
                "half_layers",
                "categorical_inds",
            }
            assert all(
                kwarg in accepted_kwargs for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - accepted_kwargs}"
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            style = kwargs.pop("style", None)
            return self._forward(
                (style, x, args[1]), single_eval_pos=len(args[0]), **kwargs
            )
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            accepted_kwargs = {
                "src_mask",
                "single_eval_pos",
                "only_return_standard_out",
                "half_layers",
                "categorical_inds",
            }
            assert all(
                kwarg in accepted_kwargs for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - accepted_kwargs}"
            return self._forward(*args, **kwargs)
        else:
            raise ValueError(
                "Unrecognized input. Please follow the doc string exactly."
            )

    def _forward(
        self,
        src,
        src_mask=None,
        single_eval_pos=None,
        only_return_standard_out=True,
        half_layers=False,
        categorical_inds=None,
    ):
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src

        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]

        # make this optional for another encoder / old encoder!?

        x_src = self.encoder(
            x_src, single_eval_pos=single_eval_pos, categorical_inds=categorical_inds
        )

        if self.decoder_dict_once is not None:
            x_src = torch.cat(
                [x_src, self.decoder_dict_once_embeddings.repeat(1, x_src.shape[1], 1)],
                dim=0,
            )

        unsqueeze_y = (
            lambda v: v.unsqueeze(-1) if len(v.shape) < len(x_src.shape) else v
        )
        y_src = (
            {k: unsqueeze_y(v) for k, v in y_src.items()}
            if isinstance(y_src, dict)
            else unsqueeze_y(y_src)
        )

        if y_src is not None:
            y_src = self.y_encoder(y_src, single_eval_pos=single_eval_pos)

        if self.style_encoder:
            assert (
                style_src is not None
            ), "style_src must be given if style_encoder is used"
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_src.device)

        assert not (
            torch.isnan(style_src).any() or torch.isinf(style_src).any()
        ), "style_src is nan"
        assert not (
            torch.isnan(x_src).any() or torch.isinf(x_src).any()
        ), "x_src is nan"
        assert not (
            torch.isnan(y_src).any() or torch.isinf(y_src).any()
        ), "y_src is nan"

        global_src = (
            torch.tensor([], device=x_src.device)
            if self.global_att_embeddings is None
            else self.global_att_embeddings.weight.unsqueeze(1).repeat(
                1, x_src.shape[1], 1
            )
        )

        if src_mask is not None:
            assert self.global_att_embeddings is None or isinstance(src_mask, tuple)

        if src_mask is None:  # default
            if self.global_att_embeddings is None:  # default
                full_len = len(x_src) + len(style_src)
                if self.full_attention:
                    src_mask = bool_mask_to_att_mask(
                        torch.ones((full_len, full_len), dtype=torch.bool)
                    ).to(x_src.device)
                elif self.efficient_eval_masking:  # default
                    src_mask = single_eval_pos + len(style_src)
                else:
                    src_mask = self.generate_D_q_matrix(
                        full_len, len(x_src) - single_eval_pos
                    ).to(x_src.device)
            else:
                src_mask = (
                    self.global_att_embeddings.num_embeddings,
                    len(x_src) + len(style_src),
                )

        train_x = x_src[:single_eval_pos]
        if y_src is not None:
            train_x = train_x + y_src[:single_eval_pos]
        src = torch.cat([global_src, style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)

        if (
            self.transformer_decoder is not None
        ):  # Decoder is not none when in use_decoder setup
            assert not half_layers
            train_len = len(global_src) + len(style_src) + len(train_x)
            train_out = self.transformer_encoder(src[:train_len], src_mask=src_mask)

            if self.global_att_embeddings_for_compression is not None:
                train_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=train_out,
                    src_mask=0,
                )

            test_output = self.transformer_decoder(
                src[train_len:], src_mask=0, att_src=train_out
            )
            output = torch.cat([train_out, test_output], 0)
        else:
            extra_kwargs = {}
            if half_layers:
                extra_kwargs["half_layers"] = half_layers
            output = self.transformer_encoder(
                src, src_mask=src_mask, **extra_kwargs
            )  # can make this mask instead for backwards compatibility

        num_prefix_positions = len(style_src) + (
            self.global_att_embeddings.num_embeddings
            if self.global_att_embeddings
            else 0
        )
        if self.return_all_outputs:
            out_range_start = num_prefix_positions
        else:
            out_range_start = single_eval_pos + num_prefix_positions

        # In the line below, we use the indexing feature, that we have `x[i:None] == x[i:]`
        out_range_end = (
            -len(self.decoder_dict_once_embeddings)
            if self.decoder_dict_once is not None
            else None
        )

        # take care the output once are counted from the end
        output_once = (
            {
                k: v(output[-(i + 1)])
                for i, (k, v) in enumerate(self.decoder_dict_once.items())
            }
            if self.decoder_dict_once is not None
            else {}
        )

        output = (
            {
                k: v(output[out_range_start:out_range_end])
                for k, v in self.decoder_dict.items()
            }
            if self.decoder_dict is not None
            else {}
        )

        if only_return_standard_out:
            output = output["standard"]

        if output_once:
            return output, output_once

        return output

    @torch.no_grad()
    def init_from_small_model(self, small_model):
        assert (
            isinstance(self.decoder, nn.Linear)
            and isinstance(self.encoder, (nn.Linear, nn.Sequential))
            and isinstance(self.y_encoder, (nn.Linear, nn.Sequential))
        )

        def set_encoder_weights(my_encoder, small_model_encoder):
            my_encoder_linear, small_encoder_linear = (
                (my_encoder, small_model_encoder)
                if isinstance(my_encoder, nn.Linear)
                else (my_encoder[-1], small_model_encoder[-1])
            )
            small_in_dim = small_encoder_linear.out_features
            my_encoder_linear.weight.zero_()
            my_encoder_linear.bias.zero_()
            my_encoder_linear.weight[:small_in_dim] = small_encoder_linear.weight
            my_encoder_linear.bias[:small_in_dim] = small_encoder_linear.bias

        set_encoder_weights(self.encoder, small_model.encoder)
        set_encoder_weights(self.y_encoder, small_model.y_encoder)

        small_in_dim = small_model.decoder.in_features

        self.decoder.weight[:, :small_in_dim] = small_model.decoder.weight
        self.decoder.bias = small_model.decoder.bias

        for my_layer, small_layer in zip(
            self.transformer_encoder.layers, small_model.transformer_encoder.layers
        ):
            small_hid_dim = small_layer.linear1.out_features
            my_in_dim = my_layer.linear1.in_features

            # packed along q,k,v order in first dim
            my_in_proj_w = my_layer.self_attn.in_proj_weight
            small_in_proj_w = small_layer.self_attn.in_proj_weight

            my_in_proj_w.view(3, my_in_dim, my_in_dim)[
                :, :small_in_dim, :small_in_dim
            ] = small_in_proj_w.view(3, small_in_dim, small_in_dim)
            my_layer.self_attn.in_proj_bias.view(3, my_in_dim)[
                :, :small_in_dim
            ] = small_layer.self_attn.in_proj_bias.view(3, small_in_dim)

            my_layer.self_attn.out_proj.weight[
                :small_in_dim, :small_in_dim
            ] = small_layer.self_attn.out_proj.weight
            my_layer.self_attn.out_proj.bias[
                :small_in_dim
            ] = small_layer.self_attn.out_proj.bias

            my_layer.linear1.weight[
                :small_hid_dim, :small_in_dim
            ] = small_layer.linear1.weight
            my_layer.linear1.bias[:small_hid_dim] = small_layer.linear1.bias

            my_layer.linear2.weight[
                :small_in_dim, :small_hid_dim
            ] = small_layer.linear2.weight
            my_layer.linear2.bias[:small_in_dim] = small_layer.linear2.bias

            my_layer.norm1.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm1.weight
            )
            my_layer.norm2.weight[:small_in_dim] = (
                math.sqrt(small_in_dim / my_in_dim) * small_layer.norm2.weight
            )

            my_layer.norm1.bias[:small_in_dim] = small_layer.norm1.bias
            my_layer.norm2.bias[:small_in_dim] = small_layer.norm2.bias


class PerFeatureTransformer(Module):
    def init_weights(self, zero_init=True):
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.init_method is not None:
            self.apply(self.init_method)
        # print("Using zero init:", zero_init)
        if zero_init:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                if layer.linear2.bias is not None:
                    nn.init.zeros_(layer.linear2.bias)
                if hasattr(layer, "linear4"):
                    utils.print_once("Initializing linear4 to zeros")
                    nn.init.zeros_(layer.linear4.weight)
                    if layer.linear4.bias is not None:
                        nn.init.zeros_(layer.linear4.bias)
                attns = (
                    layer.self_attn_between_features,
                    layer.self_attn_between_items,
                )
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    if attn.out_proj.bias is not None:
                        nn.init.zeros_(attn.out_proj.bias)

    def __init__(
        self,
        encoder,
        ninp,
        nhead,
        nhid,
        nlayers,
        y_encoder=None,
        decoder_dict=None,
        init_method=None,
        activation="gelu",
        recompute_layer=False,
        min_num_layers_layer_dropout=None,
        repeat_same_layer=False,
        final_unified_transformer_config=None,
        dag_pos_enc_dim=0,
        features_per_group=1,
        feature_positional_embedding=None,
        zero_init=True,
        use_separate_decoder=False,
        nlayers_decoder=None,
        use_encoder_compression_layer=False,
        **layer_kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group

        print(y_encoder)
        print(encoder)

        layer_creator = lambda: PerFeatureEncoderLayer(
            ninp, nhead, nhid, activation, batch_first=True, **layer_kwargs
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator,
            nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            self.transformer_decoder = LayerStack(
                layer_creator,
                nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert use_separate_decoder

            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression, ninp
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator,
                2,
            )

        self.unified_transformer_encoder = None
        if final_unified_transformer_config is not None:
            # Example for final_unified_transfomer_config:
            # final_unified_transfomer_config = {
            #    "nlayers": 2,
            #    "aggregation": "mean",
            #    "d_model": 512,
            #    "nhead": 512 // 32,
            # }

            nlayers_unified = final_unified_transformer_config.pop("nlayers")
            self.unified_aggregation = final_unified_transformer_config.pop(
                "aggregation"
            )
            layer_creator_unfied = lambda: TransformerEncoderLayer(
                **{"batch_first": False, **final_unified_transformer_config}
            )
            if repeat_same_layer:
                layer = layer_creator()
                layer_creator_unfied = lambda: layer
            self.unified_transformer_encoder = LayerStack(
                layer_creator_unfied,
                nlayers_unified,
            )
            self.project_to_unified = nn.Linear(
                ninp, final_unified_transformer_config["d_model"]
            )

            ninp = final_unified_transformer_config["d_model"]
            nhid = self.unified_transformer_encoder.layers[0].linear1.out_features

        self.decoder_dict = make_decoder_dict(decoder_dict, ninp, nhid)

        self.feature_positional_embedding = feature_positional_embedding
        # print("using feature positional embedding", self.feature_positional_embedding)
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim

        self.init_weights(zero_init=zero_init)

    def reset_save_peak_mem_factor(self, factor=None):
        """
        Setting this factor > 1 will cause the model to save more memory during the forward pass in inference mode.
        A value of 8 is good for a 4x larger width in the fully-connected layers.
        And yields a situation were we need around 2 * num_features * num_items * emsize * 2 bytes of memory for a forward pass (using mixed precision).
        WARN: It should only be used with post-norm.
        :param factor: recommended to be 8
        :return: None
        """
        for layer in self.transformer_encoder.layers:
            assert hasattr(
                layer, "save_peak_mem_factor"
            ), "Layer does not have save_peak_mem_factor"
            layer.save_peak_mem_factor = factor

    def __setstate__(self, state):
        state.setdefault("features_per_group", 1)
        state.setdefault("feature_positional_embedding", None)
        super().__setstate__(state)

    def forward(self, *args, **kwargs):
        """
        This will perform a forward-pass (possibly recording gradients) of the model.
        We have multiple interfaces we support with this model:

        model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
        model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        """
        if "train_x" in kwargs:
            assert len(args) == 0
            args = [kwargs["train_x"], kwargs["train_y"], kwargs["test_x"]]
            del kwargs["train_x"]
            del kwargs["train_y"]
            del kwargs["test_x"]

        if len(args) == 3:
            # case model(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
            assert all(
                kwarg
                in {
                    "style",
                    "only_return_standard_out",
                    "data_dags",
                    "categorical_inds",
                    "half_layers",
                }
                for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'style', 'only_return_standard_out'}}"
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            return self._forward(x, args[1], single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case model((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case model((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            assert all(
                kwarg
                in {
                    "src_mask",
                    "single_eval_pos",
                    "only_return_standard_out",
                    "data_dags",
                    "categorical_inds",  # todo reuse this for below
                    "half_layers",
                }
                for kwarg in kwargs.keys()
            ), f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'single_eval_pos', 'only_return_standard_out'}}"
            if len(args[0]) == 3:
                return self._forward(*args[0][1:], style=args[0][0], **kwargs)
            else:
                assert (
                    len(args[0]) == 2
                ), f"Expected tuple of length 2 or 3, got {len(args[0])}"
                return self._forward(*args[0], **kwargs)
        else:
            raise ValueError(
                "Unrecognized input. Please follow the doc string exactly."
            )

    def _forward(
        self,
        x,
        y,
        single_eval_pos=None,
        only_return_standard_out=True,
        style=None,
        data_dags=None,
        categorical_inds=None,
        half_layers=False,
    ):
        assert style is None
        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}

        if isinstance(y, dict):
            assert "main" in set(y.keys()), f"Main must be in input keys: {y.keys()}."
        else:
            y = {"main": y}

        seq_len, batch_size, num_features = x["main"].shape

        # print('PerFeatureTransformer._forward before', x["main"][:5, :, :5])
        # print('PerFeatureTransformer._forward before SHAPE', x["main"].shape)

        # print(f"missing_to_next: {missing_to_next}", 'num_features', num_features, 'features_per_group', self.features_per_group)
        for k in x:
            num_features_ = x[k].shape[2]

            # pad to multiple of features_per_group
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )
        for k in x:
            # print('x.shape', x.shape)
            x[k] = einops.rearrange(
                x[k], "s b (f n) -> b s f n", n=self.features_per_group
            )  # s b f -> b s #groups #features_per_group



        if categorical_inds is not None:
            new_categorical_inds = []
            for ci in categorical_inds:
                num_subgroups = x["main"].shape[2]
                new_categorical_inds += [
                    [
                        i - subgroup * self.features_per_group
                        for i in ci
                        if (
                            subgroup * self.features_per_group
                            <= i
                            < (subgroup + 1) * self.features_per_group
                        )
                    ]
                    for subgroup in range(num_subgroups)
                ]
            categorical_inds = new_categorical_inds

        for k in y:
            if len(y[k].shape) == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert (
                    k != "main" or y[k].shape[1] == single_eval_pos
                ), "For main y, y must not be given for target time steps (Otherwise the solution is leaked)."
                if y[k].shape[1] == single_eval_pos:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)

        embedded_y = self.y_encoder(y, single_eval_pos=single_eval_pos).transpose(0, 1)
        del y
        assert not torch.isnan(
            embedded_y
        ).any(), f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers to the ys that are not fully provided (test set missing)"

        extra_encoders_args = {}
        if categorical_inds is not None and isinstance(
            self.encoder, encoders.SequentialEncoder
        ):
            extra_encoders_args["categorical_inds"] = categorical_inds

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")

        # print('PerFeatureTransformer._forward after', x["main"])
        # print('PerFeatureTransformer._forward after SHAPE', x["main"].shape)

        embedded_x = einops.rearrange(
            self.encoder(
                x,
                single_eval_pos=single_eval_pos,
                **extra_encoders_args,
            ),
            "s (b f) e -> b s f e",
            b=embedded_y.shape[0],
        )  # b s f 1 -> b s f e
        del x

        embedded_x, embedded_y = self.add_embeddings(
            embedded_x, embedded_y, data_dags, num_features, seq_len
        )
        del data_dags

        embedded_input = torch.cat(
            (embedded_x, embedded_y.unsqueeze(2)), dim=2
        )  # b s f e + b s 1 e -> b s f+1 e

        assert not torch.isnan(
            embedded_input
        ).any(), (
            f"{torch.isnan(embedded_x).any()=} and {torch.isnan(embedded_y).any()=}"
        )
        del embedded_y, embedded_x

        encoder_out = self.transformer_encoder(
            (
                embedded_input
                if not self.transformer_decoder
                else embedded_input[:, :single_eval_pos]
            ),
            single_eval_pos=single_eval_pos,
            half_layers=half_layers,
        )  # b s f+1 e -> b s f+1 e

        # If we are using a decoder
        if self.transformer_decoder:
            print_once("Using separate decoder")
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos

            if self.global_att_embeddings_for_compression is not None:
                # TODO: fixed number of compression tokens
                train_encoder_out = self.encoder_compression_layer(
                    self.global_att_embeddings_for_compression,
                    att_src=encoder_out[:, single_eval_pos],
                    single_eval_pos=single_eval_pos,
                )

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)
            del test_encoder_out

        del embedded_input

        if self.unified_transformer_encoder:
            projected_out = self.project_to_unified(encoder_out)
            if self.unified_aggregation == "max":
                encoder_out = projected_out.max(-2).values
            elif self.unified_aggregation == "mean":
                encoder_out = projected_out.mean(-2)
            elif self.unified_aggregation == "last":
                encoder_out = projected_out[:, :, -1]
            else:
                raise NotImplementedError("unified_aggregation must be max or mean")
            encoder_out = self.unified_transformer_encoder(
                encoder_out.transpose(0, 1), src_mask=single_eval_pos
            )
            test_encoder_out = encoder_out[single_eval_pos:]  # out: s b e
            train_encoder_out = encoder_out[:single_eval_pos]  # out: s b e
        else:
            test_encoder_out = encoder_out[:, single_eval_pos:, -1].transpose(
                0, 1
            )  # out: s b e
            train_encoder_out = encoder_out[:, :single_eval_pos, -1].transpose(
                0, 1
            )  # out: s b e

        output_decoded = (
            {k: v(test_encoder_out) for k, v in self.decoder_dict.items()}
            if self.decoder_dict is not None
            else {}
        )
        output_decoded["train_embeddings"] = train_encoder_out

        if only_return_standard_out:
            output_decoded = output_decoded["standard"]

        return output_decoded

    def add_embeddings(self, x, y, data_dags, num_features, seq_len):
        if self.feature_positional_embedding == "normal_rand_vec":
            embs = torch.randn((x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
            x += embs[None, None]
        elif self.feature_positional_embedding == "uni_rand_vec":
            embs = (
                torch.rand((x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype) * 2
                - 1
            )
            x += embs[None, None]
        elif self.feature_positional_embedding == "learned":
            w = self.feature_positional_embedding_embeddings.weight
            embs = w[torch.randint(0, w.shape[0], (x.shape[2],))]
            x += embs[None, None]
        elif self.feature_positional_embedding == "subspace":
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4), device=x.device, dtype=x.dtype
            )
            embs = self.feature_positional_embedding_embeddings(embs)
            x += embs[None, None]
        else:
            assert self.feature_positional_embedding is None

        # TODO should this go into encoder?
        # could also be made a bit more concise by moving down to operate on full_x
        if data_dags is not None:
            for b_i, data_dag in enumerate(data_dags):
                g_ = data_dag.copy()  # type: nx.DiGraph
                while utils.add_direct_connections(g_):
                    pass
                subgraph = g_.subgraph(
                    [
                        n
                        for n, info in g_.nodes.items()
                        if (info["is_feature"] or info["is_target"])
                    ]
                )
                k = self.dag_pos_enc_dim
                assert k
                utils.add_pos_emb(subgraph, k=k)
                graph_pos_embs_features = torch.zeros(
                    (num_features, k)
                )  # shape: (num_features, k)
                graph_pos_embs_targets = torch.zeros((1, k))  # shape: (num_targets, k)
                for node, node_info in subgraph.nodes.items():
                    for feature_idx in node_info.get("feature_idxs", []):
                        graph_pos_embs_features[feature_idx] = node_info[
                            "positional_encoding"
                        ]
                    for target_idx in node_info.get("target_idxs", []):
                        graph_pos_embs_targets[target_idx] = node_info[
                            "positional_encoding"
                        ]

                # assert ((graph_pos_embs_features == -1000)[:-1].int() <= (graph_pos_embs_features == -1000)[1:].int()).all(), graph_pos_embs_features.mean(1)

                # print('n',torch.isnan(graph_pos_embs_features).any(), torch.isnan(graph_pos_embs_targets).any())
                # print('o', torch.isnan(x).any(), torch.isnan(y).any())

                graph_pos_embs_targets -= graph_pos_embs_features.mean(0, keepdim=True)
                graph_pos_embs_features -= graph_pos_embs_features.mean(0, keepdim=True)

                graph_pos_embs_features = graph_pos_embs_features[None].expand(
                    seq_len, -1, -1
                )
                x[b_i, :, :, :k] += graph_pos_embs_features.to(y.device, y.dtype)

                graph_pos_embs_targets = (
                    graph_pos_embs_targets[None].expand(seq_len, -1, -1).squeeze(-2)
                )
                y[b_i, :, :k] += graph_pos_embs_targets.to(y.device, y.dtype)
        else:
            assert not hasattr(self, "dag_pos_enc_dim") or not self.dag_pos_enc_dim

        return x, y


from torch.utils.checkpoint import checkpoint
from functools import partial


class LayerStack(Module):
    """
    Similar to nn.Sequential, but with support for passing keyword arguments to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        layer_creator,
        num_layers,
        recompute_each_layer=False,
        min_num_layers_layer_dropout=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(self, x, half_layers=False, **kwargs):
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                self.min_num_layers_layer_dropout, self.num_layers + 1, (1,)
            ).item()
        for i, layer in enumerate(self.layers[:n_layers]):
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x)
            else:
                x = layer(x, **kwargs)

        return x


class Ensemble(torch.nn.Module):
    """
    Ensemble of models with the same input and output structure.
    This could for example be a list of `TransformerModel`s and `PerFeatureTransformer`s.
    """

    def __init__(self, models: list[PerFeatureTransformer]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.criterion = models[0].criterion

    def forward(self, *args, **kwargs):
        return mean_nested_structures([m(*args, **kwargs) for m in self.models])

    def reset_save_peak_mem_factor(self, factor=None):
        for m in self.models:
            m.reset_save_peak_mem_factor(factor)
