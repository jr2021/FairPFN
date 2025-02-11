import math
from functools import partial
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import (
    _get_activation_fn,
    Module,
    Tensor,
    Optional,
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
)

from torch.utils.checkpoint import checkpoint

import utils


def get_cu_seqlens(batch_size, seqlen, device):
    return torch.arange(
        0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
    )


def flash_attn_forward(
    mha: torch.nn.modules.transformer.MultiheadAttention,
    x_q: torch.Tensor,
    x_kv: torch.Tensor,
    share_kv: bool = False,
    scale_softmax_w_dataset=False,
    multi_query_factor: tp.Optional[int] = None,
):
    import einops
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_kvpacked_func,
        flash_attn_unpadded_qkvpacked_func,
        flash_attn_unpadded_func,
    )

    assert not mha.dropout
    if x_q.numel() == 0:
        return torch.empty_like(x_q)
    x_q_is_x_kv = x_q is x_kv
    if not mha.batch_first:
        x_q = x_q.transpose(0, 1)
        x_kv = x_kv.transpose(0, 1)

    d = mha.in_proj_weight.shape[1]
    if not share_kv:
        in_proj_weight = mha.in_proj_weight  # 3 * em x em
    else:
        in_proj_weight = torch.cat(
            (
                mha.in_proj_weight[:d],
                mha.in_proj_weight[:d],
                mha.in_proj_weight[2 * d :],
            ),
            dim=0,
        )
        assert in_proj_weight.shape == mha.in_proj_weight.shape, (
            in_proj_weight.shape,
            mha.in_proj_weight.shape,
        )

    softmax_scale = None
    if scale_softmax_w_dataset is not False:
        attend_to_count = x_kv.shape[1]
        if scale_softmax_w_dataset is True:  # use the default
            softmax_scale = (
                math.sqrt(attend_to_count) / math.sqrt(1000) / math.sqrt(mha.head_dim)
            )  # 1000 coming since 1000 is a common dataset size
        elif isinstance(scale_softmax_w_dataset, tuple):
            exponent, constant_divider = scale_softmax_w_dataset
            softmax_scale = (
                (attend_to_count**exponent)
                / constant_divider
                / math.sqrt(mha.head_dim)
            )

    if x_q_is_x_kv:
        assert not multi_query_factor
        x = x_q
        in_projected_x = torch.einsum("...d,ed->...e", x, in_proj_weight)
        if mha.in_proj_bias is not None:
            in_projected_x = in_projected_x + mha.in_proj_bias
        projected = einops.rearrange(
            in_projected_x,
            "bs sl (three nh hdim) -> (bs sl) three nh hdim",
            three=3,
            nh=mha.num_heads,
        )
        batch_size, seqlen, d = x.shape
        cu_seqlens = get_cu_seqlens(batch_size, seqlen, x.device)
        post_attention = flash_attn_unpadded_qkvpacked_func(
            projected.half(), cu_seqlens, seqlen, 0.0, softmax_scale=softmax_scale
        )  # projected[:,:,0], projected[:,:,1], projected[:,:,2])
    else:
        in_projected_q = torch.einsum("...d,ed->...e", x_q, in_proj_weight[:d])
        in_projected_kv = torch.einsum("...d,ed->...e", x_kv, in_proj_weight[d:])
        if mha.in_proj_bias is not None:
            in_projected_q = in_projected_q + mha.in_proj_bias[:d]
            in_projected_kv = in_projected_kv + mha.in_proj_bias[d:]
        q = einops.rearrange(
            in_projected_q, "bs sl (nh hdim) -> (bs sl) nh hdim", nh=mha.num_heads
        )
        kv = einops.rearrange(
            in_projected_kv,
            "bs sl (two nh hdim) -> (bs sl) two nh hdim",
            two=2,
            nh=mha.num_heads,
        )
        if multi_query_factor:
            utils.print_once("using multi query")
            # a multi_query_factor of 1 would be the same as not using multi_query_factor
            if mha.num_heads > multi_query_factor:
                kv = kv[:, :, : mha.num_heads // multi_query_factor, :].repeat(
                    1, 1, multi_query_factor, 1
                )
            else:
                # expand only works for dimension with size 1
                kv = kv[:, :, :1, :].expand(-1, -1, mha.num_heads, -1)

        post_attention = flash_attn_unpadded_kvpacked_func(
            q.half(),
            kv.half(),
            get_cu_seqlens(*x_q.shape[:2], x_q.device),
            get_cu_seqlens(*x_kv.shape[:2], x_kv.device),
            x_q.shape[1],
            x_kv.shape[1],
            0.0,
            softmax_scale=softmax_scale,
        )
    out = mha.out_proj(
        post_attention.to(x_q.dtype).view(x_q.shape[0], x_q.shape[1], -1)
    )
    if not mha.batch_first:
        out = out.transpose(0, 1)
    return out


def custom_attn_forward(
    mha, x_q, x_kv, attention_style="standard", activation="none", attention_scale=1.0
):
    import einops

    assert not mha.dropout
    if x_q.numel() == 0:
        return torch.empty_like(x_q)
    if not mha.batch_first:
        x_q = x_q.transpose(0, 1)
        x_kv = x_kv.transpose(0, 1)
    d = mha.in_proj_weight.shape[1]
    q = einops.rearrange(
        torch.einsum("...d,ed->...e", x_q, mha.in_proj_weight[:d])
        + mha.in_proj_bias[:d],
        "bs sl (nh hdim) -> bs sl nh hdim",
        nh=mha.num_heads,
    )
    kv = einops.rearrange(
        torch.einsum("...d,ed->...e", x_kv, mha.in_proj_weight[d:])
        + mha.in_proj_bias[d:],
        "bs sl (two nh hdim) -> bs sl two nh hdim",
        two=2,
        nh=mha.num_heads,
    )
    batch_size, seqlen_q = q.shape[0], q.shape[1]
    assert (
        kv.shape[0] == batch_size
        and kv.shape[3] == q.shape[2]
        and kv.shape[4] == q.shape[3]
    )
    k, v = kv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if attention_style == "distribute":
        attention = torch.softmax(scores, dim=-2, dtype=v.dtype)
    elif attention_style == "standard":
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    elif attention_style == "sigmoid":
        attention = torch.sigmoid(scores)
    else:
        raise ValueError(f"Unknown attention_style {attention_style}")
    post_attention = torch.einsum("bhts,bshd->bthd", attention, v)
    if activation == "none" or activation is None:
        pass
    elif activation == "relu":
        post_attention = torch.relu(post_attention)
    elif activation == "gelu":
        post_attention = torch.nn.functional.gelu(post_attention)
    else:
        raise ValueError(f"Unknown activation {activation}")

    if attention_scale == "linear":
        post_attention = post_attention / x_kv.shape[1]
    elif attention_scale == "sqrt":
        post_attention = post_attention / math.sqrt(x_kv.shape[1])
    else:
        post_attention = post_attention * attention_scale
    out = mha.out_proj(post_attention.reshape(x_q.shape[0], x_q.shape[1], -1))
    if not mha.batch_first:
        out = out.transpose(0, 1)
    return out


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.0,
        activation="gelu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        recompute_attn=False,
        save_trainingset_representations=False,
        use_flash_attention=True,
        custom_attention_style_and_activation_and_scale=None,
        use_zero_attention=False,
        share_kv_proj_weights=False,
        scale_softmax_w_dataset_size=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            add_zero_attn=use_zero_attention,
            **factory_kwargs,
        )
        if use_zero_attention:
            assert (not use_flash_attention) and (
                custom_attention_style_and_activation_and_scale is None
            )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            assert (d_model // nhead) in (
                32,
                64,
            ), "Flash attention only supports d_model // nhead == 32 or 64"
        self.custom_attention_style_and_activation_and_scale = (
            custom_attention_style_and_activation_and_scale
        )
        if custom_attention_style_and_activation_and_scale is not None:
            assert (
                not use_flash_attention
            ), "Custom attention style and activation only supported with flash attention"
        self.share_kv_proj_weights = share_kv_proj_weights
        self.scale_softmax_w_dataset_size = scale_softmax_w_dataset_size
        if self.share_kv_proj_weights or self.scale_softmax_w_dataset_size:
            assert (
                use_flash_attention
            ), "share_kv_proj_weights and scale_softmax_w_dataset_size only supported with flash attention"

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault("save_trainingset_representations", False)
        self.__dict__.setdefault("use_flash_attention", False)
        self.__dict__.setdefault(
            "custom_attention_style_and_activation_and_scale", None
        )
        self.__dict__.setdefault("share_kv_proj_weights", False)
        self.__dict__.setdefault("scale_softmax_w_dataset_size", False)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        att_src: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.save_trainingset_representations:
            assert (
                isinstance(src_mask, int) and not self.training
            ), "save_trainingset_representations is only supported in eval mode and requires src_mask to be an int"

        if self.use_flash_attention:
            assert isinstance(src_mask, int) or isinstance(
                src_mask, tuple
            ), f"Flash Attention requires the efficient mode or global attention, found {src_mask=}"

        if self.use_flash_attention and not src.is_cuda:
            utils.print_once(
                "Warning: Flash Attention is only supported on CUDA devices. "
                "Falling back to regular attention."
            )

        if att_src is not None:
            assert (
                src_mask == 0
            ), "att_src is only supported in efficient mode with only test part"

        if self.custom_attention_style_and_activation_and_scale is not None:
            assert isinstance(src_mask, int)

        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

    
        if isinstance(src_mask, int):
            # efficient attention, by splitting the src into a training set and an eval set and using full attention on both
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_to_attend_to = src_[:single_eval_position]
            if att_src is not None:
                assert not self.save_trainingset_representations
                src_to_attend_to = att_src
            elif self.save_trainingset_representations:
                if (
                    single_eval_position == src_.shape[0]
                    or single_eval_position is None
                ):
                    self.saved_src_to_attend_to = src_to_attend_to
                elif single_eval_position == 0:
                    if self.saved_src_to_attend_to is None:
                        raise ValueError(
                            "First save the trainingset representations by passing in a src_mask of None or the length of the src"
                        )
                    src_to_attend_to = self.saved_src_to_attend_to
                else:
                    raise ValueError(
                        "save_trainingset_representations only supports single_eval_position == 0 or single_eval_position == src.shape[0]"
                    )
            if self.use_flash_attention and src.is_cuda:
                attn = lambda q, kv: flash_attn_forward(
                    self.self_attn,
                    q,
                    kv,
                    share_kv=self.share_kv_proj_weights,
                    scale_softmax_w_dataset=self.scale_softmax_w_dataset_size,
                )

            elif self.custom_attention_style_and_activation_and_scale is not None:
                (
                    attn_style,
                    attn_activation,
                    attn_scale,
                ) = self.custom_attention_style_and_activation_and_scale
                attn = lambda q, kv: custom_attn_forward(
                    self.self_attn, q, kv, attn_style, attn_activation
                )
            else:
                attn = lambda q, kv: self.self_attn(q, kv, kv)[0]
                attn = partial(checkpoint, attn) if self.recompute_attn else attn

            src_left = attn(
                src_[:single_eval_position],
                src_[:single_eval_position],
            )
            src_right = attn(src_[single_eval_position:], src_to_attend_to)
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                src2 = checkpoint(
                    self.self_attn,
                    src_,
                    src_,
                    src_,
                    src_key_padding_mask,
                    True,
                    src_mask,
                )[0]
            else:
                src2 = self.self_attn(
                    src_,
                    src_,
                    src_,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                )[0]
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src


import einops


class PerFeatureEncoderLayer(Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        pre_norm=False,
        device=None,
        dtype=None,
        use_flash_attention=False,
        recompute_attn=False,
        second_mlp=False,
        bias=True,
        triton_ln=False,
        # this will slow down the forward pass by around 20%
        # TODO: allow this to be adaptively turned off when the standard implementation fits into memory
        save_peak_mem_factor=None,  # save peak mem, only effective with post-norm, a value of 8 is effective
        multi_query_factor=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn_between_features = MultiheadAttention(
            d_model, nhead, batch_first=True, bias=bias, **factory_kwargs
        )

        self.self_attn_between_items = MultiheadAttention(
            d_model, nhead, batch_first=True, bias=bias, **factory_kwargs
        )

        # Implementation of Feedforward model
        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.activation = _get_activation_fn(activation)

        self.triton_ln = triton_ln
        if triton_ln == "parameter_free":
            from .parameter_free_layer_norm import ParameterFreeLayerNormTriton

            layer_norm_constructor = ParameterFreeLayerNormTriton
        elif triton_ln == "pt_pf":
            layer_norm_constructor = partial(nn.LayerNorm, elementwise_affine=False)
        elif triton_ln is True or triton_ln == "standard":
            from .layer_norm import LayerNormTriton

            layer_norm_constructor = LayerNormTriton
        else:
            layer_norm_constructor = nn.LayerNorm

        self.norm1 = layer_norm_constructor(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )
        self.norm2 = layer_norm_constructor(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )
        self.norm3 = layer_norm_constructor(
            d_model, eps=layer_norm_eps, **factory_kwargs
        )

        if second_mlp:
            self.linear3 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.linear4 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
            self.norm4 = layer_norm_constructor(
                d_model, eps=layer_norm_eps, **factory_kwargs
            )

        self.batch_first = batch_first
        self.pre_norm = pre_norm
        self.use_flash_attention = use_flash_attention
        self.recompute_attn = recompute_attn
        self.second_mlp = second_mlp
        self.save_peak_mem_factor = save_peak_mem_factor
        self.multi_query_factor = multi_query_factor

    def __setstate__(self, state):
        state.setdefault("save_peak_mem_factor", False)
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,  # shape of (batch_size, num_items, num_features, d_model) if batch_first else (num_items, num_features, batch_size, d_model)
        single_eval_pos: int,
        att_src: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            single_eval_pos: the position from which on everything is treated as test set
        Shape:
            see the docs in Transformer class.
        """
        assert (
            len(src.shape) == 4
        ), "src must be of shape (batch_size, num_items, num_features, d_model) if batch_first else (num_items, num_features, batch_size, d_model)"

        if att_src is not None:
            assert (
                single_eval_pos == 0
            ), "att_src is only supported in efficient mode with only test part"

        if self.use_flash_attention and not src.is_cuda:
            utils.print_once(
                "Warning: Flash Attention is only supported on CUDA devices. "
                "Falling back to regular attention."
            )

        if self.save_peak_mem_factor is not None:
            assert (
                not src.requires_grad
            ), "save_peak_mem_factor only works with inference mode"
            assert not self.pre_norm, "save_peak_mem_factor only works with post-norm"

        if not self.batch_first:
            src = einops.rearrange(
                src,
                "num_items num_features batch_size d_model -> batch_size num_items num_features d_model",
            )

        batch_size, num_items, num_features, d_model = src.shape

        if self.use_flash_attention and src.is_cuda:
            attn = lambda attn_module, q, kv=None, **kwargs: flash_attn_forward(
                attn_module,
                q,
                kv,
                **kwargs,
            )
        else:
            assert not self.multi_query_factor
            attn = lambda attn_module, q, kv, multi_query_factor=None: attn_module(
                q, kv, kv
            )[0]

        def attn_between_features(src):
            if self.pre_norm:
                src_ = self.norm1(src)
            else:
                src_ = src

            src_ = einops.rearrange(
                src_,
                "batch_size num_items num_features d_model -> (batch_size num_items) num_features d_model",
            )

            if self.save_peak_mem_factor:
                num_splits = self.save_peak_mem_factor
                split_size = (src_.shape[0] // num_splits) + 1
                for i in range(num_splits):
                    if i * split_size >= src_.shape[0]:
                        break
                    src_[i * split_size : (i + 1) * split_size] += attn(
                        self.self_attn_between_features,
                        src_[i * split_size : (i + 1) * split_size],
                        src_[i * split_size : (i + 1) * split_size],
                    )
                src2 = src_
            else:
                src2 = attn(self.self_attn_between_features, src_, src_)

            src2 = einops.rearrange(
                src2,
                "(batch_size num_items) num_features d_model -> batch_size num_items num_features d_model",
                batch_size=batch_size,
                num_items=num_items,
            )

            return src2

        def attn_between_items(src, att_src=None):
            if self.pre_norm:
                src_ = self.norm2(src)
            else:
                src_ = src

            src_ = einops.rearrange(
                src_,
                "batch_size num_items num_features d_model -> (batch_size num_features) num_items d_model",
            )
            if att_src is not None:
                att_src = einops.rearrange(
                    att_src,
                    "batch_size num_items num_features d_model -> (batch_size num_features) num_items d_model",
                )

            if self.save_peak_mem_factor:
                split_size = (src_.shape[0] // self.save_peak_mem_factor) + 1
                for i in range(self.save_peak_mem_factor):
                    if i * split_size >= src_.shape[0]:
                        break

                    #  start with right, as it depends on the left
                    src_[
                        split_size * i : split_size * (i + 1), single_eval_pos:
                    ] += attn(
                        self.self_attn_between_items,
                        src_[split_size * i : split_size * (i + 1), single_eval_pos:],
                        src_[split_size * i : split_size * (i + 1), :single_eval_pos],
                        multi_query_factor=self.multi_query_factor,
                    )

                    src_to_attend_to = (
                        src_[split_size * i : split_size * (i + 1), :single_eval_pos]
                        if att_src is None
                        else att_src[split_size * i : split_size * (i + 1)]
                    )

                    src_[
                        split_size * i : split_size * (i + 1), :single_eval_pos
                    ] += attn(
                        self.self_attn_between_items,
                        src_[split_size * i : split_size * (i + 1), :single_eval_pos],
                        src_to_attend_to,
                    )
                    del src_to_attend_to

                src2 = src_
            else:
                src_left_ = src_[:, :single_eval_pos]

                src_left = attn(
                    self.self_attn_between_items,
                    src_left_,
                    src_left_,
                )

                src_to_attend_to = src_left_ if att_src is None else att_src

                src_right = attn(
                    self.self_attn_between_items,
                    src_[:, single_eval_pos:],
                    src_to_attend_to,
                    multi_query_factor=self.multi_query_factor,
                )
                del src_to_attend_to

                src2 = torch.cat([src_left, src_right], dim=1)

            src2 = einops.rearrange(
                src2,
                "(batch_size num_features) num_items d_model -> batch_size num_items num_features d_model",
                batch_size=batch_size,
                num_items=num_items,
            )
            return src2

        def ffn(src, norm, linear1, linear2):
            if self.pre_norm:
                src_ = norm(src)
            else:
                src_ = src

            if self.save_peak_mem_factor:
                # split sequence into chunks of size seq_len // 8
                num_chunks = self.save_peak_mem_factor * 8
                chunk_size = (src_.shape[1] // num_chunks) + 1
                for i in range(num_chunks):
                    if i * chunk_size >= src_.shape[1]:
                        break
                    src_[:, i * chunk_size : (i + 1) * chunk_size] += linear2(
                        self.activation(
                            linear1(src_[:, i * chunk_size : (i + 1) * chunk_size])
                        )
                    )
                src2 = src_
            else:
                src2 = linear2(self.activation(linear1(src_)))
            return src2

        if self.recompute_attn:
            ffn = partial(checkpoint, ffn)
            attn_between_features = partial(checkpoint, attn_between_features)
            attn_between_items = partial(checkpoint, attn_between_items)

        src_ = attn_between_features(src)

        if not self.pre_norm:
            if not self.save_peak_mem_factor:
                src = self.norm1(src + src_)
            else:
                src.set_(self.norm1(src_))
        else:
            src = src + src_

        # here everything is fine

        if self.second_mlp:
            # src = src + ffn2(src)
            src_ = ffn(src, self.norm4, self.linear3, self.linear4)
            if not self.pre_norm:
                src = self.norm4(src + src_)
            else:
                src = src + src_

        # src = src + attn_between_items(src)
        src_ = attn_between_items(src, att_src)
        # print('max memory after attn btw items', torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 'GB')

        if not self.pre_norm:
            if not self.save_peak_mem_factor:
                src = self.norm2(src + src_)
            else:
                src.set_(self.norm2(src_))
        else:
            src = src + src_
        # print('max memory after norm2', torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 'GB')

        # src = src + ffn(src)
        src_ = ffn(src, self.norm3, self.linear1, self.linear2)
        # print('max memory after ffn', torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 'GB')

        if not self.pre_norm:
            if not self.save_peak_mem_factor:
                src = self.norm3(src + src_)
            else:
                src.set_(self.norm3(src_))
        else:
            src = src + src_
        # print('max memory after nrom3', torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 'GB')

        if not self.batch_first:
            src = einops.rearrange(
                src,
                "batch_size num_items num_features d_model -> num_items num_features batch_size d_model",
            )
        return src
