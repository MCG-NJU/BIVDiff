# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.models.cross_attention import CrossAttention
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from torch import nn

from .modules import get_sin_pos_embedding
from .utils import zero_module

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        add_temp_attn: bool = False,
        prepend_first_frame: bool = False,
        add_temp_embed: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # temporal embedding
        self.add_temp_embed = add_temp_embed

        if add_temp_attn:
            if prepend_first_frame:
                # SC-Attn
                self.attn1 = SparseCausalAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim
                    if only_cross_attention
                    else None,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
            else:
                # Normal CrossAttn
                self.attn1 = CrossAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim
                    if only_cross_attention
                    else None,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )

            # Temp-Attn
            self.temp_norm = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim)
            )
            self.temp_attn = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            zero_module(self.temp_attn.to_out)
        else:
            # Normal Attention
            self.attn1 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim
                if only_cross_attention
                else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            self.temp_attn = None

        self.norm1 = (
            AdaLayerNorm(dim, num_embeds_ada_norm)
            if self.use_ada_layer_norm
            else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        )

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.attn2 = None
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Optional[Callable] = None,
    ):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1.set_use_memory_efficient_attention_xformers(
                use_memory_efficient_attention_xformers, attention_op=attention_op
            )
            if self.attn2 is not None:
                self.attn2.set_use_memory_efficient_attention_xformers(
                    use_memory_efficient_attention_xformers, attention_op=attention_op
                )
            if self.temp_attn is not None:
                self.temp_attn.set_use_memory_efficient_attention_xformers(
                    use_memory_efficient_attention_xformers, attention_op=attention_op
                )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        video_length=None,
    ):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep)
            if self.use_ada_layer_norm
            else self.norm1(hidden_states)
        )

        attn1_args = dict(
            hidden_states=norm_hidden_states, attention_mask=attention_mask
        )
        if self.temp_attn is not None and isinstance(self.attn1, SparseCausalAttention):
            attn1_args.update({"video_length": video_length})
        # Self-/Sparse-Attention
        if self.only_cross_attention:
            hidden_states = (
                self.attn1(**attn1_args, encoder_hidden_states=encoder_hidden_states)
                + hidden_states
            )
        else:
            hidden_states = self.attn1(**attn1_args) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                + hidden_states
            )

        if self.temp_attn is not None:
            identity = hidden_states
            d = hidden_states.shape[1]
            # add temporal embedding
            if self.add_temp_embed:
                temp_emb = get_sin_pos_embedding(
                    hidden_states.shape[-1], video_length
                ).to(hidden_states)
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> b d f c", f=video_length
                )
                hidden_states += temp_emb
                hidden_states = rearrange(hidden_states, "b d f c -> (b f) d c")
            # normalization
            hidden_states = rearrange(
                hidden_states, "(b f) d c -> (b d) f c", f=video_length
            )
            norm_hidden_states = (
                self.temp_norm(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.temp_norm(hidden_states)
            )
            # apply temporal attention
            hidden_states = self.temp_attn(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
            # ignore effects of temporal layers on image inputs
            if video_length <= 1:
                hidden_states = identity + 0.0 * hidden_states

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        add_temp_attn: bool = False,
        prepend_first_frame: bool = False,
        add_temp_embed: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    add_temp_attn=add_temp_attn,
                    prepend_first_frame=prepend_first_frame,
                    add_temp_embed=add_temp_embed,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        return_dict=False,
    ):
        # Input
        assert (
            hidden_states.dim() == 5
        ), f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(
            encoder_hidden_states, "b n c -> (b f) n c", f=video_length
        )

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * weight, inner_dim
            )
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size x num_frames, num_channels, height, width)`)
            Hidden states conditioned on `encoder_hidden_states` input.
    """

    sample: torch.FloatTensor


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        add_temp_embed: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    add_temp_embed=add_temp_embed,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.proj_out = zero_module(self.proj_out)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] or `tuple`:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        # 1. Input
        batch_size, channel, num_frames, height, width = hidden_states.shape

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(
            batch_size * height * width, num_frames, channel
        )
        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=num_frames,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, channel, num_frames)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


class SparseCausalAttention(CrossAttention):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        video_length = cross_attention_kwargs.get("video_length", 8)
        attention_mask = self.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = self.to_q(hidden_states)
        dim = query.shape[-1]

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.cross_attention_norm:
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        if video_length > 1:
            key = torch.cat(
                [key[:, [0] * video_length], key[:, former_frame_index]], dim=2
            )
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        if video_length > 1:
            value = torch.cat(
                [value[:, [0] * video_length], value[:, former_frame_index]], dim=2
            )
        value = rearrange(value, "b f d c -> (b f) d c")

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        # attention, what we cannot get enough of
        if hasattr(self.processor, "attention_op"):
            hidden_states = xformers.ops.memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=attention_mask,
                op=self.processor.attention_op,
            )
            hidden_states = hidden_states.to(query.dtype)
        elif hasattr(self.processor, "slice_size"):
            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads),
                device=query.device,
                dtype=query.dtype,
            )
            for i in range(hidden_states.shape[0] // self.processor.slice_size):
                start_idx = i * self.slice_size
                end_idx = (i + 1) * self.slice_size
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
                attn_mask_slice = (
                    attention_mask[start_idx:end_idx]
                    if attention_mask is not None
                    else None
                )
                attn_slice = self.get_attention_scores(
                    query_slice, key_slice, attn_mask_slice
                )
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
                hidden_states[start_idx:end_idx] = attn_slice
        else:
            attention_probs = self.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
