import torch.nn as nn
from diffusers.models.unet_2d_blocks import ResnetBlock2D, Upsample2D

from ..model.modules.utils import zero_module


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016
    """

    def __init__(self, in_dim, out_dim=None, num_layers=3, dropout=0.0):
        super().__init__()
        out_dim = out_dim or in_dim

        # conv layers
        convs = []
        prev_dim, next_dim = in_dim, out_dim
        for i in range(num_layers):
            if i == num_layers - 1:
                next_dim = out_dim
            convs.extend(
                [
                    nn.GroupNorm(32, prev_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Conv3d(prev_dim, next_dim, (3, 1, 1), padding=(1, 0, 0)),
                ]
            )
            prev_dim, next_dim = next_dim, prev_dim
        self.convs = nn.ModuleList(convs)

    def forward(self, hidden_states):
        video_length = hidden_states.shape[2]
        identity = hidden_states

        for conv in self.convs:
            hidden_states = conv(hidden_states)

        # ignore these convolution layers on image input
        hidden_states = identity + hidden_states if video_length > 1 else identity

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        add_temp_conv=False,
    ):
        super().__init__()
        resnets = []
        if add_temp_conv:
            self.temp_convs = None
            temp_convs = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if add_temp_conv:
                temp_convs.append(
                    TemporalConvLayer(out_channels, out_channels, dropout=0.1)
                )
                temp_convs[-1].convs[-1] = zero_module(temp_convs[-1].convs[-1])

        self.resnets = nn.ModuleList(resnets)
        if add_temp_conv:
            self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states, num_frames=1):
        for layer_idx in range(len(self.resnets)):
            hidden_states = self.resnets[layer_idx](hidden_states, temb=None)
            if hasattr(self, "temp_convs"):
                hidden_states = hidden_states.reshape(
                    hidden_states.shape[0] // num_frames,
                    num_frames,
                    *hidden_states.shape[1:],
                )
                hidden_states = hidden_states.swapaxes(1, 2)
                hidden_states = self.temp_convs[layer_idx](hidden_states)
                hidden_states = hidden_states.swapaxes(1, 2)
                hidden_states = hidden_states.reshape(
                    hidden_states.shape[0] * num_frames, *hidden_states.shape[2:]
                )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    add_temp_conv=False,
):
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_temp_conv=add_temp_conv,
        )

    raise ValueError(
        f"{up_block_type} does not exist. Please refer to: `from diffusers.models.vae import get_up_block'"
    )
