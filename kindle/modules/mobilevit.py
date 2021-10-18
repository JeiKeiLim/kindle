"""Mobile ViT Block module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn

from kindle.modules.activation import Activation
from kindle.modules.conv import Conv


class PreNorm(nn.Module):
    """Pre-normalization layer."""

    def __init__(self, channels: int, layer: nn.Module) -> None:
        """Initialize PreNorm module.

        Args:
            channels: number of channels to normalize.
            layer: layer module to pre-norm.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.layer = layer

    def forward(self, x, **kwargs):
        """Normalize input and forward."""
        return self.layer(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """FeedForward module for Transformer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        dropout: float = 0.0,
        activation: Union[str, None] = "SiLU",
    ) -> None:
        """Initialize FeedForward module.

        Args:
            channels: input channels
            hidden_channels: hidden channels
            dropout: dropout probability.
            activation: Name of the activation to use in the middle of Linear modules.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            Activation(activation)(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward module."""
        return self.net(x)


class Attention(nn.Module):
    """Attention module for Transformer."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        channels_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        """Initialize Attention module.

        Args:
            channels: input channels.
            heads: number of heads in multi-head attention.
            channels_head: number of channes to use in heads.
            dropout: dropout probability.
        """
        super().__init__()
        hidden_channels = channels_head * heads
        project_out = not (heads == 1 and channels_head == channels)

        self.heads = heads
        self.scale = channels_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(channels, hidden_channels * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(hidden_channels, channels), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """Forward attention module."""
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(  # pylint: disable=invalid-name
            lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer module for MobileViTBlock."""

    def __init__(
        self,
        channels: int,
        depth: int,
        heads: int,
        channels_head: int,
        channels_mlp: int,
        dropout: float = 0.0,
        activation: Union[str, None] = "SiLU",
    ) -> None:
        """Initialize Transformer module.

        Args:
            channels: input channels.
            depth: depth of the transformer.
            heads: number of heads to use in multi-head attention.
            channels_head: number of channes to use in heads.
            channels_mlp: number of channes to use in MLP.
            dropout: dropout probability.
            activation: Name of the activation to use in the middle of Linear modules.
        """
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            channels, Attention(channels, heads, channels_head, dropout)
                        ),
                        PreNorm(
                            channels,
                            FeedForward(
                                channels, channels_mlp, dropout, activation=activation
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x):
        """Forward Transformer layer."""
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x
        return x


class MobileViTBlock(nn.Module):
    """Mobile ViT Block (https://arxiv.org/pdf/2110.02178.pdf)."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        mlp_channels: int,
        depth: int,
        kernel_size: int = 3,
        patch_size: Union[int, Tuple[int, int]] = 2,
        dropout: float = 0.0,
        activation: Union[str, None] = "SiLU",
    ) -> None:
        """Initialize Mobile ViT Block.

        Args:
            in_channels: number of incoming channels
            conv_channels: number of channels to use in convolution.
            mlp_channels: number of channels to use in MLP.
            depth: depth of the transformer.
            kernel_size: kernel size in nxn convolution.
            dropout: dropout probability.
            activation: Name of the activation to use in the middle of Linear modules.
        """
        super().__init__()
        self.patch_w, self.patch_h = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.conv1_nxn = Conv(
            in_channels, in_channels, kernel_size, activation=activation
        )
        self.conv2_1x1 = Conv(in_channels, conv_channels, 1, activation=activation)
        self.transformer = Transformer(
            conv_channels, depth, 1, 32, mlp_channels, dropout=dropout
        )
        self.conv3_1x1 = Conv(conv_channels, in_channels, 1, activation=activation)
        self.conv4_nxn = Conv(
            2 * in_channels, in_channels, kernel_size, activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward MobileViTBlock."""
        y = x.clone()
        x = self.conv1_nxn(x)
        x = self.conv2_1x1(x)

        _, _, h, w = x.shape  # pylint: disable=invalid-name
        x = rearrange(
            x,
            "b d (h ph) (w pw) -> b (ph pw) (h w) d",
            ph=self.patch_h,
            pw=self.patch_w,
        )
        x = self.transformer(x)
        x = rearrange(
            x,
            "b (ph pw) (h w) d -> b d (h ph) (w pw)",
            h=h // self.patch_h,
            w=w // self.patch_w,
            ph=self.patch_h,
            pw=self.patch_w,
        )

        x = self.conv3_1x1(x)
        x = torch.cat((x, y), 1)
        x = self.conv4_nxn(x)

        return x
