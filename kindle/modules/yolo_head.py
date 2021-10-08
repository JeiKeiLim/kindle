"""YOLOv5 head module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class YOLOHead(nn.Module):
    """YOLOv5 head module."""

    export: bool = False

    def __init__(  # pylint: disable=not-callable
        self,
        n_classes: int,
        anchors: Tuple[Tuple[float, ...]],
        n_channels: Tuple[int, ...],
        strides: Tuple[float, ...],
    ) -> None:
        super().__init__()

        # Number of classes
        self.n_classes = n_classes
        # Number of outputs per anchor
        self.n_outputs = n_classes + 5
        # Number of detection layers
        self.n_layers = len(anchors)
        # Number of anchors per a layer
        self.n_anchors = len(anchors[0]) // 2

        # YOLOv5 compatability
        self.nc = self.n_classes  # pylint: disable=invalid-name
        self.no = self.n_outputs  # pylint: disable=invalid-name
        self.nl = self.n_layers  # pylint: disable=invalid-name
        self.na = self.n_anchors  # pylint: disable=invalid-name

        self.grid = [
            torch.zeros(1) for _ in range(self.n_layers)
        ]  # Grid initialization

        anchor_buf = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer("anchors", anchor_buf)  # (n_layer, n_anchors, 2)
        self.register_buffer(
            "anchor_grid", anchor_buf.clone().view(self.n_layers, 1, -1, 1, 1, 2)
        )  # (n_layers, 1, n_anchors, 1, 1, 2)

        # TODO(jeikeilim): Conv type can be choosable.
        self.conv = nn.ModuleList(
            nn.Conv2d(x, self.n_outputs * self.n_anchors, 1) for x in n_channels
        )
        self.stride = torch.tensor(strides)
        self.anchors /= self.stride.view(-1, 1, 1)  # pylint: disable=no-member

        # TODO(jeikeilim): Consider what to do with NMS layer

    def forward(
        self, x: List[torch.Tensor]
    ) -> Union[
        Tuple[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor], List[torch.Tensor]
    ]:
        """Forward YOLO head.

        Args:
            x: List of feature maps

        Returns:
            1. (batch_size, n_anchors, height, width, n_classes + 5) in training mode.
            2. (batch_size, n_anchors * heights * widths, n_classes + 5) in export mode
            3. (1) + (2) in inference mode
        """
        preds = []
        for i in range(self.n_layers):
            x[i] = self.conv[i](x[i])
            batch_size, _, height, width = x[i].shape

            # Reshape to (batch_size, n_anchor, height, width, outputs)
            x[i] = (
                x[i]
                .view(batch_size, self.n_anchors, self.n_outputs, height, width)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(width, height).to(x[i].device)
                else:
                    self.grid[i] = self.grid[i].to(x[i].device)

                y = x[i].sigmoid()
                box_xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                box_wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # type: ignore
                box_xyxy = self._xywh2xyxy(box_xy, box_wh).view(batch_size, -1, 4)
                score = y[..., 4:].float().view(batch_size, -1, self.n_classes + 1)
                preds.append(torch.cat([box_xyxy, score], -1))

        if self.training:
            return x
        if self.export:
            return (torch.cat(preds, 1),)

        return (torch.cat(preds, 1), x)

    def initialize_biases(
        self,
        class_frequency: Optional[torch.Tensor] = None,
        n_object_per_image: Tuple[int, int] = (8, 640),
        class_probability: float = 0.6,
    ) -> None:
        """Initialize biases for objectness and class probabilities.

        Args:
        class_frequency: Class histogram values. If this is provided,
            class biases will be initialized by
            initial bias + log(class_frequency / class_frequency.sum())

        n_object_per_image: Number of object per image. (Number of object, per image).
            objectness biases will be initialized by
            initial bias + log(n_object / (per_image / stride) ** 2)

        class_probability: Global class probability.
            This will be ignored if class_frequency is provided.
            Otherwise, class biasese will be initialized by
            initial bias + log(class_probability / (n_classes - 0.99))
        """

        for conv, stride in zip(self.conv, self.stride):
            bias = conv.bias.view(self.n_anchors, -1)
            bias.data[:, 4] += math.log(
                n_object_per_image[0] / (n_object_per_image[1] / stride) ** 2
            )

            if class_frequency is None:
                bias.data[:, 5:] += math.log(
                    class_probability / (self.n_classes - 0.99)
                )
            else:
                bias.data[:, 5:] += torch.log(class_frequency / class_frequency.sum())

            conv.bias = nn.Parameter(bias.view(-1), requires_grad=True)

    @staticmethod
    def _make_grid(width: int = 20, height: int = 20) -> torch.Tensor:
        """Make grid."""
        y_grid, x_grid = torch.meshgrid([torch.arange(height), torch.arange(width)])
        return torch.stack((x_grid, y_grid), 2).view((1, 1, height, width, 2)).float()

    @staticmethod
    def _xywh2xyxy(  # pylint: disable=invalid-name
        xy: torch.Tensor, wh: torch.Tensor
    ) -> torch.Tensor:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        # where xy1=top-left, xy2=bottom-right
        half_wh = wh / 2
        x1y1 = xy - half_wh
        x2y2 = xy + half_wh
        return torch.cat([x1y1, x2y2], -1)
