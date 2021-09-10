"""YOLOv5 head module.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import List, Tuple, Union

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
        self.grid = [
            torch.zeros(1) for _ in range(self.n_layers)
        ]  # Grid initialization

        anchor_buf = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer("anchors", anchor_buf)  # (n_layer, n_anchors, 2)
        self.register_buffer(
            "anchor_grid", anchor_buf.clone().view(self.n_layers, 1, -1, 1, 1, 2)
        )  # (n_layers, 1, n_anchors, 1, 1, 2)
        self.conv = nn.ModuleList(
            nn.Conv2d(x, self.n_outputs * self.n_anchors, 1) for x in n_channels
        )
        self.stride = torch.tensor(strides)

        # TODO(jeikeilim): Add initialize biases

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
