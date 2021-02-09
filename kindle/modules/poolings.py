"""Module generator related to pooling operations.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from torch import nn


class GlobalAvgPool(nn.AdaptiveAvgPool2d):
    """Global average pooling module.

    Arguments: []
    """

    def __init__(self):
        """Initialize."""
        super().__init__(output_size=1)
