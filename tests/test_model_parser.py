"""Model parse test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import torch

from kindle.model import Model


class TestModelParser:
    """Test model parser."""

    # pylint: disable=no-self-use

    INPUT = torch.rand(1, 3, 32, 32)

    def test_show_case(self):
        """Test show case model."""
        model = Model(os.path.join("tests", "test_configs", "show_case.yaml"))
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])

    def test_vgg(self):
        """Test vgg model."""
        model = Model(os.path.join("tests", "test_configs", "vgg.yaml"))
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])

    def test_example(self):
        """Test example model."""
        model = Model(os.path.join("tests", "test_configs", "example.yaml"))
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
