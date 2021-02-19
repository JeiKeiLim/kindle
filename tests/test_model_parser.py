"""Model parse test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import torch

from kindle.model import Model
from kindle.torch_utils import count_model_params


class TestModelParser:
    """Test model parser."""

    # pylint: disable=no-self-use
    INPUT = torch.rand(1, 3, 32, 32)

    def test_show_case(self, verbose: bool = False):
        """Test show case model."""
        model = Model(
            os.path.join("tests", "test_configs", "show_case.yaml"),
            verbose=verbose,
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 180034

    def test_vgg(self, verbose: bool = False):
        """Test vgg model."""
        model = Model(
            os.path.join("tests", "test_configs", "vgg.yaml"), verbose=verbose
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 3732970

    def test_example(self, verbose: bool = False):
        """Test example model."""
        model = Model(
            os.path.join("tests", "test_configs", "example.yaml"), verbose=verbose
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 137862


if __name__ == "__main__":
    tester = TestModelParser()
    tester.test_show_case(verbose=True)
    # tester.test_example(verbose=True)
