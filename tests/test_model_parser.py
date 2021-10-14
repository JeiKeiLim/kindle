"""Model parse test.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import os

import torch

from kindle.model import Model
from kindle.utils.torch_utils import count_model_params


class TestModelParser:
    """Test model parser."""

    # pylint: disable=no-self-use
    INPUT = torch.rand(1, 3, 32, 32)

    def _validate_export_model(self, model: Model) -> bool:
        model.eval()
        model_out = model(TestModelParser.INPUT)
        # model.fuse().eval()
        model.export()
        model_out_fused = model(TestModelParser.INPUT)

        return torch.all(torch.isclose(model_out, model_out_fused, rtol=1e-5))

    def test_show_case(self, verbose: bool = False):
        """Test show case model."""
        model = Model(
            os.path.join("tests", "test_configs", "show_case.yaml"),
            verbose=verbose,
        )
        assert count_model_params(model) == 185666
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert self._validate_export_model(model)

    def test_nn_model(self, verbose: bool = False):
        """Test nn model."""
        model = Model(
            os.path.join("tests", "test_configs", "nn_model.yaml"),
            verbose=verbose,
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 28506
        assert self._validate_export_model(model)

    def test_vgg(self, verbose: bool = False):
        """Test vgg model."""
        model = Model(
            os.path.join("tests", "test_configs", "vgg.yaml"), verbose=verbose
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 3732970
        assert self._validate_export_model(model)

    def test_example(self, verbose: bool = False):
        """Test example model."""
        model = Model(
            os.path.join("tests", "test_configs", "example.yaml"), verbose=verbose
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 137862
        assert self._validate_export_model(model)

    def test_gap_model(self, verbose: bool = False):
        """Test example model."""
        model = Model(
            os.path.join("tests", "test_configs", "gap_test_model.yaml"),
            verbose=verbose,
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 20148
        assert self._validate_export_model(model)

    def test_pretrained(self, verbose: bool = False):
        """Test show case model."""
        model = Model(
            os.path.join("tests", "test_configs", "pretrained_example.yaml"),
            verbose=verbose,
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 3621866
        assert self._validate_export_model(model)

    def test_pretrained2(self, verbose: bool = False):
        """Test show case model."""
        model = Model(
            os.path.join("tests", "test_configs", "pretrained_example2.yaml"),
            verbose=verbose,
        )
        assert model(TestModelParser.INPUT).shape == torch.Size([1, 10])
        assert count_model_params(model) == 3760122
        assert self._validate_export_model(model)


if __name__ == "__main__":
    tester = TestModelParser()
    # tester.test_pretrained2(verbose=True)
    # tester.test_pretrained(verbose=True)
    # tester.test_nn_model(verbose=True)
    tester.test_show_case(verbose=True)
    # tester.test_gap_model(verbose=True)
    # tester.test_example(verbose=True)
