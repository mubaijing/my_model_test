import torch

from motoropt.models.factory import build_model


def test_mlp_forward_shape():
    model = build_model({"name": "mlp", "hidden_layers": [8]}, input_dim=3, output_dim=1)
    y = model(torch.randn(4, 3))
    assert y.shape == (4, 1)
