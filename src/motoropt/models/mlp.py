from collections.abc import Sequence

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = (64, 64, 32),
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act = self._activation(activation)
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def _activation(name: str):
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "gelu":
            return nn.GELU
        if name == "tanh":
            return nn.Tanh
        if name == "silu":
            return nn.SiLU
        raise ValueError(f"Unsupported activation: {name}")
