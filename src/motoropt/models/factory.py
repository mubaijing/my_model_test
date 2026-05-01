from typing import Any, Dict

from .mlp import MLPRegressor


def build_model(config: Dict[str, Any], input_dim: int, output_dim: int):
    name = config.get("name", "mlp").lower()
    if name == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=config.get("hidden_layers", [64, 64, 32]),
            activation=config.get("activation", "relu"),
            dropout=float(config.get("dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model name: {name}")
