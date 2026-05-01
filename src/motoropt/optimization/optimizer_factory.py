from typing import Any, Dict

from motoropt.optimization.ga import GeneticAlgorithmOptimizer


def build_optimizer(config: Dict[str, Any]):
    name = config.get("name", "ga").lower()
    if name == "ga":
        return GeneticAlgorithmOptimizer(config)
    raise ValueError(f"Unsupported optimizer name: {name}")
