from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class DesignProblem:
    variables: Dict[str, Dict[str, float]]

    @property
    def names(self) -> list[str]:
        return list(self.variables.keys())

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(
            [[self.variables[name]["lower"], self.variables[name]["upper"]] for name in self.names],
            dtype=float,
        )

    def vector_to_dict(self, x: Sequence[float]) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, x)}

    def clip(self, x: np.ndarray) -> np.ndarray:
        bounds = self.bounds
        return np.clip(x, bounds[:, 0], bounds[:, 1])
