from typing import Tuple

import numpy as np

from motoropt.inference.predictor import Predictor
from motoropt.optimization.problem import DesignProblem


class SurrogateObjective:
    def __init__(
        self,
        predictor: Predictor,
        problem: DesignProblem,
        target: str,
        direction: str = "maximize",
    ):
        self.predictor = predictor
        self.problem = problem
        self.target = target
        self.direction = direction.lower()
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")

    def evaluate(self, x: np.ndarray) -> Tuple[float, dict]:
        design = self.problem.vector_to_dict(x)
        performance = self.predictor.predict(design)
        value = float(performance[self.target])
        objective_value = -value if self.direction == "maximize" else value
        return objective_value, performance
