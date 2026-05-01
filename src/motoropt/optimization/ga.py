from typing import Any, Dict

import numpy as np
import pandas as pd

from motoropt.optimization.objective import SurrogateObjective
from motoropt.optimization.problem import DesignProblem


class GeneticAlgorithmOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.population_size = int(config.get("population_size", 50))
        self.generations = int(config.get("generations", 100))
        self.crossover_rate = float(config.get("crossover_rate", 0.8))
        self.mutation_rate = float(config.get("mutation_rate", 0.1))
        self.mutation_scale = float(config.get("mutation_scale", 0.08))
        self.elite_size = int(config.get("elite_size", 2))
        self.tournament_size = int(config.get("tournament_size", 3))
        self.rng = np.random.default_rng(config.get("random_state", None))

    def run(self, problem: DesignProblem, objective: SurrogateObjective) -> Dict[str, Any]:
        bounds = problem.bounds
        population = self._initialize(bounds)
        history = []

        best_x = None
        best_obj = float("inf")
        best_perf = None

        for generation in range(1, self.generations + 1):
            obj_values, performances = self._evaluate(population, objective)
            order = np.argsort(obj_values)
            population = population[order]
            obj_values = obj_values[order]
            performances = [performances[i] for i in order]

            if obj_values[0] < best_obj:
                best_obj = float(obj_values[0])
                best_x = population[0].copy()
                best_perf = performances[0]

            history.append(
                {
                    "generation": generation,
                    "best_objective": float(obj_values[0]),
                    "mean_objective": float(np.mean(obj_values)),
                    "best_target": float(best_perf[objective.target]),
                }
            )

            elites = population[: self.elite_size]
            children = []
            while len(children) < self.population_size - self.elite_size:
                p1 = self._tournament(population, obj_values)
                p2 = self._tournament(population, obj_values)
                c1, c2 = self._crossover(p1, p2)
                children.append(self._mutate(c1, bounds))
                if len(children) < self.population_size - self.elite_size:
                    children.append(self._mutate(c2, bounds))

            population = np.vstack([elites, np.asarray(children)])
            population = np.asarray([problem.clip(x) for x in population])

        return {
            "best_design": problem.vector_to_dict(best_x),
            "predicted_performance": best_perf,
            "objective_value": best_obj,
            "history": pd.DataFrame(history),
        }

    def _initialize(self, bounds: np.ndarray) -> np.ndarray:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        return self.rng.uniform(lower, upper, size=(self.population_size, len(bounds)))

    def _evaluate(self, population: np.ndarray, objective: SurrogateObjective):
        obj_values = []
        performances = []
        for x in population:
            obj, perf = objective.evaluate(x)
            obj_values.append(obj)
            performances.append(perf)
        return np.asarray(obj_values, dtype=float), performances

    def _tournament(self, population: np.ndarray, obj_values: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        winner = idx[np.argmin(obj_values[idx])]
        return population[winner].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray):
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        alpha = self.rng.random(size=p1.shape)
        c1 = alpha * p1 + (1.0 - alpha) * p2
        c2 = alpha * p2 + (1.0 - alpha) * p1
        return c1, c2

    def _mutate(self, x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        span = bounds[:, 1] - bounds[:, 0]
        mask = self.rng.random(size=x.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_scale, size=x.shape) * span
        x = x + mask * noise
        return np.clip(x, bounds[:, 0], bounds[:, 1])
