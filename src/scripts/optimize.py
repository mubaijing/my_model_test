import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from motoropt.inference.predictor import Predictor
from motoropt.optimization.optimizer_factory import build_optimizer
from motoropt.optimization.objective import SurrogateObjective
from motoropt.optimization.problem import DesignProblem
from motoropt.utils.config import load_yaml, resolve_path
from motoropt.utils.io import ensure_dir, save_json
from motoropt.visualization.plot_optimization import plot_optimization_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimize_config.yaml")
    args = parser.parse_args()

    config = load_yaml(resolve_path(args.config, PROJECT_ROOT))
    schema = load_yaml(resolve_path(config["schema_path"], PROJECT_ROOT))
    model_cfg = config["model"]

    predictor = Predictor(
        model_path=resolve_path(model_cfg["model_path"], PROJECT_ROOT),
        input_scaler_path=resolve_path(model_cfg["input_scaler_path"], PROJECT_ROOT),
        output_scaler_path=resolve_path(model_cfg["output_scaler_path"], PROJECT_ROOT),
        model_meta_path=resolve_path(model_cfg["model_meta_path"], PROJECT_ROOT),
    )

    problem = DesignProblem(schema["design_variables"])
    objective = SurrogateObjective(
        predictor=predictor,
        problem=problem,
        target=config["objective"]["target"],
        direction=config["objective"].get("direction", "maximize"),
    )

    if config["optimizer"].get("name", "ga").lower() != "ga":
        raise ValueError("Only GA is implemented in this first version.")

    optimizer = build_optimizer(config["optimizer"])
    result = optimizer.run(problem, objective)

    output_dir = ensure_dir(resolve_path(config["paths"]["output_dir"], PROJECT_ROOT))
    history = result.pop("history")
    history.to_csv(output_dir / "history.csv", index=False)
    plot_optimization_history(history, output_dir / "optimization_history.png")
    save_json(result, output_dir / "best_result.json")

    print("Optimization finished.")
    print(f"Best result: {result}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
