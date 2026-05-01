import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from motoropt.inference.predictor import Predictor
from motoropt.utils.config import load_yaml, resolve_path


def parse_unknown_pairs(items):
    data = {}
    i = 0
    while i < len(items):
        key = items[i]
        if not key.startswith("--") or i + 1 >= len(items):
            raise ValueError(f"Invalid input argument near: {key}")
        data[key[2:]] = float(items[i + 1])
        i += 2
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/optimize_config.yaml")
    parser.add_argument("--input", type=str, default=None)
    args, unknown = parser.parse_known_args()

    config = load_yaml(resolve_path(args.config, PROJECT_ROOT))
    model_cfg = config["model"]

    predictor = Predictor(
        model_path=resolve_path(model_cfg["model_path"], PROJECT_ROOT),
        input_scaler_path=resolve_path(model_cfg["input_scaler_path"], PROJECT_ROOT),
        output_scaler_path=resolve_path(model_cfg["output_scaler_path"], PROJECT_ROOT),
        model_meta_path=resolve_path(model_cfg["model_meta_path"], PROJECT_ROOT),
    )

    if args.input:
        data = json.loads(args.input)
    else:
        data = parse_unknown_pairs(unknown)

    result = predictor.predict(data)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
