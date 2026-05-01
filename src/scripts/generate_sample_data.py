import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from motoropt.utils.io import ensure_dir


def synthetic_torque(air_gap, magnet_thickness, slot_opening, rng):
    torque = (
        12.0
        - 3.8 * air_gap
        + 2.15 * magnet_thickness
        - 0.22 * (magnet_thickness - 6.2) ** 2
        - 0.85 * (slot_opening - 2.1) ** 2
        - 0.18 * air_gap * magnet_thickness
        + 0.35 * np.sin(1.8 * slot_opening)
    )
    return torque + rng.normal(0.0, 0.25, size=np.shape(torque))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--output", type=str, default="data/examples/sample_motor_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    air_gap = rng.uniform(0.5, 1.5, size=args.n)
    magnet_thickness = rng.uniform(2.0, 8.0, size=args.n)
    slot_opening = rng.uniform(1.0, 4.0, size=args.n)
    torque = synthetic_torque(air_gap, magnet_thickness, slot_opening, rng)

    df = pd.DataFrame(
        {
            "air_gap_mm": air_gap,
            "magnet_thickness_mm": magnet_thickness,
            "slot_opening_mm": slot_opening,
            "torque_Nm": torque,
        }
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    ensure_dir(output.parent)

    if output.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output, index=False)
    else:
        df.to_csv(output, index=False)

    print(f"Saved sample data to: {output}")


if __name__ == "__main__":
    main()
