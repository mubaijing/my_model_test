from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_optimization_history(history: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure()
    plt.plot(history["generation"], history["best_target"])
    plt.xlabel("Generation")
    plt.ylabel("Best target")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
