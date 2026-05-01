from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_curve(history: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    plt.figure()
    plt.scatter(y_true, y_pred, s=20)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
