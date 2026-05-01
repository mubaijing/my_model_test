import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from motoropt.data.dataset import TabularDataset
from motoropt.data.loader import read_table, select_numeric_data
from motoropt.data.preprocessor import split_and_scale
from motoropt.models.factory import build_model
from motoropt.training.metrics import regression_metrics
from motoropt.training.trainer import Trainer
from motoropt.utils.config import load_yaml, resolve_path
from motoropt.utils.io import ensure_dir, save_json
from motoropt.utils.seed import set_seed
from motoropt.visualization.plot_training import plot_loss_curve, plot_pred_vs_true


def select_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config_path = resolve_path(args.config, PROJECT_ROOT)
    config = load_yaml(config_path)
    schema = load_yaml(resolve_path(config["schema_path"], PROJECT_ROOT))

    seed = int(config["data"].get("random_state", 42))
    set_seed(seed)

    model_dir = ensure_dir(resolve_path(config["paths"]["model_dir"], PROJECT_ROOT))
    log_dir = ensure_dir(resolve_path(config["paths"]["log_dir"], PROJECT_ROOT))
    figure_dir = ensure_dir(resolve_path(config["paths"]["figure_dir"], PROJECT_ROOT))

    data_path = resolve_path(config["data"]["file_path"], PROJECT_ROOT)
    df = read_table(data_path, sheet_name=config["data"].get("sheet_name"))
    df = select_numeric_data(df, schema["input_columns"], schema["target_columns"])

    processed = split_and_scale(
        df=df,
        input_columns=schema["input_columns"],
        target_columns=schema["target_columns"],
        test_size=float(config["data"].get("test_size", 0.15)),
        val_size=float(config["data"].get("val_size", 0.15)),
        random_state=seed,
    )

    train_ds = TabularDataset(processed.X_train, processed.y_train)
    val_ds = TabularDataset(processed.X_val, processed.y_val)
    test_ds = TabularDataset(processed.X_test, processed.y_test)

    batch_size = int(config["training"].get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        config["model"],
        input_dim=len(schema["input_columns"]),
        output_dim=len(schema["target_columns"]),
    )
    device = select_device(config["training"].get("device", "auto"))

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
        early_stopping_patience=int(config["training"].get("early_stopping_patience", 50)),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(config["training"].get("epochs", 300)),
    )

    y_true_s, y_pred_s = trainer.predict_loader(test_loader)
    y_true = processed.output_scaler.inverse_transform(y_true_s.numpy())
    y_pred = processed.output_scaler.inverse_transform(y_pred_s.numpy())
    metrics = regression_metrics(y_true, y_pred)

    torch.save(trainer.model.state_dict(), model_dir / "surrogate_model.pt")
    joblib.dump(processed.input_scaler, model_dir / "input_scaler.pkl")
    joblib.dump(processed.output_scaler, model_dir / "output_scaler.pkl")

    meta = {
        "input_columns": schema["input_columns"],
        "target_columns": schema["target_columns"],
        "model_config": config["model"],
        "test_metrics": metrics,
    }
    save_json(meta, model_dir / "model_meta.json")

    history_df = pd.DataFrame(history)
    history_df.to_csv(log_dir / "train_log.csv", index=False)
    plot_loss_curve(history_df, figure_dir / "loss_curve.png")
    plot_pred_vs_true(y_true, y_pred, figure_dir / "pred_vs_true.png")

    print("Training finished.")
    print(f"Model directory: {model_dir}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()
