from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import numpy as np
import torch

from motoropt.models.factory import build_model
from motoropt.utils.io import load_json


class Predictor:
    def __init__(
        self,
        model_path: str | Path,
        input_scaler_path: str | Path,
        output_scaler_path: str | Path,
        model_meta_path: str | Path,
        device: str = "cpu",
    ):
        self.device = device
        self.meta = load_json(model_meta_path)
        self.input_columns = self.meta["input_columns"]
        self.target_columns = self.meta["target_columns"]
        self.input_scaler = joblib.load(input_scaler_path)
        self.output_scaler = joblib.load(output_scaler_path)

        self.model = build_model(
            self.meta["model_config"],
            input_dim=len(self.input_columns),
            output_dim=len(self.target_columns),
        )
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, data: Dict[str, float] | Iterable[Dict[str, float]] | np.ndarray):
        X, single = self._to_array(data)
        X_scaled = self.input_scaler.transform(X).astype(np.float32)
        tensor = torch.as_tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_scaled = self.model(tensor).cpu().numpy()
        y = self.output_scaler.inverse_transform(y_scaled)

        results = [
            {name: float(value) for name, value in zip(self.target_columns, row)}
            for row in y
        ]
        return results[0] if single else results

    def _to_array(self, data: Any) -> tuple[np.ndarray, bool]:
        if isinstance(data, dict):
            row = [float(data[col]) for col in self.input_columns]
            return np.asarray([row], dtype=np.float32), True

        if isinstance(data, np.ndarray):
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
                return arr, True
            return arr, False

        rows = []
        for item in data:
            rows.append([float(item[col]) for col in self.input_columns])
        return np.asarray(rows, dtype=np.float32), False
