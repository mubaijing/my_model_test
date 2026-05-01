from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ProcessedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    input_scaler: StandardScaler
    output_scaler: StandardScaler


def split_and_scale(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> ProcessedData:
    X = df[list(input_columns)].to_numpy(dtype=np.float32)
    y = df[list(target_columns)].to_numpy(dtype=np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / max(1e-12, 1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_train_s = input_scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = input_scaler.transform(X_val).astype(np.float32)
    X_test_s = input_scaler.transform(X_test).astype(np.float32)

    y_train_s = output_scaler.fit_transform(y_train).astype(np.float32)
    y_val_s = output_scaler.transform(y_val).astype(np.float32)
    y_test_s = output_scaler.transform(y_test).astype(np.float32)

    return ProcessedData(
        X_train=X_train_s,
        y_train=y_train_s,
        X_val=X_val_s,
        y_val=y_val_s,
        X_test=X_test_s,
        y_test=y_test_s,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
    )
