from pathlib import Path
from typing import Sequence

import pandas as pd


def read_table(file_path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)

    raise ValueError(f"Unsupported data file type: {suffix}")


def validate_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def select_numeric_data(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    target_columns: Sequence[str],
) -> pd.DataFrame:
    required = list(input_columns) + list(target_columns)
    validate_columns(df, required)
    selected = df[required].copy()
    for col in required:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(axis=0).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No valid numeric rows after cleaning.")
    return selected
