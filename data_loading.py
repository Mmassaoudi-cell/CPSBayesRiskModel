from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PowerDatasetSpec:
    task: str
    path: Path
    label_col: str = "marker"


def paper2_dataset_specs(data_root: str | Path) -> list[PowerDatasetSpec]:
    root = Path(data_root)
    return [
        PowerDatasetSpec(task="binary", path=root / "Binary" / "TrainingDataBinary.csv"),
        PowerDatasetSpec(task="multiclass", path=root / "Multi" / "TrainingDataMulti.csv"),
    ]


def load_power_dataset(spec: PowerDatasetSpec):
    df = pd.read_csv(spec.path)
    df.columns = [str(c).strip() for c in df.columns]
    y = df[spec.label_col].astype(int)
    X = df.drop(columns=[spec.label_col]).replace([np.inf, -np.inf], np.nan)
    return X, y


def stratified_holdout(X, y, seed: int, test_size: float = 0.25):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
