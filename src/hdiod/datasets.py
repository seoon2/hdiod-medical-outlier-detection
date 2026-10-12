"""Loaders for the six medical benchmark datasets.

Every loader returns ``(X, y)`` where ``X`` is Min-Max scaled to [0, 1] and ``y`` is a
binary label (1 = anomaly, 0 = normal). The label definitions follow the dissertation.

Raw files are not redistributed. See ``data/README.md`` for download instructions.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Raw files live in data/raw/ by default; set HDIOD_DATA_DIR to use another folder.
DATA_DIR = Path(os.environ.get("HDIOD_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "raw"))

# The six continuous (hormone-level) attributes of the UCI thyroid ANN data.
THYROID_CONTINUOUS = list(range(15, 21))


def _scale(X) -> np.ndarray:
    return MinMaxScaler().fit_transform(np.asarray(X, dtype=float))


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download instructions.")
    return path


def load_pima(data_dir: Path = DATA_DIR):
    """Pima Indians Diabetes. Anomaly = diabetes onset (Outcome == 1)."""
    df = pd.read_csv(_require(data_dir / "diabetes.csv"))
    return _scale(df.drop(columns="Outcome")), df["Outcome"].to_numpy()


def _read_thyroid(*names: str, data_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(_require(data_dir / n), header=None, sep=r"\s+") for n in names]
    df = pd.concat(frames, ignore_index=True)
    df.columns = list(range(21)) + ["cls"]
    return df


def load_thyroid(data_dir: Path = DATA_DIR):
    """UCI thyroid (ann-train). Anomaly = class 1 (hyperfunction). 6 continuous features."""
    df = _read_thyroid("ann-train.data", data_dir=data_dir)
    return _scale(df[THYROID_CONTINUOUS]), (df["cls"] == 1).astype(int).to_numpy()


def load_annthyroid(data_dir: Path = DATA_DIR):
    """UCI thyroid (ann-train + ann-test). Anomaly = class 1 or 2. 6 continuous features."""
    df = _read_thyroid("ann-train.data", "ann-test.data", data_dir=data_dir)
    return _scale(df[THYROID_CONTINUOUS]), (df["cls"] != 3).astype(int).to_numpy()


def load_vertebral(data_dir: Path = DATA_DIR):
    """UCI Vertebral Column (3 classes). Anomaly = Hernia or Spondylolisthesis."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=212)
    y = (ds.data.targets.iloc[:, 0] != "Normal").astype(int).to_numpy()
    return _scale(ds.data.features), y


def load_arrhythmia(data_dir: Path = DATA_DIR):
    """UCI Arrhythmia. Anomaly = any class other than 1 (normal).

    Columns 10-14 contain many missing values and are dropped.
    """
    df = pd.read_csv(_require(data_dir / "arrhythmia.data"), header=None, na_values="?")
    df = df.drop(columns=df.columns[[10, 11, 12, 13, 14]])
    X, cls = df.iloc[:, :-1], df.iloc[:, -1]
    return _scale(X), (cls != 1).astype(int).to_numpy()


def load_wpbc(data_dir: Path = DATA_DIR):
    """UCI Wisconsin Prognostic Breast Cancer. Anomaly = recurrence (Outcome == 'R')."""
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=16)
    X = ds.data.features.copy()
    X["lymph_node_status"] = X["lymph_node_status"].fillna(X["lymph_node_status"].median())
    y = (ds.data.targets["Outcome"] == "R").astype(int).to_numpy()
    return _scale(X), y


LOADERS = {
    "Pima": load_pima,
    "Vertebral": load_vertebral,
    "Thyroid": load_thyroid,
    "Annthyroid": load_annthyroid,
    "Arrhythmia": load_arrhythmia,
    "WPBC": load_wpbc,
}
