"""Shared evaluation helpers."""

from __future__ import annotations

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .hdiod import HDIOD
from .rdos import RDOS

# Dataset-specific settings used in the dissertation.
# HDIOD: median + 1.5 MAD threshold on sparse, high-dimensional data (2.5 elsewhere).
# RDOS: wider bandwidth on Arrhythmia, where h = 0.1 gave near-zero densities.
MAD_MULTIPLIER = {"Arrhythmia": 1.5, "WPBC": 1.5}
RDOS_BANDWIDTH = {"Arrhythmia": 0.3}


def build_models(dataset: str, k: int = 20, legacy_rdos: bool = False) -> dict:
    """Return the five detectors configured as in the dissertation."""
    return {
        "HDIOD": HDIOD(k=k, mad_multiplier=MAD_MULTIPLIER.get(dataset, 2.5)),
        "RDOS": RDOS(k=k, bandwidth=RDOS_BANDWIDTH.get(dataset, 0.1), legacy=legacy_rdos),
        "LOF": LOF(n_neighbors=k),
        "KNN": KNN(n_neighbors=k),
        "IForest": IForest(contamination=0.15, random_state=42),
    }


def run_model(model, X):
    """Return ``(labels, scores)`` for either our detectors or PyOD detectors."""
    if hasattr(model, "decision_function"):  # PyOD API
        model.fit(X)
        return model.predict(X), model.decision_function(X)
    return model.fit_predict(X)


def metrics(y_true, labels, scores) -> dict:
    return {
        "AUC-ROC": roc_auc_score(y_true, scores),
        "F1-score": f1_score(y_true, labels, zero_division=0),
        "Precision": precision_score(y_true, labels, zero_division=0),
        "Recall": recall_score(y_true, labels, zero_division=0),
    }


def stratified_subsample(X, y, n: int, seed: int = 42):
    """Stratified subsample, used to keep the k-sweep on Annthyroid tractable."""
    from sklearn.model_selection import train_test_split

    if len(y) <= n:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_s, np.asarray(y_s)
