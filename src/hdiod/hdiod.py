"""High-Density Iteration-based Outlier Detection (HDIOD).

Re-implementation of the method proposed in

    Y. Zhou, H. Xia, D. Yu, J. Cheng and J. Li, "Outlier detection method based on
    high-density iteration", Information Sciences, vol. 662, 120286, 2024.
    doi:10.1016/j.ins.2024.120286

Each point repeatedly moves to its densest k-nearest neighbour until no neighbour is
denser (or a maximum number of steps is reached). All neighbours visited along the
way form an extended neighbourhood (EkNN). The Centripetal Outlier Factor (COF) of
a point is the highest local density inside its EkNN divided by its own density, so
points far from any dense region receive large scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class HDIOD:
    """HDIOD outlier detector.

    Parameters
    ----------
    k : int
        Number of nearest neighbours used for density estimation and iteration.
    mad_multiplier : float
        Points with ``COF > median + mad_multiplier * MAD`` are labelled as outliers.
        2.5 is the conventional value. 1.5 was used for the sparse, high-dimensional
        Arrhythmia and WPBC datasets, where 2.5 flagged almost no points.
    max_iter : int
        Maximum number of high-density moves per point (guards against cycles).
    """

    def __init__(self, k: int = 20, mad_multiplier: float = 2.5, max_iter: int = 100):
        self.k = k
        self.mad_multiplier = mad_multiplier
        self.max_iter = max_iter

    def _local_density(self, X: np.ndarray):
        """Gaussian kernel density over the k nearest neighbours.

        The bandwidth is the standard deviation of all k-NN distances, which keeps the
        kernel on a sensible scale across datasets of different dimensionality.
        """
        knn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X)
        distances, indices = knn.kneighbors(X)

        h = np.std(distances)
        if h == 0:
            h = 1e-5  # all points identical: avoid division by zero

        kernel = np.exp(-(distances ** 2) / (2 * h ** 2)) / ((2 * np.pi) ** (X.shape[1] / 2))
        return kernel.mean(axis=1), indices

    def _high_density_iteration(self, local_density: np.ndarray, indices: np.ndarray):
        """Move each point towards denser neighbours and collect its EkNN set."""
        n_samples = local_density.shape[0]
        eknn_sets = []

        for i in range(n_samples):
            current = i
            eknn = set()
            steps = 0

            while True:
                neighbours = indices[current]
                eknn.update(neighbours)

                densest = neighbours[np.argmax(local_density[neighbours])]
                if local_density[current] >= local_density[densest] or steps >= self.max_iter:
                    break  # reached a local density peak (or the step limit)
                current = densest
                steps += 1

            eknn_sets.append(eknn)

        return eknn_sets

    def score_samples(self, X) -> np.ndarray:
        """Return the COF score of every sample (higher = more anomalous)."""
        X = np.asarray(X, dtype=float)
        local_density, indices = self._local_density(X)
        eknn_sets = self._high_density_iteration(local_density, indices)

        cof = np.empty(X.shape[0])
        for i, eknn in enumerate(eknn_sets):
            cof[i] = local_density[list(eknn)].max() / local_density[i]
        return cof

    def fit_predict(self, X):
        """Return ``(labels, scores)`` where labels are 1 for outliers and 0 otherwise."""
        scores = self.score_samples(X)
        median = np.median(scores)
        mad = 1.4826 * np.median(np.abs(scores - median))
        threshold = median + self.mad_multiplier * mad
        return (scores > threshold).astype(int), scores
