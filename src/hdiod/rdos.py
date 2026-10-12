"""Relative Density-based Outlier Score (RDOS) baseline.

Re-implementation of

    B. Tang and H. He, "A local density-based approach for outlier detection",
    Neurocomputing, vol. 241, pp. 171-180, 2017. doi:10.1016/j.neucom.2017.02.051

The local density of a point is a Gaussian kernel estimate over an extended
neighbourhood made of its k-nearest neighbours (kNN), reverse nearest neighbours
(RNN) and shared nearest neighbours (SNN). The RDOS score is the mean density of that
neighbourhood divided by the point's own density.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class RDOS:
    """RDOS outlier detector.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    bandwidth : float
        Gaussian kernel bandwidth.
    percentile : float
        Points scoring above this percentile are labelled as outliers.
    legacy : bool
        If True, use the simplified setting from the dissertation, in which every
        point's neighbourhood covers the whole dataset. Use this to reproduce the
        dissertation numbers.
    """

    def __init__(self, k: int = 20, bandwidth: float = 0.1, percentile: float = 85, legacy: bool = False):
        self.k = k
        self.bandwidth = bandwidth
        self.percentile = percentile
        self.legacy = legacy

    def _neighbourhoods(self, X: np.ndarray):
        """Return a sparse boolean matrix whose row i marks the neighbourhood of point i."""
        n = X.shape[0]
        if self.legacy:
            return None  # every point is in every neighbourhood

        indices = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X).kneighbors(X, return_distance=False)
        rows = np.repeat(np.arange(n), self.k)
        knn = sparse.csr_matrix((np.ones(rows.size, dtype=bool), (rows, indices.ravel())), shape=(n, n))

        rnn = knn.T                       # j is a reverse neighbour of i if i is in kNN(j)
        snn = (knn @ knn.T).astype(bool)  # i and j share at least one nearest neighbour
        neighbourhood = (knn + rnn + snn).astype(bool).tolil()
        neighbourhood.setdiag(True)       # the point itself is part of its density estimate
        return neighbourhood.tocsr()

    def _kernel_mean(self, X: np.ndarray, i: int, members: np.ndarray) -> float:
        distances = np.linalg.norm(X[members] - X[i], axis=1)
        return np.exp(-(distances ** 2) / (2 * self.bandwidth ** 2)).mean()

    def score_samples(self, X) -> np.ndarray:
        """Return the RDOS score of every sample (higher = more anomalous)."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        neighbourhood = self._neighbourhoods(X)
        everyone = np.arange(n)

        members = [everyone if neighbourhood is None else neighbourhood[i].indices for i in range(n)]
        density = np.array([self._kernel_mean(X, i, members[i]) for i in range(n)])

        scores = np.array([density[members[i]].mean() / (density[i] + 1e-10) for i in range(n)])
        return scores

    def fit_predict(self, X):
        """Return ``(labels, scores)`` where labels are 1 for outliers and 0 otherwise."""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, self.percentile)
        return (scores > threshold).astype(int), scores
