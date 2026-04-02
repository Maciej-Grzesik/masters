from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


class MahalanobisOOD(BaseEstimator):
    def __init__(self, regularization: float = 1e-6, threshold: float | None = None):
        self.regularization = regularization
        self.threshold = threshold
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "MahalanobisOOD":
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array")

        self.mean_ = x_train.mean(axis=0)
        centered = x_train - self.mean_
        cov = np.cov(centered, rowvar=False)

        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=np.float64)

        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * self.regularization
        self.precision_ = np.linalg.pinv(cov)

        train_scores = self.score_samples(x_train)
        self.threshold = float(np.quantile(train_scores, 0.95))
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("Model has to be fitted before scoring")

        centered = x - self.mean_
        sq_mahal = np.einsum("bi,ij,bj->b", centered, self.precision_, centered)
        return sq_mahal.astype(np.float64)

    def predict(self, x: np.ndarray, threshold: float | None = None) -> np.ndarray:
        th = self.threshold if threshold is None else threshold
        if th is None:
            raise RuntimeError("Threshold not available. Fit first or pass threshold")
        return (self.score_samples(x) > th).astype(np.int64)
