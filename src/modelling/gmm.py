from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture


class GMMOOD(BaseEstimator):
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int = 1410,
        threshold: float | None = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.threshold = threshold
        self.model_: GaussianMixture | None = None

    def fit(self, x_train: np.ndarray) -> "GMMOOD":
        if x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array")

        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        )
        self.model_.fit(x_train)

        train_scores = self.score_samples(x_train)
        self.threshold = float(np.quantile(train_scores, 0.95))
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has to be fitted before scoring")

        return (-self.model_.score_samples(x)).astype(np.float64)

    def predict(self, x: np.ndarray, threshold: float | None = None) -> np.ndarray:
        th = self.threshold if threshold is None else threshold
        if th is None:
            raise RuntimeError("Threshold not available. Fit first or pass threshold")
        return (self.score_samples(x) > th).astype(np.int64)