# src/pipeline/weighted_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from inspect import signature

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .weights import make_sample_weights, WeightingConfig

@dataclass
class WeightedModel(BaseEstimator, RegressorMixin):
    base_estimator: Any
    weighting: WeightingConfig

    def __post_init__(self):
        self._fit_accepts_sample_weight = None
        self.est_: Any = None

    def _estimator_accepts_sample_weight(self, est) -> bool:
        try:
            sig = signature(est.fit)
            return "sample_weight" in sig.parameters
        except (ValueError, TypeError):
            return False

    def fit(self, X, y):
        y = np.asarray(y)
        w = make_sample_weights(y, **self.weighting.as_kwargs())
        self.est_ = clone(self.base_estimator)
        if self._fit_accepts_sample_weight is None:
            self._fit_accepts_sample_weight = self._estimator_accepts_sample_weight(self.est_)
        if self._fit_accepts_sample_weight:
            self.est_.fit(X, y, sample_weight=w)
        else:
            # graceful fallback
            self.est_.fit(X, y)
        return self

    def predict(self, X):
        return self.est_.predict(X)

    # delegate attrs like feature_importances_ if present
    def __getattr__(self, name):
        est = object.__getattribute__(self, "est_") if "est_" in self.__dict__ else None
        if est is not None and hasattr(est, name):
            return getattr(est, name)
        raise AttributeError(name)
