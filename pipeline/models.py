from __future__ import annotations

from typing import Callable, Any, Dict

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

__all__ = ["ModelFactory", "DEFAULT_MODELS", "make_model"]

from sklearn.tree import DecisionTreeRegressor


ModelFactory = Callable[[], Any]

def _lr() -> Any:
    return LinearRegression()

def _dt_regressor() -> Any:
    return DecisionTreeRegressor(
        random_state=42
    )

def _rf_regressor() -> Any:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

def _lasso() -> Any:
    return Lasso(alpha=0.001, max_iter=1000000)

def _hgb() -> Any:
    return HGBR(
        loss="absolute_error",
        learning_rate=0.06,
        max_leaf_nodes=63,  # more capacity than default
        min_samples_leaf=20,  # regularize a bit
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
    )

def _hgb_log() -> Any:
    # log1p target transform + absolute error loss
    return TransformedTargetRegressor(
        regressor=_hgb(),
        func=np.log1p,
        inverse_func=np.expm1,
)





DEFAULT_MODELS: Dict[str, ModelFactory] = {
    "linreg": _lr,
    "lasso": _lasso,
    "dt_regressor": _dt_regressor,
    "rf_regressor": _rf_regressor,
    "hgbr": _hgb(),
    "hgb_log": _hgb_log,
}

def make_model(name: str, registry: Dict[str, ModelFactory] | None = None) -> Any:
    reg = DEFAULT_MODELS if registry is None else registry
    if name not in reg:
        raise KeyError(f"Unknown model '{name}'. Available: {list(reg)}")
    return clone(reg[name]())
