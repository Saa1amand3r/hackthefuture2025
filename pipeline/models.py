from __future__ import annotations

from typing import Callable, Any, Dict

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso

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

DEFAULT_MODELS: Dict[str, ModelFactory] = {
    "linreg": _lr,
    "lasso": _lasso,
    "dt_regressor": _dt_regressor,
    "rf_regressor": _rf_regressor
}

def make_model(name: str, registry: Dict[str, ModelFactory] | None = None) -> Any:
    reg = DEFAULT_MODELS if registry is None else registry
    if name not in reg:
        raise KeyError(f"Unknown model '{name}'. Available: {list(reg)}")
    return clone(reg[name]())
