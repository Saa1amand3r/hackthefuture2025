from __future__ import annotations

from typing import Callable, Any, Dict

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

__all__ = ["ModelFactory", "DEFAULT_MODELS", "make_model"]



ModelFactory = Callable[[], Any]


def _lr() -> Any:
    return LinearRegression()

DEFAULT_MODELS: Dict[str, ModelFactory] = {
    "linreg": _lr,
}

def make_model(name: str, registry: Dict[str, ModelFactory] | None = None) -> Any:
    reg = DEFAULT_MODELS if registry is None else registry
    if name not in reg:
        raise KeyError(f"Unknown model '{name}'. Available: {list(reg)}")
    return clone(reg[name]())
