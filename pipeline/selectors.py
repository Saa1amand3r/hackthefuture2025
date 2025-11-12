from __future__ import annotations

from typing import Callable, Dict

from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

__all__ = ["SCORE_FUNCS", "make_selector"]

SCORE_FUNCS: Dict[str, Callable] = {
    "anova": f_regression,            # F-score for regression (linear-friendly)
    "mi":    mutual_info_regression,  # Mutual information for regression (nonlinear-friendly)
}

def make_selector(kind: str | None, k: int | str | None) -> SelectKBest | None:
    if kind is None or k is None:
        return None
    try:
        return SelectKBest(score_func=SCORE_FUNCS[kind], k=k)
    except KeyError as e:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {list(SCORE_FUNCS)}") from e
