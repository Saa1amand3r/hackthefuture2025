from __future__ import annotations

from typing import Callable, Any, Dict

from sklearn.base import clone
from xgboost import XGBClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier

__all__ = ["ModelFactory", "DEFAULT_MODELS", "make_model"]

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier


ModelFactory = Callable[[], Any]

def _lr() -> Any:
    return LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs",
    class_weight="balanced",
    n_jobs=None,
    random_state=42,
)

def _adaboost() -> Any:
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=300,
        learning_rate=0.5,
        algorithm="SAMME",
        random_state=42,
    )


def _knnr() -> Any:
    return KNeighborsClassifier(
    n_neighbors=11,
    weights="distance",
    p=2
)

def _nb() -> Any:
    return GaussianNB()

def _qda() -> Any:
    return QuadraticDiscriminantAnalysis(
        reg_param=1e-3,
        tol=1e-4
    )

def _hgb() -> Any:
    return HistGradientBoostingClassifier(
        random_state=42
    )

def _rf() -> Any:
    return RandomForestClassifier(
        n_estimators=1369,
        max_depth=16,
        min_samples_leaf= 8,
        min_samples_split= 2,
        max_features=None,
        bootstrap=True,
    )

def _xgb() -> Any:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False
    )

DEFAULT_MODELS: Dict[str, ModelFactory] = {
    "logreg": _lr,
    "rf": _rf,
    "xgb": _xgb,
    "adaboost": _adaboost,
    "hgb": _hgb,
    "qda": _qda,
    "nb": _nb,
    "knnr": _knnr
}

def make_model(name: str, registry: Dict[str, ModelFactory] | None = None) -> Any:
    reg = DEFAULT_MODELS if registry is None else registry
    if name not in reg:
        raise KeyError(f"Unknown model '{name}'. Available: {list(reg)}")
    return clone(reg[name]())
