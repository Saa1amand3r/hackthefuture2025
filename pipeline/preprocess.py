from __future__ import annotations
from typing import List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from imblearn.pipeline import Pipeline

__all__ = ["make_default_preprocessor"]

def make_default_preprocessor(
    num_cols: List[str],
    log_cols: List[str],
) -> ColumnTransformer:
    num_plain = [c for c in num_cols if c not in set(log_cols)]

    log_pipe = Pipeline(steps=[
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline(steps=[
        ("scale", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_plain", num_pipe, num_plain),
            ("num_log",   log_pipe, log_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )