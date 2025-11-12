from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

__all__ = ["PipelineConfig"]

from pipeline.weights import WeightingConfig


@dataclass(frozen=True)
class PipelineConfig:
    numeric: List[str]
    log: List[str]
    categorical: List[str]
    selector_kind: Optional[str] = None
    selector_k: Optional[int | str] = None
    model_name: Optional[str] = None

    weighting: Optional[WeightingConfig] = None

    resampler: Optional[str] = None