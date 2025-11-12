from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

__all__ = ["PipelineConfig"]

@dataclass(frozen=True)
class PipelineConfig:
    numeric: list[str]
    log: list[str]

    selector_kind: str | None = None
    selector_k:    Optional[int | str] = None

    resampler: str | None = None
    model_name: Optional[str] = None
