from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Optional, List

from imblearn.pipeline import Pipeline

from .config       import PipelineConfig
from .preprocess import make_default_preprocessor
from .selectors    import make_selector
from .samplers     import SAMPLER_REGISTRY
from .models       import make_model, DEFAULT_MODELS, ModelFactory

__all__ = ["PipelineFactory"]

@dataclass
class PipelineFactory:
    model_registry: Dict[str, ModelFactory] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    preprocess_factory: Callable[[List[str], List[str]], Any] = make_default_preprocessor
    selector_factory: Callable[[str | None, int | str | None], Any] = make_selector

    def build(self, cfg: PipelineConfig) -> Pipeline:
        steps: list[tuple[str, Any]] = []

        steps.append(("preprocess",
                      self.preprocess_factory(cfg.numeric, cfg.log)))

        selector = self.selector_factory(cfg.selector_kind, cfg.selector_k)
        if selector is not None:
            steps.append(("select", selector))

        sampler = SAMPLER_REGISTRY[cfg.resampler]   # "none" → None
        if sampler is not None:
            if (isinstance(sampler,list)):
                steps.extend(sampler)
            else:
                steps.append(("resample", sampler))

        if cfg.model_name:
            steps.append(("model", make_model(cfg.model_name,
                                              registry=self.model_registry)))

        return Pipeline(steps, verbose=False)

    def add_model(self, name: str, factory: ModelFactory) -> None:
        self.model_registry[name] = factory

    @staticmethod
    def with_model(pipeline: Pipeline, model: Any) -> Pipeline:
        from sklearn.base import clone

        if pipeline.steps and pipeline.steps[-1][0] == "model":
            new_steps = pipeline.steps[:-1] + [("model", clone(model))]
        else:
            new_steps = pipeline.steps + [("model", clone(model))]
        return Pipeline(new_steps, verbose=getattr(pipeline, "verbose", False))
