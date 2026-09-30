"""MOOForge - multi-objective evolutionary optimization toolkit.

Top-level API: build optimizers, run benchmarks, compute quality indicators.
Flagship: AHVA-MOEA (adaptive hypervolume-driven operator-credit MOEA).
Author: 晨星 (CJX0712).
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import DEFAULT_CONFIG, MOOForgeConfig
from .core.seed import set_all
from .core.types import Budget, OptimizationResult, ProblemSpec
from .metrics.hypervolume import hypervolume
from .metrics.indicators import igd_plus, normalized_hv, spacing, spread
from .optimizers.factory import (
    FLAGSHIP,
    PRIMARY_BASELINE,
    available,
    baseline_algorithms,
    build,
    list_algorithms,
    offline_fallback_algorithms,
)
from .pipeline.moo_pipeline import MOOForgePipeline, run_pipeline
from .problems.registry import default_problems, list_problems, make_problem

__all__ = [
    "DEFAULT_CONFIG",
    "FLAGSHIP",
    "PRIMARY_BASELINE",
    "Budget",
    "MOOForgeConfig",
    "MOOForgePipeline",
    "OptimizationResult",
    "ProblemSpec",
    "__author__",
    "__version__",
    "available",
    "baseline_algorithms",
    "build",
    "default_problems",
    "hypervolume",
    "igd_plus",
    "list_algorithms",
    "list_problems",
    "make_problem",
    "normalized_hv",
    "offline_fallback_algorithms",
    "run_pipeline",
    "set_all",
    "spacing",
    "spread",
]
