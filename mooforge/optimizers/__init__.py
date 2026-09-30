"""Optimizers: Tier-0 pymoo SOTA + Tier-1 numpy handrolled + AHVA flagship."""

from .base import (
    BaseOptimizer,
    crowding_distance,
    de_operator,
    fast_non_dominated_sort,
    polynomial_mutation,
    sbx_crossover,
)
from .factory import (
    FLAGSHIP,
    PRIMARY_BASELINE,
    available,
    baseline_algorithms,
    build,
    list_algorithms,
    offline_fallback_algorithms,
)
from .numpy_impl import AHVAMOEA, DENumpy, NSGA2Numpy
from .pymoo_backends import (
    PYMOO_AVAILABLE,
    MOEADPymoo,
    NSGA2Pymoo,
    NSGA3Pymoo,
    SMSEMOAPymoo,
    SPEA2Pymoo,
)

__all__ = [
    "AHVAMOEA",
    "FLAGSHIP",
    "PRIMARY_BASELINE",
    "PYMOO_AVAILABLE",
    "BaseOptimizer",
    "DENumpy",
    "MOEADPymoo",
    "NSGA2Numpy",
    "NSGA2Pymoo",
    "NSGA3Pymoo",
    "SMSEMOAPymoo",
    "SPEA2Pymoo",
    "available",
    "baseline_algorithms",
    "build",
    "crowding_distance",
    "de_operator",
    "fast_non_dominated_sort",
    "list_algorithms",
    "offline_fallback_algorithms",
    "polynomial_mutation",
    "sbx_crossover",
]
