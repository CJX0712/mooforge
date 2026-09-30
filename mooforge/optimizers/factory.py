"""Optimizer factory: names -> instances, with availability probing."""

from __future__ import annotations

from .numpy_impl import AHVAMOEA, DENumpy, NSGA2Numpy
from .pymoo_backends import (
    PYMOO_AVAILABLE,
    MOEADPymoo,
    NSGA2Pymoo,
    NSGA3Pymoo,
    SMSEMOAPymoo,
    SPEA2Pymoo,
)

# name -> constructor (no args)
_REGISTRY = {
    "NSGA2-Pymoo": NSGA2Pymoo,
    "NSGA3-Pymoo": NSGA3Pymoo,
    "MOEAD-Pymoo": MOEADPymoo,
    "SMSEMOA-Pymoo": SMSEMOAPymoo,
    "SPEA2-Pymoo": SPEA2Pymoo,
    "NSGA2-Numpy": NSGA2Numpy,
    "DE-Numpy": DENumpy,
    "AHVA-MOEA": AHVAMOEA,
}

FLAGSHIP = "AHVA-MOEA"
# Strong SOTA baseline used for the significance gate.
PRIMARY_BASELINE = "NSGA2-Pymoo"


def build(name: str, **kwargs):
    if name not in _REGISTRY:
        from ..core.errors import E301_UNKNOWN_ALGORITHM, OptimizerError

        raise OptimizerError(f"unknown optimizer: {name}", E301_UNKNOWN_ALGORITHM)
    return _REGISTRY[name](**kwargs)


def available(name: str) -> bool:
    try:
        inst = build(name)
        return inst.available()
    except Exception:  # noqa: BLE001
        return False


def list_algorithms() -> list[str]:
    return list(_REGISTRY.keys())


def baseline_algorithms() -> list[str]:
    """Deterministic baseline set; pymoo backends dropped if unavailable."""
    order = ["NSGA2-Pymoo", "NSGA3-Pymoo", "MOEAD-Pymoo", "SMSEMOA-Pymoo", "SPEA2-Pymoo"]
    present = [a for a in order if available(a)]
    present += ["NSGA2-Numpy", "DE-Numpy"]
    return present


def offline_fallback_algorithms() -> list[str]:
    """Pure-numpy set that runs with zero external dependencies."""
    return ["NSGA2-Numpy", "DE-Numpy", "AHVA-MOEA"]


__all__ = [
    "FLAGSHIP",
    "PRIMARY_BASELINE",
    "PYMOO_AVAILABLE",
    "available",
    "baseline_algorithms",
    "build",
    "list_algorithms",
    "offline_fallback_algorithms",
]
