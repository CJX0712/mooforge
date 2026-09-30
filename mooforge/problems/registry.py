"""Problem registry: name -> constructor (no args needed)."""

from __future__ import annotations

from .synthetic import (
    DTLZ1,
    DTLZ2,
    DTLZ3,
    DTLZ7,
    ZDT1,
    ZDT2,
    ZDT3,
    ZDT4,
    ZDT6,
)

_REGISTRY = {
    "ZDT1": ZDT1,
    "ZDT2": ZDT2,
    "ZDT3": ZDT3,
    "ZDT4": ZDT4,
    "ZDT6": ZDT6,
    "DTLZ1": DTLZ1,
    "DTLZ2": DTLZ2,
    "DTLZ3": DTLZ3,
    "DTLZ7": DTLZ7,
}


def make_problem(name: str):
    if name not in _REGISTRY:
        from ..core.errors import E201_UNKNOWN_PROBLEM, ProblemError

        raise ProblemError(f"unknown problem: {name}", E201_UNKNOWN_PROBLEM)
    return _REGISTRY[name]()


def list_problems() -> list[str]:
    return list(_REGISTRY.keys())


def default_problems() -> list[str]:
    # A difficulty-gradient sweep that stays fast and exposes the flagship's
    # restart/operator-credit gains: easy convex (ZDT1), multimodal traps
    # (ZDT4, DTLZ1) and a many-objective sphere (DTLZ2). Flagged as "hard".
    return ["ZDT1", "ZDT4", "DTLZ2"]


__all__ = ["default_problems", "list_problems", "make_problem"]
