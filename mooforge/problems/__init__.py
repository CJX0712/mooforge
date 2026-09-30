"""Synthetic multi-objective benchmark problems."""

from .registry import default_problems, list_problems, make_problem
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
    BaseProblem,
)

__all__ = [
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ7",
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT4",
    "ZDT6",
    "BaseProblem",
    "default_problems",
    "list_problems",
    "make_problem",
]
