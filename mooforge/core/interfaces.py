"""Structural contracts (Protocol) so modules only depend on abstractions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Budget, OptimizationResult, ProblemSpec


@runtime_checkable
class Problem(Protocol):
    """A multi-objective optimisation problem with an analytic Pareto front."""

    spec: ProblemSpec

    def evaluate(self, X: np.ndarray) -> np.ndarray:  # (n, n_var) -> (n, n_obj)
        ...

    def pareto_front(self, n_points: int = 200) -> np.ndarray:
        ...

    def reference_point(self) -> np.ndarray:
        ...

    def nadir(self) -> np.ndarray:
        ...


@runtime_checkable
class Optimizer(Protocol):
    """An optimiser consuming a fixed evaluation budget."""

    name: str
    backend: str

    def available(self) -> bool:
        ...

    def run(
        self, problem: Problem, budget: Budget, seed: int
    ) -> OptimizationResult:
        ...


@runtime_checkable
class Indicator(Protocol):
    """A quality indicator. Convention: lower is better unless stated."""

    name: str

    def compute(self, F: np.ndarray, problem: Problem) -> float:
        ...


__all__ = ["Indicator", "Optimizer", "Problem"]
