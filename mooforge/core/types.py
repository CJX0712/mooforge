"""Core value objects: problem spec, run result, evaluation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import E102_BOUNDS, ConfigError


@dataclass
class ProblemSpec:
    """Static description of a multi-objective problem."""

    name: str
    n_var: int
    n_obj: int
    xl: np.ndarray
    xu: np.ndarray
    # Analytic knowledge used for exact quality indicators.
    has_analytic_front: bool = True
    difficulty: str = "medium"
    # Reference point covering the worst local optimum so HV stays comparable
    # (>=0 and <1 for every optimizer) even on multimodal landscapes.
    ref_point: np.ndarray | None = None

    def __post_init__(self) -> None:
        xl = np.asarray(self.xl, dtype=float).reshape(-1)
        xu = np.asarray(self.xu, dtype=float).reshape(-1)
        if xl.size != self.n_var or xu.size != self.n_var:
            raise ConfigError(
                E102_BOUNDS,
                f"bounds size {xl.size}/{xu.size} != n_var {self.n_var}",
            )
        if np.any(xu <= xl):
            raise ConfigError(E102_BOUNDS, "xu must be strictly greater than xl")
        object.__setattr__(self, "xl", xl)
        object.__setattr__(self, "xu", xu)
        if self.ref_point is not None:
            rp = np.asarray(self.ref_point, dtype=float).reshape(-1)
            if rp.size != self.n_obj:
                raise ConfigError(
                    E102_BOUNDS, f"ref_point size {rp.size} != n_obj {self.n_obj}"
                )
            object.__setattr__(self, "ref_point", rp)


@dataclass
class OptimizationResult:
    """Outcome of one optimizer run on one problem."""

    problem: str
    algorithm: str
    backend: str
    X: np.ndarray
    F: np.ndarray
    n_evals: int
    elapsed_sec: float
    hv: float | None = None
    igd: float | None = None
    spacing: float | None = None
    spread: float | None = None
    history: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "algorithm": self.algorithm,
            "backend": self.backend,
            "n_evals": self.n_evals,
            "elapsed_sec": round(float(self.elapsed_sec), 6),
            "hv": None if self.hv is None else float(self.hv),
            "igd": None if self.igd is None else float(self.igd),
            "spacing": None if self.spacing is None else float(self.spacing),
            "spread": None if self.spread is None else float(self.spread),
            "n_solutions": int(np.asarray(self.X).shape[0]),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class Budget:
    """Evaluation budget contract shared by every optimizer."""

    n_evals: int
    pop_size: int

    @property
    def n_gen(self) -> int:
        return max(1, int(self.n_evals) // max(1, int(self.pop_size)))


__all__ = ["Budget", "OptimizationResult", "ProblemSpec"]
