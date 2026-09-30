"""Tier-0 SOTA backends via pymoo 0.6.2 (NSGA-II / NSGA-III / MOEA-D / SMS-EMOA)."""

from __future__ import annotations

import time

import numpy as np

from ..core.errors import E302_BACKEND_MISSING, OptimizerError
from ..core.types import Budget
from .base import BaseOptimizer

PYMOO_AVAILABLE = False
try:
    from pymoo.core.problem import Problem as _PymooProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    PYMOO_AVAILABLE = True
except Exception:  # pragma: no cover  # noqa: BLE001
    PYMOO_AVAILABLE = False


class _Adapter(_PymooProblem if PYMOO_AVAILABLE else object):
    def __init__(self, moo_problem) -> None:
        self._moo = moo_problem
        super().__init__(
            n_var=moo_problem.spec.n_var,
            n_obj=moo_problem.spec.n_obj,
            xl=moo_problem.spec.xl,
            xu=moo_problem.spec.xu,
            vtype=float,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self._moo.evaluate(X)


def _build_algorithm(name: str, pop_size: int, n_obj: int):
    if not PYMOO_AVAILABLE:  # pragma: no cover
        raise OptimizerError("pymoo backend unavailable", E302_BACKEND_MISSING)
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.algorithms.moo.spea2 import SPEA2
    from pymoo.util.ref_dirs import get_reference_directions

    if name == "NSGA2":
        return NSGA2(pop_size=pop_size)
    if name == "SMSEMOA":
        return SMSEMOA(pop_size=pop_size)
    if name == "SPEA2":
        return SPEA2(pop_size=pop_size)
    if name == "MOEAD":
        n_part = 12 if n_obj == 2 else 10
        ref = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=n_part)
        return MOEAD(ref_dirs=ref, n_neighbors=min(20, ref.shape[0] // 2))
    if name == "NSGA3":
        n_part = 12 if n_obj == 2 else 10
        ref = get_reference_directions("das-dennis", n_dim=n_obj, n_partitions=n_part)
        return NSGA3(pop_size=pop_size, ref_dirs=ref)
    raise OptimizerError(f"unknown pymoo algorithm {name}", E302_BACKEND_MISSING)


class PymooOptimizer(BaseOptimizer):
    """Generic pymoo-backed optimizer identified by its algorithm name."""

    backend = "pymoo"

    def __init__(self, name: str) -> None:
        self.name = name

    def available(self) -> bool:
        return PYMOO_AVAILABLE

    def run(self, problem, budget: Budget, seed: int) -> object:
        if not PYMOO_AVAILABLE:  # pragma: no cover
            raise OptimizerError(f"{self.name} backend unavailable", E302_BACKEND_MISSING)
        t0 = time.perf_counter()
        adapter = _Adapter(problem)
        algo = _build_algorithm(self.name, budget.pop_size, problem.spec.n_obj)
        term = get_termination("n_eval", budget.n_evals)
        res = minimize(adapter, algo, termination=term, seed=int(seed), verbose=False)
        X = np.asarray(res.X)
        F = np.asarray(res.F)
        if X.size == 0 or F.size == 0:
            raise OptimizerError(f"{self.name} returned no result", E302_BACKEND_MISSING)
        return self._result(
            problem, X, F, n_evals=budget.n_evals, elapsed=time.perf_counter() - t0,
            history=[], meta={"algorithm": self.name},
        )


class NSGA2Pymoo(PymooOptimizer):
    def __init__(self) -> None:
        super().__init__("NSGA2")


class NSGA3Pymoo(PymooOptimizer):
    def __init__(self) -> None:
        super().__init__("NSGA3")


class MOEADPymoo(PymooOptimizer):
    def __init__(self) -> None:
        super().__init__("MOEAD")


class SMSEMOAPymoo(PymooOptimizer):
    def __init__(self) -> None:
        super().__init__("SMSEMOA")


class SPEA2Pymoo(PymooOptimizer):
    def __init__(self) -> None:
        super().__init__("SPEA2")


__all__ = [
    "PYMOO_AVAILABLE",
    "MOEADPymoo",
    "NSGA2Pymoo",
    "NSGA3Pymoo",
    "PymooOptimizer",
    "SMSEMOAPymoo",
    "SPEA2Pymoo",
]
