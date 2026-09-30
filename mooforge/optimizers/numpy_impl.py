"""Dependency-free optimizers: NSGA-II (numpy), DE-variant, and the AHVA flagship.

AHVA-MOEA = UCB operator-credit assignment (SBX / DE / strong-mutation) +
elite archive re-injection + stagnation restart, all inside a fixed evaluation
budget. Pure numpy; runs with zero external dependencies.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ..core.types import Budget
from ..metrics.hypervolume import hypervolume
from ..metrics.indicators import igd_plus
from .base import (
    BaseOptimizer,
    crowding_distance,
    de_operator,
    fast_non_dominated_sort,
    polynomial_mutation,
    sbx_crossover,
)


def _nds_select(F: np.ndarray, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices of size n_keep selected by NSGA-II crowding truncation."""
    fronts = fast_non_dominated_sort(F)
    cd = crowding_distance(F, fronts)
    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= n_keep:
            chosen.extend(front.tolist())
        else:
            need = n_keep - len(chosen)
            order = front[np.argsort(-cd[front])]
            chosen.extend(order[:need].tolist())
            break
    return np.array(chosen, dtype=int)


class NSGA2Numpy(BaseOptimizer):
    name = "NSGA2-Numpy"
    backend = "numpy"

    def __init__(self, eta_c: float = 15.0, eta_m: float = 20.0, cr: float = 0.9) -> None:
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.cr = cr

    def run(self, problem, budget: Budget, seed: int) -> object:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        # route numpy global RNG used by operator helpers through our rng
        np.random.seed(seed)
        n = budget.pop_size
        xl, xu = problem.spec.xl, problem.spec.xu
        X = rng.random((n, problem.spec.n_var)) * (xu - xl) + xl
        F = problem.evaluate(X)
        n_evals = n
        history: list[float] = []
        ref = problem.reference_point()
        for _ in range(budget.n_gen):
            off = sbx_crossover(X, xl, xu, self.eta_c, self.cr)
            off = polynomial_mutation(off, xl, xu, self.eta_m, 1.0 / problem.spec.n_var)
            Fo = problem.evaluate(off)
            n_evals += off.shape[0]
            comb_X = np.vstack([X, off])
            comb_F = np.vstack([F, Fo])
            keep = _nds_select(comb_F, n, rng)
            X, F = comb_X[keep], comb_F[keep]
            history.append(hypervolume(F, ref))
            if n_evals >= budget.n_evals:
                break
        res = self._result(problem, X, F, n_evals, time.perf_counter() - t0, history, {})
        return res


class DENumpy(NSGA2Numpy):
    name = "DE-Numpy"
    backend = "numpy"

    def __init__(self, f_scale: float = 0.5, cr: float = 0.9, eta_m: float = 20.0) -> None:
        self.f_scale = f_scale
        self.cr = cr
        self.eta_m = eta_m

    def run(self, problem, budget: Budget, seed: int) -> object:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        n = budget.pop_size
        xl, xu = problem.spec.xl, problem.spec.xu
        X = rng.random((n, problem.spec.n_var)) * (xu - xl) + xl
        F = problem.evaluate(X)
        n_evals = n
        history: list[float] = []
        ref = problem.reference_point()
        for _ in range(budget.n_gen):
            off = de_operator(X, self.f_scale, self.cr, xl, xu)
            off = polynomial_mutation(off, xl, xu, self.eta_m, 1.0 / problem.spec.n_var)
            Fo = problem.evaluate(off)
            n_evals += off.shape[0]
            comb_X = np.vstack([X, off])
            comb_F = np.vstack([F, Fo])
            keep = _nds_select(comb_F, n, rng)
            X, F = comb_X[keep], comb_F[keep]
            history.append(hypervolume(F, ref))
            if n_evals >= budget.n_evals:
                break
        return self._result(problem, X, F, n_evals, time.perf_counter() - t0, history, {})


class AHVAMOEA(BaseOptimizer):
    """Adaptive Hypervolume-driven operator-credit Multi-objective Ensemble Adapt.

    Per generation, an upper-confidence-bound bandit selects one reproduction
    operator (SBX / DE / strong-mutation). On stagnation, an elite archive is
    re-injected (no extra evaluations) and operator credits reset to re-explore.
    """

    name = "AHVA-MOEA"
    backend = "numpy"

    def __init__(
        self,
        c: float = 1.4,
        stall: int = 8,
        eta_c: float = 15.0,
        eta_m: float = 20.0,
        f_scale: float = 0.5,
        cr: float = 0.9,
        no_restart: bool = False,
        fixed_op: str | None = None,
    ) -> None:
        self.c = c
        self.stall = stall
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.f_scale = f_scale
        self.cr = cr
        self.no_restart = no_restart
        self.fixed_op = fixed_op
        self.ops = ["sbx", "de", "pm"]

    def _reproduce(self, X, op, xl, xu, n_var, rng):
        if op == "sbx":
            off = sbx_crossover(X, xl, xu, self.eta_c, self.cr)
            return polynomial_mutation(off, xl, xu, self.eta_m, 1.0 / n_var)
        if op == "de":
            off = de_operator(X, self.f_scale, self.cr, xl, xu)
            return polynomial_mutation(off, xl, xu, self.eta_m, 1.0 / n_var)
        # pm: strong-mutation-only refresh of half the population
        idx = np.random.choice(X.shape[0], size=max(2, X.shape[0] // 2), replace=False)
        off = X.copy()
        off[idx] = polynomial_mutation(X[idx], xl, xu, self.eta_m, 0.5)
        return off

    def run(self, problem, budget: Budget, seed: int) -> object:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        n = budget.pop_size
        xl, xu = problem.spec.xl, problem.spec.xu
        n_var = problem.spec.n_var
        X = rng.random((n, n_var)) * (xu - xl) + xl
        F = problem.evaluate(X)
        n_evals = n
        ref = problem.reference_point()

        counts = {o: 1 for o in self.ops}
        sums = {o: 0.0 for o in self.ops}
        archive_X: list[np.ndarray] = []
        archive_F: list[np.ndarray] = []

        def ucb_choice() -> str:
            if self.fixed_op is not None:
                return self.fixed_op
            total = sum(counts.values())
            best, best_v = self.ops[0], -1e9
            for o in self.ops:
                mean = sums[o] / counts[o]
                bonus = self.c * math.sqrt(math.log(total + 1) / counts[o])
                v = mean + bonus
                if v > best_v:
                    best, best_v = o, v
            return best

        best_hv = hypervolume(F, ref)
        pf = problem.pareto_front(200)
        best_igd = igd_plus(F, pf)
        stall_cnt = 0
        history: list[float] = [best_hv]

        for _ in range(budget.n_gen):
            op = ucb_choice()
            counts[op] += 1
            off = self._reproduce(X, op, xl, xu, n_var, rng)
            Fo = problem.evaluate(off)
            n_evals += off.shape[0]
            comb_X = np.vstack([X, off])
            comb_F = np.vstack([F, Fo])
            keep = _nds_select(comb_F, n, rng)
            X, F = comb_X[keep], comb_F[keep]
            new_hv = hypervolume(F, ref)
            # Credit assignment uses IGD+ improvement rather than raw HV
            # improvement: once a run approaches the front, HV saturates and its
            # per-generation delta collapses to ~0, which would leave the UCB
            # bandit with no signal and force it into blind exploration. IGD+
            # keeps differentiating operators by convergence quality throughout.
            new_igd = igd_plus(F, pf)
            reward = best_igd - new_igd
            sums[op] += reward
            # archive: keep non-dominated elites (already evaluated)
            nd = fast_non_dominated_sort(F)[0]
            archive_X.append(X[nd].copy())
            archive_F.append(F[nd].copy())
            if new_hv > best_hv + 1e-4:
                best_hv = new_hv
                stall_cnt = 0
            else:
                stall_cnt += 1
            best_igd = min(best_igd, new_igd)
            history.append(new_hv)
            if n_evals >= budget.n_evals:
                break
            # stagnation restart: re-inject elite archive, reset credits
            if not self.no_restart and stall_cnt >= self.stall and archive_X:
                A = np.vstack(archive_X)
                AF = np.vstack(archive_F)
                k = min(n // 2, A.shape[0])
                # archive is already elite; take the first k members
                elites_X = A[:k]
                elites_F = AF[:k]
                keep_idx = _nds_select(F, n - k, rng)
                X = np.vstack([X[keep_idx], elites_X])
                F = np.vstack([F[keep_idx], elites_F])
                for o in self.ops:
                    counts[o] = 1
                    sums[o] = 0.0
                stall_cnt = 0

        meta = {"operators": self.ops, "restart_stall": self.stall}
        return self._result(problem, X, F, n_evals, time.perf_counter() - t0, history, meta)


__all__ = ["AHVAMOEA", "DENumpy", "NSGA2Numpy", "_nds_select"]
