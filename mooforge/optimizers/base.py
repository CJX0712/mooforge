"""Optimizer base + shared population operators (numpy, dependency-free)."""

from __future__ import annotations

import numpy as np

from ..core.types import OptimizationResult


class BaseOptimizer:
    """Common plumbing: name/backend bookkeeping and result assembly."""

    name: str = "base"
    backend: str = "numpy"

    def available(self) -> bool:
        return True

    def _result(
        self,
        problem,
        X: np.ndarray,
        F: np.ndarray,
        n_evals: int,
        elapsed: float,
        history: list[float],
        meta: dict,
    ) -> OptimizationResult:
        from ..metrics.indicators import igd_plus, normalized_hv, spacing, spread

        pf = problem.pareto_front(300)
        ref = problem.reference_point()
        hv = normalized_hv(F, pf, ref)
        igd = igd_plus(F, pf)
        sp = spacing(F)
        spr = spread(F)
        return OptimizationResult(
            problem=problem.name,
            algorithm=self.name,
            backend=self.backend,
            X=X,
            F=F,
            n_evals=n_evals,
            elapsed_sec=float(elapsed),
            hv=hv,
            igd=igd,
            spacing=sp,
            spread=spr,
            history=history,
            meta=meta,
        )


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Deb's fast non-dominated sort. Returns list of fronts (index arrays)."""
    n = F.shape[0]
    S: list[list[int]] = [[] for _ in range(n)]
    ndom = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        fp = F[p]
        for q in range(p + 1, n):
            fq = F[q]
            le_p = np.all(fp <= fq)
            lt_p = np.any(fp < fq)
            le_q = np.all(fq <= fp)
            lt_q = np.any(fq < fp)
            if le_p and lt_p:
                S[p].append(q)
                ndom[q] += 1
            elif le_q and lt_q:
                S[q].append(p)
                ndom[p] += 1
    for p in range(n):
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [np.array(f, dtype=int) for f in fronts if len(f) > 0]


def crowding_distance(F: np.ndarray, fronts: list[np.ndarray]) -> np.ndarray:
    n = F.shape[0]
    cd = np.zeros(n)
    for front in fronts:
        if len(front) == 0:
            continue
        m = F.shape[1]
        for obj in range(m):
            order = front[np.argsort(F[front, obj])]
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            fmin = F[order[0], obj]
            fmax = F[order[-1], obj]
            if fmax == fmin:
                continue
            step = (F[order[1:-1], obj] - F[order[:-2], obj]) / (fmax - fmin)
            cd[order[1:-1]] += step
    return cd


def sbx_crossover(
    X_parents: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta: float = 15.0, prob: float = 0.9
) -> np.ndarray:
    """Vectorised simulated binary crossover (whole-population pairing)."""
    n = X_parents.shape[0]
    if n < 2:
        return X_parents.copy()
    half = (n // 2) * 2
    perm = np.random.permutation(half)
    a = X_parents[perm[: half // 2]]
    b = X_parents[perm[half // 2 : half]]
    u = np.random.rand(*a.shape)
    beta = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1)),
    )
    c1 = 0.5 * ((1.0 + beta) * a + (1.0 - beta) * b)
    c2 = 0.5 * ((1.0 - beta) * a + (1.0 + beta) * b)
    swap = np.random.rand(a.shape[0]) < 0.5
    c1, c2 = np.where(swap[:, None], c2, c1), np.where(swap[:, None], c1, c2)
    do_x = np.random.rand(a.shape[0]) < prob
    c1 = np.where(do_x[:, None], c1, a)
    c2 = np.where(do_x[:, None], c2, b)
    out = np.empty_like(X_parents)
    out[perm[: half // 2]] = np.clip(c1, xl, xu)
    out[perm[half // 2 : half]] = np.clip(c2, xl, xu)
    if half < n:
        out[half:] = X_parents[half:]
    return out


def polynomial_mutation(
    X: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta: float = 20.0, prob: float = 0.1
) -> np.ndarray:
    """Vectorised polynomial mutation (per-gene mutation with prob/d)."""
    out = X.copy()
    n, d = X.shape
    r = np.random.rand(n, d)
    delta = np.where(
        r < 0.5,
        (2.0 * r) ** (1.0 / (eta + 1)) - 1.0,
        1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta + 1)),
    )
    mask = np.random.rand(n, d) < (prob / d)
    vals = np.clip(X + delta * (xu - xl) * mask, xl, xu)
    out[mask] = vals[mask]
    return out


def de_operator(
    X_pop: np.ndarray,
    F_scale: float = 0.5,
    cr: float = 0.9,
    xl: np.ndarray | None = None,
    xu: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised DE/rand/1/bin producing a trial population sized like X_pop."""
    n, d = X_pop.shape
    idx = np.random.randint(0, n, size=(n, 3))
    a = X_pop[idx[:, 0]]
    b = X_pop[idx[:, 1]]
    c = X_pop[idx[:, 2]]
    j_rand = np.random.randint(0, d, size=n)
    r = np.random.rand(n, d)
    trial = np.where((r < cr) | (np.arange(d)[None, :] == j_rand[:, None]),
                     a + F_scale * (b - c), X_pop)
    if xl is not None and xu is not None:
        trial = np.clip(trial, xl, xu)
    return trial


__all__ = [
    "BaseOptimizer",
    "crowding_distance",
    "de_operator",
    "fast_non_dominated_sort",
    "polynomial_mutation",
    "sbx_crossover",
]
