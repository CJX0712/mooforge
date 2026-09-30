"""Analytic multi-objective test problems (ZDT / DTLZ families).

Every problem exposes its *analytic* Pareto front, so quality indicators
(IGD+, normalised hypervolume) can be computed against ground truth instead of
against an empirically merged front. Each problem also declares a reference
point that covers the worst local optimum, so HV stays comparable (in [0,1])
even on multimodal landscapes.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E200_PROBLEM, ProblemError
from ..core.types import ProblemSpec


def _non_dominated(F: np.ndarray) -> np.ndarray:
    """Return the subset of F that is not strictly dominated by any other row."""
    if F.size == 0:
        return F
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        leq = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        dom = leq & lt
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return F[keep]


class BaseProblem:
    """Shared plumbing: bounds, evaluation counting, reference point logic."""

    def __init__(self, n_var: int, n_obj: int, difficulty: str = "medium") -> None:
        xl = np.zeros(n_var)
        xu = np.ones(n_var)
        self.spec = ProblemSpec(
            name=type(self).__name__,
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
            difficulty=difficulty,
        )
        self._n_calls = 0

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def n_evals(self) -> int:
        return self._n_calls

    def reset_counter(self) -> None:
        self._n_calls = 0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.spec.n_var:
            raise ProblemError(
                f"expected {self.spec.n_var} variables, got {X.shape[1]}"
            )
        F = self._evaluate(X)
        F = np.atleast_2d(np.asarray(F, dtype=float))
        if F.shape != (X.shape[0], self.spec.n_obj):
            raise ProblemError(
                f"objective shape {F.shape} != {(X.shape[0], self.spec.n_obj)}",
                E200_PROBLEM,
            )
        self._n_calls += X.shape[0]
        return F

    def _evaluate(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def pareto_front(self, n_points: int = 300) -> np.ndarray:
        return self._pareto_front(n_points)

    def _pareto_front(self, n_points: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reference_point(self) -> np.ndarray:
        if self.spec.ref_point is not None:
            return self.spec.ref_point.astype(float)
        pf = self.pareto_front(400)
        ideal = pf.min(axis=0)
        nadir = pf.max(axis=0)
        return nadir + 0.1 * np.maximum(nadir - ideal, 1e-9)

    def nadir(self) -> np.ndarray:
        return self.pareto_front(400).max(axis=0)

    def ideal(self) -> np.ndarray:
        return self.pareto_front(400).min(axis=0)


class ZDT1(BaseProblem):
    """Convex, continuous, unimodal front: f2 = 1 - sqrt(f1)."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, 2, difficulty="easy")
        self.spec.ref_point = np.array([1.0, 11.0])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1) if X.shape[1] > 1 else np.ones(len(X))
        h = 1.0 - np.sqrt(np.clip(f1 / g, 0.0, None))
        return np.column_stack([f1, g * h])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT2(BaseProblem):
    """Non-convex continuous front: f2 = 1 - f1^2."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, 2, difficulty="easy")
        self.spec.ref_point = np.array([1.0, 11.0])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1) if X.shape[1] > 1 else np.ones(len(X))
        h = 1.0 - (np.clip(f1 / g, 0.0, None) ** 2)
        return np.column_stack([f1, g * h])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - f1**2])


class ZDT3(BaseProblem):
    """Disconnected front (5 disjoint segments) - the classic NSGA-II trap."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, 2, difficulty="hard")
        self.spec.ref_point = np.array([1.0, 11.0])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        f1 = X[:, 0]
        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1) if X.shape[1] > 1 else np.ones(len(X))
        r = np.clip(f1 / g, 0.0, None)
        h = 1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * f1)
        return np.column_stack([f1, g * h])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, max(n_points * 40, 4000))
        r = f1
        f2 = 1.0 - np.sqrt(r) - r * np.sin(10.0 * np.pi * f1)
        cand = np.column_stack([f1, f2])
        front = _non_dominated(cand)
        front = front[np.argsort(front[:, 0])]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front


class ZDT4(BaseProblem):
    """Highly multimodal landscape (21^9 local Pareto-optima)."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, 2, difficulty="hard")
        self.spec.ref_point = np.array([1.0, 450.0])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        f1 = X[:, 0]
        tail = X[:, 1:] if X.shape[1] > 1 else np.zeros((len(X), 0))
        g = 1.0 + 10.0 * (X.shape[1] - 1) + np.sum(tail**2 - 10.0 * np.cos(4.0 * np.pi * tail), axis=1)
        h = 1.0 - np.sqrt(np.clip(f1 / g, 0.0, None))
        return np.column_stack([f1, g * h])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])


class ZDT6(BaseProblem):
    """Non-uniformly sampled, thin front: f2 = 1 - f1^2 on f1 in [~0.28, 1]."""

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, 2, difficulty="medium")
        self.spec.ref_point = np.array([1.0, 11.0])

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4.0 * X[:, 0]) * (np.sin(6.0 * np.pi * X[:, 0]) ** 6)
        g = 1.0 + 9.0 * (np.mean(X[:, 1:], axis=1) ** 0.25) if X.shape[1] > 1 else np.ones(len(X))
        h = 1.0 - (np.clip(f1 / g, 0.0, None) ** 2)
        return np.column_stack([f1, g * h])

    def _pareto_front(self, n_points: int) -> np.ndarray:
        x1 = np.linspace(0.0, 1.0, max(n_points * 20, 2000))
        f1 = 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
        f2 = 1.0 - f1**2
        cand = np.column_stack([f1, f2])
        front = _non_dominated(cand)
        front = front[np.argsort(front[:, 0])]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front


class DTLZ1(BaseProblem):
    """Linear PF (sum f = 0.5) buried in a multimodal (1e5 local) landscape."""

    def __init__(self, n_obj: int = 3, k: int = 4) -> None:
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, difficulty="hard")
        self._k = k
        self.spec.ref_point = np.full(n_obj, 900.0)

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        m = self.spec.n_obj
        k = self._k
        xm = X[:, m - 1 :]
        g = 100.0 * (
            k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5)), axis=1)
        )
        F = []
        for i in range(m):
            head = X[:, : m - 1 - i] if (m - 1 - i) > 0 else np.zeros((len(X), 0))
            prod = np.prod(head, axis=1) if head.shape[1] else np.ones(len(X))
            if i > 0:
                prod = prod * (1.0 - X[:, m - 1 - i])
            F.append(0.5 * (1.0 + g) * prod)
        return np.column_stack(F)

    def _pareto_front(self, n_points: int) -> np.ndarray:
        m = self.spec.n_obj
        rng = np.random.default_rng(20240921)
        if m == 2:
            f1 = np.linspace(0.0, 0.5, n_points)
            return np.column_stack([f1, 0.5 - f1])
        w = rng.dirichlet(np.ones(m), size=max(n_points, 500))
        pf = 0.5 * w
        pf = pf[np.argsort(pf[:, 0])]
        return pf[:n_points]


class DTLZ2(BaseProblem):
    """Spherical PF (sum f^2 = 1), scalable to many objectives."""

    def __init__(self, n_obj: int = 3, k: int = 4) -> None:
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, difficulty="medium")
        self._k = k
        self.spec.ref_point = np.full(n_obj, 2.0)

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        m = self.spec.n_obj
        xm = X[:, m - 1 :]
        g = np.sum((xm - 0.5) ** 2, axis=1)
        F = []
        for i in range(m):
            head = X[:, : m - 1 - i] if (m - 1 - i) > 0 else np.zeros((len(X), 0))
            prod = np.prod(np.cos(head * np.pi / 2.0), axis=1) if head.shape[1] else np.ones(len(X))
            if i > 0:
                prod = prod * np.sin(X[:, m - 1 - i] * np.pi / 2.0)
            F.append((1.0 + g) * prod)
        return np.column_stack(F)

    def _pareto_front(self, n_points: int) -> np.ndarray:
        m = self.spec.n_obj
        rng = np.random.default_rng(31415)
        if m == 2:
            a = np.linspace(0.0, np.pi / 2.0, n_points)
            return np.column_stack([np.cos(a), np.sin(a)])
        w = rng.normal(size=(max(n_points, 800), m))
        w = np.abs(w) / np.linalg.norm(w, axis=1, keepdims=True)
        w = w[np.argsort(w[:, 0])]
        return w[:n_points]


class DTLZ3(BaseProblem):
    """DTLZ2 front with a Rastrigin-style multimodal distance function."""

    def __init__(self, n_obj: int = 3, k: int = 4) -> None:
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, difficulty="hard")
        self._k = k
        self.spec.ref_point = np.full(n_obj, 2.0)

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        m = self.spec.n_obj
        xm = X[:, m - 1 :]
        g = 100.0 * (
            self._k + np.sum((xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (xm - 0.5)), axis=1)
        )
        F = []
        for i in range(m):
            head = X[:, : m - 1 - i] if (m - 1 - i) > 0 else np.zeros((len(X), 0))
            prod = np.prod(np.cos(head * np.pi / 2.0), axis=1) if head.shape[1] else np.ones(len(X))
            if i > 0:
                prod = prod * np.sin(X[:, m - 1 - i] * np.pi / 2.0)
            F.append((1.0 + g) * prod)
        return np.column_stack(F)

    def _pareto_front(self, n_points: int) -> np.ndarray:
        return DTLZ2(self.spec.n_obj, self._k)._pareto_front(n_points)


class DTLZ7(BaseProblem):
    """Degenerate, disconnected PF with 2^m disjoint regions."""

    def __init__(self, n_obj: int = 3, k: int = 4) -> None:
        n_var = n_obj + k - 1
        super().__init__(n_var, n_obj, difficulty="hard")
        self._k = k
        rp = np.ones(n_obj)
        rp[-1] = 10.0 * n_obj
        self.spec.ref_point = rp

    def _evaluate(self, X: np.ndarray) -> np.ndarray:
        m = self.spec.n_obj
        xm = X[:, m - 1 :]
        g = 1.0 + 9.0 * np.mean(xm, axis=1)
        F = [X[:, i] for i in range(m - 1)]
        tail = X[:, m - 1 :]
        frac = np.sum(
            tail / (1.0 + g[:, None]) * (1.0 + np.sin(3.0 * np.pi * tail)),
            axis=1,
        )
        h = m - frac
        F.append((1.0 + g) * h)
        return np.column_stack(F)

    def _pareto_front(self, n_points: int) -> np.ndarray:
        m = self.spec.n_obj
        rng = np.random.default_rng(777)
        samples = max(n_points * 60, 6000)
        tail = rng.random((samples, self._k))
        frac = np.sum(tail * (1.0 + np.sin(3.0 * np.pi * tail)), axis=1)
        fm = m - frac
        head = np.zeros((samples, m - 1))
        cand = np.column_stack([head, fm])
        front = _non_dominated(cand)
        front = front[np.argsort(front[:, 0])]
        if len(front) > n_points:
            idx = np.linspace(0, len(front) - 1, n_points).astype(int)
            front = front[idx]
        return front


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
    "_non_dominated",
]
