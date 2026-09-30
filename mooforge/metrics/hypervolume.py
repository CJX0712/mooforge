"""Hypervolume (HV): exact for 1-3 objectives, seeded Monte-Carlo fallback.

All problems are minimisation. HV is the volume of the union of axis-aligned
boxes [f_i, reference] dominated by the solution set.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E401_REFERENCE, E402_EMPTY_SET, MetricError


def _to_points(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise MetricError("F must be 2D (n_points, n_obj)", E402_EMPTY_SET)
    if F.shape[0] == 0 or F.shape[1] == 0:
        raise MetricError("empty solution set", E402_EMPTY_SET)
    if not np.all(np.isfinite(F)):
        raise MetricError("non-finite objective values", E402_EMPTY_SET)
    return F


def _normalize(F: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = _to_points(F)
    ref = np.asarray(ref, dtype=float).reshape(-1)
    if ref.shape[0] != F.shape[1]:
        raise MetricError(
            f"reference dim {ref.shape[0]} != n_obj {F.shape[1]}", E401_REFERENCE
        )
    # Keep only points that are actually dominated by ref.
    mask = np.all(F <= ref, axis=1)
    if not np.any(mask):
        return np.empty((0, F.shape[1])), ref
    return F[mask], ref


def _hv_2d(F: np.ndarray, ref: np.ndarray) -> float:
    # Pareto surface against objective 1 (sort ascending), keep non-dominated.
    order = np.argsort(F[:, 0])
    fx, fy = F[order, 0], F[order, 1]
    sx, sy = [fx[0]], [fy[0]]
    for x, y in zip(fx[1:], fy[1:]):
        if y < sy[-1]:
            sx.append(x)
            sy.append(y)
    sx = np.array(sx)
    sy = np.array(sy)
    nxt = np.append(sx[1:], ref[0])
    vol = float(np.sum((nxt - sx) * (ref[1] - sy)))
    return max(vol, 0.0)


def _hv_rec(P: np.ndarray, ref: np.ndarray, depth: int) -> float:
    n, m = P.shape
    if m == 1:
        return max(0.0, ref[0] - P[:, 0].min())
    if m == 2:
        return _hv_2d(P, ref)
    # Slicing along the first objective (ascending).
    order = np.argsort(P[:, 0])
    Ps = P[order]
    total = 0.0
    for i in range(n):
        width = (Ps[i + 1, 0] if i + 1 < n else ref[0]) - Ps[i, 0]
        if width <= 0:
            continue
        # Remaining objectives of points j<=i that are still dominated by Ps[i,1:].
        sub = Ps[: i + 1, 1:]
        dom = np.all(sub <= Ps[i, 1:], axis=1)
        sub = sub[dom]
        if sub.shape[0] == 0:
            continue
        total += width * _hv_rec(sub, ref[1:], depth + 1)
    return total


def hypervolume(F: np.ndarray, ref: np.ndarray, mc_samples: int = 200_000, seed: int = 1) -> float:
    """Exact HV for d<=3; seeded Monte-Carlo estimate for d>=4."""
    F, ref = _normalize(F, ref)
    if F.shape[0] == 0:
        return 0.0
    m = F.shape[1]
    if m <= 3:
        return _hv_rec(F, ref.copy(), 0)
    # Monte-Carlo over the dominated box, via the ideal point for tightness.
    ideal = F.min(axis=0)
    lo = np.minimum(ideal, np.zeros_like(ideal))
    hi = ref
    rng = np.random.default_rng(seed)
    samples = rng.random((mc_samples, m)) * (hi - lo) + lo
    dominated = np.all(samples[:, None, :] >= F[None, :, :], axis=2)
    inside = dominated.any(axis=1)
    vol_box = float(np.prod(hi - lo))
    return vol_box * float(inside.mean())


def reference_hypervolume(pf: np.ndarray, ref: np.ndarray, mc_samples: int = 200_000, seed: int = 2) -> float:
    """HV of the analytic Pareto front (used to normalise)."""
    return hypervolume(pf, ref, mc_samples=mc_samples, seed=seed)


__all__ = ["hypervolume", "reference_hypervolume"]
