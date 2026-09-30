"""Quality indicators: normalised HV, IGD+, Spacing, Spread.

All are minimisation-friendly where noted. Convention for the summary table:
  * hv_norm : higher is better (relative hypervolume, in [0,1])
  * igd     : lower is better (IGD+ vs analytic front)
  * spacing : lower is better (uniformity)
  * spread  : lower is better (extent coverage)
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E401_REFERENCE, E402_EMPTY_SET, MetricError
from .hypervolume import hypervolume


def igd_plus(F: np.ndarray, pf: np.ndarray) -> float:
    """Inverted Generational Distance plus (Ishibuchi 2015). Lower=better."""
    F = np.asarray(F, float)
    pf = np.asarray(pf, float)
    if F.size == 0 or pf.size == 0:
        raise MetricError("empty set for IGD+", E402_EMPTY_SET)
    if F.shape[1] != pf.shape[1]:
        raise MetricError("objective dim mismatch in IGD+", E401_REFERENCE)
    # d+(p,s) = sqrt(sum(max(s_i-p_i,0)^2))
    diff = F[None, :, :] - pf[:, None, :]
    diff = np.clip(diff, 0.0, None)
    d = np.sqrt(np.sum(diff**2, axis=2))  # (|pf|, |F|)
    mins = d.min(axis=1)  # per PF point, nearest solution (positive part)
    return float(np.mean(mins))


def spacing(F: np.ndarray) -> float:
    """Schott's spacing: variance of neighbour distances. Lower=better."""
    F = np.asarray(F, float)
    if F.shape[0] < 2:
        return 0.0
    # pairwise Euclidean distance matrix
    diff = F[:, None, :] - F[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    d_i = dist.min(axis=1)
    d_mean = d_i.mean()
    return float(np.sqrt(np.mean((d_i - d_mean) ** 2)))


def spread(F: np.ndarray) -> float:
    """Extent coverage relative to the bounding box of the set. Lower=better."""
    F = np.asarray(F, float)
    if F.shape[0] < 2:
        return 0.0
    rng = F.max(axis=0) - F.min(axis=0)
    span = np.prod(rng) if F.shape[1] >= 1 else 0.0
    if span <= 0:
        return 0.0
    return float(span)


def normalized_hv(
    F: np.ndarray,
    pf: np.ndarray,
    ref: np.ndarray | None = None,
    mc_samples: int = 200_000,
    seed: int = 1,
) -> float:
    """HV of F divided by the reference-box volume -> in [0,1], higher=better.

    Using the reference-box volume (prod over objectives of ref_i) keeps the
    metric bounded and makes it comparable across problems, since every solver
    is measured against the same box.
    """
    F = np.asarray(F, float)
    pf = np.asarray(pf, float)
    if ref is None:
        ideal = pf.min(axis=0)
        nadir = pf.max(axis=0)
        ref = nadir + 0.1 * np.maximum(nadir - ideal, 1e-9)
    ref = np.asarray(ref, float).reshape(-1)
    if F.shape[0] == 0:
        return 0.0
    box_vol = float(np.prod(ref))
    if box_vol <= 0:
        return 0.0
    hv_f = hypervolume(F, ref, mc_samples=mc_samples, seed=seed)
    return float(hv_f / box_vol)


__all__ = ["igd_plus", "normalized_hv", "spacing", "spread"]
