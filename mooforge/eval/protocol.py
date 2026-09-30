"""Benchmark protocol: run a single optimizer/problem/seed and aggregate."""

from __future__ import annotations

import hashlib

import numpy as np

from ..core.errors import E300_OPTIMIZER, OptimizerError
from ..core.seed import spawn
from ..core.types import Budget, OptimizationResult
from ..optimizers.factory import available, build
from ..problems.registry import make_problem


def _stable_hash(s: str) -> int:
    """Process-stable hash (PYTHONHASHSEED-independent) for reproducible seeds."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 1_000_003


def run_single(
    algo_name: str,
    problem_name: str,
    budget: Budget,
    seed: int,
    params: dict | None = None,
) -> OptimizationResult | None:
    """Run one algorithm on one problem with one seed.

    Returns None when the backend is unavailable (so the benchmark can skip and
    report, instead of failing the whole run).
    """
    if not available(algo_name):
        return None
    try:
        params = dict(params or {})
        # `_label` is a pipeline-internal tag (e.g. ablation variant name) and is
        # never a valid optimizer constructor argument; strip it before build.
        label = params.pop("_label", algo_name)
        algo = build(algo_name, **params)
        problem = make_problem(problem_name)
        # Derive an independent, reproducible RNG stream per (algorithm, problem)
        # so that different solvers are not compared on the same random draw and
        # ablation variants do not degenerate into their base counterparts.
        stream_key = f"{algo_name}|{label}|{problem_name}"
        return algo.run(problem, budget, spawn(seed, _stable_hash(stream_key)))
    except OptimizerError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise OptimizerError(f"{algo_name} on {problem_name}: {exc}", E300_OPTIMIZER) from exc


def aggregate(records: list[dict]) -> dict:
    """Aggregate per (algorithm, problem): mean +- std for hv / igd."""
    out: dict = {}
    for rec in records:
        key = (rec["algorithm"], rec["problem"])
        out.setdefault(key, {"hv": [], "igd": [], "spacing": [], "spread": []})
        if rec.get("hv") is not None:
            out[key]["hv"].append(rec["hv"])
        if rec.get("igd") is not None:
            out[key]["igd"].append(rec["igd"])
        if rec.get("spacing") is not None:
            out[key]["spacing"].append(rec["spacing"])
        if rec.get("spread") is not None:
            out[key]["spread"].append(rec["spread"])
    summary: dict = {}
    for key, vals in out.items():
        summary[f"{key[0]}::{key[1]}"] = {
            "hv_mean": float(np.mean(vals["hv"])) if vals["hv"] else None,
            "hv_std": float(np.std(vals["hv"])) if vals["hv"] else None,
            "igd_mean": float(np.mean(vals["igd"])) if vals["igd"] else None,
            "igd_std": float(np.std(vals["igd"])) if vals["igd"] else None,
            "spacing_mean": float(np.mean(vals["spacing"])) if vals["spacing"] else None,
            "spread_mean": float(np.mean(vals["spread"])) if vals["spread"] else None,
            "n": len(vals["hv"]),
        }
    return summary


def significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> tuple[bool, float]:
    """Simple significance gate: |mean_a - mean_b| > 0.5*(std_a + std_b)."""
    margin = 0.5 * (std_a + std_b)
    return (abs(mean_a - mean_b) > margin), float(abs(mean_a - mean_b))


__all__ = ["aggregate", "run_single", "significant"]
