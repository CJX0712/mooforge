"""Optimizer build, determinism, and offline-fallback tests."""

import numpy as np

from mooforge.core.types import Budget
from mooforge.eval.protocol import run_single
from mooforge.metrics.indicators import normalized_hv
from mooforge.optimizers.factory import (
    FLAGSHIP,
    PRIMARY_BASELINE,
    available,
    build,
    list_algorithms,
    offline_fallback_algorithms,
)

PROB = "ZDT1"
BUD = Budget(n_evals=300, pop_size=24)


def test_offline_fallback_builds_without_pymoo():
    for name in offline_fallback_algorithms():
        algo = build(name)
        assert algo is not None
        assert algo.available() is True


def test_flagship_and_baseline_build():
    assert build(FLAGSHIP) is not None
    # baseline may be pymoo-backed; only require it builds when available
    if available(PRIMARY_BASELINE):
        assert build(PRIMARY_BASELINE) is not None


def test_run_is_deterministic():
    r1 = run_single(FLAGSHIP, PROB, BUD, 0)
    r2 = run_single(FLAGSHIP, PROB, BUD, 0)
    assert r1 is not None and r2 is not None
    assert np.isclose(r1.hv, r2.hv, atol=1e-9)
    assert np.isclose(r1.igd, r2.igd, atol=1e-9)


def test_ablation_variant_is_distinct_from_base():
    # With independent RNG streams per algorithm, the fixed-SBX ablation must
    # not degenerate into the plain numpy NSGA-II run.
    base = run_single("NSGA2-Numpy", PROB, BUD, 0)
    fix = run_single(FLAGSHIP, PROB, BUD, 0, {"fixed_op": "sbx", "_label": "AHVA-FixedSBX"})
    assert base is not None and fix is not None
    # Different RNG streams => different (but valid) outcomes.
    assert not np.isclose(base.hv, fix.hv, atol=1e-9)


def test_all_available_algorithms_run_and_stay_in_range():
    for name in list_algorithms():
        if not available(name):
            continue
        res = run_single(name, PROB, BUD, 1)
        assert res is not None, f"{name} returned None"
        assert 0.0 <= res.hv <= 1.0, f"{name} hv out of [0,1]: {res.hv}"


def test_normalized_hv_bounds():
    rng = np.random.default_rng(0)
    F = rng.random((40, 2))
    pf = rng.random((60, 2))
    ref = np.array([1.2, 1.2])
    val = normalized_hv(F, pf, ref, mc_samples=2000)
    assert 0.0 <= val <= 1.0
