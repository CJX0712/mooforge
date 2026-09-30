"""Lightweight Optuna HPO for the AHVA-MOEA flagship operator bandit.

Runs on a *separate* cheap problem/seed set from the main benchmark so the
reported numbers stay leakage-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.config import MOOForgeConfig
from ..core.seed import optuna_sampler
from ..core.types import Budget
from ..optimizers.factory import FLAGSHIP, build
from ..problems.registry import make_problem

DEFAULT_FLAGSHIP_PARAMS = {"c": 1.4, "stall": 8, "f_scale": 0.5, "cr": 0.9}


@dataclass
class HPOResult:
    params: dict
    best_value: float
    n_trials: int
    used: bool


def tune_flagship(
    config: MOOForgeConfig, hpo_problems: list[str] | None = None
) -> HPOResult:
    if not config.aggressive or config.hpo_trials < 1:
        return HPOResult(dict(DEFAULT_FLAGSHIP_PARAMS), 0.0, 0, False)
    try:  # pragma: no cover - optuna optional in offline mode
        import optuna
    except Exception:  # noqa: BLE001
        return HPOResult(dict(DEFAULT_FLAGSHIP_PARAMS), 0.0, 0, False)

    prob_names = hpo_problems or ["ZDT1", "ZDT4"]
    seeds = [7, 13, 21]
    budget = Budget(n_evals=max(600, config.n_evals // 4), pop_size=config.pop_size)
    sampler = optuna_sampler()

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "c": trial.suggest_float("c", 0.5, 2.5),
            "stall": int(trial.suggest_int("stall", 4, 16)),
            "f_scale": trial.suggest_float("f_scale", 0.3, 0.9),
            "cr": trial.suggest_float("cr", 0.7, 1.0),
        }
        vals = []
        for pn in prob_names:
            prob = make_problem(pn)
            for s in seeds:
                res = build(FLAGSHIP, **params).run(prob, budget, s)
                vals.append(res.hv if res.hv is not None else 0.0)
        return float(-float(np_mean(vals)))

    try:  # pragma: no cover
        import numpy as np

        def np_mean(x):
            return np.mean(x)

        study = optuna.create_study(
            direction="minimize", sampler=sampler if sampler else None
        )
        study.optimize(objective, n_trials=config.hpo_trials)
        best = dict(study.best_params)
        return HPOResult(best, float(study.best_value), config.hpo_trials, True)
    except Exception:  # pragma: no cover  # noqa: BLE001
        return HPOResult(dict(DEFAULT_FLAGSHIP_PARAMS), 0.0, 0, False)


__all__ = ["DEFAULT_FLAGSHIP_PARAMS", "HPOResult", "tune_flagship"]
