"""Configuration with ENV_MOOFORGE_* overrides and schema validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from .errors import ConfigError

ENV_PREFIX = "MOOFORGE_"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be an int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class MOOForgeConfig:
    """Runtime configuration. Every field may be overridden by env vars."""

    seed: int = 42
    n_evals: int = 2400
    pop_size: int = 48
    n_seeds: int = 3
    ablation_seeds: int = 2
    hpo_trials: int = 12
    hv_mc_samples: int = 200_000
    aggressive: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"seed must be >= 0, got {self.seed}")
        if self.n_evals < 100:
            raise ConfigError(f"n_evals too small: {self.n_evals}")
        if self.pop_size < 4:
            raise ConfigError(f"pop_size too small: {self.pop_size}")
        if self.n_seeds < 1:
            raise ConfigError(f"n_seeds must be >= 1, got {self.n_seeds}")
        if self.hpo_trials < 1:
            raise ConfigError(f"hpo_trials must be >= 1, got {self.hpo_trials}")

    @classmethod
    def from_env(cls) -> MOOForgeConfig:
        base = cls()
        return replace(
            base,
            seed=_env_int("SEED", base.seed),
            n_evals=_env_int("N_EVALS", base.n_evals),
            pop_size=_env_int("POP_SIZE", base.pop_size),
            n_seeds=_env_int("N_SEEDS", base.n_seeds),
            ablation_seeds=_env_int("ABLATION_SEEDS", base.ablation_seeds),
            hpo_trials=_env_int("HPO_TRIALS", base.hpo_trials),
            hv_mc_samples=_env_int("HV_MC_SAMPLES", base.hv_mc_samples),
            aggressive=os.environ.get(ENV_PREFIX + "AGGRESSIVE", "0") == "1",
        )


DEFAULT_CONFIG = MOOForgeConfig()

__all__ = ["DEFAULT_CONFIG", "ENV_PREFIX", "MOOForgeConfig"]
