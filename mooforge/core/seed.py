"""Global determinism: one entry point sets every RNG the system may touch."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

DEFAULT_SEED = 42

# Optuna samplers are seeded explicitly through this module so that hyper
# parameter searches are bit-reproducible as well.
_OPTUNA_SEED: int = DEFAULT_SEED


def set_all(seed: int = DEFAULT_SEED) -> None:
    """Seed python / numpy / optuna (if importable) with a single value."""
    global _OPTUNA_SEED
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed)!r}")
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"seed out of range [0, 2**32-1]: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _OPTUNA_SEED = seed
    try:  # pragma: no cover - optuna is optional
        import optuna  # pyright: ignore[reportMissingImports]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:  # pragma: no cover  # noqa: BLE001, S110
        pass


def optuna_sampler(n_startup: int = 8, **kwargs: Any) -> Any:
    """Return a deterministic Optuna sampler (TPE), or None when unavailable."""
    try:  # pragma: no cover - optuna is optional
        import optuna  # pyright: ignore[reportMissingImports]

        return optuna.samplers.TPESampler(
            seed=_OPTUNA_SEED, n_startup_trials=n_startup, **kwargs
        )
    except Exception:  # pragma: no cover  # noqa: BLE001
        return None


def current_seed() -> int:
    return _OPTUNA_SEED


def spawn(seed: int, offset: int) -> int:
    """Derive a deterministic child seed from a parent seed and an offset."""
    return (int(seed) * 1_000_003 + int(offset) * 7919 + 12345) % (2**32 - 1)


__all__ = ["DEFAULT_SEED", "current_seed", "optuna_sampler", "set_all", "spawn"]
