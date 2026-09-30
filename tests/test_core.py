"""Core determinism, config, and seed tests."""

import numpy as np
import pytest

from mooforge.core.config import ConfigError, MOOForgeConfig
from mooforge.core.seed import set_all, spawn


def test_set_all_is_deterministic():
    set_all(42)
    a = np.random.rand(5)
    set_all(42)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_set_all_changes_stream():
    set_all(1)
    a = np.random.rand(3)
    set_all(2)
    b = np.random.rand(3)
    assert not np.allclose(a, b)


def test_spawn_is_deterministic_and_in_range():
    s = spawn(42, 123)
    assert s == spawn(42, 123)
    assert spawn(99, 7) == spawn(99, 7)
    assert 0 <= s < 2**32 - 1


def test_spawn_distinct_for_distinct_inputs():
    assert spawn(42, 1) != spawn(42, 2)
    assert spawn(1, 5) != spawn(2, 5)


def test_config_validation_rejects_bad_budget():
    with pytest.raises(ConfigError):
        MOOForgeConfig(n_evals=50)
    with pytest.raises(ConfigError):
        MOOForgeConfig(pop_size=2)
    with pytest.raises(ConfigError):
        MOOForgeConfig(n_seeds=0)


def test_config_defaults_are_sane():
    cfg = MOOForgeConfig()
    assert cfg.seed == 42
    assert cfg.n_evals == 2400
    assert cfg.pop_size == 48
    assert cfg.n_seeds == 3
