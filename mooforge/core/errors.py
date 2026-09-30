"""MOOForge error taxonomy (E100~E500).

Error codes are stable and documented; every code maps to one failure class so
that the CLI can report actionable messages instead of raw tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorCode:
    code: str
    message: str


E100_CONFIG = ErrorCode("E100", "invalid configuration")
E101_SEED = ErrorCode("E101", "invalid random seed")
E102_BOUNDS = ErrorCode("E102", "invalid variable bounds")

E200_PROBLEM = ErrorCode("E200", "problem definition error")
E201_UNKNOWN_PROBLEM = ErrorCode("E201", "unknown problem name")
E202_EVAL_SHAPE = ErrorCode("E202", "objective evaluation returned wrong shape")

E300_OPTIMIZER = ErrorCode("E300", "optimizer error")
E301_UNKNOWN_ALGORITHM = ErrorCode("E301", "unknown algorithm name")
E302_BACKEND_MISSING = ErrorCode("E302", "required backend is unavailable")
E303_NO_RESULT = ErrorCode("E303", "optimizer produced no feasible result")

E400_METRIC = ErrorCode("E400", "metric computation error")
E401_REFERENCE = ErrorCode("E401", "reference point / front is invalid")
E402_EMPTY_SET = ErrorCode("E402", "empty or non-finite objective set")

E500_IO = ErrorCode("E500", "input/output error")


class MOOForgeError(Exception):
    """Base class carrying a stable error code."""

    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        msg = f"[{code.code}] {code.message}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)


class ConfigError(MOOForgeError):
    def __init__(self, detail: str = "", code: ErrorCode = E100_CONFIG) -> None:
        super().__init__(code, detail)


class ProblemError(MOOForgeError):
    def __init__(self, detail: str = "", code: ErrorCode = E200_PROBLEM) -> None:
        super().__init__(code, detail)


class OptimizerError(MOOForgeError):
    def __init__(self, detail: str = "", code: ErrorCode = E300_OPTIMIZER) -> None:
        super().__init__(code, detail)


class MetricError(MOOForgeError):
    def __init__(self, detail: str = "", code: ErrorCode = E400_METRIC) -> None:
        super().__init__(code, detail)


__all__ = [
    "E100_CONFIG",
    "E101_SEED",
    "E102_BOUNDS",
    "E200_PROBLEM",
    "E201_UNKNOWN_PROBLEM",
    "E202_EVAL_SHAPE",
    "E300_OPTIMIZER",
    "E301_UNKNOWN_ALGORITHM",
    "E302_BACKEND_MISSING",
    "E303_NO_RESULT",
    "E400_METRIC",
    "E401_REFERENCE",
    "E402_EMPTY_SET",
    "E500_IO",
    "ConfigError",
    "ErrorCode",
    "MOOForgeError",
    "MetricError",
    "OptimizerError",
    "ProblemError",
]
