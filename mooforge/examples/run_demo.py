"""End-to-end demo: run the full benchmark, write benchmark.json, print summary.

Deterministic: re-running with the same config reproduces the core metrics
bit-for-bit (unless a backend is unavailable and the run is skipped).
"""

from __future__ import annotations

import os
import sys

# make the package importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mooforge.cli import main

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark.json")
    rc = main(["run", "--out", os.path.abspath(out)])
    sys.exit(rc)
